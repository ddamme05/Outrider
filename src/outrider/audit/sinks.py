# Sink Protocols for agent-node-emitted audit events.
"""Sink Protocols for agent-node-emitted audit events.

Nodes consume sink Protocols defined here rather than touching DB code
directly — this keeps `nodes-receive-deps-via-closure` honest (real sinks
inject at graph-build time, test sinks inject at fixture-setup time) and
keeps audit-table writes out of node call sites.

V1 ships four sinks from this module: `PhaseEventSink` for
`ReviewPhaseEvent` (per `phase-events-bound-work`), `FileExaminationSink`
for `FileExaminationEvent` (per intake + analyze per-file outcomes),
`AnalyzeEventSink` bundling the four analyze-emitted event types
(`FindingEvent`, `FindingProposalRejectedEvent`,
`AnalyzeResponseRejectedEvent`, `AnalyzeCompletedEvent`), and
`PublishEventSink` bundling the four publish-emitted event types
(`PublishRoutingEvent`, `PublishEligibilityEvent`, `PublishAttemptEvent`,
`PublishEvent`) per DECISIONS.md #023 routing-vs-eligibility decoupling.
`LLMCallEvent` emission lives inside `LLMProvider.complete()` and uses
the sibling `LLMExchangePersister` Protocol in `llm/base.py` — no node
code emits `LLMCallEvent` directly.

The durable `AuditPersister` in `outrider.audit.persister` implements
ALL FIVE (`PhaseEventSink` + `FileExaminationSink` + `AnalyzeEventSink`
+ `PublishEventSink` + `LLMExchangePersister`) from one body, sharing
DB transaction lifecycle and session-per-call discipline. Test-only
no-op implementations (`NoOpPersister`, `RecordingPhaseEventSink`,
`RecordingPublishEventSink`) live in `tests/conftest.py` for fixtures
that don't need durable persistence.
"""

from typing import Protocol, runtime_checkable
from uuid import UUID

from outrider.audit.events import (
    AnalyzeCompletedEvent,
    AnalyzeResponseRejectedEvent,
    FileExaminationEvent,
    FindingEvent,
    FindingProposalRejectedEvent,
    PublishAttemptEvent,
    PublishEligibilityEvent,
    PublishEvent,
    PublishRoutingEvent,
    ReviewPhaseEvent,
)


@runtime_checkable
class PhaseEventSink(Protocol):
    """Sink for ReviewPhaseEvent emissions per `phase-events-bound-work`.

    Nodes call `emit_phase()` at entry (marker='start') and on success-exit
    (marker='end') so replay has causal barriers. Real implementations
    write to the audit_events table; test implementations record to a list
    for assertion.

    Production / durable implementations MUST:
      - Be idempotent on `(review_id, phase_id, marker)`. A future
        checkpoint replay or retry can re-emit the same start/end pair;
        the audit row must not duplicate. (`phase_id` is UUID4 —
        collision-free in practice; the node pre-mints via `str(uuid4())`.)
      - Be safe under concurrent invocations. V1.5's parallel-analyze
        fan-out will emit per-file phase pairs concurrently from multiple
        worker tasks; the sink must serialize or per-task its DB writes
        (mirrors `LLMExchangePersister`'s "fresh AsyncSession per call"
        rule).
      - Either persist the event before returning, OR raise. Silent drop
        is never acceptable — `phase-events-bound-work` requires the row
        to land, and a sink that returns success-without-persistence is
        the failure mode the structural `isinstance` gate at `build_graph`
        construction time cannot catch alone.

    Test / in-memory recording implementations (e.g., the
    `RecordingPhaseEventSink` fixture in `tests/conftest.py`) are
    DELIBERATELY exempt from the idempotency rule: they record every
    emission to a list for assertion. Tests that need to verify start/end
    pairs must check the recorded list themselves; making the recorder
    dedupe would silently swallow legitimate test signals about
    double-emissions. Concurrency-safety likewise relaxes for recorders
    that target single-test fixtures.

    Same shape as `LLMExchangePersister` but for phase events only —
    `LLMCallEvent` emission stays inside `LLMProvider.complete()`. The
    durable `AuditPersister` in `outrider.audit.persister` implements
    `PhaseEventSink` alongside `FileExaminationSink`, `AnalyzeEventSink`,
    and `LLMExchangePersister` from one class.

    `@runtime_checkable` matches the `LLMExchangePersister` precedent and
    enables `build_graph` to reject sinks lacking the `emit_phase` member
    at construction time via `isinstance(...)`. Note: runtime-checkable
    Protocols verify MEMBER PRESENCE only (per PEP 544) — they don't
    validate signature, async-vs-sync nature, or types. Wrong-signature
    `emit_phase` still surfaces at the first emission call; mypy strict
    mode is the write-time gate for signature shape.
    """

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        """Persist a single phase-boundary marker; raise on persistence failure."""
        ...


@runtime_checkable
class FileExaminationSink(Protocol):
    """Sink for FileExaminationEvent emissions per intake's content-fetch path.

    Intake emits one `FileExaminationEvent` per file fetched (`parse_status`
    in `clean` / `degraded` / `failed` / `skipped`). The cross-field rule
    that `skip_reason` is non-None iff `parse_status="skipped"` is enforced
    by the event model itself (per `DECISIONS.md#018`) — the sink only
    persists; it does not re-validate.

    Same shape and discipline as `PhaseEventSink`:
      - Idempotent on `event.event_id`. Re-emission of an identical event
        (from a retry or checkpoint replay) must not duplicate the row.
      - Safe under concurrent invocation. Intake's phase-2 content fan-out
        uses `asyncio.TaskGroup` under a semaphore; multiple worker
        coroutines may emit concurrently from one node invocation. The
        durable sink serializes per call (fresh `AsyncSession` per
        emission, matching the `emit_phase` precedent at
        `persister.py:1170`).
      - Persist before returning, OR raise. Silent drop is never acceptable
        — `phase-events-bound-work`'s sibling discipline applies here:
        intake's `FileExaminationEvent` is the structural-evidence row that
        proves a file was actually examined; losing it silently breaks
        replay equivalence.

    The durable `AuditPersister` implements this Protocol alongside
    `PhaseEventSink`, `AnalyzeEventSink`, `PublishEventSink`, and
    `LLMExchangePersister` — one class, one transaction-lifecycle
    discipline, five sinks. Test fixtures may record to a list or
    persist directly per the same recorder-vs-durable split documented
    on `PhaseEventSink`.

    `@runtime_checkable` matches the `PhaseEventSink` precedent and enables
    `build_graph` to reject sinks lacking the `emit_file_examination` member
    at construction time via `isinstance(...)`. PEP 544 caveat applies:
    member-presence only, not signature shape — wrong-signature
    `emit_file_examination` still surfaces at first emission. mypy strict
    is the write-time gate for signature shape.
    """

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        """Persist a single file-examination event; raise on persistence failure."""
        ...


@runtime_checkable
class AnalyzeEventSink(Protocol):
    """Sink for the four audit event types the analyze node emits.

    The analyze-node spec (`specs/2026-05-19-analyze-node.md` §7) bundles
    four event types under one Protocol rather than four separate sinks
    because:

    - All four are emitted by ONE node body in one transaction window;
      the durable `AuditPersister` implements them under one session
      lifecycle.
    - The node body's local-bookkeeping counter discipline (per
      `_enforce_proposal_accounting` on `AnalyzeCompletedEvent`) treats
      these four events as cardinal-related: one
      `AnalyzeCompletedEvent` per pass, N `FindingEvent`s + M
      `FindingProposalRejectedEvent`s + at-most-one
      `AnalyzeResponseRejectedEvent` per per-file LLM call.
    - Four separate kwargs on the analyze function signature would
      crowd the deps surface and invite test-time mock proliferation;
      one sink keeps the test setup focused.

    `LLMCallEvent` is NOT here — the provider's
    `LLMExchangePersister` emits it autonomously inside
    `LLMProvider.complete()`.

    Production / durable implementations MUST:
      - Be idempotent on `event.event_id`. Retry / checkpoint replay
        must not duplicate rows.
      - Be safe under concurrent invocations. V1.5's parallel-analyze
        fan-out emits per-file events concurrently.
      - Persist before returning, OR raise. Silent drop is never
        acceptable.

    Test recorders (e.g., `RecordingAnalyzeEventSink`) record every
    emission into per-type lists for assertion; they are deliberately
    exempt from the idempotency rule (so double-emit bugs surface in
    tests rather than being silently deduped).

    `@runtime_checkable` matches the sibling-Protocol precedent —
    `build_graph` can reject sinks lacking any of the four `emit_*`
    members at construction time. PEP 544 caveat: member-presence
    only, not signature shape; mypy strict is the write-time gate.
    """

    async def emit_finding(self, event: FindingEvent) -> None:
        """Persist a `FindingEvent` for an admitted `ReviewFinding`."""
        ...

    async def emit_finding_proposal_rejected(self, event: FindingProposalRejectedEvent) -> None:
        """Persist a per-proposal admission-failure event."""
        ...

    async def emit_analyze_response_rejected(self, event: AnalyzeResponseRejectedEvent) -> None:
        """Persist a per-response parse-failure event."""
        ...

    async def emit_analyze_completed(self, event: AnalyzeCompletedEvent) -> None:
        """Persist the per-pass aggregate event."""
        ...


@runtime_checkable
class PublishEventSink(Protocol):
    """Sink for the four publish-node audit event types.

    Per DECISIONS.md #023 (publish routing and eligibility are separate
    decisions, not one combined gate): the publish node emits one
    `PublishRoutingEvent` per finding (coordinates-derived destination),
    one `PublishEligibilityEvent` per finding (policy-derived
    materialization gate), at most one `PublishAttemptEvent` per
    `publisher.create_review` attempt (terminal GitHub-call outcome),
    and at most one `PublishEvent` per logical publication (success-path
    review-level summary, including external-record recovery).

    Production / durable implementations MUST:
      - Be idempotent on `event_id`. LangGraph checkpoint replay can
        re-emit the same event; the persister handles dedup via the
        `audit_events.event_id` PK + payload-equality check (raises
        `AuditPersisterIdempotencyConflict` on mismatch). Consumer-side
        replay-equivalence dedup keys off the canonical
        `decision_content_hash` / `attempt_content_hash` carried by the
        events themselves.
      - Be concurrent-safe across reviews (one persister per process,
        one session per emit call). V1 publish is per-finding sequential,
        but the same persister instance services multiple reviews.

    Test / recording implementations capture each event for assertion;
    `RecordingPublishEventSink` in `tests/conftest.py` is the canonical
    test double.
    """

    async def emit_publish_routing(self, event: PublishRoutingEvent) -> None:
        """Persist a `PublishRoutingEvent` for the per-finding routing decision.

        Fires for EVERY finding processed regardless of eligibility — the
        audit trail records what coordinates classified, even when the
        eligibility gate later withholds materialization.
        """
        ...

    async def emit_publish_eligibility(self, event: PublishEligibilityEvent) -> None:
        """Persist a `PublishEligibilityEvent` for the per-finding policy gate.

        Fires alongside the routing event under the interleaved
        per-finding loop. Carries `eligibility` (`eligible`/`withheld`)
        and `policy_version` for severity-versioned replay.
        """
        ...

    async def emit_publish_attempt(self, event: PublishAttemptEvent) -> None:
        """Persist a `PublishAttemptEvent` for the per-attempt GitHub-call outcome.

        Single emission per attempt, AFTER the GitHub call resolves
        (no in_flight pre-call emission — would conflict with
        `audit-events-append-only`). On `failed` outcome, `failure_class`
        carries the exception class name (bounded at 128 chars per
        DECISIONS.md #023 append-only contract + the schema-layer
        defense against attacker-influenced 422 error strings).
        """
        ...

    async def emit_publish_result(self, event: PublishEvent) -> None:
        """Persist a `PublishEvent` for the success-path review-level summary.

        Named `emit_publish_result` (NOT `emit_publish_event`) to avoid
        confusion with the four other event types this sink emits — the
        canonical `PublishEvent` IS the publish-level result row,
        carrying `github_review_id` + `comments_posted` + `review_status`.
        Not emitted on `failed` / `no_op_empty` / `idempotently_skipped*`
        paths.
        """
        ...

    async def query_prior_publish_event(self, review_id: UUID) -> PublishEvent | None:
        """Return the most-recent prior `PublishEvent` for `review_id`,
        or `None` if no prior event was emitted.

        Backs the V1 publish node's intra-Outrider idempotency check per
        FUP-064: a same-`review_id` redispatch (e.g., dispatcher re-fires
        the webhook after agent crash + restart) hits this query BEFORE
        the GitHub call. On hit, the publish node emits
        `PublishAttemptEvent(outcome=idempotently_skipped)` and returns
        `PublishResult.skipped()` — no GitHub round-trip burned.

        Multi-row semantics (replay re-emission divergence): if multiple
        `PublishEvent` rows exist for `review_id`, return the most-recent
        by `timestamp`. The persister is append-only; consumer-side
        drift detection joins on `(review_id, github_review_id)` per
        Q5 withdrawal — divergent rows surface as two logical
        `PublishEvent`s and are caught by V1.5 anomaly rules
        (FOLLOWUPS.md FUP-063), not by this query method.

        Implementation parallels the per-emit session discipline:
        each call opens its own `AsyncSession` (read-only, no
        `session.begin()`), runs one `SELECT ... LIMIT 1`, and
        deserializes the JSONB payload via `PublishEvent.model_validate`.
        """
        ...


__all__ = [
    "AnalyzeEventSink",
    "FileExaminationSink",
    "PhaseEventSink",
    "PublishEventSink",
]
