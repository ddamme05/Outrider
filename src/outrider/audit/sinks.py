# See DECISIONS.md#016-llm-exchanges-stored-locally-under-retention-logs-stay-metadata-only
# (this module defines the sink Protocols consumed by the persister
# whose atomic-emit contract `#016` establishes).
"""Sink Protocols for agent-node-emitted audit events.

Nodes consume sink Protocols defined here rather than touching DB code
directly — this keeps `nodes-receive-deps-via-closure` honest (real sinks
inject at graph-build time, test sinks inject at fixture-setup time) and
keeps audit-table writes out of node call sites.

V1 defines eight sink Protocols in this module (the ninth that
`AuditPersister` implements, `LLMExchangePersister`, lives at
`llm/base.py` because LLM-exchange persistence is owned by the LLM
provider boundary, not by the audit-events surface): `PhaseEventSink`
for
`ReviewPhaseEvent` (per `phase-events-bound-work`), `FileExaminationSink`
for `FileExaminationEvent` (per intake + analyze per-file outcomes),
`AnalyzeEventSink` bundling the eight analyze-emitted event types
(`FindingEvent`, `FindingProposalRejectedEvent`,
`AnalyzeResponseRejectedEvent`, `AnalyzeCompletedEvent`,
`ScopeExclusionEvent`, `CacheLookupEvent`, `CacheServeEvent`,
`ObservedSkipShadowEvent`),
`PublishEventSink` bundling the four publish-emitted event types
(`PublishRoutingEvent`, `PublishEligibilityEvent`, `PublishAttemptEvent`,
`PublishEvent`) per DECISIONS.md #023 routing-vs-eligibility decoupling,
`TraceEventSink` for `TraceDecisionEvent` per
`specs/2026-05-23-trace-node.md` Q4 + M7, `HITLEventSink` bundling
`emit_hitl_request` + `emit_hitl_decision` per
`specs/2026-05-26-hitl-node.md` Q7, `SynthesizeEventSink` for
`SynthesizeCompletedEvent` per `specs/2026-05-28-synthesize-node.md`
(per-review aggregate emitted at end-of-synthesize-work), and
`SlackEventSink` for `SlackNotificationEvent` per
`specs/2026-06-15-slack-dashboard-in-slack.md` (one per Slack post).
`LLMCallEvent` emission lives inside `LLMProvider.complete()` and uses
the sibling `LLMExchangePersister` Protocol in `llm/base.py` — no node
code emits `LLMCallEvent` directly.

The durable `AuditPersister` in `outrider.audit.persister` implements
ALL NINE (`PhaseEventSink` + `FileExaminationSink` + `AnalyzeEventSink`
+ `PublishEventSink` + `TraceEventSink` + `HITLEventSink` +
`SynthesizeEventSink` + `SlackEventSink` + `LLMExchangePersister`) from one body, sharing
DB transaction lifecycle and session-per-call discipline. Test-only
no-op implementations
(`NoOpPersister`, `RecordingPhaseEventSink`) live in
`tests/conftest.py` for fixtures that don't need durable persistence;
sink-specific recording doubles (HITL, publish, analyze) live in
per-test-file scope inside `tests/unit/` per the existing convention.
"""

from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable
from uuid import UUID

from outrider.audit.aggregates import ReviewLLMAggregates
from outrider.audit.events import (
    AnalyzeCompletedEvent,
    AnalyzeResponseRejectedEvent,
    CacheLookupEvent,
    CacheServeEvent,
    FileExaminationEvent,
    FindingProposalRejectedEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    ObservedSkipShadowEvent,
    PublishAttemptEvent,
    PublishEligibilityEvent,
    PublishEvent,
    PublishRoutingEvent,
    ReviewPhaseEvent,
    ScopeExclusionEvent,
    SlackNotificationEvent,
    SynthesizeCompletedEvent,
    TraceDecisionEvent,
)
from outrider.schemas.review_finding import ReviewFinding


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
        the audit row must not duplicate. (`phase_id` is a SHA-256
        hex digest the node mints via
        `policy.canonical.compute_phase_id(review_id, node_id, attempt_key)` —
        deterministic across body re-runs so the idempotency key is
        stable across checkpoint replay.)
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
    `PhaseEventSink`, `AnalyzeEventSink`, `PublishEventSink`,
    `TraceEventSink`, `HITLEventSink`, `SynthesizeEventSink`,
    `SlackEventSink`, and `LLMExchangePersister` — one class, one
    transaction-lifecycle discipline, nine sinks. Test
    fixtures may record to a list or persist directly per the same
    recorder-vs-durable split documented on `PhaseEventSink`.

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
    """Sink for the eight audit event types the analyze node emits.

    The analyze-node spec (`specs/2026-05-19-analyze-node.md` §7) bundled
    the original four event types under one Protocol rather than separate
    sinks (the trivial-scope-filter spec added the fifth,
    `ScopeExclusionEvent`; the file-hash-analyze-cache spec added the
    sixth, `CacheLookupEvent`; the analyze-cache-serve spec added the
    seventh, `CacheServeEvent`; the OBSERVED-query-library spec added the
    eighth, `ObservedSkipShadowEvent`) because:

    - All eight are emitted by ONE node body in one transaction window;
      the durable `AuditPersister` implements them under one session
      lifecycle.
    - The node body's local-bookkeeping counter discipline (per
      `_enforce_proposal_accounting` on `AnalyzeCompletedEvent`) treats
      the original four as cardinal-related: one
      `AnalyzeCompletedEvent` per pass, N `FindingEvent`s + M
      `FindingProposalRejectedEvent`s + at-most-one
      `AnalyzeResponseRejectedEvent` per per-file LLM call.
      `ScopeExclusionEvent` sits outside that accounting relation:
      at-most-one per examined pass-0 file, emitted whether or not an
      LLM call follows (the all-trivial skip has no call at all).
      `CacheLookupEvent` likewise: at-most-one per pass-0 clean-mode
      candidate file when the analyze cache is enabled (shadow-stage
      would-hit/miss telemetry); `CacheServeEvent` replaces it on a
      served hit under `cache_mode=serve`. `ObservedSkipShadowEvent` is the
      OBSERVED skip-routing shadow record — at-most-one per pass-0 clean-mode
      file, recording the coverage decision + whether an enforced skip was
      taken (`skip_enforced`; dormant in production), outside the proposal
      accounting.
    - Separate kwargs on the analyze function signature would
      crowd the deps surface and invite test-time mock proliferation;
      one sink keeps the test setup focused.

    `LLMCallEvent` is NOT here — the provider's
    `LLMExchangePersister` emits it autonomously inside
    `LLMProvider.complete()`.

    Production / durable implementations MUST:
      - Be safe under concurrent invocations. V1.5's parallel-analyze
        fan-out emits per-file events concurrently.
      - Persist before returning, OR raise. Silent drop is never
        acceptable.
      - Idempotency keys differ by method:
        - `emit_finding` mints a fresh `FindingEvent.event_id` per call,
          so re-emission appends a new audit row (NOT deduped on
          `event_id`). The `findings` *content* row is keyed on the
          stable `finding_id` (PK): re-emit writes at most one content
          row and NEVER resurrects a row the retention sweep purged
          (`specs/2026-05-30-findings-content-writer.md`). So a single
          finding may have N append-only `FindingEvent` rows but exactly
          one (or zero, post-purge) `findings` content row.
        - The other seven methods (`emit_finding_proposal_rejected`,
          `emit_analyze_response_rejected`, `emit_analyze_completed`,
          `emit_scope_exclusion`, `emit_cache_lookup`, `emit_cache_serve`,
          `emit_observed_skip_shadow`)
          are `event_id`-PK idempotent per `DECISIONS.md#026`: retry /
          replay minting a fresh `event_id` is acceptable, and
          consumer-side dedup handles read-time collapse.

    Test recorders (e.g., `RecordingAnalyzeEventSink`) record every
    emission into per-type lists for assertion; they are deliberately
    exempt from the idempotency rule (so double-emit bugs surface in
    tests rather than being silently deduped).

    `@runtime_checkable` matches the sibling-Protocol precedent —
    `build_graph` can reject sinks lacking any of the eight `emit_*`
    members at construction time. PEP 544 caveat: member-presence
    only, not signature shape; mypy strict is the write-time gate.
    """

    async def emit_finding(self, finding: ReviewFinding, *, is_eval: bool) -> None:
        """Persist an admitted finding: BOTH the `FindingEvent` audit row AND
        the `findings` content row, co-inserted in one transaction.

        The persister lifts the metadata-only `FindingEvent` from the
        `ReviewFinding` internally (per DECISIONS.md#014) and writes the full
        content row alongside it, mirroring the DECISIONS.md#016 `LLMCallEvent`
        + `llm_call_content` co-insert. `is_eval` threads through to both rows.
        See specs/2026-05-30-findings-content-writer.md.
        """
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

    async def emit_scope_exclusion(self, event: ScopeExclusionEvent) -> None:
        """Persist the per-file trivial-scope classification record.

        `event_id`-PK idempotent per `DECISIONS.md#026` — resume
        re-emission appends a fresh row; consumer-side dedup collapses.
        """
        ...

    async def emit_cache_lookup(self, event: CacheLookupEvent) -> None:
        """Persist the per-file analyze-cache lookup record (shadow-stage
        would-hit/miss telemetry; event_id-PK per `DECISIONS.md#026`).
        """
        ...

    async def emit_cache_serve(self, event: CacheServeEvent) -> None:
        """Persist the per-file analyze-cache SERVE record (serve-flip stage:
        a live hit was served and the LLM call skipped; event_id-PK per
        `DECISIONS.md#026`). Served-finding content rides re-emitted
        `FindingEvent`s via `emit_finding`, not this event.
        """
        ...

    async def emit_observed_skip_shadow(self, event: ObservedSkipShadowEvent) -> None:
        """Persist the per-file OBSERVED skip-routing record (Cost Lever 3: the
        coverage decision + `skip_enforced` — whether the enforced pre-LLM skip
        was taken; dormant in production, so the LLM still runs there).
        event_id-PK idempotent per `DECISIONS.md#026`.
        """
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

    async def query_prior_publish_event(self, *, review_id: UUID) -> PublishEvent | None:
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

    # See DECISIONS.md#027 — V1 per-review publish-side advisory lock.
    def acquire_publish_lock(self, *, review_id: UUID) -> AbstractAsyncContextManager[None]:
        """Acquire a per-review advisory lock for the publish path.

        Returns an async context manager that yields once the lock
        is held. Acquisition is bounded by a deadline (default 120s
        in the durable implementation); on timeout the implementation
        raises rather than yielding.

        Backs the V1 defense against concurrent identical resume
        paths both reaching `publisher.create_review` and POSTing
        twice. The durable implementation uses
        `pg_try_advisory_xact_lock(<lock_id>)` where `lock_id` is the
        first 8 bytes of `review_id.bytes` as a signed int8, in a
        loop with exponential backoff: each probe opens its own
        session+transaction, releases on not-acquired, sleeps, retries.
        On acquired, holds the session+transaction for the lifetime of
        the context manager. Recording sinks no-op (no real
        serialization).

        Why try-lock + bounded backoff (NOT plain blocking
        `pg_advisory_xact_lock`): a blocking variant holds a connection
        for the entire wait. N same-review contenders would pin N pool
        connections simultaneously, starving the winner's
        `emit_publish_*` calls (each opens a fresh session per the
        per-emit discipline). Backoff releases the connection between
        probes — the winner's emit path stays unstarved.

        Why NOT single-shot `pg_try_advisory_xact_lock` with loser-
        skip: the immediate loser cannot observe whether the winner
        actually committed the POST. Skipping → emit
        `IDEMPOTENTLY_SKIPPED` even when the winner crashed between
        lock acquisition and POST → publish lost. Bounded retry puts
        the loser BEHIND the winner's transaction boundary on the
        eventual successful acquire, so the post-lock
        `query_prior_publish_event` observes the winner's committed
        `PublishEvent` (success → authentic skip) OR its absence
        (winner crashed → loser POSTs). False-skip class eliminated.

        The lock auto-releases on context exit (transaction
        commit/rollback). Callers use `async with
        publish_event_sink.acquire_publish_lock(review_id):` — no
        bool result to inspect (acquisition is unconditional;
        failure to acquire within the deadline raises rather than
        returning False).
        """
        ...


@runtime_checkable
class TraceEventSink(Protocol):
    """Sink for `TraceDecisionEvent` emissions per the trace node spec
    (`specs/2026-05-23-trace-node.md` Q4 + M7).

    Why sibling Protocol (not extending `AnalyzeEventSink`): trace is a
    separate node with separate responsibility. Per Q4 resolution,
    conflating the sinks would make the analyze sink's name lie. The
    `build_graph` deps surface gains one kwarg (`trace_sink: TraceEventSink`);
    cost is bounded and matches the existing one-sink-per-node pattern.

    **Audit-first emission contract per M7 (b) — non-None return.**
    `emit_trace_decision` returns the canonical persisted
    `TraceDecisionEvent` — either the just-inserted incoming event
    (insert path) OR the existing row's event (no-op path on
    natural-key match with identity-subset equality). The producer
    node (trace) MUST use the returned event to construct the state-
    layer `TraceDecision` for the state delta, ensuring state and
    audit stay in lockstep across retry/replay even when per-emission
    fields (`reason`, `proposed_import_strings`, `resolved_candidate_paths`,
    `trace_path`) differ between attempts. Without this lockstep, the
    crash-after-audit-before-state scenario would diverge state from
    audit on retry.

    **Persister-side natural-key idempotency on `(review_id,
    source_finding_id)` per `DECISIONS.md#026`** — the durable
    `AuditPersister` implementation runs `postgresql_insert(...)
    .on_conflict_do_nothing(...)` against a partial unique index
    introduced by an Alembic migration; on conflict, a follow-up SELECT
    loads the existing row and the identity-subset comparison
    (`source_finding_id`, `target_file`, `resolution_status`, `is_eval`)
    distinguishes legitimate retry (no-op return) from real divergence
    (raise `AuditPersisterNaturalKeyConflict`). The persister + migration
    + identity-subset helper land in Group 4; this Protocol declares
    the contract.

    Production / durable implementations MUST:
      - Implement the audit-first return contract per M7 (b) — return
        the canonical persisted event, not the incoming one.
      - Be idempotent on the semantic key `(review_id, source_finding_id)`
        (NOT on `event_id` PK — that's a different idempotency mode
        per `DECISIONS.md#026`). Replay producing the same logical
        decision twice MUST collapse to one audit row, not two.
      - Be concurrent-safe: V1 is single-threaded per review via the
        `BackgroundTasksDispatcher`, but the partial unique index from
        Group 4's Alembic migration is the DB-level safety net for
        V1.5 parallel-analyze + the webhook-redispatch edge case.
      - Persist before returning, OR raise. Silent drop is never
        acceptable — trace's audit-first contract relies on
        `TraceDecision` in state ↔ matching `TraceDecisionEvent` row
        in audit_events (joined on source_finding_id within the review).

    Test recorders capture each emission for assertion; per the
    audit-first contract, they MUST also return the incoming event
    (no idempotency dedup in test sinks — recorders are
    deliberately exempt so double-emit bugs surface in tests rather
    than being silently deduped).

    `@runtime_checkable` matches the sibling-Protocol precedent —
    `build_graph` can reject sinks lacking `emit_trace_decision` at
    construction time via `isinstance(...)`. PEP 544 caveat: member-
    presence only, not signature shape; mypy strict is the write-time
    gate for the non-None return type.
    """

    async def emit_trace_decision(self, event: TraceDecisionEvent) -> TraceDecisionEvent:
        """Persist a `TraceDecisionEvent` and return the canonical
        persisted event (incoming on insert; existing on no-op match).

        See class docstring for the audit-first return contract.
        """
        ...

    async def get_trace_decisions(self, *, review_id: UUID) -> tuple[TraceDecisionEvent, ...]:
        """Return every persisted `TraceDecisionEvent` for the review.

        Read-side recovery surface for the audit-ahead-of-state crash
        window: a decision row persisted by a run that crashed after
        `emit_trace_decision` but before the node's state delta merged
        is invisible to the trace node's state-side `already_traced`
        gate. The node consults this method once per pass and ADOPTS
        the persisted row for any pending finding it covers instead of
        re-probing — re-deciding under changed resolver semantics
        (deploy between crash and resume) would trip the natural-key
        identity guard and abort the resumed review. Read-only; the
        append-only guarantee is untouched. Test recorders may return
        `()` (no recovery path exercised).
        """
        ...


@runtime_checkable
class HITLEventSink(Protocol):
    """Sink for the HITL node's two emit moments per the HITL node spec.

    Why a single sink bundling both methods (vs separate request/decision
    sinks): one node emits BOTH events at distinct moments
    (`HITLRequestEvent` BEFORE `interrupt(...)`, `HITLDecisionEvent`
    AFTER resume returns). Two separate Protocol injection slots in
    `build_graph(...)` would gain nothing — the audit responsibility is
    "the hitl node", not "request emission" + "decision emission". This
    mirrors the `PublishEventSink` precedent (one node, four methods,
    one Protocol).

    **Audit-first emission contract: non-None return.** Both methods
    return the canonical persisted event — either the just-inserted
    incoming event (insert path) OR the existing row's event (no-op
    path on natural-key match with identity-subset equality). The
    producer node MUST use the returned event to construct the state-
    layer `HITLRequest` / `HITLDecision` for the state delta, ensuring
    state and audit stay in lockstep across retry / replay. Without
    this lockstep, the crash-after-audit-before-state scenario would
    diverge state from audit on retry.

    **Persister-side natural-key idempotency on `(review_id)`** — the
    durable `AuditPersister` implementation runs
    `postgresql_insert(...).on_conflict_do_nothing(...)` against the
    partial unique indexes (one per event_type) introduced by the HITL
    Alembic migration; on conflict, a follow-up SELECT loads the
    existing row and the identity-subset comparison distinguishes
    legitimate retry (no-op return) from real divergence (raise
    `AuditPersisterHITLRequestNaturalKeyConflict` or
    `AuditPersisterHITLDecisionNaturalKeyConflict`). The decision-side
    identity subset is `{decisions_content_hash, is_eval}`; the
    request-side subset is `{findings_requiring_approval,
    auto_post_findings, created_at, expires_at, is_eval}`.

    Test recorders capture each emission for assertion; per the
    audit-first contract, they MUST also return the incoming event (no
    idempotency dedup in test sinks — recorders are deliberately
    exempt so double-emit bugs surface in tests rather than being
    silently deduped).
    """

    async def emit_hitl_request(self, event: HITLRequestEvent) -> HITLRequestEvent:
        """Persist a `HITLRequestEvent` and return the canonical
        persisted event (incoming on insert; existing on no-op match).

        See class docstring for the audit-first return contract.
        Natural-key idempotent on `(review_id)`.
        """
        ...

    async def emit_hitl_decision(self, event: HITLDecisionEvent) -> HITLDecisionEvent:
        """Persist a `HITLDecisionEvent` and return the canonical
        persisted event (incoming on insert; existing on no-op match).

        See class docstring for the audit-first return contract.
        Natural-key idempotent on `(review_id)` with identity-subset
        check on `decisions_content_hash` (divergent concurrent
        submissions raise `AuditPersisterHITLDecisionNaturalKeyConflict`).
        """
        ...


@runtime_checkable
class SynthesizeEventSink(Protocol):
    """Sink for the synthesize node's one audit event type.

    Synthesize emits exactly one audit event per review:
    `SynthesizeCompletedEvent` at end-of-work (after the summary
    call returns, after deduplication + sort, before returning the state
    delta). One emit method (plus a read query below) matches the analyze-completed
    pattern conceptually but with simpler shape (no per-pass
    multiplicity, no per-finding sub-events — every finding in the
    deduplicated `ReviewReport.findings` was already emitted as its own
    `FindingEvent` upstream by analyze).

    Production / durable implementations MUST:
      - Be idempotent on `event.event_id`. Retry / checkpoint replay
        must not duplicate rows. Per pre-spec gate #1, synthesize uses
        event_id-PK idempotency (NOT natural-key) — see
        `SynthesizeCompletedEvent` class docstring for the rationale.
      - Be safe under concurrent invocations. `DECISIONS.md#027` line
        946 says LangGraph does NOT serialize concurrent `ainvoke` on
        the same `thread_id` — the durable persister relies on
        event_id-PK uniqueness, NOT on serial-execution assumptions.
      - Persist before returning, OR raise. Silent drop is never
        acceptable.

    Test recorders (e.g., `RecordingSynthesizeEventSink`) record every
    emission into a per-type list for assertion; they are deliberately
    exempt from the idempotency rule (so double-emit bugs surface in
    tests rather than being silently deduped).

    `@runtime_checkable` matches the sibling-Protocol precedent —
    `build_graph` can reject sinks lacking `emit_synthesize_completed` or
    `query_review_llm_aggregates` at construction time. PEP 544 caveat:
    member-presence only, not signature shape; mypy strict is the write-time gate.
    """

    async def emit_synthesize_completed(self, event: SynthesizeCompletedEvent) -> None:
        """Persist a `SynthesizeCompletedEvent` row (per-review aggregate)."""
        ...

    async def query_review_llm_aggregates(
        self, *, review_id: UUID, is_eval: bool
    ) -> ReviewLLMAggregates:
        """Sum the review's `LLMCallEvent` rows (count + tokens + cost).

        Synthesize calls this at metric-build time to populate the `ReviewMetrics`
        LLM aggregates from the audit stream (FUP-093) — the same single-source SUM
        the dashboard read-API uses, so the persisted audit row and the dashboard
        badge cannot diverge. Read-only; the durable `AuditPersister` implements it
        alongside `emit_synthesize_completed` (mirrors `PublishEventSink`'s
        emit + `query_prior_publish_event` shape).
        """
        ...


@runtime_checkable
class SlackEventSink(Protocol):
    """Sink for the Slack notifier's one audit event type.

    `notify/slack.py` emits exactly one `SlackNotificationEvent` per Slack
    post — the rich HITL-pending card or the compact review-posted FYI. An emit
    method + a read query (`query_slack_notification`), mirroring `SynthesizeEventSink`.

    Production / durable implementations MUST:
      - Be idempotent on `event.event_id` (event_id-PK per `DECISIONS.md#026`,
        the `CacheLookupEvent` / `SynthesizeCompletedEvent` precedent) — retry
        / checkpoint replay must not duplicate rows.
      - Persist before returning, OR raise. Silent drop is never acceptable.

    The `(review_id, channel_id, kind)` de-duplication the spec describes is a
    best-effort PRE-POST check in the notifier (query before posting), NOT a
    persister natural-key constraint — see
    `specs/2026-06-15-slack-dashboard-in-slack.md` (Audit append-only). The
    persist path here is the plain event_id-PK append.

    `@runtime_checkable` matches the sibling-Protocol precedent — `build_graph`
    can reject sinks lacking `emit_slack_notification` or `query_slack_notification`
    at construction time.
    """

    async def emit_slack_notification(self, event: SlackNotificationEvent) -> None:
        """Persist a `SlackNotificationEvent` row (one per Slack post)."""
        ...

    async def query_slack_notification(
        self, *, review_id: UUID, channel_id: str, kind: str
    ) -> SlackNotificationEvent | None:
        """Most-recent `SlackNotificationEvent` for `(review_id, channel_id, kind)`, or None.

        The notifier's read surface: backs the best-effort pre-post dedup (None →
        safe to post) and the status-mirror target (the returned event's
        `message_ts`). Read-only; the durable `AuditPersister` implements it
        alongside `emit_slack_notification` (mirrors `PublishEventSink`'s
        emit + `query_prior_publish_event` shape).
        """
        ...


__all__ = [
    "AnalyzeEventSink",
    "FileExaminationSink",
    "HITLEventSink",
    "PhaseEventSink",
    "PublishEventSink",
    "SlackEventSink",
    "SynthesizeEventSink",
    "TraceEventSink",
]
