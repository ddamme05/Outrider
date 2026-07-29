# See DECISIONS.md#016-llm-exchanges-stored-locally-under-retention-logs-stay-metadata-only
# (this module defines the sink Protocols consumed by the persister
# whose atomic-emit contract `#016` establishes).
"""Sink Protocols for agent-node-emitted audit events.

Nodes consume these Protocols instead of touching DB code directly, which keeps
audit-table writes out of node call sites and lets tests inject recorders.

Eight Protocols live here, one per emitting node or notifier. The ninth,
`LLMExchangePersister`, lives in `llm/base.py` because LLM-exchange persistence
belongs to the LLM provider boundary — no node emits `LLMCallEvent` directly.
`AuditPersister` implements all nine from one body, sharing its transaction and
session-per-call discipline.

Rules every durable implementation shares, stated once here rather than per
Protocol:

  - Persist before returning, or raise. A sink that returns success without
    persisting is the failure the `isinstance` gate at build time cannot catch.
  - Be safe under concurrent invocation — a fresh `AsyncSession` per call.
  - Test recorders are deliberately exempt from idempotency: they record every
    emission so double-emit bugs surface instead of being silently deduped.

Each Protocol below documents only what is unique to it: its events, its
idempotency key, and any behaviour that departs from the above.

`@runtime_checkable` throughout lets `build_graph` reject a sink missing a member
at construction time. Per PEP 544 that check sees member presence only, never
signature shape; mypy strict is the write-time gate for signatures.
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
    """Sink for `ReviewPhaseEvent` per `phase-events-bound-work`.

    Nodes emit at entry (marker='start') and success-exit (marker='end') so replay
    has causal barriers.

    Idempotency key: `(review_id, phase_id, marker)`. `phase_id` is a SHA-256 digest
    from `policy.canonical.compute_phase_id(review_id, node_id, attempt_key)`,
    deterministic across body re-runs so the key survives checkpoint replay.
    """

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        """Persist a single phase-boundary marker."""
        ...


@runtime_checkable
class FileExaminationSink(Protocol):
    """Sink for `FileExaminationEvent` — one per file intake fetches.

    Idempotency key: `event.event_id`.

    The cross-field rule that `skip_reason` is non-None iff `parse_status="skipped"`
    is enforced by the event model (`DECISIONS.md#018`); the sink only persists.
    """

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        """Persist a single file-examination event."""
        ...


@runtime_checkable
class AnalyzeEventSink(Protocol):
    """Sink for the eight event types the analyze node emits.

    One Protocol rather than eight because a single node body emits all of them
    inside one transaction window, and separate kwargs would crowd the deps surface.

    Idempotency differs by method, which is why it is not stated once:

      - `emit_finding` mints a fresh `FindingEvent.event_id` per call, so re-emission
        APPENDS a row rather than deduping. The `findings` content row is keyed on
        the stable `finding_id`, so a finding may have N audit rows but at most one
        content row — and re-emission never resurrects a row the retention sweep
        purged.
      - The other seven are `event_id`-PK idempotent (`DECISIONS.md#026`);
        consumer-side dedup collapses replayed rows at read time.

    `LLMCallEvent` is not here — the provider emits it via `LLMExchangePersister`.
    """

    async def emit_finding(
        self, finding: ReviewFinding, *, is_eval: bool, phase_key: str | None = None
    ) -> None:
        """Persist BOTH the `FindingEvent` audit row and the `findings` content row,
        co-inserted in one transaction.

        The persister lifts the metadata-only event from the `ReviewFinding`
        itself (`DECISIONS.md#014`); `is_eval` threads to both rows.
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
        """Persist the per-file trivial-scope classification record."""
        ...

    async def emit_cache_lookup(self, event: CacheLookupEvent) -> None:
        """Persist the per-file analyze-cache lookup record (would-hit/miss telemetry)."""
        ...

    async def emit_cache_serve(self, event: CacheServeEvent) -> None:
        """Persist the per-file record of a served cache hit, where the LLM call was
        skipped. Served-finding content rides re-emitted `FindingEvent`s via
        `emit_finding`, not this event.
        """
        ...

    async def emit_observed_skip_shadow(self, event: ObservedSkipShadowEvent) -> None:
        """Persist the per-file OBSERVED skip-routing record: the coverage decision
        plus `skip_enforced`, which is dormant in production, so the LLM still runs
        there.
        """
        ...


@runtime_checkable
class PublishEventSink(Protocol):
    """Sink for the four publish-node event types, plus its idempotency reads.

    Routing and eligibility are separate decisions, not one gate (`DECISIONS.md#023`):
    one `PublishRoutingEvent` and one `PublishEligibilityEvent` per finding, at most
    one `PublishAttemptEvent` per attempt, and at most one `PublishEvent` per logical
    publication.

    Idempotency key: `event_id`, with a payload-equality check that raises
    `AuditPersisterIdempotencyConflict` on mismatch. Replay-equivalence dedup keys off
    the `decision_content_hash` / `attempt_content_hash` the events carry.
    """

    async def emit_publish_routing(self, event: PublishRoutingEvent) -> None:
        """Persist the per-finding routing decision.

        Fires for EVERY finding regardless of eligibility, so the trail records
        what coordinates classified even when the gate later withholds it.
        """
        ...

    async def emit_publish_eligibility(self, event: PublishEligibilityEvent) -> None:
        """Persist the per-finding policy gate.

        Carries `eligibility` and `policy_version` for severity-versioned replay.
        """
        ...

    async def emit_publish_attempt(self, event: PublishAttemptEvent) -> None:
        """Persist the per-attempt GitHub-call outcome, emitted AFTER the call
        resolves — there is no in-flight pre-call emission, which would conflict
        with `audit-events-append-only`.

        On `failed`, `failure_class` carries the exception class name, bounded at
        128 chars against attacker-influenced 422 strings.
        """
        ...

    async def emit_publish_result(self, event: PublishEvent) -> None:
        """Persist the success-path review-level summary.

        Named `emit_publish_result` rather than `emit_publish_event` to keep it
        distinct from the sink's other event types. Not emitted on `failed`,
        `no_op_empty`, or `idempotently_skipped*` paths.
        """
        ...

    async def query_prior_publish_event(self, *, review_id: UUID) -> PublishEvent | None:
        """Return the most-recent prior `PublishEvent` for `review_id`, or None.

        Backs the publish node's idempotency check: a same-`review_id` redispatch
        hits this BEFORE the GitHub call, and on a hit the node skips without
        burning a round-trip.

        If replay left several rows, the most recent by `timestamp` wins;
        divergence between them is an anomaly-rule concern, not this query's.
        """
        ...

    # See DECISIONS.md#027 — V1 per-review publish-side advisory lock.
    def acquire_publish_lock(self, *, review_id: UUID) -> AbstractAsyncContextManager[None]:
        """Acquire a per-review advisory lock, yielding once held and raising if the
        deadline passes. Guards against two concurrent resume paths both POSTing.

        The durable implementation try-locks with exponential backoff rather than
        blocking, for two reasons that are not obvious:

          - A blocking `pg_advisory_xact_lock` holds a connection for the whole
            wait, so N contenders pin N pool connections and starve the winner's
            own per-emit sessions. Backoff releases between probes.
          - A single-shot try-lock that lets the loser skip cannot tell whether the
            winner actually committed. Retrying puts the loser behind the winner's
            transaction boundary, so its `query_prior_publish_event` either sees a
            committed `PublishEvent` (authentic skip) or its absence (winner
            crashed, so post). That removes the false-skip class.

        Releases on context exit. Recording sinks no-op.
        """
        ...


@runtime_checkable
class TraceEventSink(Protocol):
    """Sink for `TraceDecisionEvent` emissions from the trace node.

    Two departures from the shared rules:

      - **Audit-first, non-None return.** `emit_trace_decision` returns the CANONICAL
        persisted event — the incoming one on insert, the existing row's on a
        natural-key no-op. The node must build its state-layer `TraceDecision` from
        the returned event, or a crash between audit and state diverges the two on
        retry.
      - **Natural-key idempotency on `(review_id, source_finding_id)`**, not
        `event_id` (`DECISIONS.md#026`). The persister inserts with
        `on_conflict_do_nothing` against a partial unique index; on conflict it
        re-reads and compares the identity subset (`source_finding_id`, `target_file`,
        `resolution_status`, `is_eval`) to tell a legitimate retry from real
        divergence, raising `AuditPersisterNaturalKeyConflict` for the latter.

    Test recorders must still return the incoming event to honour the first rule.
    """

    async def emit_trace_decision(self, event: TraceDecisionEvent) -> TraceDecisionEvent:
        """Persist and return the canonical event — incoming on insert, existing on a
        no-op match. See the class docstring for why the return value is load-bearing.
        """
        ...

    async def get_trace_decisions(self, *, review_id: UUID) -> tuple[TraceDecisionEvent, ...]:
        """Return every persisted `TraceDecisionEvent` for the review.

        Recovery surface for the crash window where a decision persisted but the
        node's state delta never merged, leaving the row invisible to the
        state-side `already_traced` gate. The node adopts the persisted row rather
        than re-probing: re-deciding under resolver semantics that changed between
        crash and resume would trip the natural-key identity guard and abort the
        resumed review.
        """
        ...


@runtime_checkable
class HITLEventSink(Protocol):
    """Sink for the HITL node's two emit moments.

    One Protocol because one node emits both — `HITLRequestEvent` before
    `interrupt(...)`, `HITLDecisionEvent` after resume returns.

    Same two departures as `TraceEventSink`: both methods return the canonical
    persisted event (audit-first lockstep), and idempotency is natural-key on
    `(review_id)` rather than `event_id`. Identity subsets differ by side — decision:
    `{decisions_content_hash, is_eval}`; request: `{findings_requiring_approval,
    auto_post_findings, created_at, expires_at, is_eval}` — and a mismatch raises
    `AuditPersisterHITLRequestNaturalKeyConflict` or its decision-side counterpart.
    """

    async def emit_hitl_request(self, event: HITLRequestEvent) -> HITLRequestEvent:
        """Persist and return the canonical `HITLRequestEvent`."""
        ...

    async def emit_hitl_decision(self, event: HITLDecisionEvent) -> HITLDecisionEvent:
        """Persist and return the canonical `HITLDecisionEvent`. Divergent concurrent
        submissions raise `AuditPersisterHITLDecisionNaturalKeyConflict`.
        """
        ...


@runtime_checkable
class SynthesizeEventSink(Protocol):
    """Sink for `SynthesizeCompletedEvent`, plus the review's LLM-aggregate read.

    Synthesize emits exactly one event per review, at end-of-work.

    Idempotency key: `event.event_id`, NOT natural-key — see the
    `SynthesizeCompletedEvent` docstring for why. Concurrency safety rests on that PK
    rather than on serial execution: LangGraph does not serialize concurrent
    `ainvoke` on one `thread_id` (`DECISIONS.md#027`).
    """

    async def emit_synthesize_completed(self, event: SynthesizeCompletedEvent) -> None:
        """Persist the per-review aggregate event."""
        ...

    async def query_review_llm_aggregates(
        self, *, review_id: UUID, is_eval: bool
    ) -> ReviewLLMAggregates:
        """Sum the review's `LLMCallEvent` rows (count, tokens, cost).

        Synthesize populates `ReviewMetrics` from this — the same single-source SUM
        the dashboard read-API uses, so the persisted row and the badge cannot
        diverge.
        """
        ...


@runtime_checkable
class SlackEventSink(Protocol):
    """Sink for `SlackNotificationEvent` — one per Slack post.

    Idempotency key: `event.event_id` (`DECISIONS.md#026`).

    The `(review_id, channel_id, kind)` de-duplication is a best-effort PRE-POST
    query in the notifier, NOT a persister constraint; this path is a plain append.
    """

    async def emit_slack_notification(self, event: SlackNotificationEvent) -> None:
        """Persist one Slack-post record."""
        ...

    async def query_slack_notification(
        self, *, review_id: UUID, channel_id: str, kind: str
    ) -> SlackNotificationEvent | None:
        """Most-recent event for `(review_id, channel_id, kind)`, or None.

        Backs the notifier's best-effort pre-post dedup and supplies the
        `message_ts` the status mirror updates.
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
