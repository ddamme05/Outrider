# Sink Protocols for agent-node-emitted audit events.
"""Sink Protocols for agent-node-emitted audit events.

Nodes consume sink Protocols defined here rather than touching DB code
directly — this keeps `nodes-receive-deps-via-closure` honest (real sinks
inject at graph-build time, test sinks inject at fixture-setup time) and
keeps audit-table writes out of node call sites.

V1 ships three sinks from this module: `PhaseEventSink` for
`ReviewPhaseEvent` (per `phase-events-bound-work`), `FileExaminationSink`
for `FileExaminationEvent` (per intake + analyze per-file outcomes), and
`AnalyzeEventSink` bundling the four analyze-emitted event types
(`FindingEvent`, `FindingProposalRejectedEvent`,
`AnalyzeResponseRejectedEvent`, `AnalyzeCompletedEvent`). `LLMCallEvent`
emission lives inside `LLMProvider.complete()` and uses the sibling
`LLMExchangePersister` Protocol in `llm/base.py` — no node code emits
`LLMCallEvent` directly.

The durable `AuditPersister` in `outrider.audit.persister` implements
ALL FOUR (`PhaseEventSink` + `FileExaminationSink` + `AnalyzeEventSink`
+ `LLMExchangePersister`) from one body, sharing DB transaction
lifecycle and session-per-call discipline. Test-only no-op
implementations (`NoOpPersister`, `RecordingPhaseEventSink`) live in
`tests/conftest.py` for fixtures that don't need durable persistence.
"""

from typing import Protocol, runtime_checkable

from outrider.audit.events import (
    AnalyzeCompletedEvent,
    AnalyzeResponseRejectedEvent,
    FileExaminationEvent,
    FindingEvent,
    FindingProposalRejectedEvent,
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
    `PhaseEventSink`, `AnalyzeEventSink`, and `LLMExchangePersister` —
    one class, one transaction-lifecycle discipline, four sinks. Test
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


__all__ = [
    "AnalyzeEventSink",
    "FileExaminationSink",
    "PhaseEventSink",
]
