# Sink Protocols for agent-node-emitted audit events.
"""Sink Protocols for agent-node-emitted audit events.

Nodes consume sink Protocols defined here rather than touching DB code
directly — this keeps `nodes-receive-deps-via-closure` honest (real sinks
inject at graph-build time, test sinks inject at fixture-setup time) and
keeps audit-table writes out of node call sites.

V1 ships one sink: `PhaseEventSink` for `ReviewPhaseEvent` emission per
`phase-events-bound-work`. `LLMCallEvent` emission lives inside
`LLMProvider.complete()` and uses the sibling `LLMExchangePersister`
Protocol in `llm/base.py` — no node code emits `LLMCallEvent` directly.

The durable `AuditPersister` in `outrider.audit.persister` implements
BOTH `PhaseEventSink` and `LLMExchangePersister` from one body, sharing
DB transaction lifecycle and session-per-call discipline. Test-only no-op
implementations (`NoOpPersister`, `RecordingPhaseEventSink`) live in
`tests/conftest.py` for fixtures that don't need durable persistence.
"""

from typing import Protocol, runtime_checkable

from outrider.audit.events import ReviewPhaseEvent


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
    BOTH `PhaseEventSink` and `LLMExchangePersister` from one class.

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


__all__ = [
    "PhaseEventSink",
]
