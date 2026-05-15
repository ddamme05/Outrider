"""PhaseEventSink Protocol contract tests.

Pins the runtime-checkable Protocol contract introduced by the
triage-node spec. PhaseEventSink is consumed by agent nodes (starting
with triage) and lands its concrete implementation in FUP-007's
audit-persister spec.

These tests document what the Protocol enforces structurally vs what
falls through to runtime — important for callers and for the
`build_graph` validation gate that uses `isinstance(sink, PhaseEventSink)`.
"""

from typing import Literal
from uuid import uuid4

import pytest

from outrider.audit import PhaseEventSink
from outrider.audit.events import ReviewPhaseEvent


class _ProtocolSatisfying:
    """Minimal class that satisfies the PhaseEventSink Protocol structurally."""

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        return None


class _MissingEmitPhase:
    """Class that does NOT have an `emit_phase` member."""

    async def some_other_method(self, event: ReviewPhaseEvent) -> None:
        return None


def _build_phase_event(*, marker: Literal["start", "end"] = "start") -> ReviewPhaseEvent:
    return ReviewPhaseEvent(
        review_id=uuid4(),
        phase_id=str(uuid4()),
        node_id="triage",
        marker=marker,
    )


def test_protocol_satisfying_class_is_isinstance() -> None:
    """A class with the right method NAME satisfies isinstance(...). Pins
    the structural-validation contract that `build_graph` relies on."""
    sink = _ProtocolSatisfying()
    assert isinstance(sink, PhaseEventSink)


def test_class_missing_emit_phase_fails_isinstance() -> None:
    """A class without an `emit_phase` attribute fails isinstance. This IS
    the failure mode the `build_graph` structural gate catches."""
    sink = _MissingEmitPhase()
    assert not isinstance(sink, PhaseEventSink)


def test_plain_object_fails_isinstance() -> None:
    """Plain object() — no methods at all — fails isinstance. Documents the
    most common production failure case: passing None-equivalent objects."""
    assert not isinstance(object(), PhaseEventSink)


def test_protocol_member_presence_only_documented_limit() -> None:
    """PEP 544: @runtime_checkable verifies MEMBER PRESENCE only — not
    signature, not async-vs-sync, not arity, not types. This test pins the
    documented limitation so future readers know the gate's bounds.

    A class with `emit_phase = "not callable"` (an attribute of any kind)
    PASSES isinstance — this is broken at the first call site, not at
    `build_graph`. The spec's PEP 544 caveat documents this; mypy strict
    is the write-time gate for signature shape."""

    class _WrongShape:
        emit_phase = "not callable at all"

    sink = _WrongShape()
    assert isinstance(sink, PhaseEventSink), (
        "PEP 544 runtime-checkable checks member presence only. "
        "If this assertion ever flips, the spec's PEP 544 caveat needs revisiting."
    )


def test_protocol_satisfying_class_can_be_awaited() -> None:
    """End-to-end happy path: the satisfying class actually works when
    `await sink.emit_phase(event)` is called — not just a static-type
    fiction. Catches the case where someone defines an emit_phase that
    raises immediately or is misshapen."""
    sink = _ProtocolSatisfying()
    event = _build_phase_event(marker="start")

    import asyncio

    asyncio.run(sink.emit_phase(event))  # must not raise


@pytest.mark.parametrize("marker", ["start", "end"])
def test_protocol_works_for_both_markers(marker: Literal["start", "end"]) -> None:
    """Both marker values flow through the sink without the Protocol
    enforcing anything about marker contents (marker validation is on
    ReviewPhaseEvent, not on the sink). Pins separation of concerns."""
    sink = _ProtocolSatisfying()
    event = _build_phase_event(marker=marker)

    import asyncio

    asyncio.run(sink.emit_phase(event))
