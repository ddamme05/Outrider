"""FileExaminationSink Protocol contract tests.

Mirrors `test_phase_event_sink_protocol.py` for the sibling Protocol
introduced by the intake-and-webhook spec. The Protocol's contract:
runtime-checkable, single `async emit_file_examination` member, no
signature-shape enforcement at runtime (PEP 544 caveat).

`AuditPersister` satisfies this Protocol AND `PhaseEventSink` from one
class body, sharing transaction discipline.
"""

import asyncio
from uuid import uuid4

import pytest

from outrider.audit.events import FileExaminationEvent
from outrider.audit.sinks import FileExaminationSink


class _ProtocolSatisfying:
    """Minimal class that structurally satisfies FileExaminationSink."""

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        return None


class _MissingEmit:
    """Lacks the `emit_file_examination` member entirely."""

    async def some_other_method(self, event: FileExaminationEvent) -> None:
        return None


def _build_event(*, parse_status: str = "clean") -> FileExaminationEvent:
    return FileExaminationEvent(
        review_id=uuid4(),
        file_path="src/example.py",
        examination_type="intake_fetch",
        node_id="intake",
        parse_status=parse_status,
    )


def test_protocol_satisfying_class_is_isinstance() -> None:
    """Class with the right method NAME satisfies isinstance — pins the
    member-presence gate that `build_graph` will rely on once intake ships."""
    assert isinstance(_ProtocolSatisfying(), FileExaminationSink)


def test_class_missing_emit_method_fails_isinstance() -> None:
    """Class without `emit_file_examination` attribute fails isinstance."""
    assert not isinstance(_MissingEmit(), FileExaminationSink)


def test_plain_object_fails_isinstance() -> None:
    """Plain object() fails isinstance — the most common production failure."""
    assert not isinstance(object(), FileExaminationSink)


def test_protocol_member_presence_only() -> None:
    """PEP 544: member-presence only; signature/async/types not enforced
    at runtime. Mirrors the same caveat documented on PhaseEventSink."""

    class _WrongShape:
        emit_file_examination = "not callable"

    assert isinstance(_WrongShape(), FileExaminationSink), (
        "PEP 544 runtime-checkable checks member presence only."
    )


def test_protocol_satisfying_class_can_be_awaited() -> None:
    """End-to-end: the satisfying class is actually awaitable."""
    sink = _ProtocolSatisfying()
    event = _build_event()
    asyncio.run(sink.emit_file_examination(event))


@pytest.mark.parametrize("parse_status", ["clean", "degraded", "failed"])
def test_protocol_works_for_non_skipped_parse_statuses(parse_status: str) -> None:
    """Non-skipped parse statuses flow through without the Protocol
    enforcing anything about content. The skip_reason cross-field rule
    is enforced by FileExaminationEvent itself, not by the sink."""
    sink = _ProtocolSatisfying()
    event = _build_event(parse_status=parse_status)
    asyncio.run(sink.emit_file_examination(event))


def test_audit_persister_satisfies_protocol() -> None:
    """The durable `AuditPersister` satisfies FileExaminationSink via the
    `emit_file_examination` method added by this spec. Pins the
    'one class, multiple Protocols' design — same precedent as
    `PhaseEventSink`."""
    from outrider.audit.persister import AuditPersister

    # Member presence is what the runtime-checkable Protocol verifies.
    # Construction of an AuditPersister requires a session_factory and
    # retention_settings; for the membership check we only need to verify
    # the class has the attribute.
    assert hasattr(AuditPersister, "emit_file_examination")
    assert callable(AuditPersister.emit_file_examination)


def test_protocol_declares_exact_method_set() -> None:
    """Protocol surface check — exact membership, not just presence.

    Class-10 (centrally-pinned-contract registration) doctrine: a new
    method on `FileExaminationSink` must surface here AND at every
    sink consumer + test fixture. Exact-membership check fails loudly
    on silent drift.
    """
    expected = {"emit_file_examination"}
    actual = {name for name in dir(FileExaminationSink) if not name.startswith("_")}
    assert actual == expected, (
        f"FileExaminationSink method set drift: missing={expected - actual}, "
        f"extra={actual - expected}."
    )
