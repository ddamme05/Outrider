"""Unit tests for the `HITLEventSink` Protocol and its recording test double.

Mirrors the per-test-file recording-sink precedent in
`tests/unit/test_publish_routing.py` and `tests/unit/test_analyze_node.py`:
the durable `AuditPersister` is integration-tested elsewhere; this file
asserts the Protocol surface (member presence, runtime-checkable
membership, non-None return contract) and pins the recording double's
audit-first return shape.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from outrider.audit.events import HITLDecisionEvent, HITLRequestEvent
from outrider.audit.sinks import HITLEventSink
from outrider.policy.canonical import compute_hitl_decision_content_hash
from outrider.schemas.hitl import PerFindingDecision, PerFindingOutcome


class _RecordingHITLEventSink:
    """Test double: records every emit into per-type lists; returns the
    incoming event verbatim per the audit-first contract.

    Recording sinks are deliberately exempt from idempotency dedup so
    double-emit bugs surface in tests rather than being silently absorbed
    (mirrors `audit/sinks.py:75-83` recorder-vs-durable split).
    """

    def __init__(self) -> None:
        self.requests: list[HITLRequestEvent] = []
        self.decisions: list[HITLDecisionEvent] = []

    async def emit_hitl_request(self, event: HITLRequestEvent) -> HITLRequestEvent:
        self.requests.append(event)
        return event

    async def emit_hitl_decision(self, event: HITLDecisionEvent) -> HITLDecisionEvent:
        self.decisions.append(event)
        return event


def _make_request(*, review_id: UUID) -> HITLRequestEvent:
    now = datetime.now(UTC)
    return HITLRequestEvent(
        event_id=uuid4(),
        review_id=review_id,
        timestamp=now,
        is_eval=False,
        findings_requiring_approval=(uuid4(),),
        auto_post_findings=(),
        created_at=now,
        expires_at=now,
    )


def _make_decision(*, review_id: UUID) -> HITLDecisionEvent:
    now = datetime.now(UTC)
    decision = PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.APPROVE,
        reason="ok",
    )
    annotation = "approved"
    return HITLDecisionEvent(
        event_id=uuid4(),
        review_id=review_id,
        timestamp=now,
        is_eval=False,
        reviewer_id="admin",
        decisions=(decision,),
        annotation=annotation,
        decided_at=now,
        decision_latency_seconds=0.0,
        decisions_content_hash=compute_hitl_decision_content_hash(
            decisions=(decision,), annotation=annotation
        ),
    )


def test_protocol_is_runtime_checkable() -> None:
    sink = _RecordingHITLEventSink()
    assert isinstance(sink, HITLEventSink)


def test_recording_sink_records_request_and_returns_event() -> None:
    sink = _RecordingHITLEventSink()
    review_id = uuid4()
    event = _make_request(review_id=review_id)

    import asyncio

    returned = asyncio.run(sink.emit_hitl_request(event))

    assert returned is event
    assert sink.requests == [event]
    assert sink.decisions == []


def test_recording_sink_records_decision_and_returns_event() -> None:
    sink = _RecordingHITLEventSink()
    review_id = uuid4()
    event = _make_decision(review_id=review_id)

    import asyncio

    returned = asyncio.run(sink.emit_hitl_decision(event))

    assert returned is event
    assert sink.decisions == [event]
    assert sink.requests == []


def test_recording_sink_does_not_dedup_double_emit() -> None:
    """Recording sinks deliberately exempt from idempotency dedup per
    `audit/sinks.py:75-83`."""
    sink = _RecordingHITLEventSink()
    review_id = uuid4()
    e1 = _make_request(review_id=review_id)
    e2 = _make_request(review_id=review_id)  # same review_id, different event_id

    import asyncio

    asyncio.run(sink.emit_hitl_request(e1))
    asyncio.run(sink.emit_hitl_request(e2))

    assert sink.requests == [e1, e2]


def test_protocol_membership_rejects_missing_method() -> None:
    """A class that only implements one of the two methods is NOT a
    structural HITLEventSink. PEP 544 `runtime_checkable` checks
    member presence; missing methods fail `isinstance`."""

    class _PartialSink:
        async def emit_hitl_request(self, event: HITLRequestEvent) -> HITLRequestEvent:
            return event

    sink = _PartialSink()
    assert not isinstance(sink, HITLEventSink)


def test_protocol_membership_rejects_completely_unrelated_class() -> None:
    """A class that implements neither method is NOT a structural
    HITLEventSink."""

    class _UnrelatedSink:
        pass

    sink = _UnrelatedSink()
    assert not isinstance(sink, HITLEventSink)


@pytest.mark.parametrize("method_name", ["emit_hitl_request", "emit_hitl_decision"])
def test_protocol_declares_both_methods(method_name: str) -> None:
    """Protocol surface check: HITLEventSink declares both methods."""
    assert hasattr(HITLEventSink, method_name)
