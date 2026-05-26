"""Unit tests for the `ReviewStatusSink` Protocol + its recording test
double.

Durable persister behavior is covered by integration tests; this file
asserts the Protocol surface (member presence, runtime-checkable
membership) and pins the recording double's idempotency-exempt
recording semantics.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from outrider.db.sinks import ReviewStatusReader, ReviewStatusSink


class _RecordingReviewStatusSink:
    """Test double: records every call into per-method lists. Records
    are not idempotency-deduped so double-call bugs surface in tests."""

    def __init__(self) -> None:
        self.awaiting_approval_calls: list[dict[str, Any]] = []
        self.running_calls: list[dict[str, Any]] = []
        self.expired_calls: list[dict[str, Any]] = []

    async def mark_awaiting_approval(
        self,
        *,
        review_id: UUID,
        expires_at: datetime,
        hitl_request_payload: dict[str, Any],
    ) -> None:
        self.awaiting_approval_calls.append(
            {
                "review_id": review_id,
                "expires_at": expires_at,
                "hitl_request_payload": hitl_request_payload,
            }
        )

    async def mark_running(
        self,
        *,
        review_id: UUID,
        hitl_decision_payload: dict[str, Any],
    ) -> None:
        self.running_calls.append(
            {
                "review_id": review_id,
                "hitl_decision_payload": hitl_decision_payload,
            }
        )

    async def mark_awaiting_approval_expired(self, *, review_id: UUID) -> None:
        self.expired_calls.append({"review_id": review_id})


def test_protocol_is_runtime_checkable() -> None:
    sink = _RecordingReviewStatusSink()
    assert isinstance(sink, ReviewStatusSink)


def test_recording_sink_records_awaiting_approval() -> None:
    sink = _RecordingReviewStatusSink()
    review_id = uuid4()
    expires_at = datetime.now(UTC)
    payload = {"findings_requiring_approval": [], "auto_post_findings": []}

    asyncio.run(
        sink.mark_awaiting_approval(
            review_id=review_id,
            expires_at=expires_at,
            hitl_request_payload=payload,
        )
    )

    assert len(sink.awaiting_approval_calls) == 1
    assert sink.awaiting_approval_calls[0]["review_id"] == review_id


def test_recording_sink_records_running() -> None:
    sink = _RecordingReviewStatusSink()
    review_id = uuid4()
    payload = {"reviewer_id": "admin", "decisions": []}

    asyncio.run(
        sink.mark_running(review_id=review_id, hitl_decision_payload=payload),
    )

    assert len(sink.running_calls) == 1
    assert sink.running_calls[0]["hitl_decision_payload"] == payload


def test_recording_sink_records_expired() -> None:
    sink = _RecordingReviewStatusSink()
    review_id = uuid4()

    asyncio.run(sink.mark_awaiting_approval_expired(review_id=review_id))

    assert len(sink.expired_calls) == 1
    assert sink.expired_calls[0]["review_id"] == review_id


def test_recording_sink_does_not_dedup_idempotent_calls() -> None:
    """Recording sinks deliberately exempt from idempotency semantics."""
    sink = _RecordingReviewStatusSink()
    review_id = uuid4()

    asyncio.run(sink.mark_awaiting_approval_expired(review_id=review_id))
    asyncio.run(sink.mark_awaiting_approval_expired(review_id=review_id))

    assert len(sink.expired_calls) == 2


def test_partial_sink_rejected_by_runtime_check() -> None:
    """A class missing `mark_awaiting_approval_expired` fails
    `isinstance(ReviewStatusSink)`."""

    class _Partial:
        async def mark_awaiting_approval(
            self, *, review_id: UUID, expires_at: datetime, hitl_request_payload: dict[str, Any]
        ) -> None:
            pass

        async def mark_running(
            self, *, review_id: UUID, hitl_decision_payload: dict[str, Any]
        ) -> None:
            pass

    assert not isinstance(_Partial(), ReviewStatusSink)


@pytest.mark.parametrize(
    "method_name",
    ["mark_awaiting_approval", "mark_running", "mark_awaiting_approval_expired"],
)
def test_protocol_declares_three_methods(method_name: str) -> None:
    assert hasattr(ReviewStatusSink, method_name)


def test_reader_protocol_is_runtime_checkable_with_durable_persister() -> None:
    """The durable `ReviewStatusPersister` is a structural
    `ReviewStatusReader`."""
    from outrider.db.review_status_persister import ReviewStatusPersister

    class _FakeFactory:
        def __call__(self) -> Any:
            raise NotImplementedError

    persister = ReviewStatusPersister(session_factory=_FakeFactory())  # type: ignore[arg-type]
    assert isinstance(persister, ReviewStatusReader)
    assert isinstance(persister, ReviewStatusSink)
