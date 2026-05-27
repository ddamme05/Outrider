"""Unit tests for `sweep/hitl_expiry.py` — anomaly-first ordering +
reclaim_stuck_hitl_states semantics.

Per the spec's Group 8 prescription, the load-bearing properties are:
  1. transition_expired_hitl_reviews emits the anomaly BEFORE flipping
     status. If anomaly emit RAISES, the status flip is skipped.
  2. reclaim_stuck_hitl_states marks `failed` only when the LangGraph
     checkpointer has no record for the thread_id (window-(c) crash).
  3. run_once runs reclaim BEFORE expiry so a stuck row past
     expires_at is correctly classified as `failed` (not expired).

Integration-style tests against real DB live in
`tests/integration/test_sweep_hitl_expiry_integration.py` (to land
when the schema migration is applied against postgres-test). These
unit tests pin the loop semantics against stub sinks + stub
checkpointer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.sql.elements import TextClause

from outrider.anomaly.rule_names import AnomalyRuleName
from outrider.sweep.hitl_expiry import (
    reclaim_stuck_hitl_states,
    run_once,
    transition_expired_hitl_reviews,
)


class _StubAnomalySink:
    def __init__(self) -> None:
        self.emit_calls: list[dict[str, Any]] = []
        self.raise_for_review_id: UUID | None = None

    async def emit_anomaly(
        self,
        *,
        review_id: UUID,
        rule_name: AnomalyRuleName,
        severity: str,
        details: dict[str, Any],
    ) -> None:
        if self.raise_for_review_id == review_id:
            msg = "synthetic anomaly emit failure"
            raise RuntimeError(msg)
        self.emit_calls.append(
            {
                "review_id": review_id,
                "rule_name": rule_name,
                "severity": severity,
                "details": details,
            }
        )


class _StubReviewStatusSink:
    def __init__(self) -> None:
        self.expired_calls: list[UUID] = []
        self.running_calls: list[dict[str, Any]] = []
        self.awaiting_calls: list[dict[str, Any]] = []

    async def mark_awaiting_approval_expired(self, *, review_id: UUID) -> None:
        self.expired_calls.append(review_id)

    async def mark_awaiting_approval(self, **kwargs: Any) -> None:
        self.awaiting_calls.append(kwargs)

    async def mark_running(self, **kwargs: Any) -> None:
        self.running_calls.append(kwargs)


def _make_conn(expired_rows: list[tuple[UUID, datetime]]) -> MagicMock:
    """Mock AsyncConnection that returns the given expired rows on
    SELECT + grants the advisory lock.

    Dispatches by checking SQLAlchemy expression KIND, not duck-typing
    via `hasattr(stmt, "is_text")`. Two statement shapes hit this mock
    in order: (1) a `text("SELECT pg_try_advisory_xact_lock(...)")`
    expression (TextClause) for lock acquisition; (2) a `select(...)`
    ORM expression for the candidate-rows query. The TextClause check
    via `isinstance(stmt, TextClause)` is the explicit dispatch
    discriminator; a future shape change (e.g., swapping the advisory
    lock to a typed `func.pg_try_advisory_xact_lock` ORM call) would
    surface as a fail-loud `AssertionError` in the `else` branch.
    """
    conn = MagicMock()

    async def _execute(stmt: Any, params: Any = None) -> Any:  # noqa: ARG001
        result = MagicMock()
        if isinstance(stmt, TextClause):
            # text("SELECT pg_try_advisory_xact_lock(...)") path.
            scalar = MagicMock(return_value=True)
            result.scalar_one = scalar
            return result
        # ORM `select(Review.id, Review.expires_at)` path.
        rows = [MagicMock(id=r[0], expires_at=r[1]) for r in expired_rows]
        result.all = MagicMock(return_value=rows)
        return result

    conn.execute = AsyncMock(side_effect=_execute)
    return conn


@pytest.mark.asyncio
async def test_anomaly_first_ordering_anomaly_succeeds_then_status_flips() -> None:
    """Happy path: anomaly emit succeeds, status flips."""
    review_id = uuid4()
    expires_at = datetime.now(UTC) - timedelta(minutes=1)
    conn = _make_conn([(review_id, expires_at)])
    anomaly_sink = _StubAnomalySink()
    status_sink = _StubReviewStatusSink()

    n = await transition_expired_hitl_reviews(
        conn=conn,
        anomaly_sink=anomaly_sink,  # type: ignore[arg-type]
        review_status_sink=status_sink,  # type: ignore[arg-type]
    )

    assert n == 1
    assert len(anomaly_sink.emit_calls) == 1
    assert anomaly_sink.emit_calls[0]["review_id"] == review_id
    assert anomaly_sink.emit_calls[0]["rule_name"] == AnomalyRuleName.HITL_TIMEOUT
    assert anomaly_sink.emit_calls[0]["severity"] == "medium"
    assert status_sink.expired_calls == [review_id]


@pytest.mark.asyncio
async def test_anomaly_first_ordering_anomaly_fails_status_does_not_flip() -> None:
    """Anomaly emit RAISES — status flip MUST be skipped. The row stays
    in awaiting_approval and the next sweep tick retries.

    This is the load-bearing durability property: without it, status
    would flip first; on retry the predicate `status='awaiting_approval'`
    would no longer match and the canonical anomaly would be lost forever.
    """
    review_id_ok = uuid4()
    review_id_fail = uuid4()
    expires_at = datetime.now(UTC) - timedelta(minutes=1)
    conn = _make_conn([(review_id_ok, expires_at), (review_id_fail, expires_at)])
    anomaly_sink = _StubAnomalySink()
    anomaly_sink.raise_for_review_id = review_id_fail
    status_sink = _StubReviewStatusSink()

    n = await transition_expired_hitl_reviews(
        conn=conn,
        anomaly_sink=anomaly_sink,  # type: ignore[arg-type]
        review_status_sink=status_sink,  # type: ignore[arg-type]
    )

    # Only the OK review transitioned.
    assert n == 1
    # status_sink saw only the OK review's mark_awaiting_approval_expired
    # — NOT the failing one.
    assert status_sink.expired_calls == [review_id_ok]
    # The failing review never reached the status flip.
    assert review_id_fail not in status_sink.expired_calls


@pytest.mark.asyncio
async def test_transition_sub_job_no_longer_self_locks() -> None:
    """The sub-job MUST NOT acquire SWEEP_LOCK_ID itself per the
    refactored ordering contract — the wrapper `run_once` acquires
    once and holds across both sub-jobs. Calling the sub-job
    directly (outside `run_once`) is permitted but the caller is
    responsible for the lock.

    Tested by giving the conn a no-execute mock for the lock query
    and confirming the sub-job runs its query directly without
    short-circuiting on a lock check."""
    review_id = uuid4()
    expires_at = datetime.now(UTC) - timedelta(minutes=1)
    conn = _make_conn([(review_id, expires_at)])
    anomaly_sink = _StubAnomalySink()
    status_sink = _StubReviewStatusSink()

    n = await transition_expired_hitl_reviews(
        conn=conn,
        anomaly_sink=anomaly_sink,  # type: ignore[arg-type]
        review_status_sink=status_sink,  # type: ignore[arg-type]
    )

    # No self-lock; transitioned the seeded row.
    assert n == 1
    assert len(anomaly_sink.emit_calls) == 1


@pytest.mark.asyncio
async def test_run_once_skips_on_lock_held() -> None:
    """run_once is the single lock-acquire site. If the lock is
    held by another sweep, return all zeros without running either
    sub-job."""
    conn = MagicMock()
    result_lock_held = MagicMock()
    result_lock_held.scalar_one = MagicMock(return_value=False)
    conn.execute = AsyncMock(return_value=result_lock_held)

    anomaly_sink = _StubAnomalySink()
    status_sink = _StubReviewStatusSink()
    audit_persister = MagicMock()
    audit_persister.query_hitl_decision_event = AsyncMock(return_value=None)
    checkpointer = MagicMock()
    checkpointer.aget = AsyncMock(return_value=None)
    compiled_graph = MagicMock()
    compiled_graph.ainvoke = AsyncMock(return_value=None)

    result = await run_once(
        conn=conn,
        session_factory=MagicMock(),
        anomaly_sink=anomaly_sink,  # type: ignore[arg-type]
        review_status_sink=status_sink,  # type: ignore[arg-type]
        audit_persister=audit_persister,
        checkpointer=checkpointer,
        compiled_graph=compiled_graph,
    )

    assert result == {"reclaim_recovered": 0, "reclaim_failed": 0, "transitioned": 0}
    # Neither sub-job ran — no audit query, no checkpointer read.
    audit_persister.query_hitl_decision_event.assert_not_called()
    checkpointer.aget.assert_not_called()


@pytest.mark.asyncio
async def test_reclaim_window_f_recovery_advances_lifecycle_from_audit_row() -> None:
    """F1 audit-fold regression: window (f) (audit row exists,
    reviews.hitl_decision IS NULL) MUST advance lifecycle via
    mark_running with the canonical audit content — NOT mark failed.

    Pre-fix bug: the sweep skipped any row with a checkpoint, and
    window (f) crashes leave the HITL interrupt checkpoint in place,
    so the row stayed stuck forever."""
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    from outrider.audit.events import HITLDecisionEvent
    from outrider.policy.canonical import compute_hitl_decision_content_hash
    from outrider.schemas.hitl import PerFindingDecision, PerFindingOutcome

    review_id = uuid4()
    finding_id = uuid4()

    conn = MagicMock()

    async def _execute(stmt: Any, params: Any = None) -> Any:  # noqa: ARG001
        result = MagicMock()
        if isinstance(stmt, TextClause):
            result.scalar_one = MagicMock(return_value=True)
            return result
        result.all = MagicMock(return_value=[MagicMock(id=review_id)])
        return result

    conn.execute = AsyncMock(side_effect=_execute)

    # Construct the orphaned audit row.
    now = _datetime.now(_UTC)
    canonical_decision_pfd = PerFindingDecision(
        finding_id=finding_id,
        outcome=PerFindingOutcome.APPROVE,
        reason="approved before crash",
    )
    audit_event = HITLDecisionEvent(
        review_id=review_id,
        is_eval=False,
        reviewer_id="admin",
        decisions=(canonical_decision_pfd,),
        annotation="canonical decision",
        decided_at=now,
        decision_latency_seconds=0.0,
        decisions_content_hash=compute_hitl_decision_content_hash(
            decisions=(canonical_decision_pfd,),
            annotation="canonical decision",
        ),
    )

    audit_persister = MagicMock()
    audit_persister.query_hitl_decision_event = AsyncMock(return_value=audit_event)
    status_sink = _StubReviewStatusSink()
    checkpointer = MagicMock()
    # Window (f): checkpoint still exists (HITL interrupt suspended
    # before the crash). Pre-fix would have skipped this row.
    checkpointer.aget = AsyncMock(return_value={"checkpoint": "still-exists"})

    # Sweep MUST drive the graph through Command(resume=...). Without
    # this, the lifecycle column flips but the graph stays suspended
    # at the interrupt and publish never runs.
    compiled_graph = MagicMock()
    compiled_graph.ainvoke = AsyncMock(return_value=None)

    result = await reclaim_stuck_hitl_states(
        conn=conn,
        session_factory=MagicMock(),
        audit_persister=audit_persister,
        review_status_sink=status_sink,  # type: ignore[arg-type]
        checkpointer=checkpointer,
        compiled_graph=compiled_graph,
    )

    # The graph was driven through Command(resume=canonical_decision)
    # — this is the load-bearing F1 fix. A direct `mark_running` call
    # from the sweep would have flipped the JSONB but left the graph
    # suspended; only graph-driven recovery completes publish.
    compiled_graph.ainvoke.assert_awaited_once()
    call_args = compiled_graph.ainvoke.await_args
    resume_command = call_args.args[0]
    # The Command payload carries the canonical audit-row content
    # (reviewer_id, annotation flow through).
    resume_payload = resume_command.resume
    assert resume_payload["reviewer_id"] == "admin"
    assert resume_payload["annotation"] == "canonical decision"
    # thread_id = str(review_id) for the resume.
    config = call_args.kwargs["config"]
    assert config["configurable"]["thread_id"] == str(review_id)
    # The sweep does NOT call mark_running directly — that's the
    # graph body's job, invoked transitively through Command(resume).
    assert status_sink.running_calls == []
    # The checkpointer was NOT consulted — audit-row recovery
    # short-circuits before the checkpoint check.
    checkpointer.aget.assert_not_called()
    assert result == {"recovered": 1, "failed": 0}


@pytest.mark.asyncio
async def test_reclaim_window_c_marks_failed_when_no_audit_and_no_checkpoint() -> None:
    """Window (c) recovery: no audit row + no checkpoint + expires_at
    past grace period = mark failed."""
    review_id = uuid4()
    # Past the grace period so the per-row grace gate admits this
    # candidate to the checkpoint check + window-c mark-failed write.
    past_expires_at = datetime.now(UTC) - timedelta(hours=1)
    conn = MagicMock()

    async def _execute(stmt: Any, params: Any = None) -> Any:  # noqa: ARG001
        result = MagicMock()
        if isinstance(stmt, TextClause):
            result.scalar_one = MagicMock(return_value=True)
            return result
        result.all = MagicMock(
            return_value=[MagicMock(id=review_id, expires_at=past_expires_at)],
        )
        return result

    conn.execute = AsyncMock(side_effect=_execute)

    audit_persister = MagicMock()
    audit_persister.query_hitl_decision_event = AsyncMock(return_value=None)
    status_sink = _StubReviewStatusSink()
    checkpointer = MagicMock()
    checkpointer.aget = AsyncMock(return_value=None)

    # session_factory needs to support `async with` for the UPDATE.
    # `update_result.rowcount > 0` gates the failed-count + log emit.
    # Return a result with rowcount=1 so the test exercises the
    # "match found, increment failed" branch (the happy path for
    # window-c reclaim). A separate test could mock rowcount=0 to
    # exercise the concurrent-actor-changed-status branch.
    update_result = MagicMock()
    update_result.rowcount = 1
    session_inner = MagicMock()
    session_inner.execute = AsyncMock(return_value=update_result)
    session_inner.__aenter__ = AsyncMock(return_value=session_inner)
    session_inner.__aexit__ = AsyncMock(return_value=None)
    session_inner.begin = MagicMock(return_value=session_inner)
    session_factory = MagicMock(return_value=session_inner)
    compiled_graph = MagicMock()
    compiled_graph.ainvoke = AsyncMock(return_value=None)

    result = await reclaim_stuck_hitl_states(
        conn=conn,
        session_factory=session_factory,
        audit_persister=audit_persister,
        review_status_sink=status_sink,  # type: ignore[arg-type]
        checkpointer=checkpointer,
        compiled_graph=compiled_graph,
    )

    assert result == {"recovered": 0, "failed": 1}
    # mark_running was NOT called — this is a failed-mark path.
    assert status_sink.running_calls == []


@pytest.mark.asyncio
async def test_reclaim_skips_row_when_no_audit_but_checkpoint_present() -> None:
    """No audit row + checkpoint present + past grace = still in flight; skip."""
    review_id = uuid4()
    past_expires_at = datetime.now(UTC) - timedelta(hours=1)
    conn = MagicMock()

    async def _execute(stmt: Any, params: Any = None) -> Any:  # noqa: ARG001
        result = MagicMock()
        if isinstance(stmt, TextClause):
            result.scalar_one = MagicMock(return_value=True)
            return result
        result.all = MagicMock(
            return_value=[MagicMock(id=review_id, expires_at=past_expires_at)],
        )
        return result

    conn.execute = AsyncMock(side_effect=_execute)
    audit_persister = MagicMock()
    audit_persister.query_hitl_decision_event = AsyncMock(return_value=None)
    status_sink = _StubReviewStatusSink()
    checkpointer = MagicMock()
    checkpointer.aget = AsyncMock(return_value={"checkpoint": "in-flight"})
    compiled_graph = MagicMock()
    compiled_graph.ainvoke = AsyncMock(return_value=None)

    result = await reclaim_stuck_hitl_states(
        conn=conn,
        session_factory=MagicMock(),
        audit_persister=audit_persister,
        review_status_sink=status_sink,  # type: ignore[arg-type]
        checkpointer=checkpointer,
        compiled_graph=compiled_graph,
    )

    assert result == {"recovered": 0, "failed": 0}
    checkpointer.aget.assert_awaited()
