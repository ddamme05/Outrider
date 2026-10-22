# HITL-expiry sweep per specs/2026-05-26-hitl-node.md Group 8.
"""Recover interrupted approvals, then expire undecided requests.

`reclaim_stuck_hitl_states` resumes saved decisions from immutable audit rows
(`audit-events-append-only`);
without a decision or checkpoint, it can mark an old waiting review failed.
`transition_expired_hitl_reviews` records `hitl_timeout` before changing status,
so an anomaly-write failure leaves the review eligible for retry.

`run_once` holds `SWEEP_LOCK_ID` once across both jobs. Reclaim runs first so
saved decisions are recovered before timeout handling. `run_all_sweeps` keeps
the same transaction lock through the wider maintenance sequence.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from langgraph.types import Command
from sqlalchemy import select, text, update

from outrider.anomaly.rule_names import AnomalyRuleName, AnomalySeverity
from outrider.db.models.reviews import Review
from outrider.schemas.hitl import HITLDecision
from outrider.sweep.purge_expired import SWEEP_LOCK_ID

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

    from outrider.anomaly.sinks import AnomalySink
    from outrider.audit.persister import AuditPersister
    from outrider.db.sinks import ReviewStatusSink


logger = logging.getLogger(__name__)


# Grace period before `reclaim_stuck_hitl_states` marks a stuck row
# `failed`. Operators wanting longer triage windows can extend; the
# default trades fast-recovery (5 min) against the cost of a
# spuriously-classified-failed row for a slow checkpointer write.
_DEFAULT_RECLAIM_GRACE_PERIOD: Final[timedelta] = timedelta(minutes=5)


# Severity for the canonical `hitl_timeout` anomaly per
# `docs/spec.md` §16 line 1421.
_HITL_TIMEOUT_SEVERITY: Final[AnomalySeverity] = AnomalySeverity.MEDIUM


async def _try_acquire_sweep_lock(conn: AsyncConnection) -> bool:
    """Try to acquire the transaction-scoped sweep advisory lock."""
    result = await conn.execute(
        text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
        {"lock_id": SWEEP_LOCK_ID},
    )
    return bool(result.scalar_one())


async def transition_expired_hitl_reviews(
    *,
    conn: AsyncConnection,
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
    audit_persister: AuditPersister,
) -> int:
    """Expire production approvals after recording their timeout anomaly.

    A saved `HITLDecisionEvent` means the reviewer decided: skip timeout handling
    and leave graph recovery to reclaim. Otherwise emit `hitl_timeout` first,
    then mark `awaiting_approval_expired`. Per-row errors leave the next tick
    able to retry; anomaly inserts deduplicate by review and rule.

    Count status-sink calls that return, including conditional-update no-ops.
    The caller must already hold
    `SWEEP_LOCK_ID` across reclaim and transition in one transaction.
    """
    # Query expired rows. Uses the partial index
    # `ix_reviews_awaiting_approval_expires_at` from Group 3.
    # `is_eval=False` filter per `docs/testing.md:59` ("Sweep jobs:
    # ignore is_eval=True rows. An eval review that checkpoints
    # mid-HITL won't be marked stuck by the sweep.") — the contract
    # the dashboard's is_eval filter depends on. Latent under test-DB
    # isolation today; the filter is the structural defense if
    # eval data ever shares a DB with prod.
    result = await conn.execute(
        select(Review.id, Review.expires_at, Review.is_eval).where(
            Review.status == "awaiting_approval",
            Review.expires_at < datetime.now(UTC),
            Review.is_eval.is_(False),
        )
    )
    expired_rows = list(result.all())

    transitioned = 0
    for row in expired_rows:
        review_id = row.id
        expires_at = row.expires_at
        # Read is_eval from the row directly rather than hardcoding
        # `False` at the call site. The WHERE clause above filters to
        # production rows; the row-derived value is `False` today, but
        # if the WHERE clause is ever weakened (e.g., a future "eval
        # also expires" feature flag) the emit's `is_eval` argument
        # stays self-consistent with the actual review row. Sharp-edges
        # Phase 2 audit finding: hardcoded value drift-risk under
        # future WHERE-clause changes — derive from row instead.
        review_is_eval = row.is_eval
        # Window-(f) skip: if a HITLDecisionEvent already exists for
        # this review_id, the reviewer decided — this is a stuck
        # window-f row, not a true timeout. `reclaim_stuck_hitl_states`
        # owns the graph-drive recovery; skip to avoid double-classifying
        # as both hitl_timeout (here) and recovered-via-reclaim (next
        # tick). If the audit query itself fails (transient DB error),
        # log and continue — defensive against blocking the entire sweep
        # on one bad row.
        try:
            audit_decision_event = await audit_persister.query_hitl_decision_event(
                review_id=review_id,
            )
        except Exception:
            logger.exception(
                "hitl_timeout_audit_query_failed",
                extra={"review_id": str(review_id)},
            )
            continue
        if audit_decision_event is not None:
            logger.info(
                "hitl_timeout_skip_decided_row",
                extra={
                    "review_id": str(review_id),
                    "note": (
                        "row in awaiting_approval past expires_at carries an "
                        "orphaned HITLDecisionEvent (window-f); skipping "
                        "hitl_timeout — reclaim_stuck_hitl_states owns recovery."
                    ),
                },
            )
            continue

        try:
            # `is_eval` derived from the row above — self-consistent
            # with the actual review even if the WHERE clause's
            # `is_eval=False` filter is ever weakened. Loud-failure
            # discipline per docs/testing.md: every is_eval-bearing
            # producer call site sets the flag explicitly.
            await anomaly_sink.emit_anomaly(
                review_id=review_id,
                rule_name=AnomalyRuleName.HITL_TIMEOUT,
                severity=_HITL_TIMEOUT_SEVERITY,
                details={
                    "expired_at": datetime.now(UTC).isoformat(),
                    "expires_at": expires_at.isoformat() if expires_at else None,
                },
                is_eval=review_is_eval,
            )
        except Exception:
            # Anomaly-first ordering: do NOT flip status if anomaly
            # emit failed. The row stays in awaiting_approval and the
            # next sweep tick retries.
            logger.exception(
                "hitl_timeout_anomaly_emit_failed",
                extra={"review_id": str(review_id)},
            )
            continue

        # Anomaly persisted (or already-existed via on-conflict-do-
        # nothing). Now flip status. Per-row try/except so one failing
        # write (transient DB error, lock timeout) doesn't abort the
        # entire sweep — the anomaly is already durable, the next
        # sweep tick will retry the mark via the same anomaly-already-
        # exists no-op pattern.
        try:
            await review_status_sink.mark_awaiting_approval_expired(review_id=review_id)
        except Exception:
            logger.exception(
                "hitl_timeout_mark_expired_failed",
                extra={"review_id": str(review_id)},
            )
            continue
        transitioned += 1
        logger.info(
            "hitl_timeout_transitioned",
            extra={"review_id": str(review_id)},
        )

    return transitioned


async def reclaim_stuck_hitl_states(
    *,
    conn: AsyncConnection,
    session_factory: async_sessionmaker[AsyncSession],
    audit_persister: AuditPersister,
    review_status_sink: ReviewStatusSink,
    checkpointer: BaseCheckpointSaver[Any],
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    grace_period: timedelta = _DEFAULT_RECLAIM_GRACE_PERIOD,
) -> dict[str, int]:
    """Recover saved decisions, then check running rows still paused at HITL.

    The awaiting scan can either resume a canonical audit decision or mark an
    old row failed when neither decision nor checkpoint exists. The running
    scan resumes only saved decisions still waiting at the HITL node.
    Return `recovered` across both scans and `failed` from the awaiting scan.

    The caller must hold `SWEEP_LOCK_ID`; direct callers also need a positive
    grace period. Each scan reads its own clock for its distinct age check.
    """
    # This public entry point can be called without the validating runner.
    # A non-positive grace period moves the failure cutoff into the future.
    if grace_period <= timedelta(0):
        msg = (
            f"reclaim_stuck_hitl_states: grace_period must be > timedelta(0); "
            f"got {grace_period!r}. A non-positive grace_period makes "
            f"the grace gate admit every awaiting-approval row regardless "
            f"of age — mass-marks-failed."
        )
        raise ValueError(msg)
    grace_cutoff = datetime.now(UTC) - grace_period
    awaiting_recovered, failed = await _reclaim_awaiting_approval_rows(
        conn=conn,
        session_factory=session_factory,
        audit_persister=audit_persister,
        checkpointer=checkpointer,
        compiled_graph=compiled_graph,
        grace_cutoff=grace_cutoff,
    )
    running_recovered = await _reclaim_running_rows_stuck_at_hitl(
        conn=conn,
        audit_persister=audit_persister,
        compiled_graph=compiled_graph,
        grace_period=grace_period,
    )

    return {
        "recovered": awaiting_recovered + running_recovered,
        "failed": failed,
    }


async def _reclaim_awaiting_approval_rows(
    *,
    conn: AsyncConnection,
    session_factory: async_sessionmaker[AsyncSession],
    audit_persister: AuditPersister,
    checkpointer: BaseCheckpointSaver[Any],
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    grace_cutoff: datetime,
) -> tuple[int, int]:
    """Recover saved decisions from awaiting states; fail old rows with no checkpoint.

    Both `awaiting_approval` and `awaiting_approval_expired` remain candidates,
    so a prior timeout transition cannot orphan a saved decision.

    Window (f): a canonical `HITLDecisionEvent` exists. Resume the graph with it
    immediately, without a grace gate. A direct lifecycle update would leave the
    graph paused and make dashboard retries reject its cached decision.

    Window (c): no audit decision exists. Only after `expires_at < grace_cutoff`,
    check the checkpoint; any checkpoint protects the row from this failure
    classification. With neither, conditionally mark the awaiting row failed.
    Count only matched updates, since another actor can change status after
    the candidate read. Return `(recovered, failed)`.
    """
    # Keep expired awaiting rows reachable for canonical-decision recovery.
    # Production maintenance must never mutate eval review lifecycles.
    result = await conn.execute(
        select(Review.id, Review.expires_at).where(
            Review.status.in_(("awaiting_approval", "awaiting_approval_expired")),
            Review.is_eval.is_(False),
        )
    )
    candidate_rows = list(result.all())

    recovered = 0
    failed = 0
    for row in candidate_rows:
        review_id = row.id
        row_expires_at = row.expires_at
        # The audit decision, rather than checkpoint presence, selects recovery.
        try:
            audit_decision_event = await audit_persister.query_hitl_decision_event(
                review_id=review_id,
            )
        except Exception:
            logger.exception(
                "hitl_reclaim_audit_query_failed",
                extra={"review_id": str(review_id)},
            )
            continue

        if audit_decision_event is not None:
            # Resume the graph; updating only the lifecycle would leave HITL paused.
            canonical_decision = HITLDecision(
                reviewer_id=audit_decision_event.reviewer_id,
                decisions=audit_decision_event.decisions,
                annotation=audit_decision_event.annotation,
                decided_at=audit_decision_event.decided_at,
            )
            recovery_config: RunnableConfig = {"configurable": {"thread_id": str(review_id)}}
            try:
                await compiled_graph.ainvoke(
                    Command(resume=canonical_decision.model_dump(mode="json")),
                    config=recovery_config,
                )
            except Exception:
                # The audit decision stays canonical even if graph drive fails
                # after a lifecycle write; the next tick rechecks the resulting state.
                logger.exception(
                    "hitl_reclaim_graph_drive_failed",
                    extra={
                        "review_id": str(review_id),
                        "audit_event_id": str(audit_decision_event.event_id),
                    },
                )
                continue
            recovered += 1
            logger.warning(
                "hitl_reclaim_recovered_via_command_resume",
                extra={
                    "review_id": str(review_id),
                    "audit_event_id": str(audit_decision_event.event_id),
                    "note": (
                        "Row stuck in awaiting_approval past grace period with "
                        "an orphaned HITLDecisionEvent in audit_events; "
                        "drove graph through Command(resume=canonical_decision) "
                        "to advance lifecycle AND complete publish (window-f recovery)."
                    ),
                },
            )
            continue

        # No saved decision: wait beyond expiry plus grace before classifying failure.
        if row_expires_at is None or row_expires_at >= grace_cutoff:
            continue

        # Any checkpoint protects this row from the no-checkpoint failure path.
        thread_config = {"configurable": {"thread_id": str(review_id)}}
        try:
            checkpoint = await checkpointer.aget(thread_config)  # type: ignore[arg-type]
        except Exception:
            logger.exception(
                "hitl_reclaim_checkpointer_read_failed",
                extra={"review_id": str(review_id)},
            )
            continue

        if checkpoint is not None:
            # Checkpoint presence alone does not prove the graph is stuck.
            continue

        # No decision or checkpoint after grace: conditionally fail the awaiting
        # row. Count/log only matched writes to avoid reporting a concurrent
        # recovery as failure.
        async with session_factory() as session, session.begin():
            update_result = await session.execute(
                update(Review)
                .where(
                    Review.id == review_id,
                    Review.status.in_(("awaiting_approval", "awaiting_approval_expired")),
                )
                .values(status="failed")
            )
        # Async stubs expose Result; the runtime UPDATE result supplies rowcount.
        update_rowcount: int = getattr(update_result, "rowcount", 0) or 0
        if update_rowcount > 0:
            failed += 1
            logger.warning(
                "hitl_reclaim_marked_failed",
                extra={
                    "review_id": str(review_id),
                    "note": (
                        "Row stuck past grace period with no audit row AND no "
                        "LangGraph checkpoint; reclaimed as failed for operator "
                        "triage (window-c crash)."
                    ),
                },
            )
        else:
            logger.info(
                "hitl_reclaim_status_changed_concurrently",
                extra={
                    "review_id": str(review_id),
                    "note": (
                        "Window-(c) mark-failed UPDATE matched 0 rows — status "
                        "changed between SELECT and UPDATE (parallel sweep, "
                        "manual operator action, or peer-process window-(f) "
                        "graph-driven recovery). No phantom failed-count emit."
                    ),
                },
            )

    return recovered, failed


async def _reclaim_running_rows_stuck_at_hitl(
    *,
    conn: AsyncConnection,
    audit_persister: AuditPersister,
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    grace_period: timedelta,
) -> int:
    """Resume post-decision running rows whose graph still waits at HITL (window g).

    The row must have `hitl_decision`, a canonical audit decision older than the
    grace period, and `hitl` in the graph's next nodes. This excludes pre-HITL
    work and graphs that already advanced to publish. Resume with the immutable
    audit decision so the HITL body can finish its phase end and state delta;
    its natural-key audit writes and lifecycle predicates make retries safe.

    Return the successful graph-drive count. Read a fresh clock after the
    awaiting scan: this age check uses `decided_at`, rather than `expires_at`.
    """
    # A cached decision distinguishes post-HITL running rows from earlier work.
    candidate_result = await conn.execute(
        select(Review.id).where(
            Review.status == "running",
            Review.hitl_decision.is_not(None),
            Review.is_eval.is_(False),
        )
    )
    candidate_rows = list(candidate_result.all())

    checked_at = datetime.now(UTC)

    recovered = 0
    for row in candidate_rows:
        review_id = row.id

        # The cached decision needs its canonical immutable audit row.
        try:
            decision_event = await audit_persister.query_hitl_decision_event(
                review_id=review_id,
            )
        except Exception:
            logger.exception(
                "hitl_reclaim_g_audit_query_failed",
                extra={"review_id": str(review_id)},
            )
            continue
        if decision_event is None:
            continue

        # Give a healthy resumed graph time to advance before attempting recovery.
        if decision_event.decided_at + grace_period > checked_at:
            continue

        # Resume only a graph that still waits at HITL, not publish or terminal work.
        thread_config: RunnableConfig = {"configurable": {"thread_id": str(review_id)}}
        try:
            state_snapshot = await compiled_graph.aget_state(thread_config)
        except Exception:
            logger.exception(
                "hitl_reclaim_g_state_read_failed",
                extra={"review_id": str(review_id)},
            )
            continue
        next_nodes = getattr(state_snapshot, "next", ()) or ()
        if "hitl" not in next_nodes:
            # Publish may be running or complete; this recovery does not apply.
            continue

        # Replay the canonical decision through the idempotent HITL body.
        canonical_decision = HITLDecision(
            reviewer_id=decision_event.reviewer_id,
            decisions=decision_event.decisions,
            annotation=decision_event.annotation,
            decided_at=decision_event.decided_at,
        )
        try:
            await compiled_graph.ainvoke(
                Command(resume=canonical_decision.model_dump(mode="json")),
                config=thread_config,
            )
        except Exception:
            logger.exception(
                "hitl_reclaim_g_graph_drive_failed",
                extra={
                    "review_id": str(review_id),
                    "audit_event_id": str(decision_event.event_id),
                },
            )
            continue
        recovered += 1
        logger.warning(
            "hitl_reclaim_recovered_window_g_via_command_resume",
            extra={
                "review_id": str(review_id),
                "audit_event_id": str(decision_event.event_id),
                "note": (
                    "Row at status='running' + hitl_decision populated with "
                    "LangGraph state still at hitl interrupt past grace "
                    "period; drove graph through Command(resume=) to "
                    "complete phase-end + state-delta + publish "
                    "(window-g recovery per spec line 215)."
                ),
            },
        )

    return recovered


async def run_once(
    *,
    conn: AsyncConnection,
    session_factory: async_sessionmaker[AsyncSession],
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
    audit_persister: AuditPersister,
    checkpointer: BaseCheckpointSaver[Any],
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    grace_period: timedelta = _DEFAULT_RECLAIM_GRACE_PERIOD,
) -> dict[str, int]:
    """Run reclaim before expiry under one transaction-scoped advisory lock.

    Acquire `SWEEP_LOCK_ID` once for both jobs; the caller's transaction must
    span the sequence. Recover saved decisions before timeout classification.
    Return reclaim/transition counts, or all zeros when the lock is held by a
    peer. Reject non-positive grace periods even for direct callers.
    """
    # Direct callers also need the positive-grace guard before any recovery write.
    if grace_period <= timedelta(0):
        msg = (
            f"hitl_expiry.run_once: grace_period must be > timedelta(0); "
            f"got {grace_period!r}. A non-positive grace_period makes "
            f"reclaim's grace gate admit every awaiting-approval row "
            f"regardless of age."
        )
        raise ValueError(msg)

    if not await _try_acquire_sweep_lock(conn):
        logger.info("hitl_sweep_skipped: advisory lock held by another sweep process")
        return {"reclaim_recovered": 0, "reclaim_failed": 0, "transitioned": 0}

    reclaim_result = await reclaim_stuck_hitl_states(
        conn=conn,
        session_factory=session_factory,
        audit_persister=audit_persister,
        review_status_sink=review_status_sink,
        checkpointer=checkpointer,
        compiled_graph=compiled_graph,
        grace_period=grace_period,
    )
    transitioned = await transition_expired_hitl_reviews(
        conn=conn,
        anomaly_sink=anomaly_sink,
        review_status_sink=review_status_sink,
        audit_persister=audit_persister,
    )
    return {
        "reclaim_recovered": reclaim_result["recovered"],
        "reclaim_failed": reclaim_result["failed"],
        "transitioned": transitioned,
    }


__all__ = [
    "reclaim_stuck_hitl_states",
    "run_once",
    "transition_expired_hitl_reviews",
]
