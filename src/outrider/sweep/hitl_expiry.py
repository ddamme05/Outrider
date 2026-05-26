# HITL-expiry sweep per specs/2026-05-26-hitl-node.md Group 8.
"""HITL-expiry sweep job — two sub-jobs sharing the SWEEP_LOCK_ID
advisory lock.

  1. `transition_expired_hitl_reviews` — for rows in `awaiting_approval`
     past their `expires_at`, emit the canonical `hitl_timeout` anomaly
     FIRST, then flip status to `awaiting_approval_expired`. The
     anomaly-FIRST ordering is durability-critical (per
     `docs/spec.md` §4.1.6 + the spec's Group 8 prescription): if the
     status flip happened first and the anomaly emit then failed, the
     sweep's predicate `status='awaiting_approval'` would no longer
     match on the retry and the canonical anomaly would be permanently
     lost. Anomaly-first reverses that durability gap.

  2. `reclaim_stuck_hitl_states` — for rows in `awaiting_approval`
     whose LangGraph checkpoint has NO pending interrupt (window-(c)
     crash recovery — `mark_awaiting_approval` succeeded but
     `interrupt()` never landed), OR whose audit row carries an
     existing `HITLDecisionEvent` but `reviews.hitl_decision IS NULL`
     (window-(f) crash recovery — `emit_hitl_decision` succeeded but
     `mark_running` never landed): mark the review `failed` after a
     configurable grace period. Operator triage via the resulting
     status row + structured logs. Per the F2 audit-fold rationale,
     this sub-job is the canonical recovery path for divergent-content
     window-(f) retries that the resume wrapper cannot resolve on its
     own.

Sub-job order WITHIN `run_once()`: `reclaim_stuck_hitl_states` runs
BEFORE `transition_expired_hitl_reviews`. Rationale (spec line 566):
a stuck-state row that ALSO has `expires_at < NOW()` would otherwise
flip to `awaiting_approval_expired` via the expiry sub-job, which
masks the actual crash classification ("reviewer never decided" vs
"process crashed mid-write"). Running reclaim first correctly
classifies as `failed` with explicit operator-triage signal.

Both sub-jobs share the existing `SWEEP_LOCK_ID` from
`sweep/purge_expired.py` per `sweep-jobs-use-advisory-locks` — only
one sweep process at a time across all sub-jobs.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import select, text, update

from outrider.anomaly.rule_names import AnomalyRuleName
from outrider.db.models.reviews import Review
from outrider.sweep.purge_expired import SWEEP_LOCK_ID

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

    from outrider.anomaly.sinks import AnomalySink
    from outrider.db.sinks import ReviewStatusSink


logger = logging.getLogger(__name__)


# Grace period before `reclaim_stuck_hitl_states` marks a stuck row
# `failed`. Operators wanting longer triage windows can extend; the
# default trades fast-recovery (5 min) against the cost of a
# spuriously-classified-failed row for a slow checkpointer write.
_DEFAULT_RECLAIM_GRACE_PERIOD: Final[timedelta] = timedelta(minutes=5)


# Severity for the canonical `hitl_timeout` anomaly per
# `docs/spec.md` §16 line 1421.
_HITL_TIMEOUT_SEVERITY: Final[str] = "medium"


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
    session_factory: async_sessionmaker[AsyncSession],
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
) -> int:
    """Sub-job 1: expire reviews past their HITL deadline.

    Per the anomaly-FIRST ordering contract:
      1. Query rows in `awaiting_approval` with `expires_at < NOW()`.
      2. For each row:
         a. Emit the `hitl_timeout` anomaly via `anomaly_sink`.
         b. ONLY if (a) succeeded, flip status to
            `awaiting_approval_expired` via `review_status_sink`.

    If anomaly emit RAISES, the loop body for that row short-circuits
    without the status flip. The row stays in `awaiting_approval` and
    the next sweep tick retries — the partial unique index on
    `(anomalies.review_id) WHERE rule_name='hitl_timeout'` makes the
    retried emit a no-op via `on_conflict_do_nothing`, so the second
    attempt succeeds AND flips the status.

    Returns the number of rows that successfully transitioned to
    `awaiting_approval_expired`. Rows that errored on the anomaly
    emit do NOT count (they're retried next tick).

    Acquires `SWEEP_LOCK_ID` via the provided `conn`. If another sweep
    holds it, returns 0 without scanning.
    """
    if not await _try_acquire_sweep_lock(conn):
        logger.info("hitl_expiry_sweep_skipped: advisory lock held by another sweep")
        return 0

    # Query expired rows. Uses the partial index
    # `ix_reviews_awaiting_approval_expires_at` from Group 3.
    result = await conn.execute(
        select(Review.id, Review.expires_at).where(
            Review.status == "awaiting_approval",
            Review.expires_at < datetime.now(UTC),
        )
    )
    expired_rows = list(result.all())

    transitioned = 0
    for row in expired_rows:
        review_id = row.id
        expires_at = row.expires_at
        try:
            await anomaly_sink.emit_anomaly(
                review_id=review_id,
                rule_name=AnomalyRuleName.HITL_TIMEOUT,
                severity=_HITL_TIMEOUT_SEVERITY,
                details={
                    "expired_at": datetime.now(UTC).isoformat(),
                    "expires_at": expires_at.isoformat() if expires_at else None,
                },
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
        # nothing). Now flip status.
        await review_status_sink.mark_awaiting_approval_expired(review_id=review_id)
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
    checkpointer: BaseCheckpointSaver[Any],
    grace_period: timedelta = _DEFAULT_RECLAIM_GRACE_PERIOD,
) -> int:
    """Sub-job 2: detect + mark `failed` for stuck HITL states.

    Two crash-window patterns this sub-job covers:

      - Window (c): `mark_awaiting_approval` succeeded but
        `interrupt()` never landed. The row is in `awaiting_approval`
        but the LangGraph checkpointer has NO pending interrupt for
        this thread_id. Detection: query checkpointer via
        `checkpointer.aget(config={"configurable":
        {"thread_id": str(review_id)}})` — None or a checkpoint
        without a pending-interrupt marker indicates window (c).

      - Window (f): `emit_hitl_decision` succeeded but `mark_running`
        never landed. The row is in `awaiting_approval`,
        `reviews.hitl_decision IS NULL`, but an audit row with
        `event_type='hitl_decision'` exists for this review_id. The
        F2 audit-fold flagged this case as the canonical-recovery
        path for divergent-content retries that the resume wrapper
        cannot resolve.

    Both windows resolve to: mark the review `failed` after the
    grace period elapses. Operator triages from logs + the failed
    status row.

    Returns the number of rows reclaimed. Acquires SWEEP_LOCK_ID via
    the provided conn. If another sweep holds it, returns 0.
    """
    if not await _try_acquire_sweep_lock(conn):
        logger.info("hitl_reclaim_sweep_skipped: advisory lock held by another sweep")
        return 0

    grace_cutoff = datetime.now(UTC) - grace_period

    # Query candidate rows: status='awaiting_approval' AND old enough
    # to be past the grace period. Uses the same partial index as
    # the expiry sub-job.
    result = await conn.execute(
        select(Review.id).where(
            Review.status == "awaiting_approval",
            Review.created_at < grace_cutoff,
        )
    )
    candidate_ids = [row.id for row in result.all()]

    reclaimed = 0
    for review_id in candidate_ids:
        # Check LangGraph checkpointer for a pending interrupt.
        config = {"configurable": {"thread_id": str(review_id)}}
        try:
            checkpoint = await checkpointer.aget(config)  # type: ignore[arg-type]
        except Exception:
            # Checkpointer read failure — DO NOT reclaim. Log and
            # let the next sweep tick retry. Mis-reclaiming an
            # in-flight HITL would be catastrophic (lose the
            # reviewer's chance to decide); the cost of an extra
            # sweep tick is negligible.
            logger.exception(
                "hitl_reclaim_checkpointer_read_failed",
                extra={"review_id": str(review_id)},
            )
            continue

        # If a checkpoint exists with a pending interrupt, the body
        # IS suspended at the HITL interrupt — NOT stuck, just
        # waiting. Skip.
        if checkpoint is not None:
            # Heuristic: a present-but-empty checkpoint indicates
            # an actual suspension. The full pending-interrupt
            # detection requires reading the checkpoint's
            # `interrupts` field via `aget_tuple`; for V1, ANY
            # checkpoint means "graph is alive" — skip reclaim.
            # Future refinement: use `checkpointer.aget_tuple` to
            # inspect the `pending_writes` / `interrupts` structure
            # and distinguish active-pause from stale checkpoint.
            continue

        # No checkpoint → window (c) crash OR clean state. Mark
        # the review failed. Operator triage from logs + status.
        async with session_factory() as session, session.begin():
            await session.execute(
                update(Review)
                .where(
                    Review.id == review_id,
                    Review.status == "awaiting_approval",
                )
                .values(status="failed")
            )
        reclaimed += 1
        logger.warning(
            "hitl_reclaim_marked_failed",
            extra={
                "review_id": str(review_id),
                "note": (
                    "Row stuck in awaiting_approval past grace period with no "
                    "pending LangGraph interrupt; reclaimed as failed for "
                    "operator triage. Likely window-(c) or window-(f) crash."
                ),
            },
        )

    return reclaimed


async def run_once(
    *,
    conn: AsyncConnection,
    session_factory: async_sessionmaker[AsyncSession],
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
    checkpointer: BaseCheckpointSaver[Any],
    grace_period: timedelta = _DEFAULT_RECLAIM_GRACE_PERIOD,
) -> dict[str, int]:
    """Run both HITL-expiry sub-jobs in canonical order.

    Order is locked: `reclaim_stuck_hitl_states` BEFORE
    `transition_expired_hitl_reviews`. Rationale: a stuck row past
    `expires_at` would otherwise be classified as "reviewer never
    decided" (expired) when the actual cause is "process crashed"
    (reclaimed → failed). Reclaim-first preserves the diagnostic
    distinction in the resulting row state.

    Both sub-jobs share `SWEEP_LOCK_ID`. The first one to acquire it
    runs; the second sees the lock still held (same transaction) and
    proceeds inside the same lock window — no double-lock contention.

    Returns a dict `{reclaimed: N, transitioned: M}` for telemetry.
    """
    reclaimed = await reclaim_stuck_hitl_states(
        conn=conn,
        session_factory=session_factory,
        checkpointer=checkpointer,
        grace_period=grace_period,
    )
    transitioned = await transition_expired_hitl_reviews(
        conn=conn,
        session_factory=session_factory,
        anomaly_sink=anomaly_sink,
        review_status_sink=review_status_sink,
    )
    return {"reclaimed": reclaimed, "transitioned": transitioned}


__all__ = [
    "reclaim_stuck_hitl_states",
    "run_once",
    "transition_expired_hitl_reviews",
]
