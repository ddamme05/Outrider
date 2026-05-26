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

  2. `reclaim_stuck_hitl_states` — covers two crash windows:
     - **Window (f) recovery (graph-driven via Command(resume)):**
       for rows in `awaiting_approval` past the grace period whose
       `audit_events` table carries an existing `HITLDecisionEvent`
       but `reviews.hitl_decision IS NULL`, reconstruct the
       `HITLDecision` from the canonical audit row and invoke
       `graph.ainvoke(Command(resume=...), config={thread_id})`.
       The body re-runs from the top: phase start (idempotent on
       phase_id), partition, request rebuild + emit_hitl_request
       (natural-key no-op, returns existing event), mark_awaiting_approval
       (no-op via `hitl_request IS NOT NULL` predicate), interrupt()
       returns the resume value (the canonical decision payload),
       audit no-op (returns existing event since content matches),
       mark_running (writes JSONB + flips status to `running`),
       phase end, state delta returns → graph routes to publish →
       publish runs against the recovered finding set. Lifecycle
       AND publish both reach their canonical terminal state.
       *Critical:* this is NOT just a DB status flip. A `mark_running`
       call from the sweep WITHOUT graph invocation would leave the
       graph suspended at the interrupt forever — the endpoint's
       JSONB-cache preflight would reject every retry (409) and the
       gated finding would never reach GitHub. Driving the recovery
       through `Command(resume=...)` is what makes the audit-canonical
       decision actually publish.
     - **Window (c) recovery (checkpoint-absence):** for rows in
       `awaiting_approval` past the grace period with NO audit row
       AND no LangGraph checkpoint (`mark_awaiting_approval`
       succeeded but `interrupt()` never landed), mark the review
       `failed`. Operator triage via structured logs + the failed
       status row.
     The audit-row check FIRST is the critical distinction the
     earlier checkpointer-only heuristic missed: window (f) rows
     STILL have the HITL interrupt checkpoint (the body crashed
     after interrupt() returned, before mark_running), so a pure
     checkpoint-presence heuristic skipped them indefinitely.

Sub-job order WITHIN `run_once()`: `reclaim_stuck_hitl_states` runs
BEFORE `transition_expired_hitl_reviews`. Rationale (spec line 566):
a stuck-state row that ALSO has `expires_at < NOW()` would otherwise
flip to `awaiting_approval_expired` via the expiry sub-job, which
masks the actual crash classification ("reviewer never decided" vs
"process crashed mid-write"). Running reclaim first correctly
classifies as `failed` (or advances to `running` on window-f) with
explicit operator-triage signal.

Both sub-jobs share the existing `SWEEP_LOCK_ID` from
`sweep/purge_expired.py` per `sweep-jobs-use-advisory-locks` — the
LOCK IS ACQUIRED ONCE in `run_once()` and held across both
sub-jobs. Calling the sub-jobs directly (outside `run_once`) is
permitted but the caller is responsible for the lock; each sub-job
no longer self-acquires.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from langgraph.types import Command
from sqlalchemy import select, text, update

from outrider.anomaly.rule_names import AnomalyRuleName
from outrider.db.models.reviews import Review
from outrider.schemas.hitl import HITLDecision
from outrider.sweep.purge_expired import SWEEP_LOCK_ID

if TYPE_CHECKING:
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

    **Lock-acquisition contract: this sub-job DOES NOT acquire
    `SWEEP_LOCK_ID` itself.** The caller (typically `run_once`)
    acquires the lock once and holds it across both sub-jobs in a
    single transaction. This is load-bearing for the reclaim-before-
    transition ordering: each sub-job acquiring its own lock would
    free + reacquire between sub-jobs, letting a concurrent process
    sneak in and break the order.
    """
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
    audit_persister: AuditPersister,
    review_status_sink: ReviewStatusSink,
    checkpointer: BaseCheckpointSaver[Any],
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    grace_period: timedelta = _DEFAULT_RECLAIM_GRACE_PERIOD,
) -> dict[str, int]:
    """Sub-job 2: detect + recover stuck HITL states.

    Two crash-window patterns, distinguished by an AUDIT-ROW check
    (NOT a checkpointer-only heuristic — the earlier impl missed
    window (f) because the langgraph checkpoint exists even after a
    post-interrupt crash):

      - **Window (f)** (`emit_hitl_decision` succeeded but
        `mark_running` never landed): an audit row with
        `event_type='hitl_decision'` exists for this `review_id`,
        but `reviews.hitl_decision IS NULL`. The audit row IS the
        canonical decision per `audit-events-append-only`. Recovery:
        reconstruct the `HITLDecision` domain object from the audit
        event and invoke `compiled_graph.ainvoke(Command(resume=...),
        config={thread_id})`. The body re-runs from the top,
        idempotent emits no-op via natural-key match, `mark_running`
        INSIDE the body writes JSONB + flips status, phase end fires,
        state delta returns, graph routes to publish, publish runs
        against the recovered finding set. Direct `mark_running`
        writes from the sweep are deliberately NOT used: they would
        flip the lifecycle column but leave the graph suspended at
        the interrupt forever, with the endpoint's JSONB-cache
        preflight 409-rejecting every retry. Counted under `recovered`.

      - **Window (c)** (`mark_awaiting_approval` succeeded but
        `interrupt()` never landed): no audit row exists AND the
        LangGraph checkpointer has no record for this `thread_id`.
        Recovery: mark the review `failed`. Operator triages from
        logs + the failed status row. Counted under `failed`.

    Rows where the audit row is absent AND a checkpoint exists are
    treated as "still in flight" — the body is suspended at the
    interrupt waiting for resume; skip.

    **Candidate-row freshness predicate (`Review.expires_at < grace_cutoff`).**
    The freshness gate uses `expires_at`, NOT `created_at`. Rationale:
    `created_at` is the review-row creation time (set at webhook
    receipt), so a review that spends 10 minutes in analyze/trace
    before entering HITL would match `created_at < now - 5min` the
    moment it transitions to `awaiting_approval`. The reclaim could
    then false-positive on the natural gap between
    `mark_awaiting_approval` and the langgraph checkpoint landing.
    `expires_at` is set atomically by `mark_awaiting_approval`
    (`expires_at = state.received_at + timedelta(minutes=timeout_minutes)`),
    so it reflects the END of the HITL window. A row is "stuck"
    only when `expires_at + grace_period < now`, i.e., the natural
    HITL window has already lapsed PLUS the grace period — at which
    point the transition sub-job would normally have flipped to
    `awaiting_approval_expired`, so a row still in `awaiting_approval`
    is genuinely stuck (window c or f).

    Returns `{"recovered": N, "failed": M}` for telemetry. `N + M` is
    the total number of stuck rows handled this tick.

    **Lock-acquisition contract: this sub-job DOES NOT acquire
    `SWEEP_LOCK_ID` itself.** The caller (typically `run_once`)
    acquires once and holds across both sub-jobs.
    """
    grace_cutoff = datetime.now(UTC) - grace_period

    # Query candidate rows: status='awaiting_approval' AND
    # `expires_at < grace_cutoff` (the natural HITL window has
    # lapsed past the grace period — see freshness-predicate
    # rationale in the docstring above).
    result = await conn.execute(
        select(Review.id).where(
            Review.status == "awaiting_approval",
            Review.expires_at < grace_cutoff,
        )
    )
    candidate_ids = [row.id for row in result.all()]

    recovered = 0
    failed = 0
    for review_id in candidate_ids:
        # Step 1: check the audit layer for an orphaned
        # HITLDecisionEvent. Window-(f) detection.
        try:
            audit_decision_event = await audit_persister.query_hitl_decision_event(review_id)
        except Exception:
            logger.exception(
                "hitl_reclaim_audit_query_failed",
                extra={"review_id": str(review_id)},
            )
            continue

        if audit_decision_event is not None:
            # Window (f) recovery: the audit row is canonical, but
            # the graph is still suspended at the HITL interrupt.
            # We MUST drive the graph through `Command(resume=...)`
            # so the body completes: mark_running fires INSIDE the
            # body (writing the canonical JSONB), phase end emits,
            # state delta returns, graph routes to publish, publish
            # runs against the recovered finding set. A direct
            # `mark_running` write from here would advance the
            # lifecycle column but leave the graph permanently
            # suspended — `/decide` would 409-reject all retries
            # (preflight sees hitl_decision != NULL) and the gated
            # finding never reaches GitHub.
            canonical_decision = HITLDecision(
                reviewer_id=audit_decision_event.reviewer_id,
                decisions=audit_decision_event.decisions,
                annotation=audit_decision_event.annotation,
                decided_at=audit_decision_event.decided_at,
            )
            from langchain_core.runnables import (  # noqa: PLC0415, TC002
                RunnableConfig,
            )

            recovery_config: RunnableConfig = {"configurable": {"thread_id": str(review_id)}}
            try:
                await compiled_graph.ainvoke(
                    Command(resume=canonical_decision.model_dump(mode="json")),
                    config=recovery_config,
                )
            except Exception:
                # Graph drive failed; lifecycle stays in
                # awaiting_approval. Next sweep tick retries — the
                # audit row is still canonical, and the body re-run
                # is idempotent.
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

        # Step 2: no audit row. Check the LangGraph checkpointer.
        # If a checkpoint exists, the body is suspended at the
        # interrupt waiting for resume — NOT stuck. Skip.
        config = {"configurable": {"thread_id": str(review_id)}}
        try:
            checkpoint = await checkpointer.aget(config)  # type: ignore[arg-type]
        except Exception:
            logger.exception(
                "hitl_reclaim_checkpointer_read_failed",
                extra={"review_id": str(review_id)},
            )
            continue

        if checkpoint is not None:
            # No audit row + checkpoint exists = genuinely suspended,
            # still in flight. Skip.
            continue

        # Window (c) recovery: no audit row, no checkpoint, past
        # grace period. Mark `failed` for operator triage.
        async with session_factory() as session, session.begin():
            await session.execute(
                update(Review)
                .where(
                    Review.id == review_id,
                    Review.status == "awaiting_approval",
                )
                .values(status="failed")
            )
        failed += 1
        logger.warning(
            "hitl_reclaim_marked_failed",
            extra={
                "review_id": str(review_id),
                "note": (
                    "Row stuck in awaiting_approval past grace period with no "
                    "audit row AND no LangGraph checkpoint; reclaimed as "
                    "failed for operator triage (window-c crash)."
                ),
            },
        )

    return {"recovered": recovered, "failed": failed}


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
    """Run both HITL-expiry sub-jobs in canonical order, under a
    single advisory-lock window.

    Order is locked: `reclaim_stuck_hitl_states` BEFORE
    `transition_expired_hitl_reviews`. Rationale (spec line 566): a
    stuck row past `expires_at` would otherwise be classified as
    "reviewer never decided" (expired) when the actual cause is
    "process crashed" (reclaimed -> recovered-from-audit OR failed).
    Reclaim-first preserves the diagnostic distinction in the
    resulting row state.

    **Advisory lock is acquired ONCE** by this wrapper. The
    transaction-scoped `pg_try_advisory_xact_lock` holds for the
    duration of the `conn`'s transaction, which spans both sub-jobs
    AND the SELECT/UPDATE statements inside them. Each sub-job
    deliberately does NOT self-acquire — otherwise reclaim could
    return early (lock held by a peer), then the lock could free
    before transition runs, breaking the reclaim-before-transition
    ordering.

    Returns `{"reclaim_recovered": N, "reclaim_failed": M,
    "transitioned": K}` for telemetry. `(N + M)` is the reclaim
    sub-job's total; `K` is the expiry sub-job's transition count.

    On lock contention (another sweep process holds it): returns all
    zeros without running either sub-job.
    """
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
        session_factory=session_factory,
        anomaly_sink=anomaly_sink,
        review_status_sink=review_status_sink,
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
