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
    # `is_eval=False` filter per `docs/testing.md:59` ("Sweep jobs:
    # ignore is_eval=True rows. An eval review that checkpoints
    # mid-HITL won't be marked stuck by the sweep.") — the contract
    # the dashboard's is_eval filter depends on. Latent under test-DB
    # isolation today; the filter is the structural defense if
    # eval data ever shares a DB with prod.
    result = await conn.execute(
        select(Review.id, Review.expires_at).where(
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

    **Candidate-row predicate: `status IN ('awaiting_approval',
    'awaiting_approval_expired')`** — broader than just
    `awaiting_approval` so reclaim catches window-(f) rows that
    `transition_expired_hitl_reviews` already flipped to
    `awaiting_approval_expired` in a prior sweep tick (or even the
    same tick: the spec-locked sub-job order is reclaim BEFORE
    transition, but a row near its `expires_at` boundary could be
    skipped by reclaim's grace-period gate and immediately flipped
    by transition; without the broader filter, the audit row would
    orphan permanently).

    **Audit-row presence is the canonical signal — no grace gate.**
    When a `HITLDecisionEvent` exists for the review, the emit ran;
    the audit row is canonical per `audit-events-append-only`. The
    `expires_at < grace_cutoff` gate ONLY applies to the no-audit-row
    branch (window c): there we need the grace period to avoid
    false-positive reclaim of a row that just transitioned to
    `awaiting_approval` (the body might still be running, the
    langgraph checkpoint might be in flight). For window (f),
    immediate recovery is safe.

    Per-row classification:
      - audit row exists -> graph-driven recovery (window f); counted
        as `recovered`
      - no audit row + `expires_at < grace_cutoff` + no checkpoint
        -> mark `failed` (window c); counted as `failed`
      - otherwise skip (still in flight)

    Returns `{"recovered": N, "failed": M}` for telemetry.

    **Lock-acquisition contract: this sub-job DOES NOT acquire
    `SWEEP_LOCK_ID` itself.** The caller (typically `run_once`)
    acquires once and holds across both sub-jobs.
    """
    grace_cutoff = datetime.now(UTC) - grace_period

    # Query candidate rows: status IN ('awaiting_approval',
    # 'awaiting_approval_expired'). Broader than the natural-HITL
    # set so post-transition orphans (window-f rows already flipped
    # to awaiting_approval_expired by the transition sub-job) stay
    # reachable for audit-row recovery. See docstring rationale.
    # `is_eval=False` filter per `docs/testing.md:59` contract — see
    # the sibling filter on `transition_expired_hitl_reviews` for the
    # full rationale.
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
        # Step 1: check the audit layer for an orphaned
        # HITLDecisionEvent. Window-(f) detection — audit row
        # presence is the canonical signal, no grace gate needed.
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

        # Step 2: no audit row. Apply grace gate explicitly — the
        # candidate query is broader (status IN ('awaiting_approval',
        # 'awaiting_approval_expired')) and doesn't filter by
        # expires_at, so we re-check here to avoid window-c false
        # positives. If expires_at >= grace_cutoff (or NULL), the
        # natural HITL window hasn't lapsed past grace; skip.
        if row_expires_at is None or row_expires_at >= grace_cutoff:
            continue

        # Step 3: check the LangGraph checkpointer. If a checkpoint
        # exists, the body is suspended at the interrupt waiting
        # for resume — NOT stuck. Skip.
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
            # No audit row + checkpoint exists = genuinely suspended,
            # still in flight. Skip.
            continue

        # Window (c) recovery: no audit row, no checkpoint, past
        # grace period. Mark `failed` for operator triage. Predicate
        # admits both source states so the mark-failed write works
        # whether the row was already flipped to
        # awaiting_approval_expired by transition or still sits in
        # awaiting_approval.
        #
        # Concurrent-actor defense: between the candidate SELECT and
        # this UPDATE, another actor (a parallel sweep, a manual SQL
        # operator triage, the dashboard) could have changed the
        # status (e.g., to `running` via a window-(f) graph-driven
        # recovery on a peer process). The UPDATE's WHERE clause
        # filters to the two admitted source states, so a status
        # change would land rowcount=0 — incrementing `failed` and
        # logging "reclaimed as failed" without a matched row would
        # be a phantom signal. Gate the counter + log on the actual
        # rowcount.
        async with session_factory() as session, session.begin():
            update_result = await session.execute(
                update(Review)
                .where(
                    Review.id == review_id,
                    Review.status.in_(("awaiting_approval", "awaiting_approval_expired")),
                )
                .values(status="failed")
            )
        # `AsyncSession.execute(...)` returns `Result[Any]` per the
        # async stubs; `CursorResult` (the runtime type for UPDATE/
        # INSERT/DELETE) carries `rowcount` but mypy's `Result` base
        # type doesn't expose it. `getattr` keeps the static check
        # honest without a cast (the runtime attribute is reliable
        # across SQLAlchemy 2.x).
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
