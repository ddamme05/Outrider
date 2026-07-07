# See specs/2026-05-26-hitl-node.md Group 8 scheduler-wiring bullet +
# sweep/purge_expired.py + sweep/hitl_expiry.py.
"""Single-callsite orchestrator for the V1 sweep family.

Per `docs/spec.md` §4.1.6 + the HITL spec Group 8 prescription, three
sweep responsibilities run on a periodic cadence:

  1. `hitl_expiry.run_once(...)` — reclaim stuck HITL rows + transition
     expired ones. MUST run before retention purge so a reclaim from
     window (f) advances the lifecycle to `running` BEFORE a retention
     pass could purge the row.
  2. `purge_expired.purge_expired(...)` — time-based retention sweep
     across `llm_call_content`, `findings`, `reviews`.
  3. `replay_verdict.project_replay_verdicts(...)` — append a
     replay-equivalence verdict for each completed PRODUCTION review
     lacking one (`DECISIONS.md#039` sibling). Runs LAST by ordering
     convention; flips no status, idempotent, no advisory lock. (It does
     NOT see purge's row deletions — those DELETEs are uncommitted in the
     shared outer transaction when the projector opens its own sessions —
     but verdicting a row purge will delete this tick is harmless:
     `audit_events` is append-only with no FK to `reviews`, so the verdict
     simply outlives the purged review and next tick has no candidate.)

This module exposes ONE callable, `run_all_sweeps(...)`, that an
operator-side scheduler (APScheduler, k8s CronJob, etc.) invokes per
tick. The scheduler integration itself is out of scope for V1 — adding
APScheduler is a deployment decision with its own design surface and
dep. What this module gives production today: a non-zero callsite for
the sweep code so it can be invoked from a one-shot CLI or a future
scheduler without re-wiring at every consumer.

Per `sweep-jobs-use-advisory-locks`: both `hitl_expiry.run_once` and
`purge_expired.purge_expired` acquire the SAME `SWEEP_LOCK_ID` advisory
lock via `pg_try_advisory_xact_lock(SWEEP_LOCK_ID)` (transaction-
scoped; reentrant within a single transaction). `run_all_sweeps` runs
them sequentially passing the SAME `conn` to both, so when the
caller wraps `conn` in a transaction the lock is held continuously
across both sub-jobs — operators cannot introduce a window between
hitl-expiry and retention-purge by scheduling them as separate calls.

**PRECONDITION on `conn`:** callers MUST wrap the `conn` in an
explicit transaction (`async with engine.connect() as conn,
conn.begin(): await run_all_sweeps(conn=conn, ...)`) so the advisory
lock acquired by `hitl_expiry.run_once` remains held when
`purge_expired.purge_expired` runs its own `pg_try_advisory_xact_lock`
acquire. The single in-tree caller
(`api/lifespan_sweep_loop.py::_sweep_loop`) honors this contract; a
future external caller that passes a non-transactional `conn` would
silently break the cross-sub-job lock continuity. `run_all_sweeps`
does NOT begin its own transaction to keep transaction-scope
visibility under the caller.
"""

from __future__ import annotations

import logging
from datetime import timedelta  # noqa: TC003  (runtime: function-signature annotation)
from typing import TYPE_CHECKING, Any

from outrider.sweep import hitl_expiry, purge_expired, replay_verdict

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

    from outrider.anomaly.sinks import AnomalySink
    from outrider.audit.persister import AuditPersister
    from outrider.db.sinks import ReviewStatusSink


logger = logging.getLogger(__name__)


async def run_all_sweeps(
    *,
    conn: AsyncConnection,
    session_factory: async_sessionmaker[AsyncSession],
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
    audit_persister: AuditPersister,
    checkpointer: BaseCheckpointSaver[Any],
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    grace_period: timedelta | None = None,
    purge_role: str = "sweep",
) -> dict[str, Any]:
    """Run hitl-expiry first, then retention purge. One tick.

    Returns a single telemetry dict combining both sub-runs:
      {
        "hitl": {"reclaim_recovered": N, "reclaim_failed": M,
                 "transitioned": K},
        "purge": {<target_table>: <rows_affected>, ...},
        "replay_verdict": {"projected": N, "failed": M},
      }

    Order is load-bearing:
      - HITL-expiry first: a window-(f) reclaim ADVANCES the
        lifecycle to `running`; if retention purge ran first and the
        row's `retention_expires_at` was past, the canonical decision
        could be lost before the sweep could rescue it.
      - Both share `SWEEP_LOCK_ID`; `run_once` + `purge_expired` each
        acquire it via the same `conn`'s transaction. Concurrent
        sweep processes are serialized at the DB level.

    Pass `grace_period=None` to use `hitl_expiry`'s default 5-min
    window. Override for operator triage (e.g., `timedelta(hours=1)`
    when diagnosing an outage where many stuck rows accumulated).
    """
    # Runtime check on the docstring's PRECONDITION: `conn` MUST be in
    # an explicit transaction so the `pg_try_advisory_xact_lock`
    # acquired by `hitl_expiry.run_once` stays held when
    # `purge_expired.purge_expired` runs its own acquire. Without this
    # check, a non-transactional `conn` (autocommit shape) would
    # silently free + reacquire the lock between sub-jobs, breaking
    # the cross-sub-job serialization guarantee the docstring
    # promises. Fail-loud over silent-drift per `sweep-jobs-use-
    # advisory-locks`.
    if not conn.in_transaction():
        msg = (
            "run_all_sweeps requires `conn` to be in an explicit "
            "transaction so the SWEEP_LOCK_ID advisory lock stays "
            "held across both sub-jobs. Wrap the call in `async with "
            "engine.connect() as conn, conn.begin(): await "
            "run_all_sweeps(conn=conn, ...)`."
        )
        raise RuntimeError(msg)

    # Reject non-positive `grace_period`. A `timedelta <= 0` (e.g.,
    # `timedelta(seconds=-3600)` from a manual operator-triage typo)
    # computes `grace_cutoff = now() - grace_period = now() + |Δ|`,
    # i.e., the cutoff lands in the FUTURE, and the candidate query at
    # `hitl_expiry.py:288` (`Review.expires_at < grace_cutoff` per the
    # per-row gate) admits every row regardless of age. Reclaim then
    # mass-marks-failed every awaiting-approval row in the system.
    # Catch the sign-flip at the runner entry rather than letting the
    # mass-fail land.
    if grace_period is not None and grace_period <= timedelta(0):
        msg = (
            f"run_all_sweeps: grace_period must be > timedelta(0); "
            f"got {grace_period!r}. A non-positive grace_period makes "
            f"reclaim's grace gate admit every awaiting-approval row "
            f"regardless of age — mass-marks-failed every HITL-in-flight "
            f"review in the system."
        )
        raise ValueError(msg)

    hitl_kwargs: dict[str, Any] = {
        "conn": conn,
        "session_factory": session_factory,
        "anomaly_sink": anomaly_sink,
        "review_status_sink": review_status_sink,
        "audit_persister": audit_persister,
        "checkpointer": checkpointer,
        "compiled_graph": compiled_graph,
    }
    if grace_period is not None:
        hitl_kwargs["grace_period"] = grace_period

    hitl_result = await hitl_expiry.run_once(**hitl_kwargs)
    purge_result = await purge_expired.purge_expired(conn=conn, purge_role=purge_role)
    # Arc B2: the uninstall grace→hard-delete step. `purge_installation` existed but
    # had no scheduled caller, so tombstoned installs never actually purged (#012
    # never completed on uninstall). Runs after the time-based purge, sharing the
    # tick's SWEEP_LOCK_ID transaction (the lock is reentrant within it).
    install_purge_result = await purge_expired.purge_expired_installations(
        conn=conn, purge_role=purge_role
    )
    # Replay-verdict projection runs LAST by ordering convention. It flips no status
    # and is natural-key idempotent, so it needs no advisory lock — it opens its own
    # sessions via `session_factory`, OUTSIDE `conn`'s lock transaction, and so cannot
    # see purge's uncommitted DELETEs above; verdicting a to-be-purged review is
    # harmless (append-only audit row, no FK to reviews) (DECISIONS#039 sibling).
    verdict_result = await replay_verdict.project_replay_verdicts(
        session_factory=session_factory, audit_persister=audit_persister
    )

    return {
        "hitl": hitl_result,
        "purge": purge_result,
        "install_purge": install_purge_result,
        "replay_verdict": verdict_result,
    }


__all__ = ["run_all_sweeps"]
