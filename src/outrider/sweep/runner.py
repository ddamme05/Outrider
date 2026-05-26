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

This module exposes ONE callable, `run_all_sweeps(...)`, that an
operator-side scheduler (APScheduler, k8s CronJob, etc.) invokes per
tick. The scheduler integration itself is out of scope for V1 — adding
APScheduler is a deployment decision with its own design surface and
dep. What this module gives production today: a non-zero callsite for
the sweep code so it can be invoked from a one-shot CLI or a future
scheduler without re-wiring at every consumer.

Per `sweep-jobs-use-advisory-locks`: both `hitl_expiry.run_once` and
`purge_expired.purge_expired` acquire the SAME `SWEEP_LOCK_ID` advisory
lock. `run_all_sweeps` runs them sequentially within one transaction,
so the lock is held continuously across both — operators cannot
introduce a window between hitl-expiry and retention-purge by
scheduling them as separate calls.
"""

from __future__ import annotations

import logging
from datetime import timedelta  # noqa: TC003  (runtime: function-signature annotation)
from typing import TYPE_CHECKING, Any

from outrider.sweep import hitl_expiry, purge_expired

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
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
    grace_period: timedelta | None = None,
    purge_role: str = "sweep",
) -> dict[str, Any]:
    """Run hitl-expiry first, then retention purge. One tick.

    Returns a single telemetry dict combining both sub-runs:
      {
        "hitl": {"reclaim_recovered": N, "reclaim_failed": M,
                 "transitioned": K},
        "purge": {<target_table>: <rows_affected>, ...},
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
    hitl_kwargs: dict[str, Any] = {
        "conn": conn,
        "session_factory": session_factory,
        "anomaly_sink": anomaly_sink,
        "review_status_sink": review_status_sink,
        "audit_persister": audit_persister,
        "checkpointer": checkpointer,
    }
    if grace_period is not None:
        hitl_kwargs["grace_period"] = grace_period

    hitl_result = await hitl_expiry.run_once(**hitl_kwargs)
    purge_result = await purge_expired.purge_expired(conn=conn, purge_role=purge_role)

    return {"hitl": hitl_result, "purge": purge_result}


__all__ = ["run_all_sweeps"]
