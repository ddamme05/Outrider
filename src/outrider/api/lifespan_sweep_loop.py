# Lifespan-scoped periodic sweep loop. See specs/2026-05-26-hitl-node.md
# Group 8 scheduler-wiring bullet + docs/spec.md §4.1.6.
"""Background-task scheduler for `sweep.runner.run_all_sweeps`.

Per `docs/spec.md` §4.1.6, the HITL-expiry sweep enforces the timeout
window on a 5-minute cadence. Without a periodic invoker, the timeout
enforcement + window-(c)/(f) crash recovery is inert in production.

V1 uses a minimal asyncio-based loop bound to the FastAPI lifespan:

  - Started inside the lifespan body via `start_periodic_sweep(...)`.
  - Pushed onto `AsyncExitStack` via `_cancel_task` so the task is
    cancelled cleanly at shutdown.
  - One sweep tick per `_SWEEP_INTERVAL_SECONDS` (default 300 = 5 min).
  - Wide `except Exception` inside the loop so a single tick failure
    doesn't kill the loop — logged + skipped + retried next interval.

Operators wanting a heavier scheduler (cron, k8s CronJob, APScheduler)
can disable this loop via `OUTRIDER_SWEEP_DISABLED=1` and invoke
`outrider.sweep.runner.run_all_sweeps` externally on their own cadence.

Why not APScheduler: it's an extra dep with its own design surface
(job stores, executors, persistence) for a single-tick periodic task.
The asyncio loop matches the lifespan ownership model + keeps the dep
footprint tight. If V1.5 needs multi-tenant / cross-process scheduling,
swap to APScheduler at the seam — the public surface is the single
`start_periodic_sweep` callable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Final

from outrider.sweep.runner import run_all_sweeps

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
    )

    from outrider.anomaly.sinks import AnomalySink
    from outrider.audit.persister import AuditPersister
    from outrider.db.sinks import ReviewStatusSink


logger = logging.getLogger("outrider.api.lifespan_sweep_loop")


# Per docs/spec.md §4.1.6 sweep cadence. Override via
# OUTRIDER_SWEEP_INTERVAL_SECONDS for tests / faster operator triage.
_SWEEP_INTERVAL_SECONDS: Final[float] = 300.0


async def _sweep_loop(
    *,
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
    audit_persister: AuditPersister,
    checkpointer: BaseCheckpointSaver[Any],
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    interval_seconds: float,
) -> None:
    """Run `run_all_sweeps` every `interval_seconds`.

    The loop catches `asyncio.CancelledError` at the OUTER level so
    lifespan teardown cancels cleanly. Inside each tick, a wide
    `except Exception` shields the loop from a single failure
    (network blip, transient DB error, etc.) — logged + skipped +
    retried next interval.

    Per-tick connection lifecycle: open ONE `engine.connect()` for
    the tick's work (sweep code expects an `AsyncConnection`),
    transaction-scope the SWEEP_LOCK_ID advisory lock via the
    sweep functions' internal `session.begin()` calls.
    """
    logger.info("sweep_loop_started", extra={"interval_seconds": interval_seconds})
    try:
        while True:
            try:
                async with engine.connect() as conn, conn.begin():
                    result = await run_all_sweeps(
                        conn=conn,
                        session_factory=session_factory,
                        anomaly_sink=anomaly_sink,
                        review_status_sink=review_status_sink,
                        audit_persister=audit_persister,
                        checkpointer=checkpointer,
                        compiled_graph=compiled_graph,
                    )
                logger.info("sweep_tick_completed", extra={"result": result})
            except asyncio.CancelledError:
                # Bubble up to the outer try so the loop exits.
                raise
            except Exception:
                # One failed tick doesn't kill the loop. Next interval
                # retries — the sweep code is idempotent.
                logger.exception("sweep_tick_failed")
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("sweep_loop_cancelled_at_shutdown")
        raise


def start_periodic_sweep(
    *,
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
    audit_persister: AuditPersister,
    checkpointer: BaseCheckpointSaver[Any],
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    interval_seconds: float | None = None,
) -> asyncio.Task[None]:
    """Schedule the periodic sweep as an asyncio Task.

    Returns the task so the lifespan teardown can cancel it. The
    task runs forever until cancelled — `_sweep_loop` catches
    `CancelledError` at the outer level so cancellation is
    cooperative + the in-flight tick (if any) gets to finish its
    `engine.connect()` cleanup before the task exits.

    Pass `interval_seconds=None` (default) to use
    `_SWEEP_INTERVAL_SECONDS` (300s per spec §4.1.6). Override for
    tests / operator triage.
    """
    actual_interval = interval_seconds if interval_seconds is not None else _SWEEP_INTERVAL_SECONDS
    return asyncio.create_task(
        _sweep_loop(
            engine=engine,
            session_factory=session_factory,
            anomaly_sink=anomaly_sink,
            review_status_sink=review_status_sink,
            audit_persister=audit_persister,
            checkpointer=checkpointer,
            compiled_graph=compiled_graph,
            interval_seconds=actual_interval,
        ),
        name="outrider-sweep-loop",
    )


__all__ = ["start_periodic_sweep"]
