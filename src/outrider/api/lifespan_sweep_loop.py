# Lifespan-scoped periodic sweep loop. See specs/2026-05-26-hitl-node.md
# Group 8 scheduler-wiring bullet + docs/spec.md §4.1.6.
"""Background-task scheduler for `sweep.runner.run_scheduled_tick`.

Per `docs/spec.md` §4.1.6, the HITL-expiry sweep enforces the timeout
window on a 5-minute cadence. Without a periodic invoker, the timeout
enforcement + window-(c)/(f) crash recovery is inert in production.

V1 uses a minimal asyncio-based loop bound to the FastAPI lifespan:

  - Started inside the lifespan body via `start_periodic_sweep(...)`.
  - Pushed onto `AsyncExitStack` via `_cancel_task` so the task is
    cancelled cleanly at shutdown.
  - One `run_scheduled_tick` per `_SWEEP_INTERVAL_SECONDS`
    (default 300 = 5 min) — reconcile-first, then the sweep family with
    the `#012` install hard-delete gated on reconcile confirming
    liveness this tick (see `sweep.runner.run_scheduled_tick`).
  - Wide `except Exception` inside the loop so a single tick failure
    doesn't kill the loop — logged + skipped + retried next interval.

Operators wanting a heavier scheduler (cron, k8s CronJob, APScheduler)
can disable this loop via `OUTRIDER_SWEEP_DISABLED=1` (read in
`api/lifespan.py`; when set to `"1"`, the lifespan body does NOT call
`start_periodic_sweep(...)`) and invoke
`outrider.sweep.runner.run_scheduled_tick` externally on their own
cadence — NOT `run_all_sweeps` directly, which would exclude the
reconcile janitor and its liveness-gating of the `#012` hard-delete.
The env-var disable is distinct from
`_SWEEP_INTERVAL_SECONDS` below, which still has intentionally no
env-var override — disable is a runtime operational decision; cadence
override is a code-side opt-in.

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

from outrider.sweep.runner import run_scheduled_tick

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
    from outrider.github.config import GitHubAppSettings


logger = logging.getLogger("outrider.api.lifespan_sweep_loop")


# Per docs/spec.md §4.1.6 sweep cadence. Tests / faster operator
# triage can override via the `interval_seconds` kwarg on
# `start_periodic_sweep(...)`. There is intentionally NO env-var
# override — the lifespan owns the lifecycle, and an env-var
# override would let operators silently destabilize the sweep
# cadence without an explicit code-side opt-in.
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
    github_app_settings: GitHubAppSettings | None,
    interval_seconds: float,
) -> None:
    """Run `run_scheduled_tick` every `interval_seconds`.

    Each tick is one production-tick orchestration
    (`sweep.runner.run_scheduled_tick`): reconcile the install cache
    FIRST (the `#065`/`#012`/`#067` janitor, under its own session-scoped
    advisory lock across the `GET /app/installations` call), THEN run the
    sweep family with the `#012` install hard-delete gated on that
    reconcile having confirmed liveness this tick. The reconcile + sweep
    halves have independent failure handling INSIDE the orchestrator; the
    loop adds the OUTER per-tick guard so a whole-tick failure (either
    half) is logged + skipped + retried next interval.

    The loop catches `asyncio.CancelledError` at the OUTER level so
    lifespan teardown cancels cleanly. `run_scheduled_tick` owns the
    per-tick connection lifecycle (the reconcile lock connection + the
    sweep-family transaction), so the loop opens no connection itself.
    """
    logger.info("sweep_loop_started", extra={"interval_seconds": interval_seconds})
    try:
        while True:
            try:
                result = await run_scheduled_tick(
                    engine=engine,
                    session_factory=session_factory,
                    anomaly_sink=anomaly_sink,
                    review_status_sink=review_status_sink,
                    audit_persister=audit_persister,
                    checkpointer=checkpointer,
                    compiled_graph=compiled_graph,
                    github_app_settings=github_app_settings,
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
    github_app_settings: GitHubAppSettings | None = None,
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
    # Fail-loud on non-positive intervals. A zero or negative
    # `interval_seconds` would degenerate `_sweep_loop`'s
    # `await asyncio.sleep(interval_seconds)` into a tight retry loop
    # (instant-fire + instant-retry on every tick failure), saturating
    # the event loop + the DB connection pool. Catch the misconfig at
    # task-creation time rather than letting the loop start.
    if actual_interval <= 0:
        msg = (
            f"start_periodic_sweep: interval_seconds must be > 0; got "
            f"{actual_interval}. Pass `interval_seconds=None` to use the "
            f"{_SWEEP_INTERVAL_SECONDS}s default."
        )
        raise ValueError(msg)
    return asyncio.create_task(
        _sweep_loop(
            engine=engine,
            session_factory=session_factory,
            anomaly_sink=anomaly_sink,
            review_status_sink=review_status_sink,
            audit_persister=audit_persister,
            checkpointer=checkpointer,
            compiled_graph=compiled_graph,
            github_app_settings=github_app_settings,
            interval_seconds=actual_interval,
        ),
        name="outrider-sweep-loop",
    )


__all__ = ["start_periodic_sweep"]
