# Lifespan-scoped periodic sweep loop. See specs/2026-05-26-hitl-node.md
# Group 8 scheduler-wiring bullet + docs/spec.md §4.1.6.
"""Run background maintenance while the FastAPI lifespan is active.

The first tick starts immediately; each finished or failed tick is followed by
an interval wait (default five minutes). `run_scheduled_tick` owns reconciliation
before sweeps and gates installation deletion on confirmed liveness (#065/#012/#067).
The loop logs ordinary tick failures and retries; cancellation propagates so
lifespan teardown can await cleanup.

Operators can disable this task with OUTRIDER_SWEEP_DISABLED (1/true/yes, ignoring
case and surrounding whitespace) and schedule `run_scheduled_tick` externally.
Calling `run_all_sweeps` alone omits reconciliation and its deletion-safety gate.
The interval has a code-side override, not an environment setting.
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
    from outrider.github.credentials import GitHubCredentialProvider


logger = logging.getLogger("outrider.api.lifespan_sweep_loop")


# Default wait between ticks (§4.1.6). Tests/operators can override in code.
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
    provider: GitHubCredentialProvider | None,
    interval_seconds: float,
) -> None:
    """Run a tick, log its outcome, then wait before trying again.

    The orchestrator owns its connections and reconcile/sweep failure handling.
    This outer guard contains unexpected ordinary tick errors; cancellation
    interrupts the current tick or wait and propagates to lifespan teardown.
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
                    provider=provider,
                )
                logger.info("sweep_tick_completed", extra={"result": result})
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep maintenance running after an ordinary tick failure.
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
    provider: GitHubCredentialProvider | None = None,
    interval_seconds: float | None = None,
) -> asyncio.Task[None]:
    """Start the lifespan-owned task and return it for cancellation and cleanup.

    None selects the five-minute default; a non-positive override raises before
    scheduling. The first tick runs immediately, then waits follow each tick.
    """
    actual_interval = interval_seconds if interval_seconds is not None else _SWEEP_INTERVAL_SECONDS
    # Reject a tight retry loop before allocating its task.
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
            provider=provider,
            interval_seconds=actual_interval,
        ),
        name="outrider-sweep-loop",
    )


__all__ = ["start_periodic_sweep"]
