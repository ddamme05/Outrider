"""Review dispatcher seam between the webhook handler and graph execution.

`ReviewDispatcher.dispatch(state)` is the contract for handing a seed
`ReviewState` from the webhook receiver to background graph execution.
V1 ships `BackgroundTasksDispatcher` (FastAPI in-process); V2 will ship
`CeleryDispatcher` (durable broker). The Protocol is uniform; the
composition seam differs — per the intake-and-webhook spec, V1 is
**per-request** (built via FastAPI `Depends` from the request-scoped
`BackgroundTasks`), V2 will be **lifespan-singleton** (one instance
constructed at startup).

Why state, not review_id, per `DECISIONS.md#020` (Amended 2026-05-08):
the seed `ReviewState` carries the seed `PRContext` required at graph
start; the alternative (loading state from durable storage at run_graph
start) was rejected for adding a storage layer no other component
needs. `ReviewState` is JSON-serializable so V2's Celery implementation
round-trips the same payload through the broker without code changes.
"""

from outrider.dispatcher.background_tasks import BackgroundTasksDispatcher
from outrider.dispatcher.base import ReviewDispatcher
from outrider.dispatcher.concurrency import concurrency_limited
from outrider.dispatcher.config import DispatchConfig

__all__ = [
    "BackgroundTasksDispatcher",
    "DispatchConfig",
    "ReviewDispatcher",
    "concurrency_limited",
]
