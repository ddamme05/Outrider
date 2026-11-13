"""Run queued reviews after the HTTP response using FastAPI background tasks.

Dispatch validates a JSON round-trip of the seed before queueing it. The
shared runner applies its concurrency limit when the task executes. These tasks
run in the API process and do not survive its termination. See DECISIONS.md#020.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from outrider.agent.state import ReviewState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import BackgroundTasks

__all__ = ["BackgroundTasksDispatcher"]


class BackgroundTasksDispatcher:
    """Enqueue review execution on a request's background-task queue."""

    def __init__(
        self,
        *,
        background_tasks: BackgroundTasks,
        run_graph: Callable[[ReviewState], Awaitable[object]],
    ) -> None:
        """Bind the request queue and the shared app.state.run_graph runner.

        api/lifespan.py constructs the runner; the webhook supplies the queue.
        """
        self._background_tasks = background_tasks
        self._run_graph = run_graph

    async def dispatch(self, state: ReviewState) -> None:
        """Validate a JSON round-trip, then enqueue the reconstructed state.

        Serialization or validation failures propagate before work is queued;
        the webhook handler records the failed dispatch.
        """
        serialized = state.model_dump_json()
        dispatched_state = ReviewState.model_validate_json(serialized)
        self._background_tasks.add_task(self._run_graph, dispatched_state)
