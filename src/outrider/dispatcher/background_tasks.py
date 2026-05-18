# V1 in-process dispatcher per docs/spec.md §9.2.
"""`BackgroundTasksDispatcher` — V1 in-process implementation.

Wraps FastAPI's `BackgroundTasks`. Constructed **per-request** (FastAPI
`Depends(get_dispatcher)`) because `BackgroundTasks` is request-scoped —
a lifespan-bound singleton holding one `BackgroundTasks` instance would
accumulate tasks across requests and is not how FastAPI's runner
consumes the queue. The V2 `CeleryDispatcher` will be lifespan-singleton;
the Protocol stays uniform.

Per the intake-and-webhook spec, `dispatch(state)` JSON-round-trips the
state — serializes via `model_dump_json()` AND re-hydrates via
`ReviewState.model_validate_json()` — and passes the rehydrated value to
`add_task`. This is a real V1↔V2 parity gate: if a future contributor
adds a field that serializes one way and rehydrates differently (e.g.,
`set`→`list`, `tuple`→`list`, `ZoneInfo("America/New_York")`→fixed-offset),
V1's node execution sees the same shape V2's broker rehydration will
deliver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from outrider.agent.state import ReviewState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import BackgroundTasks

__all__ = ["BackgroundTasksDispatcher"]


class BackgroundTasksDispatcher:
    """V1 dispatcher: schedules graph execution on FastAPI BackgroundTasks.

    Constructed per-request from the route's `BackgroundTasks` parameter
    plus a lifespan-bound `run_graph` callable (the closure wrapping the
    compiled graph). The webhook handler's `Depends(get_dispatcher)`
    composition is the canonical wire-up.
    """

    def __init__(
        self,
        *,
        background_tasks: BackgroundTasks,
        run_graph: Callable[[ReviewState], Awaitable[object]],
    ) -> None:
        """
        Args:
            background_tasks: The request-scoped FastAPI `BackgroundTasks`
                instance. Per FastAPI semantics, tasks added here run AFTER
                the response is sent but BEFORE the request scope tears
                down.
            run_graph: The lifespan-bound async callable that invokes the
                compiled graph against a seed state. Bound at app startup
                in `api/lifespan.py`; stashed on `app.state.run_graph`.
        """
        self._background_tasks = background_tasks
        self._run_graph = run_graph

    async def dispatch(self, state: ReviewState) -> None:
        """Enqueue the seed `ReviewState` for background execution.

        Performs a real JSON round-trip — dump → validate — and passes
        the rehydrated state to `add_task`. The node graph executes
        against the rehydrated value, NOT the in-memory caller-side
        object, so V1's per-dispatch behavior matches what V2's broker
        rehydration will deliver. Any field whose serialize/deserialize
        cycle is not identity (e.g., `set`/`tuple` collapsing to `list`,
        timezone normalization) surfaces here at V1 dispatch rather than
        on the first V2 production review.

        Failure modes propagate as exceptions from `dispatch` (the
        webhook handler converts them to 5xx per the spec's transactional
        discipline):

          - `PydanticSerializationError` from `model_dump_json()`:
            a field not JSON-serializable (e.g., a raw `asyncio.Task`).
          - `pydantic_core.ValidationError` from `model_validate_json()`:
            a serialized shape the validator no longer accepts (rare;
            implies model drift between dump and validate).
        """
        serialized = state.model_dump_json()
        dispatched_state = ReviewState.model_validate_json(serialized)
        self._background_tasks.add_task(self._run_graph, dispatched_state)
