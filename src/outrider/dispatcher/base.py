# ReviewDispatcher Protocol per DECISIONS.md#020 (Amended 2026-05-08).
"""`ReviewDispatcher` — the V1↔V2 swap seam.

The webhook handler depends on this Protocol, not on a concrete
dispatcher. V1 binds `BackgroundTasksDispatcher` via FastAPI
`Depends(get_dispatcher)`; V2 will bind `CeleryDispatcher` once that
exists. The Protocol's method takes the seed `ReviewState` directly,
NOT a `review_id` (per `DECISIONS.md#020` Amended 2026-05-08: the
pre-amendment `dispatch(review_id)` shape was explicitly rejected
because no seed-payload storage layer exists).

`@runtime_checkable` enables `isinstance(...)` member-presence checks
at the wiring site if needed; same PEP 544 caveat applies (signature
shape not validated at runtime, mypy strict is the write-time gate).
"""

from typing import Protocol, runtime_checkable

from outrider.agent.state import ReviewState

__all__ = ["ReviewDispatcher"]


@runtime_checkable
class ReviewDispatcher(Protocol):
    """Hand a seed `ReviewState` to background graph execution.

    Implementations:
      - `BackgroundTasksDispatcher` (V1) — FastAPI `BackgroundTasks`,
        in-process; constructed per-request from the route's
        request-scoped `BackgroundTasks` instance.
      - `CeleryDispatcher` (V2, not yet shipped) — durable Celery
        broker, lifespan singleton.

    The seed `state` MUST be JSON-round-trip-safe (`state-is-pure-data`):
    V1 in-process AND V2 broker-backed implementations both serialize
    via `state.model_dump_json()` and rehydrate via
    `ReviewState.model_validate_json(...)` before handing the state to
    the graph runner. V1 performs the round-trip in-process (no broker)
    so that fields whose serialize-then-validate cycle is not identity
    (e.g., `set`→`list`, `tuple`→`list`, `ZoneInfo`→fixed-offset)
    surface as drift at V1 dispatch time, NOT on the first V2
    production review.

    `dispatch` does NOT block on graph completion — it returns once the
    state has been enqueued for background execution. Failure modes
    that propagate as exceptions from `dispatch`:

      - `pydantic_core.PydanticSerializationError` from
        `model_dump_json()` — a field that isn't JSON-serializable
        (e.g., a raw `asyncio.Task`).
      - `pydantic_core.ValidationError` from `model_validate_json()` —
        a serialized shape the validator no longer accepts (implies
        model drift between dump and validate; rare but real).
      - Implementation-specific enqueue failures (event-loop shutdown,
        broker unreachable in V2).

    Per the intake-and-webhook spec, the webhook handler converts
    dispatch failure to 5xx after marking the review row 'failed'.
    """

    async def dispatch(self, state: ReviewState) -> None:
        """Enqueue the seed `ReviewState` for background graph execution.

        Implementations MUST treat `state` as JSON-round-trippable; V1
        performs an in-process dump+validate before handing to the
        graph runner, V2 will serialize through the broker.
        """
        ...
