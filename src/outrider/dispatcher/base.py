# ReviewDispatcher Protocol per DECISIONS.md#020 (Amended 2026-05-08).
"""The webhook's background-execution contract (DECISIONS.md#020).

Dispatch takes a complete seed ReviewState because there is no seed-payload
storage layer. BackgroundTasksDispatcher is the current in-process implementation;
a future broker can implement the same contract.
"""

from typing import Protocol, runtime_checkable

from outrider.agent.state import ReviewState

__all__ = ["ReviewDispatcher"]


@runtime_checkable
class ReviewDispatcher(Protocol):
    """Enqueue a JSON-round-trippable seed without waiting for graph completion.

    The state-is-pure-data invariant applies across dispatch: serialize and validate
    before handing state to the runner. Serialization, validation, or enqueue failures
    propagate; the webhook handler records failed dispatches.
    """

    async def dispatch(self, state: ReviewState) -> None:
        """Enqueue state after validating its JSON round-trip."""
        ...
