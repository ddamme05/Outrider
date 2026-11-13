"""Define the contract for scheduling a review from saved seed data.

ReviewDispatcher lets webhook handlers submit work without owning the runner.
background_tasks.py supplies the API-process implementation.
See DECISIONS.md#020.
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
