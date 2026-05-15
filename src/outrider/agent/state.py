# Re-export shim per docs/conventions.md "File organization"
"""ReviewState re-export shim.

Lets node files write `from outrider.agent.state import ReviewState`
without reaching into `outrider.schemas`. This shim is INTENTIONALLY a
re-export, not an abstraction layer — if it grows beyond re-exports,
that's the signal to reconsider, not to grow the shim (per
docs/conventions.md "File organization").
"""

from outrider.schemas.review_state import ReviewState

__all__ = [
    "ReviewState",
]
