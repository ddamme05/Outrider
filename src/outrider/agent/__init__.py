"""Build review workflows and expose fixture-based review runners.

graph.py connects the review steps. The test runners in eval_driver.py load
on demand and can create a temporary database or use one supplied by the caller.
state.py provides the shared ReviewState import.

See specs/2026-06-01-eval-graph-driver.md and docs/conventions.md.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from outrider.agent.eval_driver import (
        EvalRunResult,
        ResumedRunResult,
        run_review,
        run_review_persisting,
        run_review_with_resume,
    )

__all__ = [
    "EvalRunResult",
    "ResumedRunResult",
    "run_review",
    "run_review_persisting",
    "run_review_with_resume",
]


# Keep eval-driver dependencies out of ordinary node/state imports. PEP 562
# loads the driver only when an exported entry point or result type is requested.
def __getattr__(name: str) -> Any:
    if name in __all__:
        from outrider.agent import eval_driver

        return getattr(eval_driver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
