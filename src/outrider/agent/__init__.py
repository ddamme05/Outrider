"""The review graph and fixture-driven eval entry points.

`graph.py` builds intake → triage → analyze ⇄ trace → synthesize → hitl → publish.
These are seven review phases; analyze uses a planner, per-file workers and an
aggregate node. Runtime dependencies stay in closures, and `state.py` re-exports
ReviewState (docs/conventions.md "File organization").

Eval drivers load on demand from `eval_driver.py`: `run_review` owns an ephemeral
database; `run_review_persisting` retains a caller-owned database; and
`run_review_with_resume` rebuilds the graph before explicitly approving a saved
HITL request. See `specs/2026-06-01-eval-graph-driver.md`."""

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
