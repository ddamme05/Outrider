"""Agent package — the 7-node LangGraph state machine.

V1 ships all seven canonical nodes: intake, triage, analyze, trace,
synthesize, hitl, and publish (per their respective node specs). The
graph factory lives in `agent/graph.py`; per-node bodies live in
`agent/nodes/`.

Module structure follows docs/conventions.md "File organization":
- `agent/state.py` — re-export shim of `outrider.schemas.review_state.ReviewState`
  so node files write `from outrider.agent.state import ReviewState` without
  reaching into `schemas/`.
- `agent/graph.py` — `build_graph(...)` factory exposing the V1
  seven-node graph (intake → triage → analyze ⇄ trace → synthesize →
  hitl → publish → END). All arguments are keyword-only; the canonical
  order is documented in `docs/spec.md §9.3`. See `build_graph`'s
  signature for the full required-dep set (LLM provider, eight
  audit-side sink Protocols + one anomaly sink, GitHub publisher,
  import-path resolver, etc.).
- `agent/nodes/<node>.py` — the per-node body.
- `agent/eval_driver.py` — the eval graph drivers the non-structural eval
  scenarios import (`from outrider.agent import ...`), all three in `__all__`:
  `run_review(fixture_path)` (single pass, owns + drops an ephemeral DB);
  `run_review_with_resume(fixture_path, *, db_url)` (interrupt → restart → resume
  → publish, caller-owned DB); `run_review_persisting(fixture_path, *, db_url, ...)`
  (single pass, caller-owned DB so the audit stream + cache rows survive — the
  serve cache scenario). See `specs/2026-06-01-eval-graph-driver.md`.
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


# Lazy (PEP 562). `eval_driver` transitively imports `agent.graph`, which imports
# nodes that import `agent.reducers` — re-entering this package while it is still
# initializing. Importing eval_driver eagerly here would deadlock that chain, so
# we defer it until `outrider.agent.run_review` / `.EvalRunResult` is actually
# accessed (i.e. after this package finishes initializing — the deadlock risk is
# only re-entrancy during init, not external imports). Keeps the
# `from outrider.agent import run_review` contract.
def __getattr__(name: str) -> Any:
    if name in __all__:
        from outrider.agent import eval_driver

        return getattr(eval_driver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
