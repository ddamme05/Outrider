"""Agent package — the 7-node LangGraph state machine.

V1 ships intake, triage, analyze, and publish today (per their
respective node specs); the remaining three nodes (trace, synthesize,
hitl) land with their own specs. The graph factory lives in
`agent/graph.py`; per-node bodies live in `agent/nodes/`.

Module structure follows docs/conventions.md "File organization":
- `agent/state.py` — re-export shim of `outrider.schemas.review_state.ReviewState`
  so node files write `from outrider.agent.state import ReviewState` without
  reaching into `schemas/`.
- `agent/graph.py` — `build_graph(...)` factory exposing the V1
  four-node graph (intake → triage → analyze → publish → END). All
  arguments are keyword-only; the canonical order is documented in
  `docs/spec.md §9.3`. See `build_graph`'s signature for the full
  required-dep set (LLM provider, four sink Protocols, GitHub
  publisher, import-path resolver, etc.).
- `agent/nodes/<node>.py` — the per-node body.
"""
