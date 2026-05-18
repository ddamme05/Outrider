"""Agent package — the 7-node LangGraph state machine.

V1 ships intake + triage today (per the intake-and-webhook spec); the
remaining five nodes (analyze, trace, synthesize, hitl, publish) land
with their respective node specs. The graph factory lives in
`agent/graph.py`; per-node bodies live in `agent/nodes/`.

Module structure follows docs/conventions.md "File organization":
- `agent/state.py` — re-export shim of `outrider.schemas.review_state.ReviewState`
  so node files write `from outrider.agent.state import ReviewState` without
  reaching into `schemas/`.
- `agent/graph.py` — `build_graph(*, db_factory, github_factory, provider,
  model_config, phase_event_sink, file_examination_sink)` factory.
  All arguments are keyword-only; the canonical order is documented in
  `docs/spec.md §9.3` (note: §9.3's wording is being reconciled
  post-intake-and-webhook fold; see the spec file for the as-built
  signature).
- `agent/nodes/<node>.py` — the per-node body.
"""
