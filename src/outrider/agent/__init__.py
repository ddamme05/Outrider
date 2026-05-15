"""Agent package — the 7-node LangGraph state machine.

V1 ships triage today; intake, analyze, trace, synthesize, hitl, and
publish land with their respective node specs. The graph factory lives
in `agent/graph.py`; per-node bodies live in `agent/nodes/`.

Module structure follows docs/conventions.md "File organization":
- `agent/state.py` — re-export shim of `outrider.schemas.review_state.ReviewState`
  so node files write `from outrider.agent.state import ReviewState` without
  reaching into `schemas/`.
- `agent/graph.py` — `build_graph(provider, model_config, phase_event_sink)`
  factory.
- `agent/nodes/<node>.py` — the per-node body.
"""
