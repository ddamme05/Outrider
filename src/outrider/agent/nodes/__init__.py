"""Per-node bodies for the 7-node LangGraph state machine.

V1 ships all seven canonical nodes: intake, triage, analyze, trace,
synthesize, hitl, publish — each in its own module here. Each node is
an async callable that takes `state: ReviewState` plus closure-injected
runtime dependencies (LLMProvider, PhaseEventSink, ModelConfig values,
audit / anomaly sinks, etc.) and returns a partial-state dict for
LangGraph's reducer.
"""
