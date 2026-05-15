"""Per-node bodies for the 7-node LangGraph state machine.

V1 ships triage. Intake, analyze, trace, synthesize, hitl, publish land
with their respective node specs. Each node is an async callable that
takes `state: ReviewState` plus closure-injected runtime dependencies
(LLMProvider, PhaseEventSink, ModelConfig values, etc.) and returns a
partial-state dict for LangGraph's reducer.
"""
