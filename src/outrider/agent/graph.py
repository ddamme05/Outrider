# Single-node StateGraph(ReviewState) factory per specs/2026-05-15-triage-node.md.
"""Single-node `StateGraph(ReviewState)` factory.

V1 ships ONE node: `triage`. The factory's responsibility ends at
producing a `CompiledStateGraph` that consumers (FastAPI app at startup
in production; tests at fixture-setup) can invoke via
`await graph.ainvoke(seed_state)`.

Multi-node graph topology (intake → triage → analyze ⇄ trace →
synthesize → hitl → publish) is a downstream spec per non-goal #3 of the
triage-node spec.

## Dependency injection

All three runtime deps are required keyword arguments per
`nodes-receive-deps-via-closure`:

  - `provider: LLMProvider` — the LLM transport; closed-over inside the
    triage callable.
  - `model_config: ModelConfig` — only `triage_model` is captured at the
    callsite (NOT the whole config object) so `model-strings-from-config-
    not-hardcoded` is honored.
  - `phase_event_sink: PhaseEventSink` — required (no Optional, no no-op
    default per the triage-node spec's "no-silent-phase-drop" guarantee).

## Validation gates

Three None-rejections + two structural rejections, all at construction
time (BEFORE any `StateGraph` work happens):

  1. `provider is None` → `BuildGraphError`
  2. `model_config is None` → `BuildGraphError`
  3. `phase_event_sink is None` → `BuildGraphError`
  4. `not isinstance(provider, LLMProvider)` → `BuildGraphError`
     Member-presence check: catches objects missing `complete`.
  5. `not isinstance(phase_event_sink, PhaseEventSink)` → `BuildGraphError`
     Member-presence check: catches objects missing `emit_phase`.

PEP 544 caveat: `@runtime_checkable` Protocols verify member PRESENCE
only — not signature, async-vs-sync nature, arity, or types. Wrong-
signature `complete`/`emit_phase` falls through to fail at the first
call site. mypy strict mode is the write-time gate for signature shape.
The deliberate decision NOT to add `inspect.signature` runtime introspection
is documented in the triage-node spec — that's brittle and not worth it.

## Async invocation

Triage is `async def`. Per LangGraph 1.1.6 docs (narrative/use-graph-api.md
"Use Pydantic models for graph state" + the async section), async-node
graphs are invoked via `await graph.ainvoke(state)` / `.astream(...)`,
NOT sync `.invoke(...)`.
"""

import functools
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from outrider.agent.nodes.triage import triage
from outrider.agent.state import ReviewState
from outrider.audit.sinks import PhaseEventSink
from outrider.llm.base import LLMProvider
from outrider.llm.config import ModelConfig

# LangGraph's CompiledStateGraph is generic over [StateT, ContextT, InputT,
# OutputT]; V1 uses ReviewState for state but the output is a dict (per
# LangGraph 1.1.6 docs "the output of the graph will NOT be an instance of
# a pydantic model"). The Any params keep type checking sound without
# overcommitting to a particular dict shape we don't strictly need to
# constrain at the factory level.
_CompiledTriageGraph = CompiledStateGraph[Any, Any, Any, Any]


class BuildGraphError(ValueError):
    """build_graph received an invalid or missing dependency.

    Raised at construction time (before any StateGraph work) when a
    required dep is None or fails the structural `isinstance` Protocol
    gate. Subclass of ValueError so it's catchable by generic value-error
    handlers but distinguishable from Pydantic ValidationError.
    """


def build_graph(
    *,
    provider: LLMProvider,
    model_config: ModelConfig,
    phase_event_sink: PhaseEventSink,
) -> _CompiledTriageGraph:
    """Build the single-node triage graph.

    Keyword-only arguments to prevent positional-confusion bugs at
    callsites that will eventually pass 3+ deps. Validation order: None
    checks first (cheaper), then Protocol structural checks.

    Returns a compiled `StateGraph(ReviewState)` ready for
    `await graph.ainvoke(seed_state)` invocation.
    """
    # Fail-closed: None on any required dep.
    if provider is None:
        raise BuildGraphError("provider must not be None")
    if model_config is None:
        raise BuildGraphError("model_config must not be None")
    if phase_event_sink is None:
        raise BuildGraphError("phase_event_sink must not be None")

    # Fail-closed: structural Protocol-member checks. PEP 544 caveat per
    # module docstring — these catch missing-member, not wrong-signature.
    if not isinstance(provider, LLMProvider):
        raise BuildGraphError(
            "provider does not satisfy LLMProvider Protocol "
            "(missing `complete` member; see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(phase_event_sink, PhaseEventSink):
        raise BuildGraphError(
            "phase_event_sink does not satisfy PhaseEventSink Protocol "
            "(missing `emit_phase` member; see PEP 544 runtime-checkable semantics)"
        )

    # Close over the per-tier model id, not the whole ModelConfig.
    triage_callable = functools.partial(
        triage,
        provider=provider,
        triage_model=model_config.triage_model,
        phase_event_sink=phase_event_sink,
    )

    builder = StateGraph(ReviewState)
    builder.add_node("triage", triage_callable)
    builder.add_edge(START, "triage")
    builder.add_edge("triage", END)
    return builder.compile()


__all__ = [
    "BuildGraphError",
    "build_graph",
]
