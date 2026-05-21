# Three-node StateGraph(ReviewState) factory per specs/2026-05-19-analyze-node.md §8.
"""Three-node `StateGraph(ReviewState)` factory: intake → triage → analyze.

V1.x ships THREE nodes: `intake`, `triage`, `analyze`. Intake enriches
`pr_context.changed_files` per `DECISIONS.md#020`; triage runs a fast
LLM pass for tier classification; analyze runs one Sonnet call per
DEEP/STANDARD-tier file, emits findings, and returns analysis-round +
trace-candidate state deltas. The factory produces a
`CompiledStateGraph` that consumers invoke via
`await graph.ainvoke(seed_state)`.

Multi-node graph topology beyond intake → triage → analyze (trace →
synthesize → hitl → publish) is downstream of this spec.

## Routing: Command, not static or conditional edges from intake

Per the intake-and-webhook spec (Shape B), intake returns
`Command(goto=...)` to drive routing. No `add_edge("intake", "triage")`
and no `add_conditional_edges("intake", ...)` — per LangGraph 1.1.6
semantics, a static edge would fire ALONGSIDE the Command's dynamic
edge (sending to both destinations), and a conditional edge would
require a new state slot which conflicts with the canonical ReviewState
ownership rule (`pr_context.changed_files` enrichment only, no new
top-level slots from intake).

  - Success path: intake returns `Command(update={"pr_context": ...},
    goto="triage")` → routes to triage.
  - Size-gate skip: intake returns `Command(goto=END)` → routes to END.
  - Failure: intake re-raises after writing `reviews.status='failed'` —
    graph terminates via exception, no `Command` returned.

## Dependency injection

Required keyword arguments per `nodes-receive-deps-via-closure`:

  - `provider: LLMProvider` — LLM transport for triage AND analyze.
  - `model_config: ModelConfig` — `triage_model` and `analyze_model` are
    captured at callsite (per `model-strings-from-config-not-hardcoded`).
  - `phase_event_sink: PhaseEventSink` — required for all three nodes;
    each emits start/end phase markers.
  - `file_examination_sink: FileExaminationSink` — required for intake's
    per-file content-fetch events AND analyze's per-file examination
    outcome events (one per kept file).
  - `analyze_event_sink: AnalyzeEventSink` — required for analyze's
    `FindingEvent` / `FindingProposalRejectedEvent` /
    `AnalyzeResponseRejectedEvent` / `AnalyzeCompletedEvent` emissions.
  - `import_path_resolver: ImportPathResolver` — required for analyze's
    `parse_python(...)` call (passed through to `ast_facts/`); resolves
    same-file import paths for the registry walk.
  - `db_factory: async_sessionmaker[AsyncSession]` — required for intake's
    `reviews.status='skipped'` / `'failed'` writes. Per canonical
    `docs/spec.md §9.3`, `db_factory` is the first parameter; this spec
    adds it after the existing two sink params for backward-compat with
    the triage-node spec's signature precedent.
  - `github_factory: Callable[[int], GitHub]` — required for intake to
    construct a per-installation `GitHub` client on-demand. Per
    `DECISIONS.md#020`, minting happens at intake call-site, not at
    webhook receipt.

## Validation gates

None-rejections + structural Protocol checks at construction time
(BEFORE any `StateGraph` work). PEP 544 caveat applies: member-presence
only, not signature shape.

## Async invocation

Both nodes are `async def`. Per LangGraph 1.1.6 docs, async-node graphs
are invoked via `await graph.ainvoke(state)` / `.astream(...)`.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS, analyze
from outrider.agent.nodes.intake import intake
from outrider.agent.nodes.triage import triage
from outrider.agent.state import ReviewState
from outrider.ast_facts.base import ImportPathResolver
from outrider.audit.sinks import AnalyzeEventSink, FileExaminationSink, PhaseEventSink
from outrider.llm.base import LLMProvider

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.github import InstallationGitHubClient
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
    db_factory: async_sessionmaker[AsyncSession],
    github_factory: Callable[[int], InstallationGitHubClient],
    provider: LLMProvider,
    model_config: ModelConfig,
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    import_path_resolver: ImportPathResolver,
    total_review_budget_tokens: int = DEFAULT_REVIEW_BUDGET_TOKENS,
) -> _CompiledTriageGraph:
    """Build the three-node intake → triage → analyze graph.

    Keyword-only arguments to prevent positional-confusion bugs at
    callsites with multiple deps. Validation order: None checks first
    (cheaper), then Protocol structural checks.

    Parameter order mirrors the canonical `docs/spec.md §9.3` signature:
    `db_factory` is first, `github_factory` second, then the LLM provider
    + model_config + sinks. Because all params are keyword-only,
    reordering is source-compat for every caller.

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
    if file_examination_sink is None:
        raise BuildGraphError("file_examination_sink must not be None")
    if analyze_event_sink is None:
        raise BuildGraphError("analyze_event_sink must not be None")
    if import_path_resolver is None:
        raise BuildGraphError("import_path_resolver must not be None")
    if db_factory is None:
        raise BuildGraphError("db_factory must not be None")
    if github_factory is None:
        raise BuildGraphError("github_factory must not be None")

    # Non-callable factories would pass the None check but fail at first
    # intake execution with a confusing TypeError. The cost of one
    # `callable()` check now is the cost of avoiding a deferred crash
    # that strands the FIRST review the misconfigured app handles.
    # `async_sessionmaker` is callable (instances act as session factory
    # via `__call__`); `github_factory` is `Callable[[int], ...]`. Both
    # satisfy `callable()`.
    if not callable(db_factory):
        raise BuildGraphError(
            f"db_factory must be callable (got type: {type(db_factory).__name__})"
        )
    if not callable(github_factory):
        raise BuildGraphError(
            f"github_factory must be callable (got type: {type(github_factory).__name__})"
        )

    # Fail-closed: structural Protocol-member checks. PEP 544 caveat per
    # module docstring — these catch missing-member, not wrong-signature.
    if not isinstance(provider, LLMProvider):
        raise BuildGraphError(
            f"provider does not satisfy LLMProvider Protocol "
            f"(passed type: {type(provider).__name__}; "
            f"missing `complete` member; see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(phase_event_sink, PhaseEventSink):
        raise BuildGraphError(
            f"phase_event_sink does not satisfy PhaseEventSink Protocol "
            f"(passed type: {type(phase_event_sink).__name__}; "
            f"missing `emit_phase` member; see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(file_examination_sink, FileExaminationSink):
        raise BuildGraphError(
            f"file_examination_sink does not satisfy FileExaminationSink Protocol "
            f"(passed type: {type(file_examination_sink).__name__}; "
            f"missing `emit_file_examination` member; see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(analyze_event_sink, AnalyzeEventSink):
        raise BuildGraphError(
            f"analyze_event_sink does not satisfy AnalyzeEventSink Protocol "
            f"(passed type: {type(analyze_event_sink).__name__}; "
            f"missing one of `emit_finding` / `emit_finding_proposal_rejected` / "
            f"`emit_analyze_response_rejected` / `emit_analyze_completed`; "
            f"see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(import_path_resolver, ImportPathResolver):
        raise BuildGraphError(
            f"import_path_resolver does not satisfy ImportPathResolver Protocol "
            f"(passed type: {type(import_path_resolver).__name__}; "
            f"missing `resolve_candidate_paths` member; "
            f"see PEP 544 runtime-checkable semantics)"
        )

    # Close over the per-tier model id, not the whole ModelConfig.
    triage_callable = functools.partial(
        triage,
        provider=provider,
        triage_model=model_config.triage_model,
        phase_event_sink=phase_event_sink,
    )
    intake_callable = functools.partial(
        intake,
        github_factory=github_factory,
        db_factory=db_factory,
        phase_event_sink=phase_event_sink,
        file_examination_sink=file_examination_sink,
    )
    analyze_callable = functools.partial(
        analyze,
        provider=provider,
        analyze_model=model_config.analyze_model,
        phase_event_sink=phase_event_sink,
        file_examination_sink=file_examination_sink,
        analyze_event_sink=analyze_event_sink,
        import_path_resolver=import_path_resolver,
        total_review_budget_tokens=total_review_budget_tokens,
    )

    builder = StateGraph(ReviewState)
    builder.add_node("intake", intake_callable)
    builder.add_node("triage", triage_callable)
    builder.add_node("analyze", analyze_callable)
    builder.add_edge(START, "intake")
    # NO `builder.add_edge("intake", "triage")` here, and NO
    # `builder.add_conditional_edges("intake", ...)`. Intake routes via
    # `Command(goto=...)` per LangGraph 1.1.6 semantics — a static edge
    # would fire alongside the Command and send to BOTH destinations.
    # See module docstring "Routing" section for the full rationale.
    builder.add_edge("triage", "analyze")
    # Per analyze-node spec §8: V1 wires `analyze → END` unconditionally.
    # The future `analyze ⇄ trace` loop replaces this edge when the trace
    # spec lands; pre-wiring it now would produce an un-buildable graph
    # since trace doesn't exist as a registered node.
    builder.add_edge("analyze", END)
    return builder.compile()


__all__ = [
    "BuildGraphError",
    "build_graph",
]
