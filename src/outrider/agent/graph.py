# StateGraph(ReviewState) factory: seven LOGICAL nodes (DECISIONS.md#064).
"""`StateGraph(ReviewState)` factory.

Seven LOGICAL nodes — intake, triage, analyze, trace, synthesize, hitl, publish —
over nine physical vertices: analyze fans out into a planner, per-file `analyze_file`
Send workers, and an `analyze_aggregate` fold. Audit vocabulary, state identity and
replay grouping know only the seven (`DECISIONS.md#064`).

Intake routes with `Command(goto=...)`, which is why this module deliberately has NO
`add_edge("intake", ...)` and no `add_conditional_edges("intake", ...)`: under LangGraph
1.1.6 a static edge fires ALONGSIDE the Command's dynamic edge, sending to both.

Every runtime dependency arrives as a keyword argument and is closed over in
the node callables rather than stored in ReviewState
(`nodes-receive-deps-via-closure`). All nodes are async, so the compiled graph is
driven with `await graph.ainvoke(seed_state)`.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from outrider.agent.nodes.analyze import (
    ANALYZE_MAX_CONCURRENCY,
    DEFAULT_REVIEW_BUDGET_TOKENS,
    AnalyzeConcurrencyGate,
    analyze,
    analyze_file,
)
from outrider.agent.nodes.analyze_aggregate import analyze_aggregate
from outrider.agent.nodes.cache_config import CacheMode
from outrider.agent.nodes.hitl import hitl
from outrider.agent.nodes.hitl_config import HITLConfig
from outrider.agent.nodes.intake import intake
from outrider.agent.nodes.intake_config import IntakeConfig
from outrider.agent.nodes.patch_config import PatchConfig
from outrider.agent.nodes.publish import publish
from outrider.agent.nodes.synthesize import synthesize
from outrider.agent.nodes.trace import MAX_ANALYSIS_ROUNDS, trace
from outrider.agent.nodes.triage import triage
from outrider.agent.state import ReviewState
from outrider.anomaly import AnomalySink
from outrider.ast_facts.base import ImportPathResolver
from outrider.audit.sinks import (
    AnalyzeEventSink,
    FileExaminationSink,
    HITLEventSink,
    PhaseEventSink,
    PublishEventSink,
    SynthesizeEventSink,
    TraceEventSink,
)
from outrider.db.sinks import ReviewStatusSink
from outrider.github.publisher import GitHubPublisher
from outrider.llm.base import LLMProvider

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.cache import AnalyzeCacheStore
    from outrider.github import InstallationAuthorizer, InstallationGitHubClient
    from outrider.llm.config import ModelConfig
    from outrider.notify.orchestrator import SlackTargetResolver

# LangGraph's CompiledStateGraph is generic over [StateT, ContextT, InputT,
# OutputT]. V1 uses ReviewState for state — pin it as the first param so the
# factory return type documents the state contract. ContextT/InputT/OutputT
# stay Any: per LangGraph 1.1.6 docs "the output of the graph will NOT be
# an instance of a pydantic model" (it's a dict), and the input/context
# shapes aren't constrained at the factory level.
_CompiledTriageGraph = CompiledStateGraph[ReviewState, Any, Any, Any]


class BuildGraphError(ValueError):
    """build_graph received an invalid or missing dependency.

    Raised at construction time (before any StateGraph work) when a
    required dep is None or fails the structural `isinstance` Protocol
    gate. Subclass of ValueError so it's catchable by generic value-error
    handlers but distinguishable from Pydantic ValidationError.
    """


def _validate_provider_and_host_identity(
    *,
    provider: LLMProvider,
    model_config: ModelConfig,
    profile_id: str | None,
    reasoning_enabled: bool | None,
    profile_contract_digest: str | None,
) -> None:
    if provider is None:
        raise BuildGraphError("provider must not be None")
    if model_config is None:
        raise BuildGraphError("model_config must not be None")

    # The triad is peers: all three present (a qualified host) or all None. Lifespan
    # supplies them from one resolved profile, so a partial set is a wiring bug
    # (DECISIONS.md#056).
    triad = (profile_id, reasoning_enabled, profile_contract_digest)
    if any(v is not None for v in triad) and not all(v is not None for v in triad):
        raise BuildGraphError(
            "host-identity triad must be all-present or all-None; got a partial set "
            "(profile_id / reasoning_enabled / profile_contract_digest)"
        )

    from outrider.llm.config import is_anthropic_family_model  # noqa: PLC0415
    from outrider.llm.host_profiles import ANTHROPIC_PROFILE_ID  # noqa: PLC0415

    # Host is inferred from slug SHAPE, so a future non-anthropic host using a
    # claude-shaped slug would evade this — see the ADD-A-HOST checklist in
    # host_profiles.py (FUP-194 / DECISIONS.md#056).
    host_is_anthropic = profile_id is None or profile_id == ANTHROPIC_PROFILE_ID
    configured_models = (
        model_config.triage_model,
        model_config.analyze_model,
        model_config.standard_analyze_model,
        model_config.synthesize_model,
        model_config.trace_model,
        model_config.patch_model,
    )
    mismatched: set[str] = set()
    for model in configured_models:
        if is_anthropic_family_model(model) != host_is_anthropic:
            mismatched.add(model)
    mismatched_models = sorted(mismatched)
    if mismatched_models:
        raise BuildGraphError(
            "model_config slugs are incoherent with the host-identity triad "
            f"(profile_id={profile_id!r}); mismatched model slug(s): {mismatched_models}"
        )


def _validate_required_dependencies(
    *,
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    publish_event_sink: PublishEventSink,
    trace_sink: TraceEventSink,
    hitl_event_sink: HITLEventSink,
    synthesize_event_sink: SynthesizeEventSink,
    review_status_sink: ReviewStatusSink,
    anomaly_sink: AnomalySink,
    hitl_config: HITLConfig,
    patch_config: PatchConfig,
    checkpointer: BaseCheckpointSaver[Any],
    publisher: GitHubPublisher,
    import_path_resolver: ImportPathResolver,
    db_factory: async_sessionmaker[AsyncSession],
    github_factory: Callable[[int], Awaitable[InstallationGitHubClient]],
    installation_authorizer: InstallationAuthorizer,
) -> None:
    """Reject None on every required dependency, in declaration order.

    Order is behaviour: when several are missing the caller sees the first.
    """
    # Annotated `object` so both loops share one element type; mypy otherwise pins
    # the variable to the first tuple's union and rejects the second.
    first_group: tuple[tuple[str, object], ...] = (
        ("phase_event_sink", phase_event_sink),
        ("file_examination_sink", file_examination_sink),
        ("analyze_event_sink", analyze_event_sink),
        ("publish_event_sink", publish_event_sink),
        ("trace_sink", trace_sink),
        ("hitl_event_sink", hitl_event_sink),
        ("synthesize_event_sink", synthesize_event_sink),
        ("review_status_sink", review_status_sink),
        ("anomaly_sink", anomaly_sink),
        ("hitl_config", hitl_config),
        ("patch_config", patch_config),
    )
    for name, value in first_group:
        if value is None:
            raise BuildGraphError(f"{name} must not be None")

    # Explicit, and in position: the message carries the reason, and losing it would
    # leave a caller guessing why an in-memory graph cannot resume.
    if checkpointer is None:
        raise BuildGraphError(
            "checkpointer must not be None — HITL `interrupt(...)` + "
            "`Command(resume=...)` require a durable checkpointer for "
            "cross-process state rehydration. Use `InMemorySaver` for "
            "tests, `AsyncPostgresSaver` in production."
        )

    second_group: tuple[tuple[str, object], ...] = (
        ("publisher", publisher),
        ("import_path_resolver", import_path_resolver),
        ("db_factory", db_factory),
        ("github_factory", github_factory),
        # Required, not optional: the #065 live-auth gate is a security check, so a
        # missing authorizer must fail the build rather than silently fail open.
        ("installation_authorizer", installation_authorizer),
    )
    for name, value in second_group:
        if value is None:
            raise BuildGraphError(f"{name} must not be None")


def _validate_numeric_settings(
    *, total_review_budget_tokens: int, analyze_max_concurrency: int
) -> None:
    # `bool` is rejected explicitly: `isinstance(True, int)` is True in Python.
    if not isinstance(total_review_budget_tokens, int) or isinstance(
        total_review_budget_tokens, bool
    ):
        raise BuildGraphError(
            f"total_review_budget_tokens must be int "
            f"(got type: {type(total_review_budget_tokens).__name__})"
        )
    # >= 1 as well: a zero-permit semaphore would deadlock the first worker forever.
    if (
        not isinstance(analyze_max_concurrency, int)
        or isinstance(analyze_max_concurrency, bool)
        or analyze_max_concurrency < 1
    ):
        raise BuildGraphError(
            f"analyze_max_concurrency must be an int >= 1 (got: {analyze_max_concurrency!r})"
        )


def _validate_factories(
    *,
    db_factory: async_sessionmaker[AsyncSession],
    github_factory: Callable[[int], Awaitable[InstallationGitHubClient]],
    installation_authorizer: InstallationAuthorizer,
    resolve_slack_target: SlackTargetResolver | None,
) -> None:
    """Non-callables pass the None check and fail later with a confusing TypeError."""
    if not callable(db_factory):
        raise BuildGraphError(
            f"db_factory must be callable (got type: {type(db_factory).__name__})"
        )
    if not callable(github_factory):
        raise BuildGraphError(
            f"github_factory must be callable (got type: {type(github_factory).__name__})"
        )
    if not callable(installation_authorizer):
        raise BuildGraphError(
            "installation_authorizer must be callable (got type: "
            f"{type(installation_authorizer).__name__})"
        )
    # Slack is optional: a per-install resolver, or None to disable.
    if resolve_slack_target is not None and not callable(resolve_slack_target):
        raise BuildGraphError(
            "resolve_slack_target must be an async callable "
            f"(installation_id) -> SlackNotifyTarget | None (got type: "
            f"{type(resolve_slack_target).__name__})"
        )


def _validate_protocols(
    *,
    provider: LLMProvider,
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    publish_event_sink: PublishEventSink,
    trace_sink: TraceEventSink,
    hitl_event_sink: HITLEventSink,
    synthesize_event_sink: SynthesizeEventSink,
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
    hitl_config: HITLConfig,
    patch_config: PatchConfig,
    publisher: GitHubPublisher,
    import_path_resolver: ImportPathResolver,
) -> None:
    """Structural Protocol-member checks (PEP 544: presence, not signature shape)."""
    if not isinstance(provider, LLMProvider):
        # Compute the actually-missing member(s) rather than naming one — the
        # Protocol surface is {complete, aclose} (aclose added DECISIONS.md#035, retained #058),
        # so a hardcoded "missing complete" misdiagnoses an aclose-only gap.
        missing: list[str] = []
        for member_name in ("complete", "aclose"):
            if not hasattr(provider, member_name):
                missing.append(member_name)
        raise BuildGraphError(
            f"provider does not satisfy LLMProvider Protocol "
            f"(passed type: {type(provider).__name__}; "
            f"missing member(s): {', '.join(missing) or '<signature mismatch>'}; "
            f"see PEP 544 runtime-checkable semantics)"
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
    if not isinstance(publish_event_sink, PublishEventSink):
        raise BuildGraphError(
            f"publish_event_sink does not satisfy PublishEventSink Protocol "
            f"(passed type: {type(publish_event_sink).__name__}; "
            f"missing one of `emit_publish_routing` / `emit_publish_eligibility` / "
            f"`emit_publish_attempt` / `emit_publish_result` / "
            f"`query_prior_publish_event`; "
            f"see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(trace_sink, TraceEventSink):
        raise BuildGraphError(
            f"trace_sink does not satisfy TraceEventSink Protocol "
            f"(passed type: {type(trace_sink).__name__}; "
            f"missing `emit_trace_decision` member; "
            f"see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(hitl_event_sink, HITLEventSink):
        raise BuildGraphError(
            f"hitl_event_sink does not satisfy HITLEventSink Protocol "
            f"(passed type: {type(hitl_event_sink).__name__}; "
            f"missing one of `emit_hitl_request` / `emit_hitl_decision`; "
            f"see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(synthesize_event_sink, SynthesizeEventSink):
        raise BuildGraphError(
            f"synthesize_event_sink does not satisfy SynthesizeEventSink Protocol "
            f"(passed type: {type(synthesize_event_sink).__name__}; "
            f"missing one of `emit_synthesize_completed` / `query_review_llm_aggregates`; "
            f"see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(anomaly_sink, AnomalySink):
        raise BuildGraphError(
            f"anomaly_sink does not satisfy AnomalySink Protocol "
            f"(passed type: {type(anomaly_sink).__name__}; "
            f"missing `emit_anomaly` member; "
            f"see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(review_status_sink, ReviewStatusSink):
        raise BuildGraphError(
            f"review_status_sink does not satisfy ReviewStatusSink Protocol "
            f"(passed type: {type(review_status_sink).__name__}; "
            f"missing one of `mark_awaiting_approval` / `mark_running` / "
            f"`mark_awaiting_approval_expired` / `mark_completed`; "
            f"see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(hitl_config, HITLConfig):
        raise BuildGraphError(
            f"hitl_config must be a HITLConfig instance (passed type: {type(hitl_config).__name__})"
        )
    if not isinstance(patch_config, PatchConfig):
        raise BuildGraphError(
            f"patch_config must be a PatchConfig (passed type: {type(patch_config).__name__})"
        )
    if not isinstance(publisher, GitHubPublisher):
        raise BuildGraphError(
            f"publisher does not satisfy GitHubPublisher Protocol "
            f"(passed type: {type(publisher).__name__}; "
            f"missing one of `create_review` / `find_existing_review_on_head_sha`; "
            f"see PEP 544 runtime-checkable semantics)"
        )
    if not isinstance(import_path_resolver, ImportPathResolver):
        raise BuildGraphError(
            f"import_path_resolver does not satisfy ImportPathResolver Protocol "
            f"(passed type: {type(import_path_resolver).__name__}; "
            f"missing `resolve_candidate_paths` member; "
            f"see PEP 544 runtime-checkable semantics)"
        )


def build_graph(  # noqa: PLR0913 — closure-injected deps surface; one kwarg per node-injected resource
    *,
    db_factory: async_sessionmaker[AsyncSession],
    github_factory: Callable[[int], Awaitable[InstallationGitHubClient]],
    installation_authorizer: InstallationAuthorizer,
    provider: LLMProvider,
    model_config: ModelConfig,
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    publish_event_sink: PublishEventSink,
    trace_sink: TraceEventSink,
    hitl_event_sink: HITLEventSink,
    synthesize_event_sink: SynthesizeEventSink,
    review_status_sink: ReviewStatusSink,
    anomaly_sink: AnomalySink,
    hitl_config: HITLConfig,
    patch_config: PatchConfig,
    intake_config: IntakeConfig | None = None,
    checkpointer: BaseCheckpointSaver[Any],
    publisher: GitHubPublisher,
    import_path_resolver: ImportPathResolver,
    total_review_budget_tokens: int = DEFAULT_REVIEW_BUDGET_TOKENS,
    analyze_max_concurrency: int = ANALYZE_MAX_CONCURRENCY,
    trivial_scope_filter_enabled: bool = False,
    analyze_observed_skip_enforced: bool = False,
    analyze_cache_store: AnalyzeCacheStore | None = None,
    cache_mode: CacheMode = CacheMode.SHADOW,
    resolve_slack_target: SlackTargetResolver | None = None,
    dashboard_base_url: str | None = None,
    # Closed into the completion events, which can emit on zero-LLM-call cache-serve
    # paths and so cannot source the host from a provider response (DECISIONS.md#056).
    profile_id: str | None = None,
    reasoning_enabled: bool | None = None,
    profile_contract_digest: str | None = None,
) -> _CompiledTriageGraph:
    """Build the intake → triage → analyze ⇄ trace → synthesize → hitl → publish graph.

    Seven LOGICAL nodes, nine physical vertices — analyze spans three
    (planner / `analyze_file` workers / `analyze_aggregate`) per
    `DECISIONS.md#064`; audit vocabulary and replay grouping know only
    the seven.

    Parameter order mirrors `docs/spec.md §9.3` (`db_factory` first).

    Returns a compiled `StateGraph(ReviewState)` ready for
    `await graph.ainvoke(seed_state)`.
    """
    _validate_provider_and_host_identity(
        provider=provider,
        model_config=model_config,
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
    )
    _validate_required_dependencies(
        phase_event_sink=phase_event_sink,
        file_examination_sink=file_examination_sink,
        analyze_event_sink=analyze_event_sink,
        publish_event_sink=publish_event_sink,
        trace_sink=trace_sink,
        hitl_event_sink=hitl_event_sink,
        synthesize_event_sink=synthesize_event_sink,
        review_status_sink=review_status_sink,
        anomaly_sink=anomaly_sink,
        hitl_config=hitl_config,
        patch_config=patch_config,
        checkpointer=checkpointer,
        publisher=publisher,
        import_path_resolver=import_path_resolver,
        db_factory=db_factory,
        github_factory=github_factory,
        installation_authorizer=installation_authorizer,
    )
    _validate_numeric_settings(
        total_review_budget_tokens=total_review_budget_tokens,
        analyze_max_concurrency=analyze_max_concurrency,
    )
    _validate_factories(
        db_factory=db_factory,
        github_factory=github_factory,
        installation_authorizer=installation_authorizer,
        resolve_slack_target=resolve_slack_target,
    )
    _validate_protocols(
        provider=provider,
        phase_event_sink=phase_event_sink,
        file_examination_sink=file_examination_sink,
        analyze_event_sink=analyze_event_sink,
        publish_event_sink=publish_event_sink,
        trace_sink=trace_sink,
        hitl_event_sink=hitl_event_sink,
        synthesize_event_sink=synthesize_event_sink,
        anomaly_sink=anomaly_sink,
        review_status_sink=review_status_sink,
        hitl_config=hitl_config,
        patch_config=patch_config,
        publisher=publisher,
        import_path_resolver=import_path_resolver,
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
        installation_authorizer=installation_authorizer,
        db_factory=db_factory,
        phase_event_sink=phase_event_sink,
        file_examination_sink=file_examination_sink,
        intake_config=intake_config or IntakeConfig(),
    )
    analyze_callable = functools.partial(
        analyze,
        provider=provider,
        analyze_model=model_config.analyze_model,
        standard_analyze_model=model_config.standard_analyze_model,
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
        phase_event_sink=phase_event_sink,
        file_examination_sink=file_examination_sink,
        analyze_event_sink=analyze_event_sink,
        anomaly_sink=anomaly_sink,
        import_path_resolver=import_path_resolver,
        total_review_budget_tokens=total_review_budget_tokens,
        # Shadow default: the classifier audits would-exclude verdicts and excludes
        # nothing.
        trivial_scope_filter_enabled=trivial_scope_filter_enabled,
        # Inert by default, and the production registry seeds zero skip_safe queries,
        # so no file is skip-eligible either way (DECISIONS.md#049).
        analyze_observed_skip_enforced=analyze_observed_skip_enforced,
        # Store-or-None IS the enable switch; there is deliberately no separate flag
        # that could drift out of step with it.
        analyze_cache_store=analyze_cache_store,
        # Orthogonal to the enable switch above: a wired store in SHADOW records
        # would-hit telemetry but never serves.
        cache_mode=cache_mode,
    )
    # One gate per compiled graph. The bound is per-(graph, running loop) — see the
    # AnalyzeConcurrencyGate contract for why the semaphore cannot be minted here.
    analyze_file_callable = functools.partial(
        analyze_file,
        provider=provider,
        analyze_model=model_config.analyze_model,
        standard_analyze_model=model_config.standard_analyze_model,
        import_path_resolver=import_path_resolver,
        phase_event_sink=phase_event_sink,
        file_examination_sink=file_examination_sink,
        analyze_event_sink=analyze_event_sink,
        total_review_budget_tokens=total_review_budget_tokens,
        trivial_scope_filter_enabled=trivial_scope_filter_enabled,
        analyze_observed_skip_enforced=analyze_observed_skip_enforced,
        analyze_cache_store=analyze_cache_store,
        cache_mode=cache_mode,
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
        concurrency_semaphore=AnalyzeConcurrencyGate(analyze_max_concurrency),
    )
    analyze_aggregate_callable = functools.partial(
        analyze_aggregate,
        analyze_event_sink=analyze_event_sink,
        phase_event_sink=phase_event_sink,
        anomaly_sink=anomaly_sink,
        analyze_model=model_config.analyze_model,
        standard_analyze_model=model_config.standard_analyze_model,
        total_review_budget_tokens=total_review_budget_tokens,
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
    )
    publish_callable = functools.partial(
        publish,
        publisher=publisher,
        publish_event_sink=publish_event_sink,
        phase_event_sink=phase_event_sink,
        review_status_sink=review_status_sink,
        github_factory=github_factory,
        dashboard_base_url=dashboard_base_url,
        resolve_slack_target=resolve_slack_target,
    )
    trace_callable = functools.partial(
        trace,
        provider=provider,
        trace_model=model_config.trace_model,
        phase_event_sink=phase_event_sink,
        trace_sink=trace_sink,
        github_factory=github_factory,
    )
    hitl_callable = functools.partial(
        hitl,
        phase_event_sink=phase_event_sink,
        hitl_event_sink=hitl_event_sink,
        review_status_sink=review_status_sink,
        hitl_config=hitl_config,
        resolve_slack_target=resolve_slack_target,
    )
    synthesize_callable = functools.partial(
        synthesize,
        provider=provider,
        synthesize_model=model_config.synthesize_model,
        patch_model=model_config.patch_model,
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
        patches_enabled=patch_config.patches_enabled,
        max_suggestions=patch_config.max_patch_suggestions_per_review,
        phase_event_sink=phase_event_sink,
        synthesize_event_sink=synthesize_event_sink,
        anomaly_sink=anomaly_sink,
    )

    builder = StateGraph(ReviewState)
    builder.add_node("intake", intake_callable)
    builder.add_node("triage", triage_callable)
    builder.add_node("analyze", analyze_callable)
    # The two physical fan-out vertices. Both emit under node_id="analyze", so the
    # logical node count stays seven (DECISIONS.md#064).
    builder.add_node("analyze_file", analyze_file_callable)
    builder.add_node("analyze_aggregate", analyze_aggregate_callable)
    builder.add_node("trace", trace_callable)
    builder.add_node("synthesize", synthesize_callable)
    builder.add_node("hitl", hitl_callable)
    builder.add_node("publish", publish_callable)
    builder.add_edge(START, "intake")
    # NO `add_edge("intake", "triage")` and NO `add_conditional_edges("intake", ...)`:
    # intake routes via `Command(goto=...)`, and under LangGraph 1.1.6 a static edge
    # fires alongside the Command and sends to BOTH destinations.
    builder.add_edge("triage", "analyze")
    # NO edges out of "analyze" either, same Command rule: the planner returns
    # Send(...) fan-out on pass 0 and goto="synthesize" after pass 1.
    builder.add_edge("analyze_file", "analyze_aggregate")
    # The adaptive loop hangs off the AGGREGATE, not the planner, because that is
    # where a pass-0 round merges.
    builder.add_conditional_edges("analyze_aggregate", _analyze_router)
    builder.add_conditional_edges("trace", _trace_router)
    builder.add_edge("synthesize", "hitl")
    # A static edge is correct here: the node body's `interrupt(...)` is what pauses
    # the graph, so this fires only once the body completes or resumes.
    builder.add_edge("hitl", "publish")
    builder.add_edge("publish", END)
    return builder.compile(checkpointer=checkpointer)


def _analyze_router(state: ReviewState) -> str:
    """Route the aggregate's output: trace if round 1 has candidates, else synthesize.

    Attached to "analyze_aggregate" (the pass-0 round merges when the
    aggregate returns; the sequential pass-1 tail routes itself via
    Command). Per the trace-node spec: after pass 0 (i.e.,
    `len(state.analysis_rounds) == 1`), route to trace to consume the
    accumulated trace_candidates; otherwise synthesize.
    Round count IS the depth counter — which is why a round is one
    analyze PASS, never one parallel worker (see DECISIONS.md#063).
    Synthesize folds findings into a ReviewReport; HITL then partitions
    by severity (empty gate-set passes through to publish without an
    interrupt).
    """
    if len(state.analysis_rounds) == 1 and state.trace_candidates:
        return "trace"
    return "synthesize"


def _trace_router(state: ReviewState) -> str:
    """Route trace's output: analyze if NEW files fetched this pass, else synthesize.

    Per the trace-node spec: route back to analyze iff trace fetched at
    least one new file IN THE MOST RECENT trace() CALL AND we're still
    below `MAX_ANALYSIS_ROUNDS` (depth-2 ceiling). Otherwise proceed to
    synthesize (which then routes through HITL to publish). The depth
    bound is enforced HERE because trace runs after analyze pass N, so
    re-entering analyze produces round N+1.

    Reads `state.last_trace_pass_fetched_count` (the per-invocation
    delta trace() writes on every call), NOT the cumulative
    `len(state.trace_fetched_files)`. The cumulative check would route
    to analyze even when the latest trace() call yielded nothing new
    (replay path: cumulative list rehydrates non-empty from checkpoint
    even when the just-run trace() returned no new fetches).
    """
    if state.last_trace_pass_fetched_count > 0 and len(state.analysis_rounds) < MAX_ANALYSIS_ROUNDS:
        return "analyze"
    return "synthesize"


__all__ = [
    "BuildGraphError",
    "build_graph",
]
