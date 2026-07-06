# StateGraph(ReviewState) factory: seven LOGICAL nodes (DECISIONS.md#064).
"""`StateGraph(ReviewState)` factory — seven LOGICAL nodes, nine physical vertices.

Graph topology (logical): intake → triage → analyze ⇄ trace → synthesize → hitl → publish.

V1 ships SEVEN logical nodes: `intake`, `triage`, `analyze`, `trace`,
`synthesize`, `hitl`, `publish`. The seven are the LOGICAL graph — audit
vocabulary, state identity, replay grouping; physical LangGraph vertices
exceed them since the parallel-analyze fan-out: analyze is three physical
vertices — the `analyze` planner (Command-routing: pass-0 Sends, pass-1
sequential), the per-file `analyze_file` Send worker, and the
`analyze_aggregate` fold — all emitting under `node_id="analyze"`
distinguished by phase_key (see DECISIONS.md#064). Intake enriches
`pr_context.changed_files` per
`DECISIONS.md#020`; triage runs a fast LLM pass for tier
classification; analyze runs one Sonnet call per DEEP/STANDARD-tier
file and emits findings; trace consumes `state.trace_candidates`,
ranks via Haiku, resolves via the two-phase fetch (probe + content),
and emits `TraceDecisionEvent` audit-first — the analyze router loops
back into analyze when new trace-fetched files arrived; hitl
partitions findings by severity, optionally interrupts the graph for
human approval, and emits `HITLRequestEvent` + `HITLDecisionEvent`
audit-first; publish routes each finding through `coordinates/`,
applies the V1 eligibility gate (CRITICAL/HIGH withheld unless an
explicit HITL approval lands), and posts a single GitHub review
materializing the eligible findings across all three tiers — inline
comments, a "Related concerns" body section, and an aggregate
dashboard-only note (DECISIONS.md#050). The factory produces a `CompiledStateGraph`
that consumers invoke via `await graph.ainvoke(seed_state)`.

The adaptive `analyze ⇄ trace` loop is bounded by `MAX_ANALYSIS_ROUNDS`
(`agent/nodes/trace.py`, depth-2 ceiling). The HITL interrupt is
implemented via LangGraph's `interrupt(...)` — see
`specs/2026-05-26-hitl-node.md` for the 13-step node body contract.

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

  - `provider: LLMProvider` — LLM transport for triage, analyze, and
    trace (Haiku ranking).
  - `model_config: ModelConfig` — `triage_model`, `analyze_model`,
    `standard_analyze_model` (STANDARD-tier analyze routing), `trace_model`,
    `synthesize_model`, and `patch_model` are captured at callsite (per
    `model-strings-from-config-not-hardcoded`).
  - `phase_event_sink: PhaseEventSink` — required for all seven nodes;
    each emits start/end phase markers.
  - `file_examination_sink: FileExaminationSink` — required for intake's
    per-file content-fetch events AND analyze's per-file examination
    outcome events (one per kept file).
  - `analyze_event_sink: AnalyzeEventSink` — required for analyze's
    `FindingEvent` / `FindingProposalRejectedEvent` /
    `AnalyzeResponseRejectedEvent` / `AnalyzeCompletedEvent` emissions.
  - `publish_event_sink: PublishEventSink` — required for publish's
    `PublishRoutingEvent` / `PublishEligibilityEvent` /
    `PublishAttemptEvent` / `PublishEvent` emissions + the
    `query_prior_publish_event` idempotency lookup.
  - `trace_sink: TraceEventSink` — required for trace's
    `TraceDecisionEvent` audit-first emission per M7.
  - `hitl_event_sink: HITLEventSink` — required for hitl's
    `HITLRequestEvent` + `HITLDecisionEvent` audit-first emissions.
  - `synthesize_event_sink: SynthesizeEventSink` — required for
    synthesize's `SynthesizeCompletedEvent` per-review aggregate
    emission.
  - `anomaly_sink: AnomalySink` — required for the in-graph anomaly
    emitters: synthesize on cross-round severity divergence
    (CROSS_ROUND_SEVERITY_DIVERGENCE) and analyze on budget starvation
    (COST_BUDGET_STARVATION, Stage 2); sweep callers use the same sink
    with `SWEEP_LOCK_ID` acquired around the call.
  - `review_status_sink: ReviewStatusSink` — required for hitl's
    `reviews.status` lifecycle transitions (mark_awaiting_approval +
    mark_running), publish's terminal-success transition (mark_completed
    per canonical `docs/spec.md` §3.3 step 10), and the sweep's
    `mark_awaiting_approval_expired`.
  - `hitl_config: HITLConfig` — required for hitl's deterministic
    `expires_at = state.received_at + timedelta(minutes=...)`
    derivation. Per `nodes-receive-deps-via-closure`, config travels
    through the dependency-injection seam at `build_graph(...)` —
    the node body does not read env vars.
  - `resolve_slack_target: SlackTargetResolver | None` — optional per-install
    Slack resolver, `async (installation_id) -> SlackNotifyTarget | None`. When
    set, the hitl + publish nodes resolve the install's channel + token-bound
    orchestrator at runtime and post a best-effort notification (awaited inline,
    never gate-breaking); None (the default) disables Slack. Injected here and
    closed over in the nodes per `nodes-receive-deps-via-closure`. The resolver
    body (installations read + token decrypt + notifier construction) lives in
    the lifespan composition root, so `agent/` never imports `cryptography` /
    `slack_sdk` (FUP-186).
  - `patch_config: PatchConfig` — required for synthesize's suggested-patch
    pass (`patches_enabled` + the per-review cap; the patch model is
    `model_config.patch_model`). Same closure-injection rule — synthesize
    reads no env (DECISIONS.md#040).
  - `intake_config: IntakeConfig | None` — optional whole-PR size-gate config
    (docs/spec.md §6.10: `OUTRIDER_INTAKE_MAX_LINES` / `MAX_FILES`, defaults
    1000 / 30). None (the default) reads the env / spec defaults once here and
    injects them; intake never reads env itself.
  - `checkpointer: BaseCheckpointSaver` — required for HITL durability.
    `interrupt(...)` writes state to the checkpointer; `Command(resume=...)`
    reads it back on the next `ainvoke(..., config={"configurable":
    {"thread_id": str(review_id)}})` call. Production is
    `AsyncPostgresSaver`; tests use `InMemorySaver`. Per
    `langgraph-1.1.6/narrative/persistence.md` the checkpointer is
    load-bearing for any graph that uses `interrupt(...)` — without
    one, the suspended state is in-memory only and is lost on process
    restart.
  - `publisher: GitHubPublisher` — required for publish's GitHub
    `create_review` call (single-review-per-PR contract).
  - `import_path_resolver: ImportPathResolver` — required for analyze's
    `parse_source(...)` dispatch (threaded to whichever `ast_facts/`
    language adapter the registry selects); resolves same-file import
    paths for the registry walk.
  - `db_factory: async_sessionmaker[AsyncSession]` — required for intake's
    `reviews.status='skipped'` / `'failed'` writes. Per `docs/spec.md
    §9.3`, `db_factory` is the canonical first parameter.
  - `github_factory: Callable[[int], GitHub]` — required for intake to
    construct a per-installation `GitHub` client on-demand. Per
    `DECISIONS.md#020`, minting happens at intake call-site, not at
    webhook receipt.

## Validation gates

None-rejections + structural Protocol checks at construction time
(BEFORE any `StateGraph` work). PEP 544 caveat applies: member-presence
only, not signature shape.

## Async invocation

All seven nodes are `async def`. Per LangGraph 1.1.6 docs, async-node
graphs are invoked via `await graph.ainvoke(state)` / `.astream(...)`.
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
    from collections.abc import Callable

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.cache import AnalyzeCacheStore
    from outrider.github import InstallationGitHubClient
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


def build_graph(  # noqa: PLR0913 — closure-injected deps surface; one kwarg per node-injected resource
    *,
    db_factory: async_sessionmaker[AsyncSession],
    github_factory: Callable[[int], InstallationGitHubClient],
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
    # Host-identity triad (DECISIONS.md#056) closed into the per-node completion events (Analyze /
    # Synthesize), which can emit on zero-LLM-call cache-serve/skip paths and so cannot source it
    # from a provider response. Lifespan supplies these from the resolved profile; default None =
    # unqualified (pre-#056 / tests).
    profile_id: str | None = None,
    reasoning_enabled: bool | None = None,
    profile_contract_digest: str | None = None,
) -> _CompiledTriageGraph:
    """Build the intake → triage → analyze ⇄ trace → synthesize → hitl → publish graph.

    Seven LOGICAL nodes, nine physical vertices — analyze spans three
    (planner / `analyze_file` workers / `analyze_aggregate`) per
    `DECISIONS.md#064`; audit vocabulary and replay grouping know only
    the seven.

    Keyword-only arguments prevent positional-confusion bugs at callsites
    with multiple deps. Validation order: None checks first (cheaper),
    then Protocol structural checks. Parameter order mirrors
    `docs/spec.md §9.3` (`db_factory` first, `github_factory` second).

    Returns a compiled `StateGraph(ReviewState)` ready for
    `await graph.ainvoke(seed_state)`.
    """
    # Fail-closed: None on any required dep.
    if provider is None:
        raise BuildGraphError("provider must not be None")
    if model_config is None:
        raise BuildGraphError("model_config must not be None")
    # Host-identity triad (DECISIONS.md#056): peers — all three present (a qualified host) or all
    # None (unqualified). Lifespan supplies them from one resolved profile; a partial set is a
    # wiring bug, caught here before it reaches the completion events' coherence validator.
    _triad = (profile_id, reasoning_enabled, profile_contract_digest)
    if any(v is not None for v in _triad) and not all(v is not None for v in _triad):
        raise BuildGraphError(
            "host-identity triad must be all-present or all-None; got a partial set "
            "(profile_id / reasoning_enabled / profile_contract_digest)"
        )
    # Host/triad coherence (FUP-194 / DECISIONS.md#056): the model_config's slug family
    # MUST match the host the triad names — a slug is claude-family IFF the host is
    # anthropic (profile_id None/unqualified counts as the native anthropic default). A
    # mismatch in EITHER direction is a wiring bug whose completion events would mis-stamp
    # the host — GLM slugs tagged anthropic/None, OR claude slugs tagged a registry host
    # like "baseten" — so fail loud at build, not fail-late at the persister. Host is
    # inferred from slug SHAPE, so a future non-anthropic host using a claude-shaped slug
    # would evade this — see the ADD-A-HOST checklist in host_profiles.py.
    from outrider.llm.config import is_anthropic_family_model  # noqa: PLC0415
    from outrider.llm.host_profiles import ANTHROPIC_PROFILE_ID  # noqa: PLC0415

    host_is_anthropic = profile_id is None or profile_id == ANTHROPIC_PROFILE_ID
    mismatched_models = sorted(
        {
            m
            for m in (
                model_config.triage_model,
                model_config.analyze_model,
                model_config.standard_analyze_model,
                model_config.synthesize_model,
                model_config.trace_model,
                model_config.patch_model,
            )
            if is_anthropic_family_model(m) != host_is_anthropic
        }
    )
    if mismatched_models:
        raise BuildGraphError(
            "model_config slugs are incoherent with the host-identity triad "
            f"(profile_id={profile_id!r}); mismatched model slug(s): {mismatched_models}"
        )
    if phase_event_sink is None:
        raise BuildGraphError("phase_event_sink must not be None")
    if file_examination_sink is None:
        raise BuildGraphError("file_examination_sink must not be None")
    if analyze_event_sink is None:
        raise BuildGraphError("analyze_event_sink must not be None")
    if publish_event_sink is None:
        raise BuildGraphError("publish_event_sink must not be None")
    if trace_sink is None:
        raise BuildGraphError("trace_sink must not be None")
    if hitl_event_sink is None:
        raise BuildGraphError("hitl_event_sink must not be None")
    if synthesize_event_sink is None:
        raise BuildGraphError("synthesize_event_sink must not be None")
    if review_status_sink is None:
        raise BuildGraphError("review_status_sink must not be None")
    if anomaly_sink is None:
        raise BuildGraphError("anomaly_sink must not be None")
    if hitl_config is None:
        raise BuildGraphError("hitl_config must not be None")
    if patch_config is None:
        raise BuildGraphError("patch_config must not be None")
    if checkpointer is None:
        raise BuildGraphError(
            "checkpointer must not be None — HITL `interrupt(...)` + "
            "`Command(resume=...)` require a durable checkpointer for "
            "cross-process state rehydration. Use `InMemorySaver` for "
            "tests, `AsyncPostgresSaver` in production."
        )
    if publisher is None:
        raise BuildGraphError("publisher must not be None")
    if import_path_resolver is None:
        raise BuildGraphError("import_path_resolver must not be None")
    if db_factory is None:
        raise BuildGraphError("db_factory must not be None")
    if github_factory is None:
        raise BuildGraphError("github_factory must not be None")

    # `total_review_budget_tokens` is a public-input int; validate here so
    # a misconfigured caller fails before analyze ever runs. `bool` is
    # rejected explicitly: `isinstance(True, int)` is True in Python, but
    # a boolean budget is never intended.
    if not isinstance(total_review_budget_tokens, int) or isinstance(
        total_review_budget_tokens, bool
    ):
        raise BuildGraphError(
            f"total_review_budget_tokens must be int "
            f"(got type: {type(total_review_budget_tokens).__name__})"
        )
    # Same bool-rejection rationale; additionally require >= 1 — a
    # zero-permit semaphore would deadlock the first worker forever.
    if (
        not isinstance(analyze_max_concurrency, int)
        or isinstance(analyze_max_concurrency, bool)
        or analyze_max_concurrency < 1
    ):
        raise BuildGraphError(
            f"analyze_max_concurrency must be an int >= 1 (got: {analyze_max_concurrency!r})"
        )

    # Non-callable factories would pass the None check but fail at first
    # intake execution with a confusing TypeError. `callable()` here
    # surfaces the misconfiguration before any review is dispatched.
    if not callable(db_factory):
        raise BuildGraphError(
            f"db_factory must be callable (got type: {type(db_factory).__name__})"
        )
    if not callable(github_factory):
        raise BuildGraphError(
            f"github_factory must be callable (got type: {type(github_factory).__name__})"
        )

    # Slack is optional: a per-install resolver, or None to disable. The resolver
    # yields a real SlackNotificationOrchestrator per install, so the prior
    # member-presence + all-or-nothing channel guards are no longer needed (the
    # channel + notifier-method presence are guaranteed by the target it returns).
    # A non-callable here is a miswired composition root — fail closed at build time
    # rather than as a mid-review TypeError on the gate/publish path.
    if resolve_slack_target is not None and not callable(resolve_slack_target):
        raise BuildGraphError(
            "resolve_slack_target must be an async callable "
            f"(installation_id) -> SlackNotifyTarget | None (got type: "
            f"{type(resolve_slack_target).__name__})"
        )

    # Fail-closed: structural Protocol-member checks. PEP 544 caveat per
    # module docstring — these catch missing-member, not wrong-signature.
    if not isinstance(provider, LLMProvider):
        # Compute the actually-missing member(s) rather than naming one — the
        # Protocol surface is {complete, aclose} (aclose added DECISIONS.md#035, retained #058),
        # so a hardcoded "missing complete" misdiagnoses an aclose-only gap.
        missing = [m for m in ("complete", "aclose") if not hasattr(provider, m)]
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
        # Default reads OUTRIDER_INTAKE_* (or the §6.10 spec defaults) once at build time.
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
        # Inert default (shadow mode): the classifier runs and audits
        # would-exclude verdicts but excludes nothing. The flip to True
        # is a later evidence-backed change per the trivial-scope-filter
        # spec's #041-style lifecycle.
        trivial_scope_filter_enabled=trivial_scope_filter_enabled,
        # Step 3b-mechanism: enforced OBSERVED skip. Inert default (False) + the
        # production registry seeds zero skip_safe queries, so no file is ever
        # skip-eligible in production — the enforced branch never fires until the
        # evidence-gated flip (DECISIONS.md#049). Injected at build_graph and closed
        # over in the analyze node (nodes-receive-deps-via-closure).
        analyze_observed_skip_enforced=analyze_observed_skip_enforced,
        # None disables the analyze cache (the eval driver's default for
        # scenarios that don't exercise it); production wiring injects a
        # real AnalyzeCacheStore. Store-or-None IS the enable switch —
        # nodes-receive-deps-via-closure, no separate flag to drift.
        analyze_cache_store=analyze_cache_store,
        # Shadow default: a wired store records would-hit/miss telemetry but
        # never serves. The flip to CacheMode.SERVE is a later telemetry-gated
        # config change (analyze-cache-serve-flip spec); cache_mode is orthogonal
        # to the store-or-None enable switch.
        cache_mode=cache_mode,
    )
    # Per-file Send worker (the fan-out cutover). ONE gate per compiled
    # graph, so the in-flight bound is global across concurrent reviews
    # sharing this graph — deliberate: the provider's rate limits are
    # global, not per-review (see ANALYZE_MAX_CONCURRENCY). The gate mints
    # its semaphore PER RUNNING LOOP (a bare Semaphore binds to the first
    # loop it is contended on and crashes on any other — a latent footgun
    # for module-scoped graph fixtures or sequential asyncio.run callers).
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
    # The two physical fan-out vertices (specs/2026-07-05-parallel-analyze.md).
    # LOGICAL node count stays seven (DECISIONS.md#064): both emit under
    # node_id="analyze", the internal edges emit no AgentTransitionEvents,
    # and every audit node vocabulary stays a closed seven-name Literal.
    builder.add_node("analyze_file", analyze_file_callable)
    builder.add_node("analyze_aggregate", analyze_aggregate_callable)
    builder.add_node("trace", trace_callable)
    builder.add_node("synthesize", synthesize_callable)
    builder.add_node("hitl", hitl_callable)
    builder.add_node("publish", publish_callable)
    builder.add_edge(START, "intake")
    # NO `builder.add_edge("intake", "triage")` here, and NO
    # `builder.add_conditional_edges("intake", ...)`. Intake routes via
    # `Command(goto=...)` per LangGraph 1.1.6 semantics — a static edge
    # would fire alongside the Command and send to BOTH destinations.
    # See module docstring "Routing" section for the full rationale.
    builder.add_edge("triage", "analyze")
    # NO edges out of "analyze" either — same Command rule as intake: the
    # planner returns Command(goto=[Send("analyze_file", ...), ...]) on
    # pass 0 (or "analyze_aggregate" on the zero-worker route), and
    # Command(goto="synthesize") after the sequential pass 1.
    # Every worker feeds the single aggregate superstep (map-reduce shape
    # per langgraph-1.1.6/narrative/use-graph-api.md "Send API").
    builder.add_edge("analyze_file", "analyze_aggregate")
    # Adaptive analyze ⇄ trace loop per `specs/2026-05-23-trace-node.md`,
    # moved from "analyze" to the aggregate (the pass-0 round now merges
    # when the AGGREGATE returns): route to trace iff round 1 produced
    # trace candidates. After trace: route back to analyze iff trace's
    # content fetches produced new files AND we're below the depth-2
    # round limit. Otherwise → synthesize.
    builder.add_conditional_edges("analyze_aggregate", _analyze_router)
    builder.add_conditional_edges("trace", _trace_router)
    # synthesize is the canonical 7th node: after analyze ⇄ trace
    # terminates, synthesize aggregates findings into a ReviewReport
    # before hitl partitions and publish emits. Static edge — synthesize
    # has no router (no fan-out, no branching).
    builder.add_edge("synthesize", "hitl")
    # hitl always proceeds to publish. The node body's `interrupt(...)`
    # is what pauses the graph for human approval; the static edge fires
    # only when the body completes (either pass-through on empty gated
    # set or post-resume after `Command(resume=...)`).
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
