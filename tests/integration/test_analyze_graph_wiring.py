"""Analyze-node graph wiring integration tests per spec §8.

Four gates:

1. **`triage → analyze → END` wires correctly.** Build the compiled graph;
   assert the edge set and node membership.
2. **One clean eligible file flows through analyze.** Tier=DEEP file →
   analyze runs → AnalysisRound populated with at least one finding;
   FindingEvent emitted.
3. **One triage-excluded file does not enter analyze.** Tier=SKIM file →
   analyze iterates zero files for it; FileExaminationEvent with that
   file's path does NOT fire from analyze.
4. **One budget-skip file remains audited and does not look like 'clean'.**
   Tier=DEEP file + tiny `total_review_budget_tokens` → cost gate fires →
   `FileExaminationEvent(parse_status="skipped",
   skip_reason=COST_BUDGET_EXHAUSTED)` emitted; the file appears in
   `AnalysisRound.files_skipped` (not `files_examined`).

These exercise the COMPILED graph end-to-end (intake → triage → analyze →
publish; publish runs but is a no-op pass-through under the fixtures
here because triage routes everything to SKIP or analyze emits no
admitted findings). The unit tests in `tests/unit/test_analyze_node.py`
cover the node body in isolation; this file covers the wiring.
"""

from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

import pytest
from langgraph.graph import START
from sqlalchemy import Update

from outrider.agent.graph import build_graph
from outrider.ast_facts.models import SkipReason
from outrider.audit.aggregates import ReviewLLMAggregates
from outrider.github.authz import LiveAuthOutcome, LiveAuthResult
from outrider.llm.config import ModelConfig
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from outrider.audit.events import (
        AnalyzeCompletedEvent,
        AnalyzeResponseRejectedEvent,
        CacheLookupEvent,
        FileExaminationEvent,
        FindingEvent,
        FindingProposalRejectedEvent,
        ReviewPhaseEvent,
        ScopeExclusionEvent,
    )
    from outrider.llm.base import LLMRequest, LLMResponse
    from outrider.schemas.review_finding import ReviewFinding


# ---------------------------------------------------------------------------
# Cross-conftest fixture protocol (mirrors test_review_state_langgraph_merge)
# ---------------------------------------------------------------------------


class _RecordingPhaseEventSinkLike(Protocol):
    events: list[ReviewPhaseEvent]

    async def emit_phase(self, event: ReviewPhaseEvent) -> None: ...


# ---------------------------------------------------------------------------
# Mock LLM provider: routes by node_id
# ---------------------------------------------------------------------------


class _RoutingMockLLMProvider:
    """Returns canned responses keyed by `request.node_id`.

    Triage gets a configurable tier map; analyze gets a configurable
    findings payload. Tests construct one provider per scenario with the
    right pair of canned responses.
    """

    def __init__(self, *, triage_response: str, analyze_response: str) -> None:
        self.triage_response = triage_response
        self.analyze_response = analyze_response
        self.calls: list[LLMRequest] = []

    async def aclose(self) -> None:
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from outrider.llm.anthropic_provider import (
            _ANTHROPIC_CONTRACT_DIGEST,
            _ANTHROPIC_PROFILE_ID,
        )
        from outrider.llm.base import LLMResponse

        self.calls.append(request)
        if request.node_id == "triage":
            text = self.triage_response
        elif request.node_id == "analyze":
            text = self.analyze_response
        elif request.node_id == "synthesize":
            # Synthesize's call is prose summary; tests in this file
            # exercise analyze wiring but the graph runs to publish via
            # synthesize → hitl → publish. A minimal short prose string
            # lets ReviewReport.summary land below max_length=2000.
            text = "Test mock: synthesize summary."
        else:
            msg = f"unexpected node_id in test mock: {request.node_id!r}"
            raise AssertionError(msg)
        return LLMResponse(
            text=text,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=42,
            profile_id=_ANTHROPIC_PROFILE_ID,
            reasoning_enabled=False,
            profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
        )


# ---------------------------------------------------------------------------
# Intake stubs (cooperative — return the seed's file structure unchanged)
# ---------------------------------------------------------------------------


_SEED_INSTALLATION_ID = 12345
_SEED_REPO_ID = 500  # reviews.repo_id the #065 gate SELECT returns; forwarded to the authorizer
_SEED_OWNER = "acme"
_SEED_REPO = "widget"
_SEED_PULL_NUMBER = 42

# Two functions so analyze has scope units to intersect with.
_DEEP_FILE_PATH = "src/clean.py"
_DEEP_FILE_HEAD = b"def my_function():\n    return 42\n\ndef another(x):\n    return x + 1\n"
_DEEP_FILE_BASE = b"def my_function():\n    return 0\n\ndef another(x):\n    return x + 1\n"
_DEEP_FILE_PATCH = (
    f"--- a/{_DEEP_FILE_PATH}\n"
    f"+++ b/{_DEEP_FILE_PATH}\n"
    "@@ -1,2 +1,2 @@\n"
    " def my_function():\n"
    "-    return 0\n"
    "+    return 42\n"
)


@dataclass
class _StubFileMeta:
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None = None
    previous_filename: str | None = None


@dataclass
class _StubContentFile:
    encoding: str
    content: str


@dataclass
class _StubResponse:
    parsed_data: Any


class _StubReposAPI:
    async def async_get_content(
        self, owner: str, repo: str, path: str, *, ref: str
    ) -> _StubResponse:
        if ref == "a" * 40:
            content_bytes = _DEEP_FILE_BASE
        elif ref == "b" * 40:
            content_bytes = _DEEP_FILE_HEAD
        else:
            content_bytes = b""
        return _StubResponse(
            parsed_data=_StubContentFile(
                encoding="base64",
                content=base64.b64encode(content_bytes).decode("ascii"),
            )
        )


class _StubPullsAPI:
    async def async_list_files(
        self, owner: str, repo: str, pull_number: int, **kwargs: Any
    ) -> _StubResponse:
        return _StubResponse(
            parsed_data=[
                _StubFileMeta(
                    filename=_DEEP_FILE_PATH,
                    status="modified",
                    additions=1,
                    deletions=1,
                    patch=_DEEP_FILE_PATCH,
                )
            ]
        )


class _StubRestAPI:
    def __init__(self) -> None:
        self.repos = _StubReposAPI()
        self.pulls = _StubPullsAPI()


class _StubGitHub:
    def __init__(self) -> None:
        self.rest = _StubRestAPI()


def _stub_github_factory(installation_id: int) -> Any:
    # Pin the seed flows through build_graph's closure to the factory call;
    # a wiring regression that dropped installation_id would silently
    # produce a client for the wrong installation without this assert.
    assert installation_id == _SEED_INSTALLATION_ID, f"unexpected installation_id {installation_id}"
    return _StubGitHub()


# The #065 live-auth gate does ONE identity SELECT per intake invocation
# (`_load_review_gate_state`); this stub answers it with the seed identity + 'running' and
# still ASSERTS on any UPDATE (these tests drive intake's happy path — a status write is a
# regression). Shared by test_analyze_parallel.py, which imports `_stub_db_factory`.
class _GateReadSession:
    async def __aenter__(self) -> _GateReadSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def begin(self) -> _GateReadSession:
        return self

    async def execute(self, stmt: Any) -> Any:
        if isinstance(stmt, Update):
            raise AssertionError("intake happy-path should not write reviews.status here")
        # select(...) — the #065 identity gate load; `.first()` yields the seed identity.
        return _GateSelectResult((_SEED_INSTALLATION_ID, _SEED_REPO_ID, "running"))


@dataclass
class _GateSelectResult:
    _row: tuple[int, int, str] | None

    def first(self) -> tuple[int, int, str] | None:
        return self._row


def _stub_db_factory() -> _GateReadSession:
    return _GateReadSession()


async def _stub_authorizer(installation_id: int, repo_id: int) -> LiveAuthResult:
    """#065 live-auth stub (shared with test_analyze_parallel.py): seed identity is a live
    authorized install → intake proceeds through the gate."""
    assert installation_id == _SEED_INSTALLATION_ID, f"unexpected installation_id {installation_id}"
    return LiveAuthResult(LiveAuthOutcome.AUTHORIZED, "graph-wiring test authorized")


# ---------------------------------------------------------------------------
# Recording sinks
# ---------------------------------------------------------------------------


class _RecordingFileExaminationSink:
    def __init__(self) -> None:
        self.events: list[FileExaminationEvent] = []

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        self.events.append(event)


def _lift_finding_event(
    finding: ReviewFinding, *, is_eval: bool, phase_key: str | None = None
) -> FindingEvent:
    """Lift an admitted ``ReviewFinding`` to its metadata-only ``FindingEvent``,
    mirroring ``AuditPersister._lift_finding_event`` so the recorder captures the
    same event the production sink would emit (keeps ``is_eval`` /
    ``file_path`` assertions on recorded findings valid under the new sink
    signature)."""
    from outrider.audit.events import FindingEvent

    return FindingEvent(
        review_id=finding.review_id,
        is_eval=is_eval,
        finding_id=finding.finding_id,
        finding_type=finding.finding_type,
        severity=finding.severity,
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        dimension=finding.dimension,
        finding_content_hash=finding.content_hash,
        evidence_tier=finding.evidence_tier,
        query_match_id=finding.query_match_id,
        trace_path=finding.trace_path,
        policy_version=finding.policy_version,
        proposal_hash=finding.proposal_hash,
        phase_key=phase_key,
    )


class _RecordingAnalyzeEventSink:
    def __init__(self) -> None:
        self.findings: list[FindingEvent] = []
        self.proposal_rejections: list[FindingProposalRejectedEvent] = []
        self.response_rejections: list[AnalyzeResponseRejectedEvent] = []
        self.completed: list[AnalyzeCompletedEvent] = []
        self.scope_exclusions: list[ScopeExclusionEvent] = []
        self.cache_lookups: list[CacheLookupEvent] = []
        self.cache_serves: list[object] = []
        self.observed_skip_shadows: list[object] = []

    async def emit_finding(
        self, finding: ReviewFinding, *, is_eval: bool, phase_key: str | None = None
    ) -> None:
        self.findings.append(_lift_finding_event(finding, is_eval=is_eval, phase_key=phase_key))

    async def emit_finding_proposal_rejected(self, event: FindingProposalRejectedEvent) -> None:
        self.proposal_rejections.append(event)

    async def emit_analyze_response_rejected(self, event: AnalyzeResponseRejectedEvent) -> None:
        self.response_rejections.append(event)

    async def emit_analyze_completed(self, event: AnalyzeCompletedEvent) -> None:
        self.completed.append(event)

    async def emit_scope_exclusion(self, event: ScopeExclusionEvent) -> None:
        self.scope_exclusions.append(event)

    async def emit_cache_lookup(self, event: CacheLookupEvent) -> None:
        self.cache_lookups.append(event)

    async def emit_cache_serve(self, event: object) -> None:
        self.cache_serves.append(event)

    async def emit_observed_skip_shadow(self, event: object) -> None:
        self.observed_skip_shadows.append(event)


class _StubImportPathResolver:
    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return []

    def resolve_specifier_candidate_paths(
        self, specifier: str, importing_file_path: str, import_root: Path
    ) -> list[Path]:
        return []


# ---------------------------------------------------------------------------
# Canned LLM payload builders
# ---------------------------------------------------------------------------


def _triage_response(*, tier: str) -> str:
    return json.dumps(
        {
            "file_tiers": {_DEEP_FILE_PATH: tier},
            "overall_risk": "medium",
            "relevant_dimensions": ["security"],
            "reasoning": f"test mock: {tier} tier.",
        }
    )


def _analyze_response_one_finding() -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "finding_type": "sql_injection",
                    "evidence_tier": "judged",
                    "query_match_id": None,
                    "trace_path": None,
                    "title": "Test finding",
                    "description": "Wired analyze-graph finding.",
                    "evidence": "def my_function():\n    return 42",
                    "line_start": 1,
                    "line_end": 2,
                    "trace_candidates": [],
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# Seed builders
# ---------------------------------------------------------------------------


def _build_seed_state(*, is_eval: bool = True) -> ReviewState:
    return ReviewState(
        review_id=uuid4(),
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=_SEED_INSTALLATION_ID,
            owner=_SEED_OWNER,
            repo=_SEED_REPO,
            pr_number=_SEED_PULL_NUMBER,
            base_sha="a" * 40,
            head_sha="b" * 40,
            pr_title="Test PR",
            pr_body=None,
            author="someone",
            total_additions=1,
            total_deletions=1,
            changed_files=(
                ChangedFile(
                    path=_DEEP_FILE_PATH,
                    status="modified",
                    additions=1,
                    deletions=1,
                    patch=_DEEP_FILE_PATCH,
                    content_base=_DEEP_FILE_BASE.decode("utf-8"),
                    content_head=_DEEP_FILE_HEAD.decode("utf-8"),
                    previous_path=None,
                    language="python",
                ),
            ),
        ),
        is_eval=is_eval,
    )


class _StubPublishEventSink:
    """Recording `PublishEventSink` for analyze-graph wiring tests.

    These tests focus on intake→triage→analyze wiring assertions; the
    publish node is wired into the graph and runs as the terminal node,
    but every fixture here lands on a no-finding or SKIP-tier path so
    publish exits via the empty-eligible short-circuit without invoking
    the publisher. The stub admits the structural Protocol check at
    build_graph time and captures any emits the node would make if a
    future fixture starts producing admitted findings — keeping the
    recording lists in place so test failures point at the actual
    emitted events instead of disappearing into a discarder.
    """

    def __init__(self) -> None:
        self.routing_events: list[Any] = []
        self.eligibility_events: list[Any] = []
        self.attempt_events: list[Any] = []
        self.result_events: list[Any] = []

    async def emit_publish_routing(self, event: Any) -> None:
        self.routing_events.append(event)

    async def emit_publish_eligibility(self, event: Any) -> None:
        self.eligibility_events.append(event)

    async def emit_publish_attempt(self, event: Any) -> None:
        self.attempt_events.append(event)

    async def emit_publish_result(self, event: Any) -> None:
        self.result_events.append(event)

    async def query_prior_publish_event(self, *, review_id: Any) -> Any:  # noqa: ARG002
        return None

    @asynccontextmanager
    async def acquire_publish_lock(
        self,
        *,
        review_id: Any,  # noqa: ARG002
    ) -> AsyncIterator[None]:
        yield


class _StubTraceEventSink:
    """No-op `TraceEventSink` for analyze-graph wiring tests.

    Trace is now wired into the graph (post the trace-node arc); these
    analyze-wiring tests don't drive trace activation (no
    trace_candidates in fixtures), so the stub just admits the
    structural Protocol check at build_graph time. Returns the incoming
    event verbatim if a future fixture changes routing to invoke trace."""

    async def emit_trace_decision(self, event: Any) -> Any:
        return event

    async def get_trace_decisions(self, *, review_id: Any) -> tuple[Any, ...]:
        return ()


class _StubGitHubPublisher:
    """No-op `GitHubPublisher`. Same rationale as `_StubPublishEventSink`."""

    async def create_review(self, **kwargs: Any) -> Any:  # noqa: ARG002
        msg = "test stub — create_review unreachable in analyze-wiring tests"
        raise NotImplementedError(msg)

    async def find_existing_review_on_head_sha(self, **kwargs: Any) -> Any:  # noqa: ARG002
        msg = "test stub — find_existing_review unreachable in analyze-wiring tests"
        raise NotImplementedError(msg)


class _StubHITLEventSink:
    """No-op `HITLEventSink` for graph wiring tests.

    HITL is wired into the graph post-trace; these tests don't drive
    HITL activation (no CRITICAL/HIGH findings in fixtures), so the stub
    admits the structural Protocol check at build_graph time. Returns
    the incoming event verbatim if a future fixture changes routing to
    invoke HITL emission.
    """

    async def emit_hitl_request(self, event: Any) -> Any:
        return event

    async def emit_hitl_decision(self, event: Any) -> Any:
        return event


class _StubReviewStatusSink:
    """No-op `ReviewStatusSink`. Same rationale as `_StubHITLEventSink`."""

    async def mark_awaiting_approval(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def mark_running(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def mark_awaiting_approval_expired(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def mark_completed(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class _StubSynthesizeEventSink:
    """No-op `SynthesizeEventSink` for graph wiring tests.

    Some scenarios in this file run the full graph through synthesize
    (e.g., `test_is_eval_propagates_through_full_graph` asserts the
    `synthesize` phase event fired); others stop earlier. This stub
    just provides a structurally-valid sink without asserting on
    synthesize-side effects — assertions about whether synthesize
    actually ran live on the recording phase-event sink, not here."""

    async def emit_synthesize_completed(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def query_review_llm_aggregates(  # noqa: ARG002
        self, *, review_id: Any, is_eval: bool
    ) -> ReviewLLMAggregates:
        return ReviewLLMAggregates(
            llm_calls_made=0, total_input_tokens=0, total_output_tokens=0, total_cost_usd=0.0
        )


class _StubAnomalySink:
    """No-op `AnomalySink` (graph-caller variant per anomaly/sinks.py
    docstring). `is_eval` is mandatory at the Protocol level."""

    async def emit_anomaly(
        self,
        *,
        review_id: Any,  # noqa: ARG002
        rule_name: Any,  # noqa: ARG002
        severity: Any,  # noqa: ARG002
        details: dict[str, Any],  # noqa: ARG002
        is_eval: bool,  # noqa: ARG002
    ) -> None:
        return None


def _build_kwargs(
    *,
    provider: _RoutingMockLLMProvider,
    phase_event_sink: _RecordingPhaseEventSinkLike,
    file_examination_sink: _RecordingFileExaminationSink,
    analyze_event_sink: _RecordingAnalyzeEventSink,
    total_review_budget_tokens: int | None = None,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    from langgraph.checkpoint.memory import InMemorySaver

    from outrider.agent.nodes.hitl_config import HITLConfig
    from outrider.agent.nodes.patch_config import PatchConfig

    kwargs: dict[str, Any] = {
        "db_factory": _stub_db_factory,
        "github_factory": _stub_github_factory,
        "installation_authorizer": _stub_authorizer,
        "provider": provider,
        "model_config": model_config or ModelConfig(),
        "phase_event_sink": phase_event_sink,
        "file_examination_sink": file_examination_sink,
        "analyze_event_sink": analyze_event_sink,
        # Publish-node deps added 2026-05-22 per the publish-node arc;
        # these tests don't reach publish but build_graph's structural
        # Protocol gate requires both.
        "publish_event_sink": _StubPublishEventSink(),
        "trace_sink": _StubTraceEventSink(),
        # HITL deps added 2026-05-26 per the HITL-node arc; these tests
        # don't drive HITL activation but build_graph's structural
        # Protocol gate requires all three.
        "hitl_event_sink": _StubHITLEventSink(),
        "review_status_sink": _StubReviewStatusSink(),
        # Synthesize deps added 2026-05-28 per the synthesize-node arc;
        # these tests don't reach synthesize but `build_graph` requires
        # both at construction time.
        "synthesize_event_sink": _StubSynthesizeEventSink(),
        "anomaly_sink": _StubAnomalySink(),
        "hitl_config": HITLConfig(),
        "patch_config": PatchConfig(patches_enabled=False),
        # Checkpointer is required for any compiled graph that uses
        # `interrupt(...)` per langgraph-1.1.6/narrative/persistence.md.
        # InMemorySaver is the canonical test-only checkpointer.
        "checkpointer": InMemorySaver(),
        "publisher": _StubGitHubPublisher(),
        "import_path_resolver": _StubImportPathResolver(),
    }
    if total_review_budget_tokens is not None:
        kwargs["total_review_budget_tokens"] = total_review_budget_tokens
    return kwargs


# ---------------------------------------------------------------------------
# Gate 1 — triage → analyze → END wires correctly
# ---------------------------------------------------------------------------


def test_compiled_graph_has_analyze_node_and_correct_edges(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Build the compiled graph; assert node membership includes intake,
    triage, analyze, trace (wired per the 2026-05-24 trace spec), and
    publish, and that the only statically-renderable edge starts at
    START and lands at intake.

    LangGraph's `get_graph()` renders only static edges visible from
    START's static reachability. Since intake routes via
    `Command(goto=...)` (dynamic), `get_graph().edges` shows just
    `(START, "intake")` plus a fall-through `(intake, END)` placeholder
    — the `triage → analyze` and `analyze → END` static edges I added
    are technically present in the builder but pruned from the rendered
    graph as "unreachable from START via static edges." The functional
    `triage → analyze → END` wiring is proven by gates 2-4 below
    exercising the compiled graph end-to-end.
    """
    provider = _RoutingMockLLMProvider(
        triage_response=_triage_response(tier="deep"),
        analyze_response=_analyze_response_one_finding(),
    )
    graph = build_graph(
        **_build_kwargs(
            provider=provider,
            phase_event_sink=recording_phase_event_sink,
            file_examination_sink=_RecordingFileExaminationSink(),
            analyze_event_sink=_RecordingAnalyzeEventSink(),
        )
    )
    nodes = set(graph.get_graph().nodes)
    assert "intake" in nodes
    assert "triage" in nodes
    assert "analyze" in nodes
    assert "trace" in nodes  # trace spec landed 2026-05-24; node IS wired
    assert "publish" in nodes

    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    # START → intake is the only statically-visible START-reachable edge;
    # the rest of the topology surfaces only through `ainvoke` (gates 2-4).
    assert (START, "intake") in edges


# ---------------------------------------------------------------------------
# Gate 2 — one clean eligible file flows through analyze
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_eligible_file_flows_through_analyze(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Tier=DEEP file → analyze runs → AnalysisRound populated;
    FindingEvent emitted from analyze; AnalyzeCompletedEvent shows
    n_findings_emitted=1."""
    provider = _RoutingMockLLMProvider(
        triage_response=_triage_response(tier="deep"),
        analyze_response=_analyze_response_one_finding(),
    )
    fe_sink = _RecordingFileExaminationSink()
    ae_sink = _RecordingAnalyzeEventSink()
    state = _build_seed_state()
    graph = build_graph(
        **_build_kwargs(
            provider=provider,
            phase_event_sink=recording_phase_event_sink,
            file_examination_sink=fe_sink,
            analyze_event_sink=ae_sink,
        )
    )

    result = await graph.ainvoke(
        state, config={"configurable": {"thread_id": str(state.review_id)}}
    )

    # AnalysisRound landed in state.
    rounds = result["analysis_rounds"]
    assert len(rounds) == 1
    assert rounds[0].files_examined == (_DEEP_FILE_PATH,)
    assert rounds[0].files_skipped == ()
    assert len(rounds[0].findings) == 1

    # FindingEvent emitted from analyze.
    assert len(ae_sink.findings) == 1
    assert ae_sink.findings[0].file_path == _DEEP_FILE_PATH
    # Aggregate-keyed through the REAL graph (admission is aggregate work);
    # the recorder mirrors the persister's lift, so a dropped stamp fails here.
    assert ae_sink.findings[0].phase_key == "aggregate#0"
    assert ae_sink.completed[0].n_findings_emitted == 1

    # FileExaminationEvent shows clean parse_status for analyze.
    analyze_fe = [e for e in fe_sink.events if e.node_id == "analyze"]
    assert len(analyze_fe) == 1
    assert analyze_fe[0].parse_status == "clean"
    assert analyze_fe[0].skip_reason is None


# ---------------------------------------------------------------------------
# Gate 3 — one triage-excluded file does not enter analyze
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_excluded_file_does_not_enter_analyze(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Tier=SKIM file → analyze's triage gate excludes it BEFORE any
    per-file work. The file does NOT appear in `files_examined` or
    `files_skipped`; no analyze-emitted FileExaminationEvent fires for
    that path. The analyze provider is NEVER called."""
    provider = _RoutingMockLLMProvider(
        triage_response=_triage_response(tier="skim"),
        analyze_response=_analyze_response_one_finding(),
    )
    fe_sink = _RecordingFileExaminationSink()
    ae_sink = _RecordingAnalyzeEventSink()
    state = _build_seed_state()
    graph = build_graph(
        **_build_kwargs(
            provider=provider,
            phase_event_sink=recording_phase_event_sink,
            file_examination_sink=fe_sink,
            analyze_event_sink=ae_sink,
        )
    )

    result = await graph.ainvoke(
        state, config={"configurable": {"thread_id": str(state.review_id)}}
    )

    rounds = result["analysis_rounds"]
    assert len(rounds) == 1
    assert rounds[0].files_examined == ()
    assert rounds[0].files_skipped == ()
    assert len(rounds[0].findings) == 0

    # NO FileExaminationEvent emitted from analyze for this file.
    analyze_fe = [e for e in fe_sink.events if e.node_id == "analyze"]
    assert analyze_fe == []

    # Provider was called once (triage) — NOT twice.
    analyze_calls = [c for c in provider.calls if c.node_id == "analyze"]
    assert analyze_calls == []

    # AnalyzeCompletedEvent fires with zero counters.
    assert len(ae_sink.completed) == 1
    assert ae_sink.completed[0].n_llm_calls == 0
    assert ae_sink.completed[0].n_files_analyzed == 0


# ---------------------------------------------------------------------------
# Gate 4 — budget-skip file is audited as skipped, NOT clean
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_skip_file_is_audited_as_skipped_not_clean(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Tier=DEEP file + tiny `total_review_budget_tokens` → cost gate
    fires → FileExaminationEvent(skipped, COST_BUDGET_EXHAUSTED) emitted;
    file appears in `AnalysisRound.files_skipped` (not `files_examined`).
    The analyze provider is NOT called for that file."""
    provider = _RoutingMockLLMProvider(
        triage_response=_triage_response(tier="deep"),
        analyze_response=_analyze_response_one_finding(),
    )
    fe_sink = _RecordingFileExaminationSink()
    ae_sink = _RecordingAnalyzeEventSink()
    state = _build_seed_state()
    graph = build_graph(
        **_build_kwargs(
            provider=provider,
            phase_event_sink=recording_phase_event_sink,
            file_examination_sink=fe_sink,
            analyze_event_sink=ae_sink,
            total_review_budget_tokens=100,  # per-file cap = 25 tokens
        )
    )

    result = await graph.ainvoke(
        state, config={"configurable": {"thread_id": str(state.review_id)}}
    )

    rounds = result["analysis_rounds"]
    assert len(rounds) == 1
    assert rounds[0].files_examined == ()
    assert rounds[0].files_skipped == (_DEEP_FILE_PATH,)

    # FileExaminationEvent from analyze: parse_status=skipped, skip_reason=COST_BUDGET_EXHAUSTED.
    analyze_fe = [e for e in fe_sink.events if e.node_id == "analyze"]
    assert len(analyze_fe) == 1
    assert analyze_fe[0].parse_status == "skipped"
    assert analyze_fe[0].skip_reason == SkipReason.COST_BUDGET_EXHAUSTED
    # The audit row's parse_status is explicitly NOT "clean" — the user's
    # gate language was "does not look like 'clean'".
    assert analyze_fe[0].parse_status != "clean"

    # Provider was NOT called for analyze.
    analyze_calls = [c for c in provider.calls if c.node_id == "analyze"]
    assert analyze_calls == []

    # AnalyzeCompletedEvent shows zero LLM calls + one skipped file.
    assert len(ae_sink.completed) == 1
    assert ae_sink.completed[0].n_llm_calls == 0
    assert ae_sink.completed[0].n_files_analyzed == 0
    assert ae_sink.completed[0].n_files_skipped == 1


# ---------------------------------------------------------------------------
# Gate 5 — is_eval propagation through the full graph (production AND eval)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("eval_flag", [True, False])
async def test_is_eval_propagates_through_full_graph(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
    eval_flag: bool,
) -> None:
    """`is_eval` from the seed `ReviewState` must reach every emitted
    event — phase events (one start+end pair per node that ran),
    FileExaminationEvents (intake's per-file fetch + analyze's per-file
    outcome), and the seven analyze-specific event types (FindingEvent +
    FindingProposalRejectedEvent + AnalyzeResponseRejectedEvent +
    AnalyzeCompletedEvent + ScopeExclusionEvent — emitted by the
    trivial-scope filter's shadow mode on every analyzed pass-0 clean
    file — + CacheLookupEvent + CacheServeEvent, which this test does NOT exercise: no
    cache store is wired here, and the either-flag eval veto means an
    eval review can never emit one anyway; its propagation is pinned by
    the unit wiring tests).

    Parametrized over `is_eval=True` AND `is_eval=False` so the
    production-side propagation (`is_eval=False`) doesn't silently break
    while eval-tagged tests stay green. Eval-isolation contract is
    bidirectional: an eval-tagged row in production audit OR a
    production-tagged row in eval audit both pollute the stream.
    """
    provider = _RoutingMockLLMProvider(
        triage_response=_triage_response(tier="deep"),
        analyze_response=_analyze_response_one_finding(),
    )
    fe_sink = _RecordingFileExaminationSink()
    ae_sink = _RecordingAnalyzeEventSink()
    state = _build_seed_state(is_eval=eval_flag)
    graph = build_graph(
        **_build_kwargs(
            provider=provider,
            phase_event_sink=recording_phase_event_sink,
            file_examination_sink=fe_sink,
            analyze_event_sink=ae_sink,
        )
    )

    await graph.ainvoke(state, config={"configurable": {"thread_id": str(state.review_id)}})

    # Vacuous-pass guards: `all(...)` over an empty iterable is True, so
    # `assert all(e.is_eval is eval_flag for e in [])` passes regardless
    # of the propagation property. Pin non-empty lengths first so a
    # future fixture/schema break that empties an event list surfaces
    # the breakage rather than silently passing the propagation check.

    # Phase events: property-based — every node that ran fired its start+end
    # pair (count reflects graph wiring, not the eval-propagation contract).
    # Pin the FULL-graph node coverage so a future regression that
    # accidentally skips publish (or any node) surfaces here rather
    # than only via the absence of a publish-side audit event.
    assert len(recording_phase_event_sink.events) > 0, "no phase events emitted"
    started_nodes = {e.node_id for e in recording_phase_event_sink.events if e.marker == "start"}
    # HITL interrupts on CRITICAL/HIGH findings (sql_injection ->
    # CRITICAL per SEVERITY_POLICY) — the graph emits hitl's start
    # marker then suspends via `interrupt(...)`. publish does NOT run
    # without a resume; that's the V1 HITL gate guarantee. is_eval
    # propagation is checked across the nodes that DO run.
    assert {"intake", "triage", "analyze", "synthesize", "hitl"} <= started_nodes, (
        f"expected full-graph node coverage through hitl (7-node topology: "
        f"intake → triage → analyze ⇄ trace → synthesize → hitl → publish), "
        f"got starts from {sorted(started_nodes)}"
    )
    # HITL gate guarantee: publish MUST NOT run without an explicit
    # `Command(resume=...)`. The subset check above admits publish if
    # the gate regressed (subset relationship still holds); this
    # negative assertion locks the docstring's "publish does NOT run
    # without a resume" guarantee as a tested invariant.
    assert "publish" not in started_nodes, (
        f"publish ran without a HITL resume — gate regression; "
        f"started_nodes={sorted(started_nodes)}"
    )
    assert all(e.is_eval is eval_flag for e in recording_phase_event_sink.events), (
        f"Phase event leaked the wrong is_eval flag (expected {eval_flag})"
    )

    # FileExaminationEvents: at least intake's fetch + analyze's outcome.
    assert len(fe_sink.events) >= 2, (
        f"expected at least 2 FileExaminationEvents (intake fetch + analyze outcome), "
        f"got {len(fe_sink.events)}"
    )
    assert all(e.is_eval is eval_flag for e in fe_sink.events), (
        f"FileExaminationEvent leaked the wrong is_eval flag (expected {eval_flag})"
    )

    # Exactly one admitted finding + one AnalyzeCompletedEvent under this fixture.
    assert len(ae_sink.findings) == 1, (
        f"expected 1 FindingEvent (one admitted JUDGED finding), got {len(ae_sink.findings)}"
    )
    assert len(ae_sink.completed) == 1, (
        f"expected 1 AnalyzeCompletedEvent per pass, got {len(ae_sink.completed)}"
    )
    assert all(e.is_eval is eval_flag for e in ae_sink.findings), (
        f"FindingEvent leaked the wrong is_eval flag (expected {eval_flag})"
    )
    assert all(e.is_eval is eval_flag for e in ae_sink.completed), (
        f"AnalyzeCompletedEvent leaked the wrong is_eval flag (expected {eval_flag})"
    )
    # ScopeExclusionEvent: shadow mode classifies every analyzed pass-0
    # clean file, so this fixture MUST produce at least one — pin
    # non-empty first (vacuous-pass guard), then the propagation property.
    assert len(ae_sink.scope_exclusions) >= 1, (
        f"expected >=1 ScopeExclusionEvent (shadow-mode classification), "
        f"got {len(ae_sink.scope_exclusions)}"
    )
    assert all(e.is_eval is eval_flag for e in ae_sink.scope_exclusions), (
        f"ScopeExclusionEvent leaked the wrong is_eval flag (expected {eval_flag})"
    )
    # proposal_rejections / response_rejections are empty for this scenario
    # (clean DEEP file admits its sole JUDGED finding); the vacuous-pass on
    # `all(...) over empty` doesn't bite for them because there's no
    # contract that they MUST emit for this fixture — pinning their
    # length would over-specify.


@pytest.mark.asyncio
async def test_standard_tier_file_routes_to_standard_analyze_model(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """End-to-end build_graph wiring (specs/2026-06-08-analyze-tiered-model-routing.md):
    a STANDARD-tier file's analyze LLM call uses `model_config.standard_analyze_model`
    — distinct here from `analyze_model` — proving `build_graph` injects the STANDARD
    model into the analyze closure, and the `AnalyzeCompletedEvent` records both. The
    analyze call + its event fire before the CRITICAL-finding HITL interrupt."""
    provider = _RoutingMockLLMProvider(
        triage_response=_triage_response(tier="standard"),
        analyze_response=_analyze_response_one_finding(),
    )
    fe_sink = _RecordingFileExaminationSink()
    ae_sink = _RecordingAnalyzeEventSink()
    state = _build_seed_state()
    graph = build_graph(
        **_build_kwargs(
            provider=provider,
            phase_event_sink=recording_phase_event_sink,
            file_examination_sink=fe_sink,
            analyze_event_sink=ae_sink,
            model_config=ModelConfig(
                analyze_model="claude-sonnet-4-6",
                standard_analyze_model="claude-haiku-4-5",
            ),
        )
    )

    await graph.ainvoke(state, config={"configurable": {"thread_id": str(state.review_id)}})

    analyze_calls = [c for c in provider.calls if c.node_id == "analyze"]
    assert len(analyze_calls) == 1
    # The STANDARD-tier file routed to the cost-lever model, NOT analyze_model.
    assert analyze_calls[0].model == "claude-haiku-4-5"
    assert ae_sink.completed[0].analyze_model == "claude-sonnet-4-6"
    assert ae_sink.completed[0].standard_analyze_model == "claude-haiku-4-5"
