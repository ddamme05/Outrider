"""build_graph guard tests per the triage-node + intake-and-webhook + analyze-node specs.

Narrowly scoped to dependency / None guards on `agent/graph.py::build_graph`.
Functional behavior (the compiled graph actually runs) is covered by the
integration tests in `tests/integration/test_review_state_langgraph_merge.py`
and the analyze-wiring tests in `tests/unit/test_agent_graph_analyze_wiring.py`.

Rejection contracts pinned here (eight None gates after analyze-node landed):
  1. provider=None → BuildGraphError
  2. model_config=None → BuildGraphError
  3. phase_event_sink=None → BuildGraphError
  4. file_examination_sink=None → BuildGraphError
  5. analyze_event_sink=None → BuildGraphError
  6. import_path_resolver=None → BuildGraphError
  7. db_factory=None → BuildGraphError
  8. github_factory=None → BuildGraphError
  Plus five Protocol-structural gates and two callable() gates.

PEP 544 caveat: the isinstance gates check MEMBER PRESENCE only — wrong
signature or async-shape falls through to fail at the first call. Tests
deliberately use plain `object()` rather than wrong-signature classes to
match the gate's actual semantics.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from outrider.agent.graph import BuildGraphError, build_graph
from outrider.llm.config import ModelConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.audit.events import (
        AnalyzeCompletedEvent,
        AnalyzeResponseRejectedEvent,
        FileExaminationEvent,
        FindingEvent,
        FindingProposalRejectedEvent,
        PublishAttemptEvent,
        PublishEligibilityEvent,
        PublishEvent,
        PublishRoutingEvent,
        ReviewPhaseEvent,
        TraceDecisionEvent,
    )
    from outrider.github import InstallationGitHubClient
    from outrider.llm.base import LLMRequest, LLMResponse
    from outrider.schemas import GitHubReviewCreated, InlineComment


# ---------------------------------------------------------------------------
# Minimal protocol-satisfying stubs
# ---------------------------------------------------------------------------


class _StubProvider:
    """Satisfies LLMProvider Protocol structurally (has `complete`)."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("test stub")


class _StubPhaseSink:
    """Satisfies PhaseEventSink Protocol structurally (has `emit_phase`)."""

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        return None


class _StubFileExaminationSink:
    """Satisfies FileExaminationSink Protocol structurally."""

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        return None


class _StubAnalyzeEventSink:
    """Satisfies AnalyzeEventSink Protocol structurally (has all 4 emit_* members)."""

    async def emit_finding(self, event: FindingEvent) -> None:
        return None

    async def emit_finding_proposal_rejected(self, event: FindingProposalRejectedEvent) -> None:
        return None

    async def emit_analyze_response_rejected(self, event: AnalyzeResponseRejectedEvent) -> None:
        return None

    async def emit_analyze_completed(self, event: AnalyzeCompletedEvent) -> None:
        return None


class _StubImportPathResolver:
    """Satisfies ImportPathResolver Protocol structurally (has `resolve_candidate_paths`)."""

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return []


class _StubPublishEventSink:
    """Satisfies PublishEventSink Protocol structurally (4 emit_* + 1 query method)."""

    async def emit_publish_routing(self, event: PublishRoutingEvent) -> None:
        return None

    async def emit_publish_eligibility(self, event: PublishEligibilityEvent) -> None:
        return None

    async def emit_publish_attempt(self, event: PublishAttemptEvent) -> None:
        return None

    async def emit_publish_result(self, event: PublishEvent) -> None:
        return None

    async def query_prior_publish_event(self, *, review_id: UUID) -> PublishEvent | None:  # noqa: ARG002
        return None

    @asynccontextmanager
    async def acquire_publish_lock(
        self,
        *,
        review_id: UUID,  # noqa: ARG002
    ) -> AsyncIterator[None]:
        yield


class _StubTraceEventSink:
    """Satisfies TraceEventSink Protocol structurally (1 emit method
    returning the canonical persisted event per M7 b)."""

    async def emit_trace_decision(self, event: TraceDecisionEvent) -> TraceDecisionEvent:
        return event


class _StubHITLEventSink:
    """Satisfies HITLEventSink Protocol structurally (two emit methods,
    audit-first non-None return)."""

    async def emit_hitl_request(self, event: Any) -> Any:
        return event

    async def emit_hitl_decision(self, event: Any) -> Any:
        return event


class _StubSynthesizeEventSink:
    """Satisfies SynthesizeEventSink Protocol structurally."""

    async def emit_synthesize_completed(self, event: Any) -> None:
        return None


class _StubAnomalySink:
    """Satisfies AnomalySink Protocol structurally."""

    async def emit_anomaly(
        self,
        *,
        review_id: Any,
        rule_name: Any,
        severity: Any,
        details: dict[str, Any],
        is_eval: bool,
    ) -> None:
        return None


class _StubReviewStatusSink:
    """Satisfies ReviewStatusSink Protocol structurally (four async
    methods, all no-op stubs)."""

    async def mark_awaiting_approval(
        self, *, review_id: Any, expires_at: Any, hitl_request_payload: dict[str, Any]
    ) -> None:
        return None

    async def mark_running(self, *, review_id: Any, hitl_decision_payload: dict[str, Any]) -> None:
        return None

    async def mark_awaiting_approval_expired(self, *, review_id: Any) -> None:
        return None

    async def mark_completed(self, *, review_id: Any) -> None:
        return None


class _StubGitHubPublisher:
    """Satisfies GitHubPublisher Protocol structurally (has create_review +
    find_existing_review_on_head_sha)."""

    async def create_review(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        review_status: str,
        body_marker: str,
        comments: tuple[InlineComment, ...],
    ) -> GitHubReviewCreated:
        raise NotImplementedError("test stub")

    async def find_existing_review_on_head_sha(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        body_marker: str,
    ) -> int | None:
        raise NotImplementedError("test stub")


def _stub_db_factory() -> async_sessionmaker[AsyncSession]:
    """A callable stub satisfying both the None-check and the
    `callable()` check at construction time. The duck-typed runtime
    use (opening a session) is exercised in integration tests, not
    here."""

    def _factory() -> Any:
        msg = "test stub — db_factory should not be invoked in build-time tests"
        raise NotImplementedError(msg)

    return _factory  # type: ignore[return-value]


def _stub_github_factory(installation_id: int) -> InstallationGitHubClient:
    """A callable that satisfies the type at the call site; never invoked
    in these unit tests."""
    raise NotImplementedError("test stub")


def _valid_args() -> dict[str, Any]:
    """Build a complete, valid set of kwargs. Tests perturb one at a time."""
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

    from outrider.agent.nodes.hitl_config import HITLConfig  # noqa: PLC0415

    return {
        "provider": _StubProvider(),
        "model_config": ModelConfig(),
        "phase_event_sink": _StubPhaseSink(),
        "file_examination_sink": _StubFileExaminationSink(),
        "analyze_event_sink": _StubAnalyzeEventSink(),
        "publish_event_sink": _StubPublishEventSink(),
        "trace_sink": _StubTraceEventSink(),
        "hitl_event_sink": _StubHITLEventSink(),
        "synthesize_event_sink": _StubSynthesizeEventSink(),
        "review_status_sink": _StubReviewStatusSink(),
        "anomaly_sink": _StubAnomalySink(),
        "hitl_config": HITLConfig(),
        "checkpointer": InMemorySaver(),
        "publisher": _StubGitHubPublisher(),
        "import_path_resolver": _StubImportPathResolver(),
        "db_factory": _stub_db_factory(),
        "github_factory": _stub_github_factory,
    }


# ---------------------------------------------------------------------------
# Happy path: all three deps valid → returns compiled graph
# ---------------------------------------------------------------------------


def test_build_graph_happy_path_returns_compiled_graph() -> None:
    """Sanity test: with all three valid Protocol-satisfying deps, the
    factory returns a compiled graph object. Validates the gate doesn't
    fire on correct inputs."""
    graph = build_graph(**_valid_args())
    # CompiledStateGraph has an `ainvoke` method per LangGraph 1.1.6
    assert callable(graph.ainvoke)


# ---------------------------------------------------------------------------
# None rejections (3 sibling tests)
# ---------------------------------------------------------------------------


def test_build_graph_rejects_provider_none() -> None:
    args = _valid_args()
    args["provider"] = None
    with pytest.raises(BuildGraphError, match="provider must not be None"):
        build_graph(**args)


def test_build_graph_rejects_model_config_none() -> None:
    args = _valid_args()
    args["model_config"] = None
    with pytest.raises(BuildGraphError, match="model_config must not be None"):
        build_graph(**args)


def test_build_graph_rejects_phase_event_sink_none() -> None:
    args = _valid_args()
    args["phase_event_sink"] = None
    with pytest.raises(BuildGraphError, match="phase_event_sink must not be None"):
        build_graph(**args)


def test_build_graph_rejects_file_examination_sink_none() -> None:
    args = _valid_args()
    args["file_examination_sink"] = None
    with pytest.raises(BuildGraphError, match="file_examination_sink must not be None"):
        build_graph(**args)


def test_build_graph_rejects_analyze_event_sink_none() -> None:
    args = _valid_args()
    args["analyze_event_sink"] = None
    with pytest.raises(BuildGraphError, match="analyze_event_sink must not be None"):
        build_graph(**args)


def test_build_graph_rejects_publish_event_sink_none() -> None:
    args = _valid_args()
    args["publish_event_sink"] = None
    with pytest.raises(BuildGraphError, match="publish_event_sink must not be None"):
        build_graph(**args)


def test_build_graph_rejects_trace_sink_none() -> None:
    """`trace_sink` parity with the other required sinks — None must
    raise BuildGraphError, not silently propagate a NoneType down to
    the first `await trace_sink.emit_trace_decision(...)` call."""
    args = _valid_args()
    args["trace_sink"] = None
    with pytest.raises(BuildGraphError, match="trace_sink must not be None"):
        build_graph(**args)


def test_build_graph_rejects_publisher_none() -> None:
    args = _valid_args()
    args["publisher"] = None
    with pytest.raises(BuildGraphError, match="publisher must not be None"):
        build_graph(**args)


def test_build_graph_rejects_import_path_resolver_none() -> None:
    args = _valid_args()
    args["import_path_resolver"] = None
    with pytest.raises(BuildGraphError, match="import_path_resolver must not be None"):
        build_graph(**args)


def test_build_graph_rejects_synthesize_event_sink_none() -> None:
    """`synthesize_event_sink` parity with the other required sinks —
    None must raise BuildGraphError, not silently propagate a NoneType
    down to the first `await synthesize_event_sink.emit_synthesize_completed(...)`
    call. Per CodeRabbit catch: new constructor parameters need the
    same negative-path coverage as siblings."""
    args = _valid_args()
    args["synthesize_event_sink"] = None
    with pytest.raises(BuildGraphError, match="synthesize_event_sink must not be None"):
        build_graph(**args)


def test_build_graph_rejects_anomaly_sink_none() -> None:
    """`anomaly_sink` parity with the other required sinks — None must
    raise BuildGraphError. Per CodeRabbit catch + the two-caller-class
    contract on `AnomalySink`: synthesize is the first in-graph caller
    and structural rejection at build_graph closes the silent-None
    propagation path before the divergence-detection emit hits the
    sink."""
    args = _valid_args()
    args["anomaly_sink"] = None
    with pytest.raises(BuildGraphError, match="anomaly_sink must not be None"):
        build_graph(**args)


def test_build_graph_rejects_db_factory_none() -> None:
    args = _valid_args()
    args["db_factory"] = None
    with pytest.raises(BuildGraphError, match="db_factory must not be None"):
        build_graph(**args)


def test_build_graph_rejects_github_factory_none() -> None:
    args = _valid_args()
    args["github_factory"] = None
    with pytest.raises(BuildGraphError, match="github_factory must not be None"):
        build_graph(**args)


def test_build_graph_rejects_db_factory_non_callable() -> None:
    """A non-None but non-callable `db_factory` (e.g., a bare object())
    passes the None check but would fail at first intake invocation with
    a confusing TypeError. The callable check at build time surfaces
    misconfiguration BEFORE the first review is dispatched."""
    args = _valid_args()
    args["db_factory"] = object()  # non-callable
    with pytest.raises(BuildGraphError, match="db_factory must be callable"):
        build_graph(**args)


def test_build_graph_rejects_github_factory_non_callable() -> None:
    """Same shape as the db_factory check: non-callable github_factory
    fails at build time, not at first intake invocation."""
    args = _valid_args()
    args["github_factory"] = object()  # non-callable
    with pytest.raises(BuildGraphError, match="github_factory must be callable"):
        build_graph(**args)


# ---------------------------------------------------------------------------
# Structural Protocol rejection
# ---------------------------------------------------------------------------


def test_build_graph_rejects_provider_missing_complete_member() -> None:
    """`isinstance(provider, LLMProvider)` fails on objects without
    `complete`. Pin the gate; PEP 544 member-presence-only semantics —
    wrong-signature `complete` would NOT be caught here."""
    args = _valid_args()
    args["provider"] = object()  # no `complete` attribute
    with pytest.raises(BuildGraphError, match="provider does not satisfy LLMProvider"):
        build_graph(**args)


def test_build_graph_rejects_phase_event_sink_missing_emit_phase_member() -> None:
    """`isinstance(sink, PhaseEventSink)` fails on objects without
    `emit_phase`. Pins the no-silent-phase-drop guarantee at the
    construction-time gate."""
    args = _valid_args()
    args["phase_event_sink"] = object()  # no `emit_phase` attribute
    with pytest.raises(BuildGraphError, match="phase_event_sink does not satisfy PhaseEventSink"):
        build_graph(**args)


def test_build_graph_rejects_file_examination_sink_missing_member() -> None:
    """`isinstance(sink, FileExaminationSink)` fails on objects without
    `emit_file_examination`. Same shape as the phase-event sink gate."""
    args = _valid_args()
    args["file_examination_sink"] = object()
    with pytest.raises(
        BuildGraphError,
        match="file_examination_sink does not satisfy FileExaminationSink",
    ):
        build_graph(**args)


def test_build_graph_rejects_analyze_event_sink_missing_member() -> None:
    """`isinstance(sink, AnalyzeEventSink)` fails on objects lacking any of
    the four `emit_*` methods. PEP 544 member-presence semantics."""
    args = _valid_args()
    args["analyze_event_sink"] = object()
    with pytest.raises(
        BuildGraphError,
        match="analyze_event_sink does not satisfy AnalyzeEventSink",
    ):
        build_graph(**args)


def test_build_graph_rejects_import_path_resolver_missing_member() -> None:
    """`isinstance(resolver, ImportPathResolver)` fails on objects without
    `resolve_candidate_paths`. PEP 544 member-presence semantics."""
    args = _valid_args()
    args["import_path_resolver"] = object()
    with pytest.raises(
        BuildGraphError,
        match="import_path_resolver does not satisfy ImportPathResolver",
    ):
        build_graph(**args)


def test_build_graph_rejects_publish_event_sink_missing_member() -> None:
    """`isinstance(sink, PublishEventSink)` fails on objects lacking any
    of the five expected methods (4 emit_* + 1 query). PEP 544
    member-presence semantics."""
    args = _valid_args()
    args["publish_event_sink"] = object()
    with pytest.raises(
        BuildGraphError,
        match="publish_event_sink does not satisfy PublishEventSink",
    ):
        build_graph(**args)


def test_build_graph_rejects_trace_sink_missing_member() -> None:
    """`isinstance(sink, TraceEventSink)` fails on objects lacking
    `emit_trace_decision`. PEP 544 member-presence semantics; parity
    with `publish_event_sink_missing_member` so trace_sink's structural
    guard isn't silently undone by a future refactor."""
    args = _valid_args()
    args["trace_sink"] = object()
    with pytest.raises(
        BuildGraphError,
        match="trace_sink does not satisfy TraceEventSink",
    ):
        build_graph(**args)


def test_build_graph_rejects_publisher_missing_member() -> None:
    """`isinstance(pub, GitHubPublisher)` fails on objects lacking
    `create_review` / `find_existing_review_on_head_sha`. PEP 544
    member-presence semantics."""
    args = _valid_args()
    args["publisher"] = object()
    with pytest.raises(
        BuildGraphError,
        match="publisher does not satisfy GitHubPublisher",
    ):
        build_graph(**args)


def test_build_graph_rejects_non_int_total_review_budget_tokens() -> None:
    """`total_review_budget_tokens` is a public int; non-int input fails
    fast at construction rather than at first multiplication inside
    `_compute_per_file_cap`."""
    args = _valid_args()
    args["total_review_budget_tokens"] = "200000"  # str, not int
    with pytest.raises(BuildGraphError, match="total_review_budget_tokens must be int"):
        build_graph(**args)


def test_build_graph_rejects_bool_total_review_budget_tokens() -> None:
    """`bool` is technically an `int` subclass in Python; reject it
    explicitly so a typo'd `total_review_budget_tokens=True` doesn't
    silently become `1`."""
    args = _valid_args()
    args["total_review_budget_tokens"] = True
    with pytest.raises(BuildGraphError, match="total_review_budget_tokens must be int"):
        build_graph(**args)


# ---------------------------------------------------------------------------
# Other shapes: a class that has BOTH missing methods is rejected by the
# FIRST gate (provider). Documents the gate order.
# ---------------------------------------------------------------------------


def test_build_graph_provider_gate_fires_before_sink_gate() -> None:
    """If multiple deps fail simultaneously, the provider gate fires
    first. Useful so future hardening (e.g., aggregating errors) is an
    intentional decision, not an accidental change of behavior."""
    args = _valid_args()
    args["provider"] = object()
    args["phase_event_sink"] = object()
    with pytest.raises(BuildGraphError, match="provider"):
        build_graph(**args)


# ---------------------------------------------------------------------------
# build_graph is keyword-only — positional invocation fails
# ---------------------------------------------------------------------------


def test_build_graph_signature_is_keyword_only() -> None:
    """Defensive: build_graph uses keyword-only args. A positional call
    would be a TypeError — pinning prevents accidental signature
    relaxation."""
    with pytest.raises(TypeError):
        # Positional invocation should fail
        build_graph(
            _StubProvider(),  # type: ignore[misc]
            ModelConfig(),
            _StubPhaseSink(),
        )
