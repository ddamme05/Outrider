"""FUP-019 closure: LangGraph reducer-merge semantics on ReviewState.

Three integration tests pin the LangGraph 1.1.6 contract our spec
depends on:

1. First-node input validation fires (run-time validation only happens on
   inputs to the FIRST node per `narrative/use-graph-api.md`).
2. Partial-state merge populates `triage_result` (default reducer
   overwrites; `pr_context` survives unchanged).
3. Result-dict rehydrates as ReviewState (LangGraph returns a dict, not
   the Pydantic instance; explicit re-validation is the canonical path).

Async-node graphs require `.ainvoke()` per LangGraph docs. The compiled
graph uses MockLLMProvider + RecordingPhaseEventSink + NoOpPersister via
the root-conftest fixtures so the test doesn't depend on Anthropic
credentials or a live API call.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.agent.graph import build_graph
from outrider.llm.config import ModelConfig
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState
from outrider.schemas.triage_result import TriageResult

if TYPE_CHECKING:
    from pathlib import Path

    from outrider.audit.events import (
        AnalyzeCompletedEvent,
        AnalyzeResponseRejectedEvent,
        FileExaminationEvent,
        FindingEvent,
        FindingProposalRejectedEvent,
        ReviewPhaseEvent,
    )
    from outrider.llm.base import LLMRequest, LLMResponse


# ---------------------------------------------------------------------------
# Type stubs for cross-conftest-imported fixtures (mirrors test_triage_node.py)
# ---------------------------------------------------------------------------


class _RecordingPhaseEventSinkLike(Protocol):
    events: list[ReviewPhaseEvent]

    async def emit_phase(self, event: ReviewPhaseEvent) -> None: ...


# ---------------------------------------------------------------------------
# Minimal mock provider satisfying LLMProvider Protocol
# ---------------------------------------------------------------------------


class _MockLLMProvider:
    """Routes by `request.node_id` to a canned response per node.

    Triage gets a valid TriageResult JSON; analyze gets a canned
    `{"findings": []}` so the analyze node body iterates without emitting
    findings (consistent with the all-SKIP tier map below — the analyze
    body's triage gate excludes the file before the LLM call fires; the
    canned analyze response is defense-in-depth in case a future test
    sets a non-SKIP tier and accidentally invokes the provider).
    """

    def __init__(self, *, triage_response: str | None = None) -> None:
        # Use SKIM (analyze excludes it from iteration) rather than SKIP
        # (triage's policy forbids LLM-emitted SKIP per the size-cap-gate
        # contract; SKIP is reserved for the deterministic upstream gate).
        self.triage_response = triage_response or json.dumps(
            {
                "file_tiers": {"src/example.py": "skim"},
                "overall_risk": "low",
                "relevant_dimensions": ["code_quality"],
                "reasoning": "Test mock: SKIM tier so analyze iterates zero files.",
            }
        )
        self.analyze_response = json.dumps({"findings": []})

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from outrider.llm.base import LLMResponse

        # Strict node-id routing: an unknown `node_id` (rename, typo, new
        # node) must fail loudly at the mock boundary rather than silently
        # receiving the analyze response and confusing the failure later.
        if request.node_id == "triage":
            text = self.triage_response
        elif request.node_id == "analyze":
            text = self.analyze_response
        else:
            msg = f"_MockLLMProvider: unexpected node_id {request.node_id!r}"
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
        )


# ---------------------------------------------------------------------------
# Cooperative intake mocks
# ---------------------------------------------------------------------------
#
# Intake's deps are stubbed with values that produce a ChangedFile
# tuple BYTE-IDENTICAL to the seed's pr_context.changed_files. That way
# intake's pr_context-replacement is a structural no-op at the
# changed_files level, and tests that asserted "pr_context survives"
# (originally written under the single-node triage graph) still hold
# under the current five-node intake → triage → analyze ⇄ trace → publish graph.


_SEED_FILENAME = "src/example.py"
_SEED_BASE_BYTES = b"old\n"
_SEED_HEAD_BYTES = b"new\n"
_SEED_PATCH = "@@ -1 +1 @@\n-old\n+new\n"
_SEED_INSTALLATION_ID = 12345
_SEED_OWNER = "acme"
_SEED_REPO = "widget"
_SEED_PULL_NUMBER = 42


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
        # Validate the graph wired the right coordinates through to the
        # fetch helper. Without these, the test would still pass if
        # intake fetched the wrong file from the wrong repo and only
        # got the ref right.
        assert owner == _SEED_OWNER, f"unexpected owner {owner!r}"
        assert repo == _SEED_REPO, f"unexpected repo {repo!r}"
        assert path == _SEED_FILENAME, f"unexpected path {path!r}"

        # Return base64(content_base) for base SHA, base64(content_head)
        # for head SHA. The seed uses base_sha="a"*40 and head_sha="b"*40.
        if ref == "a" * 40:
            content_bytes = _SEED_BASE_BYTES
        elif ref == "b" * 40:
            content_bytes = _SEED_HEAD_BYTES
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
        self,
        owner: str,
        repo: str,
        pull_number: int,
        **kwargs: Any,
    ) -> _StubResponse:
        # Validate the graph passed the seed PR coordinates through.
        assert owner == _SEED_OWNER, f"unexpected owner {owner!r}"
        assert repo == _SEED_REPO, f"unexpected repo {repo!r}"
        assert pull_number == _SEED_PULL_NUMBER, f"unexpected pull_number {pull_number}"
        # Pin the load-bearing per_page contract: intake requests
        # `_SIZE_GATE_MAX_FILES + 1` (= 31) so a single API call surfaces
        # any "over the gate" PR without paginating. A regression that
        # dropped or changed this value would silently bypass the size
        # gate in production; without this assert the test stays green
        # because the stub returns a fixed 1-file response anyway.
        from outrider.agent.nodes.intake import _LIST_PR_FILES_PER_PAGE  # noqa: PLC0415

        per_page = kwargs.get("per_page")
        assert per_page == _LIST_PR_FILES_PER_PAGE, (
            f"intake must request per_page=_LIST_PR_FILES_PER_PAGE "
            f"(= _SIZE_GATE_MAX_FILES + 1 = {_LIST_PR_FILES_PER_PAGE}); "
            f"got per_page={per_page!r}"
        )

        return _StubResponse(
            parsed_data=[
                _StubFileMeta(
                    filename=_SEED_FILENAME,
                    status="modified",
                    additions=5,
                    deletions=2,
                    patch=_SEED_PATCH,
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
    # Validate the seed installation_id flows from PRContext through
    # build_graph's github_factory closure.
    assert installation_id == _SEED_INSTALLATION_ID, f"unexpected installation_id {installation_id}"
    return _StubGitHub()


# DB factory stub — intake's happy path only hits this for the size-gate /
# failure-path branches, neither of which fires for the seed's single
# small modified file.
class _NeverCalledSession:
    async def __aenter__(self) -> _NeverCalledSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def begin(self) -> _NeverCalledSession:
        return self

    async def execute(self, stmt: Any) -> Any:
        raise AssertionError("intake happy-path should not write to reviews table in these tests")


def _stub_db_factory() -> _NeverCalledSession:
    return _NeverCalledSession()


class _RecordingFileExaminationSink:
    def __init__(self) -> None:
        self.events: list[FileExaminationEvent] = []

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        self.events.append(event)


class _RecordingAnalyzeEventSink:
    """No-op sink satisfying AnalyzeEventSink Protocol structurally.

    Test bodies in this file don't assert on analyze events directly
    (analyze iterates zero files under the all-SKIP tier map); the
    recorder satisfies build_graph's structural Protocol check and
    captures any analyze emissions for future assertions.
    """

    def __init__(self) -> None:
        self.findings: list[FindingEvent] = []
        self.proposal_rejections: list[FindingProposalRejectedEvent] = []
        self.response_rejections: list[AnalyzeResponseRejectedEvent] = []
        self.completed: list[AnalyzeCompletedEvent] = []

    async def emit_finding(self, event: FindingEvent) -> None:
        self.findings.append(event)

    async def emit_finding_proposal_rejected(self, event: FindingProposalRejectedEvent) -> None:
        self.proposal_rejections.append(event)

    async def emit_analyze_response_rejected(self, event: AnalyzeResponseRejectedEvent) -> None:
        self.response_rejections.append(event)

    async def emit_analyze_completed(self, event: AnalyzeCompletedEvent) -> None:
        self.completed.append(event)


class _StubImportPathResolver:
    """No-op `ImportPathResolver` for tests where analyze iterates zero
    files. The stub satisfies the structural Protocol check; the body
    is never invoked because parse_python isn't called when no file
    reaches analyze."""

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return []


class _StubPublishEventSink:
    """No-op `PublishEventSink` (structural Protocol satisfier).

    Tests in this file exercise intake→triage→analyze; the publish node
    is wired but reaches the empty-eligible short-circuit because the
    seed state has SKIP-tier files and analyze emits nothing. The stub
    admits the structural Protocol check at build_graph; the emit
    methods would capture calls if a future fixture changed fix-tier
    routing to produce admitted findings."""

    async def emit_publish_routing(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_publish_eligibility(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_publish_attempt(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_publish_result(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def query_prior_publish_event(self, review_id: Any) -> Any:  # noqa: ARG002
        return None


class _StubTraceEventSink:
    """No-op `TraceEventSink` (structural Protocol satisfier).

    Tests in this file exercise intake→triage→analyze; trace runs only
    when analyze emits trace_candidates, which the seed state's SKIP-tier
    files prevent. The stub admits the structural Protocol check at
    build_graph; returns the incoming event verbatim if a future
    fixture changes routing to invoke trace."""

    async def emit_trace_decision(self, event: Any) -> Any:
        return event


class _StubGitHubPublisher:
    """No-op `GitHubPublisher`. Same rationale as `_StubPublishEventSink`."""

    async def create_review(self, **kwargs: Any) -> Any:  # noqa: ARG002
        msg = "test stub — create_review unreachable in this file's scenarios"
        raise NotImplementedError(msg)

    async def find_existing_review_on_head_sha(self, **kwargs: Any) -> Any:  # noqa: ARG002
        msg = "test stub — find_existing_review unreachable in this file's scenarios"
        raise NotImplementedError(msg)


class _StubHITLEventSink:
    """No-op `HITLEventSink`. Tests in this file don't drive HITL
    activation but build_graph's structural Protocol gate requires it."""

    async def emit_hitl_request(self, event: Any) -> Any:
        return event

    async def emit_hitl_decision(self, event: Any) -> Any:
        return event


class _StubReviewStatusSink:
    """No-op `ReviewStatusSink`. Same rationale."""

    async def mark_awaiting_approval(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def mark_running(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def mark_awaiting_approval_expired(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None


def _graph_kwargs(
    *,
    phase_event_sink: _RecordingPhaseEventSinkLike,
    file_examination_sink: _RecordingFileExaminationSink | None = None,
) -> dict[str, Any]:
    """Build the full set of build_graph kwargs (intake + triage + analyze + publish).

    Encapsulates the deps so test bodies stay readable. Renamed from
    `_intake_kwargs` when the analyze-node arc added `analyze_event_sink`
    + `import_path_resolver`; extended again 2026-05-22 for the
    publish-node arc (`publish_event_sink` + `publisher`).
    """
    from outrider.agent.nodes.hitl_config import HITLConfig

    return {
        "db_factory": _stub_db_factory,  # type: ignore[arg-type]
        "github_factory": _stub_github_factory,
        "provider": _MockLLMProvider(),
        "model_config": ModelConfig(),
        "phase_event_sink": phase_event_sink,
        "file_examination_sink": file_examination_sink or _RecordingFileExaminationSink(),
        "analyze_event_sink": _RecordingAnalyzeEventSink(),
        "publish_event_sink": _StubPublishEventSink(),
        "trace_sink": _StubTraceEventSink(),
        "hitl_event_sink": _StubHITLEventSink(),
        "review_status_sink": _StubReviewStatusSink(),
        "hitl_config": HITLConfig(),
        "publisher": _StubGitHubPublisher(),
        "import_path_resolver": _StubImportPathResolver(),
    }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_valid_seed_state() -> ReviewState:
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
            pr_body="Body",
            author="someone",
            total_additions=5,
            total_deletions=2,
            changed_files=(
                ChangedFile(
                    path="src/example.py",
                    status="modified",
                    additions=5,
                    deletions=2,
                    patch="@@ -1 +1 @@\n-old\n+new\n",
                    content_base="old\n",
                    content_head="new\n",
                    previous_path=None,
                    language=None,
                ),
            ),
        ),
    )


def _build_seed_dict_with_naive_datetime() -> dict[str, object]:
    """A seed dict where received_at is a NAIVE datetime. Triggers
    AwareDatetime rejection at first-node-input validation."""
    return {
        "review_id": uuid4(),
        "received_at": datetime.now(),  # naive! no tz
        "pr_context": _build_valid_seed_state().pr_context.model_dump(mode="json"),
    }


# ---------------------------------------------------------------------------
# FUP-019 test 1: first-node input validation fires
# ---------------------------------------------------------------------------
#
# Originally rebuilt 2026-05-17 for the then-current two-node intake →
# triage graph; the graph has since extended to five nodes (intake →
# triage → analyze ⇄ trace → publish) and these tests still hold
# because the intake-side stubs produce a `ChangedFile` tuple
# byte-identical to the seed's pr_context.changed_files (preserving
# pr_context-survival assertions), and the SKIP-tier triage response
# keeps analyze + trace + publish as no-op pass-throughs. The
# `_graph_kwargs` helper wires every node's
# deps via the same closure pattern.


@pytest.mark.asyncio
async def test_naive_datetime_seed_raises_validation_error_at_first_node_input(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """LangGraph 1.1.6: 'Run-time validation only occurs on inputs to the
    first node in the graph.' Build the compiled graph and call
    `await graph.ainvoke({...naive datetime...})` → pydantic.ValidationError
    raised BEFORE the triage callable runs.

    This pins the canonical behavior: validate_assignment=True on
    ReviewState is the post-first-input lifetime defense; the construction-
    time validator (AwareDatetime → reject naive) is the first-input gate.
    """
    graph = build_graph(**_graph_kwargs(phase_event_sink=recording_phase_event_sink))
    bad_seed = _build_seed_dict_with_naive_datetime()
    with pytest.raises(ValidationError):
        await graph.ainvoke(bad_seed)

    # Triage callable never ran; no phase events emitted
    assert recording_phase_event_sink.events == []


# ---------------------------------------------------------------------------
# FUP-019 test 2: partial-state merge populates triage_result; pr_context survives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_invocation_merges_partial_state_and_preserves_pr_context(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Default reducer overwrites; triage returns {'triage_result': ...};
    the result dict has triage_result populated AND pr_context survives
    byte-for-byte unchanged. The PEP 544 isinstance gate at build time
    already passed for the sink; this test verifies the runtime contract.
    """
    state = _build_valid_seed_state()
    graph = build_graph(**_graph_kwargs(phase_event_sink=recording_phase_event_sink))

    result = await graph.ainvoke(state)

    # LangGraph returns a dict, not a Pydantic instance
    assert isinstance(result, dict)
    # Both keys must be present — explicit assertion catches the case where
    # LangGraph silently drops one. Without this, an absent key would surface
    # as KeyError below, which reads as "test infra broken" rather than
    # "contract violated".
    assert "triage_result" in result, "triage_result missing from merged state"
    assert "pr_context" in result, "pr_context dropped from merged state"
    # pr_context survives unchanged (compare model_dump shapes; the dict
    # may contain Pydantic instances or dicts depending on LangGraph version)
    original_pr_context_dump = state.pr_context.model_dump(mode="json")
    result_pr_context = result["pr_context"]
    if isinstance(result_pr_context, PRContext):
        assert result_pr_context.model_dump(mode="json") == original_pr_context_dump
    else:
        # dict-shaped — direct compare after re-serializing the original
        assert result_pr_context == original_pr_context_dump

    # Phase events were emitted: every node that ran fired its start+end pair
    # (phase-events-bound-work). Property-based assertion — count reflects
    # graph wiring (intake/triage/analyze/publish today; trace/synthesize/hitl
    # later), not the contract under test. The contract is "every node that
    # ran emitted a matched start+end pair."
    events = recording_phase_event_sink.events
    assert len(events) > 0, "no phase events emitted"
    starts = [e for e in events if e.marker == "start"]
    ends = [e for e in events if e.marker == "end"]
    assert len(starts) == len(ends), (
        f"unmatched start/end pairs: {len(starts)} starts, {len(ends)} ends"
    )
    # Each emitted node_id contributed exactly one start + one end pair.
    node_ids_with_starts = {e.node_id for e in starts}
    node_ids_with_ends = {e.node_id for e in ends}
    assert node_ids_with_starts == node_ids_with_ends, (
        f"node_ids with starts ({sorted(node_ids_with_starts)}) don't match "
        f"node_ids with ends ({sorted(node_ids_with_ends)})"
    )
    # Triage MUST have run (this test exercises triage merge behavior).
    assert "triage" in node_ids_with_starts, (
        "triage did not run; this test exercises triage's merge contract"
    )


# ---------------------------------------------------------------------------
# FUP-019 test 3: result-dict rehydrates as ReviewState
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_dict_rehydrates_as_review_state(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """LangGraph 1.1.6 docs: 'the output of the graph will NOT be an
    instance of a pydantic model.' Callers do `Model(**result)` to
    re-validate the merged shape.

    This test exercises that canonical path: invoke → get dict → re-
    validate via ReviewState(**result) → instance has triage_result
    populated as a TriageResult, not a raw dict. Pins the round-trip
    contract documented in `review_state.py`.
    """
    state = _build_valid_seed_state()
    graph = build_graph(**_graph_kwargs(phase_event_sink=recording_phase_event_sink))

    result = await graph.ainvoke(state)
    rehydrated = ReviewState(**result)

    assert isinstance(rehydrated, ReviewState)
    assert rehydrated.review_id == state.review_id
    assert isinstance(rehydrated.triage_result, TriageResult)
    assert rehydrated.triage_result.overall_risk.value == "low"


# ---------------------------------------------------------------------------
# Bonus integration check: the phase-event sink receives matching phase_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_eval_survives_langgraph_merge(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """`is_eval` on the seed `ReviewState` must survive the partial-state
    merge — triage returns `{"triage_result": ...}` only, so LangGraph's
    default-reducer-overwrite path should leave the seed's `is_eval`
    untouched. Without this test, a future regression that drops the
    eval flag through the merge would silently pollute the production
    audit stream with eval-tagged runs (or vice versa).

    Tests both `is_eval=True` (eval seed) and `is_eval=False`
    (production seed) round-trip cleanly through:
      - the result-dict returned by ainvoke,
      - the rehydrated ReviewState,
      - both phase events emitted by triage during the invocation.

    The phase-event assertion specifically covers the graph-driven
    invocation path (the unit tests cover the direct-call path);
    pinning both surfaces together prevents a regression where the
    reducer drops the flag from one but not the other."""
    # Graph construction does not depend on eval_flag — hoist out of the loop.
    graph = build_graph(**_graph_kwargs(phase_event_sink=recording_phase_event_sink))
    for eval_flag in (True, False):
        state = _build_valid_seed_state()
        state.is_eval = eval_flag  # validate_assignment=True validates this
        # events_before stays in the loop: the sink is function-scoped and
        # accumulates across both iterations, so each iteration needs its
        # own pre-invocation snapshot to slice the new events.
        events_before = len(recording_phase_event_sink.events)

        result = await graph.ainvoke(state)

        # Default reducer is overwrite; triage didn't touch is_eval so it
        # comes through unchanged
        assert result["is_eval"] is eval_flag, (
            f"is_eval={eval_flag} should survive merge; got {result['is_eval']}"
        )
        # Rehydration round-trip preserves the flag
        rehydrated = ReviewState(**result)
        assert rehydrated.is_eval is eval_flag

        # Phase events emitted during THIS iteration must carry the same flag.
        # Slicing from events_before isolates per-iteration emissions since
        # the recording sink is function-scoped and accumulates across the loop.
        # Property-based: non-empty + all carry the right is_eval (count
        # reflects graph wiring, not the eval-propagation contract under test).
        new_events = recording_phase_event_sink.events[events_before:]
        assert len(new_events) > 0, f"no phase events for is_eval={eval_flag} iteration"
        for ev in new_events:
            assert ev.is_eval is eval_flag, (
                f"phase event marker={ev.marker!r} carries is_eval={ev.is_eval}, "
                f"expected {eval_flag} (eval-isolation contract broken at "
                f"the graph-driven invocation path)"
            )


@pytest.mark.asyncio
async def test_phase_events_have_matching_phase_id_through_graph(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """End-to-end: each node's start/end pair shares the same phase_id.
    Validates the closure correctly threads phase_id through both emit_phase
    calls inside the LangGraph-managed node invocation. Property-based —
    for every unique phase_id, there's exactly one start + one end, and
    they share the same node_id. Distinct phase_ids across nodes.

    The unit test pins this at the node level; the integration test
    confirms the full graph orchestration doesn't break the contract."""
    state = _build_valid_seed_state()
    graph = build_graph(**_graph_kwargs(phase_event_sink=recording_phase_event_sink))

    await graph.ainvoke(state)

    events = recording_phase_event_sink.events
    assert len(events) > 0, "no phase events emitted"

    # Group events by phase_id; every group must have exactly one start +
    # one end with the same node_id.
    by_phase_id: dict[str, list[Any]] = {}
    for ev in events:
        by_phase_id.setdefault(ev.phase_id, []).append(ev)
    for phase_id, pair in by_phase_id.items():
        assert len(pair) == 2, (
            f"phase_id {phase_id!r} has {len(pair)} events; expected 2 (start + end)"
        )
        markers = sorted(e.marker for e in pair)
        assert markers == ["end", "start"], (
            f"phase_id {phase_id!r} markers {markers!r}; expected [start, end]"
        )
        node_ids = {e.node_id for e in pair}
        assert len(node_ids) == 1, (
            f"phase_id {phase_id!r} has mixed node_ids {sorted(node_ids)!r}; "
            f"start+end pair must share node_id"
        )

    # Distinct phase_ids across distinct node invocations: total unique
    # phase_ids equals total unique node_ids (one phase_id per node invocation).
    unique_phase_ids = len(by_phase_id)
    unique_node_ids = len({e.node_id for e in events})
    assert unique_phase_ids == unique_node_ids, (
        f"{unique_phase_ids} unique phase_ids vs {unique_node_ids} unique node_ids; "
        f"each node invocation should have its own phase_id"
    )
    for ev in events:
        assert ev.review_id == state.review_id
