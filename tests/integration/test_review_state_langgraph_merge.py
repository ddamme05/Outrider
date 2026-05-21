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
# (under the single-node triage graph) still hold under the two-node
# intake → triage graph.


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


def _graph_kwargs(
    *,
    phase_event_sink: _RecordingPhaseEventSinkLike,
    file_examination_sink: _RecordingFileExaminationSink | None = None,
) -> dict[str, Any]:
    """Build the full set of build_graph kwargs (intake + triage + analyze).

    Encapsulates the seven deps so test bodies stay readable. Renamed
    from `_intake_kwargs` when the analyze-node arc added
    `analyze_event_sink` + `import_path_resolver`.
    """
    return {
        "db_factory": _stub_db_factory,  # type: ignore[arg-type]
        "github_factory": _stub_github_factory,
        "provider": _MockLLMProvider(),
        "model_config": ModelConfig(),
        "phase_event_sink": phase_event_sink,
        "file_examination_sink": file_examination_sink or _RecordingFileExaminationSink(),
        "analyze_event_sink": _RecordingAnalyzeEventSink(),
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
# Rebuilt 2026-05-17 for the two-node intake → triage graph: each test
# now passes the cooperative intake deps (stub github_factory + stub
# db_factory + recording file_examination_sink) via the `_graph_kwargs`
# helper. The intake-side stubs are designed so that the produced
# `ChangedFile` tuple is byte-identical to the seed's pr_context.changed_files,
# preserving each test's original assertions about pr_context survival.


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

    # Phase events were emitted: three start+end pairs (intake, triage, analyze).
    # Analyze fires its pair even when the all-SKIP tier map sends zero files
    # into the per-file loop (phase-events-bound-work guarantees the pair).
    events = recording_phase_event_sink.events
    assert len(events) == 6
    assert [e.node_id for e in events] == [
        "intake",
        "intake",
        "triage",
        "triage",
        "analyze",
        "analyze",
    ]
    assert [e.marker for e in events] == ["start", "end", "start", "end", "start", "end"]


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
        # Three-node graph: intake + triage + analyze start/end pairs = 6 events.
        new_events = recording_phase_event_sink.events[events_before:]
        assert len(new_events) == 6, (
            f"expected intake+triage+analyze start/end pairs for is_eval={eval_flag} "
            f"iteration, got {len(new_events)}"
        )
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
    calls inside the LangGraph-managed node invocation. Three-node graph:
    intake, triage, and analyze each produce a start/end pair, each with a
    distinct phase_id.

    The unit test pins this at the node level; the integration test
    confirms the full graph orchestration doesn't break the contract."""
    state = _build_valid_seed_state()
    graph = build_graph(**_graph_kwargs(phase_event_sink=recording_phase_event_sink))

    await graph.ainvoke(state)

    events = recording_phase_event_sink.events
    assert len(events) == 6
    # Intake / triage / analyze each share a phase_id internally; the three
    # pairs use distinct phase_ids (one per node invocation).
    assert events[0].node_id == "intake" and events[1].node_id == "intake"
    assert events[2].node_id == "triage" and events[3].node_id == "triage"
    assert events[4].node_id == "analyze" and events[5].node_id == "analyze"
    assert events[0].phase_id == events[1].phase_id  # intake pair
    assert events[2].phase_id == events[3].phase_id  # triage pair
    assert events[4].phase_id == events[5].phase_id  # analyze pair
    assert events[0].phase_id != events[2].phase_id  # intake vs triage
    assert events[2].phase_id != events[4].phase_id  # triage vs analyze
    assert events[0].phase_id != events[4].phase_id  # intake vs analyze
    for ev in events:
        assert ev.review_id == state.review_id
