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

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.agent.graph import build_graph
from outrider.llm.config import ModelConfig
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState
from outrider.schemas.triage_result import TriageResult

if TYPE_CHECKING:
    from outrider.audit.events import ReviewPhaseEvent
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
    """Returns a canned valid TriageResult JSON; no transport failures."""

    def __init__(self, *, canned_response: str | None = None) -> None:
        self.canned_response = canned_response or json.dumps(
            {
                "file_tiers": {"src/example.py": "standard"},
                "overall_risk": "low",
                "relevant_dimensions": ["code_quality"],
                "reasoning": "Standard refactor.",
            }
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from outrider.llm.base import LLMResponse

        return LLMResponse(
            text=self.canned_response,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=42,
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_valid_seed_state() -> ReviewState:
    return ReviewState(
        review_id=uuid4(),
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=12345,
            owner="acme",
            repo="widget",
            pr_number=42,
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
    graph = build_graph(
        provider=_MockLLMProvider(),
        model_config=ModelConfig(),
        phase_event_sink=recording_phase_event_sink,
    )
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
    graph = build_graph(
        provider=_MockLLMProvider(),
        model_config=ModelConfig(),
        phase_event_sink=recording_phase_event_sink,
    )

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

    # Phase events were emitted: start + end bracketing
    assert len(recording_phase_event_sink.events) == 2
    assert recording_phase_event_sink.events[0].marker == "start"
    assert recording_phase_event_sink.events[1].marker == "end"


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
    graph = build_graph(
        provider=_MockLLMProvider(),
        model_config=ModelConfig(),
        phase_event_sink=recording_phase_event_sink,
    )

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
    for eval_flag in (True, False):
        state = _build_valid_seed_state()
        state.is_eval = eval_flag  # validate_assignment=True validates this
        graph = build_graph(
            provider=_MockLLMProvider(),
            model_config=ModelConfig(),
            phase_event_sink=recording_phase_event_sink,
        )
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
        new_events = recording_phase_event_sink.events[events_before:]
        assert len(new_events) == 2, (
            f"expected start+end phase-event pair for is_eval={eval_flag} "
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
    """End-to-end: the start and end events captured by the sink share the
    same phase_id. Validates the closure correctly threads phase_id through
    both emit_phase calls inside the LangGraph-managed node invocation.

    The unit test pins this at the node level; the integration test
    confirms the full graph orchestration doesn't break the contract."""
    state = _build_valid_seed_state()
    graph = build_graph(
        provider=_MockLLMProvider(),
        model_config=ModelConfig(),
        phase_event_sink=recording_phase_event_sink,
    )

    await graph.ainvoke(state)

    events = recording_phase_event_sink.events
    assert len(events) == 2
    assert events[0].phase_id == events[1].phase_id
    assert events[0].review_id == state.review_id
    assert events[1].review_id == state.review_id


# Suppress unused-import warning on Literal — Literal IS used in fixtures
# imported via conftest; we keep it imported for ergonomic test extension.
_RESERVED = (Literal,)
