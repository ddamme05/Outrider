"""Replay the recorded empty-finding summary failures through the real node.

The bounded state projection omits source bodies that synthesis does not use.
No provider or external service is contacted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from outrider.agent.nodes.synthesize import synthesize
from outrider.audit.aggregates import ReviewLLMAggregates
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.schemas.review_state import ReviewState

if TYPE_CHECKING:
    from outrider.audit.events import SynthesizeCompletedEvent

_FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "empty_summary_live_20261112.json").read_text()
)


class _RecordedSummaryProvider:
    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Baseline must reach the actual recorded inputs, not a different bug.
        assert request.system_prompt == self.case["request"]["system_prompt"]
        assert request.user_prompt == self.case["request"]["user_prompt"]
        self.requests.append(request)
        return LLMResponse.model_validate(self.case["response"])


class _PhaseSink:
    async def emit_phase(self, event: Any) -> None:
        pass


class _AnomalySink:
    async def emit_anomaly(self, **kwargs: Any) -> None:
        raise AssertionError(f"Unexpected anomaly: {kwargs}")


class _MeasuredSink:
    def __init__(self, provider: _RecordedSummaryProvider) -> None:
        self.provider = provider
        self.completed: list[SynthesizeCompletedEvent] = []

    async def query_review_llm_aggregates(self, **kwargs: Any) -> ReviewLLMAggregates:
        # Authored accounting control: three prior calls plus this node's calls.
        return ReviewLLMAggregates(
            llm_calls_made=3 + len(self.provider.requests),
            total_input_tokens=1200 + 100 * len(self.provider.requests),
            total_output_tokens=200 + 50 * len(self.provider.requests),
            total_cost_usd=0.01 + 0.001 * len(self.provider.requests),
        )

    async def emit_synthesize_completed(self, event: SynthesizeCompletedEvent) -> None:
        self.completed.append(event)


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: case["case"])
async def test_empty_summary_uses_facts_without_a_model_call(case: dict[str, Any]) -> None:
    state = ReviewState.model_validate(
        {**case["state_projection"], "pr_context": {**case["pr_metadata"], "changed_files": []}}
    )
    provider = _RecordedSummaryProvider(case)
    sink = _MeasuredSink(provider)
    result = await synthesize(
        state,
        provider=provider,
        synthesize_model=case["request"]["model"],
        patch_model=case["request"]["model"],
        patches_enabled=True,
        max_suggestions=3,
        phase_event_sink=_PhaseSink(),
        synthesize_event_sink=sink,
        anomaly_sink=_AnomalySink(),
    )
    report = result["review_report"]
    risk = case["state_projection"]["triage_result"]["overall_risk"]
    expected = (
        f"This review retained no findings. Triage classified the pull request as {risk} risk."
    )
    # Fail on the saved out-of-scope advice before checking new metadata.
    assert report.summary == expected
    assert provider.requests == []
    assert report.findings == ()
    assert report.overall_risk.value == risk
    assert report.metrics.llm_calls_made == 3
    assert report.metrics.total_input_tokens == 1200
    assert report.metrics.total_output_tokens == 200
    assert report.metrics.total_cost_usd == 0.01
    [event] = sink.completed
    assert event.summary_source == "empty-findings-v1"
    assert event.synthesize_model is None
    assert event.summary_content_hash == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert event.selected_finding_ids == ()
    assert event.llm_calls_made == report.metrics.llm_calls_made
    assert event.total_cost_usd == report.metrics.total_cost_usd
