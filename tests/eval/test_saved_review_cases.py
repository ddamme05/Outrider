"""Replay real responses through admission; model output is never the answer key."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from outrider.llm.base import LLMRequest, LLMResponse

from .model_comparison import run_analyze_instrumented, state_from_eval_fixture
from .review_quality import assess_case
from .test_model_comparison import _GROUND_TRUTH_BY_FIXTURE

_ROOT = Path(__file__).resolve().parents[2]
_CAPTURES = Path(__file__).with_name("fixtures") / "saved_reviews"


class SavedResponseProvider:
    def __init__(self, response: dict) -> None:
        self.response = LLMResponse.model_validate(response)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        assert not self.requests, "a saved single-file response cannot serve another call"
        self.requests.append(request)
        return self.response

    async def aclose(self) -> None:
        pass


def load_capture(name: str) -> tuple[dict, SavedResponseProvider]:
    capture = json.loads((_CAPTURES / f"{name}.json").read_text())
    fixture = _ROOT / capture["fixture"]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == capture["fixture_sha256"]
    return capture, SavedResponseProvider(capture["response"])


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["valid", "reversed", "exact-quote-control"])
async def test_real_ssrf_response_requires_valid_range_and_an_exact_quote(variant: str) -> None:
    capture, provider = load_capture(
        f"haiku-ssrf-{'valid' if variant == 'exact-quote-control' else variant}"
    )
    state = state_from_eval_fixture(_ROOT / capture["fixture"])
    if variant == "exact-quote-control":
        # The saved answer contains a literal backslash-n and wrong indentation.
        # Keep it immutable; this derived control quotes the actual HEAD instead.
        payload = json.loads(provider.response.text)
        proposal = payload["findings"][0]
        source = state.pr_context.changed_files[0].content_head
        assert source is not None
        proposal["evidence"] = "\n".join(
            source.splitlines()[proposal["line_start"] - 1 : proposal["line_end"]]
        )
        provider.response = provider.response.model_copy(update={"text": json.dumps(payload)})
    result = await run_analyze_instrumented(
        state,
        provider=provider,
        model=provider.response.model,
        collect_producer_coverage=True,
    )
    assert len(provider.requests) == 1
    assert not result.producer_coverage_findings
    assert not result.response_rejections
    if variant == "exact-quote-control":
        assert len(result.findings) == 1
        assert result.findings[0].finding_type.value == "ssrf"
        assert result.proposal_rejections == 0
    else:
        assert result.proposal_rejections == 1
        assert not result.findings


@pytest.mark.asyncio
async def test_real_shipping_response_separates_quote_offset_from_unknown_test_context() -> None:
    capture, provider = load_capture("sonnet-shipping-location")
    result = await run_analyze_instrumented(
        state_from_eval_fixture(_ROOT / capture["fixture"]),
        provider=provider,
        model=provider.response.model,
        collect_producer_coverage=True,
    )
    assert len(provider.requests) == 1
    assert len(result.findings) == 1
    (assessment,) = assess_case(
        result.findings,
        _GROUND_TRUTH_BY_FIXTURE[capture["fixture"]],
        fixture_path=_ROOT / capture["fixture"],
        request=provider.requests[0],
    )
    assert assessment.status == "unscored"
    assert assessment.grounded_expected_issue_match is None
    (finding,) = assessment.findings
    assert finding.quote_spans == ((8, 13),)
    assert finding.declared_span == (9, 14)
    assert finding.location_quality == "quote-offset"
    assert finding.quote_offsets == ((1, 1),)
