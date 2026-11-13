"""Check the native OpenAI analyze request and persisted accounting exactly.

Fixed expected values cover every request argument, token split, pricing context,
and cost. The paid capture tool uses the same request builder.
See DECISIONS.md#056 and specs/2026-07-18-openai-native-host.md."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from openai.resources.chat.completions.completions import AsyncCompletions
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from pydantic import SecretStr

from outrider.audit.events import ContextManifestEntry, LLMCallEvent
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.llm.host_profiles import OPENAI_PROFILE, resolve_host_identity
from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider, _build_sdk_kwargs
from outrider.llm.pricing import CostUnpricedReason
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA_JSON

# The frozen openai request envelope for the production analyze path. The
# prompt_cache_key digest prefix is pinned LITERALLY (not recomputed from
# OPENAI_PROFILE) so any profile-contract change must acknowledge here that it
# rotates the cache key — rotation is correct, silence is not. json_object mode
# means the schema does NOT ride the wire (prompt-described per #056(b)); the
# envelope's job is to prove no json_schema block leaks in.
GOLDEN_KWARGS: dict[str, Any] = {
    "model": "gpt-5.6-sol",
    # The 5.6 wire 400s on `max_tokens` (paid probe capture, SHAPER v3) — the
    # ceiling rides under the profile-declared `max_completion_tokens`.
    "max_completion_tokens": 100,
    "temperature": 0.0,
    "messages": [
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": "Review this PR."},
    ],
    "reasoning_effort": "none",
    "service_tier": "default",
    # The contract digest rotates when supported request behavior changes.
    "prompt_cache_key": "outrider:07554de94696bea1:analyze@1.0.0",
    "response_format": {"type": "json_object"},
}
# §8a writes-reported (SHAPER v4, capture-pinned): prompt(100) INCLUDES both
# cached(30) AND writes(20) — `total == prompt + completion` on the paid
# fixture — so input = 100 - 30 - 20 = 50. Nonzero cached AND nonzero writes
# so both subtractions are exercised at once.
GOLDEN_RESPONSE_ACCOUNTING: dict[str, int] = {
    "input_tokens": 50,
    "cache_read_tokens": 30,
    "cache_write_tokens": 20,
    "output_tokens": 50,
}
# DOCUMENTED-ONLY (no longer asserted): the $ a subtractive-billing reading WOULD imply —
# Sol short-context, default tier ×1 — 50·$5 + 20·$6.25 + 30·$0.50 + 50·$30 per MTok =
# $0.00189. openai cost is Unpriced(BILLING_PENDING) until FUP-247 confirms the equation, so
# this value is a reference for the eventual adjudication, not a pinned expectation.
GOLDEN_COST_USD: float = 0.00189
GOLDEN_PRICING_VERSION = "v8"

_FIXED_REVIEW_ID = UUID("00000000-0000-0000-0000-0000000000ab")


class _RecordingPersister:
    def __init__(self) -> None:
        self.calls: list[tuple[LLMCallEvent, LLMRequest, LLMResponse]] = []

    async def persist(
        self, event: LLMCallEvent, request: LLMRequest, response: LLMResponse
    ) -> None:
        self.calls.append((event, request, response))


def _golden_request() -> LLMRequest:
    return LLMRequest(
        system_prompt="You are a code reviewer.",
        user_prompt="Review this PR.",
        model="gpt-5.6-sol",
        max_tokens=100,
        temperature=0.0,
        review_id=_FIXED_REVIEW_ID,
        node_id="analyze",
        prompt_template_version="analyze@1.0.0",
        degraded_mode=False,
        context_summary=(
            ContextManifestEntry(
                file_path="src/foo.py",
                scope_unit_name="Foo.bar",
                line_start=1,
                line_end=10,
                inclusion_reason="changed_scope",
            ),
        ),
        response_schema_json=ANALYZE_RESPONSE_SCHEMA_JSON,
    )


def _gpt56_completion() -> ChatCompletion:
    ptd_kwargs: dict[str, Any] = {"cached_tokens": 30, "cache_write_tokens": 20}
    completion = ChatCompletion(
        id="chatcmpl-openai-golden",
        created=0,
        model="gpt-5.6-sol",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="{}"),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            # cache_write_tokens is an untyped extra on the pinned SDK — preserved.
            prompt_tokens_details=PromptTokensDetails(**ptd_kwargs),
        ),
    )
    return completion.model_copy(update={"service_tier": "default"})


@pytest.mark.asyncio
async def test_openai_wire_golden_request_accounting_and_event() -> None:
    """openai request envelope (FULL kwargs equality), §8a writes-reported
    accounting, and the persisted LLMCallEvent pricing-context identity are
    frozen — the paid capture rides exactly this wire."""
    persister = _RecordingPersister()
    provider = OpenAICompatibleProvider(
        api_key=SecretStr("openai-test-key"),
        profile=OPENAI_PROFILE,
        persister=persister,
        models=("gpt-5.6-sol", "gpt-5.6-luna"),
    )
    mock = AsyncMock(return_value=_gpt56_completion())
    try:
        with patch.object(AsyncCompletions, "create", mock):
            response = await provider.complete(_golden_request())
    finally:
        await provider.aclose()

    # 1) request wire shape — the COMPLETE kwargs dict, frozen.
    assert mock.call_args.kwargs == GOLDEN_KWARGS, "openai request wire shape drifted"

    # 2) §8a writes-reported accounting + the response's pricing-context ride.
    assert {
        "input_tokens": response.input_tokens,
        "cache_read_tokens": response.cache_read_tokens,
        "cache_write_tokens": response.cache_write_tokens,
        "output_tokens": response.output_tokens,
    } == GOLDEN_RESPONSE_ACCOUNTING, "openai §8a response accounting drifted"
    assert response.billed_prompt_tokens == 100
    assert response.service_tier_actual == "default"

    # 3) the persisted LLMCallEvent's cost + pricing-context identity.
    assert len(persister.calls) == 1, "expected exactly one persisted LLMCallEvent"
    event, _req, _resp = persister.calls[0]
    assert event.input_tokens == 50
    assert event.output_tokens == 50
    assert event.cached_tokens == 30
    assert event.cache_hit is True
    assert event.billed_prompt_tokens == 100
    assert event.cache_write_tokens == 20
    assert event.service_tier == "default"
    assert event.pricing_version == GOLDEN_PRICING_VERSION
    assert event.model == "gpt-5.6-sol"
    # openai cost is WITHHELD until the SHAPER-v4 billing adjudication lands (FUP-247):
    # the token split above is wire-consistent, but the $ is Unpriced(BILLING_PENDING),
    # NOT the (documented-only) `GOLDEN_COST_USD` that a subtractive-billing reading implies.
    assert event.cost_usd is None
    assert event.cost_unpriced_reason == CostUnpricedReason.BILLING_PENDING


@pytest.mark.parametrize(
    ("model", "effort", "cache_key", "reasoning"),
    [
        ("gpt-6.1-sol", "low", "outrider:449759b03c651d18:analyze@1.0.0", True),
        ("gpt-6.1-terra", "none", "outrider:07554de94696bea1:analyze@1.0.0", False),
        ("gpt-6-luna", "none", "outrider:07554de94696bea1:analyze@1.0.0", False),
    ],
)
@pytest.mark.asyncio
async def test_current_openai_models_request_and_reasoning_accounting(
    model: str, effort: str, cache_key: str, reasoning: bool
) -> None:
    request = _golden_request().model_copy(update={"model": model})
    expected = GOLDEN_KWARGS | {
        "model": model,
        "reasoning_effort": effort,
        "prompt_cache_key": cache_key,
    }
    if reasoning:
        del expected["temperature"]
    assert _build_sdk_kwargs(request, profile=OPENAI_PROFILE) == expected
    persister = _RecordingPersister()
    provider = OpenAICompatibleProvider(
        api_key=SecretStr("openai-test-key"),
        profile=OPENAI_PROFILE,
        persister=persister,
        models=(model,),
    )
    completion = _gpt56_completion()
    assert completion.usage is not None
    completion.usage.completion_tokens_details = CompletionTokensDetails(
        reasoning_tokens=40 if reasoning else 0
    )
    mock = AsyncMock(return_value=completion)
    try:
        with patch.object(AsyncCompletions, "create", mock):
            response = await provider.complete(request)
    finally:
        await provider.aclose()
    assert mock.call_args.kwargs == expected
    assert (
        response.input_tokens,
        response.cache_read_tokens,
        response.cache_write_tokens,
        response.output_tokens,
    ) == (50, 30, 20, 50)
    event, _, persisted_response = persister.calls[0]
    assert persisted_response is response
    assert event.output_tokens == 50  # Reasoning is already included, never added or subtracted.
    assert event.reasoning_enabled is reasoning
    assert (
        response.profile_id,
        response.reasoning_enabled,
        response.profile_contract_digest,
    ) == resolve_host_identity("openai", reasoning=False, models=(model,))
    assert event.profile_contract_digest == response.profile_contract_digest
    assert event.cost_usd is None
    assert event.cost_unpriced_reason == CostUnpricedReason.BILLING_PENDING
