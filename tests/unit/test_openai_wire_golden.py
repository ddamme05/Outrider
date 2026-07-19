"""Full-envelope wire golden for the native `openai` host (DECISIONS.md#056,
openai-native-host spec).

Freezes the COMPLETE `chat.completions.create` kwargs dict the production
`OpenAICompatibleProvider(openai)` sends for an analyze-shaped request — not
individual keys — so an added, dropped, or renamed kwarg (a stray `store`, a
lost `prompt_cache_key`, a `response_format` shape change) fails the equality,
not just the behaviors the per-key pins in `test_openai_compatible_provider.py`
name. Also freezes the §8a writes-reported accounting split, the persisted
`LLMCallEvent`'s pricing-context identity, and a LITERAL cost pin.

The paid probe (`spikes/openai/probe.py`) builds its capture requests through
the same `_build_sdk_kwargs`, so the wire this golden freezes is the wire the
admission fixtures are captured on.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from openai.resources.chat.completions.completions import AsyncCompletions
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from pydantic import SecretStr

from outrider.audit.events import ContextManifestEntry, LLMCallEvent
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.llm.host_profiles import OPENAI_PROFILE
from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA_JSON

# The frozen openai request envelope for the production analyze path. The
# prompt_cache_key digest prefix is pinned LITERALLY (not recomputed from
# OPENAI_PROFILE) so any profile-contract change must acknowledge here that it
# rotates the cache key — rotation is correct, silence is not. json_object mode
# means the schema does NOT ride the wire (prompt-described per #056(b)); the
# envelope's job is to prove no json_schema block leaks in.
GOLDEN_KWARGS: dict[str, Any] = {
    "model": "gpt-5.6-sol",
    # SHAPER v3: the 5.6 wire 400s on `max_tokens` (paid probe capture) — the
    # ceiling rides under the profile-declared `max_completion_tokens`.
    "max_completion_tokens": 100,
    "temperature": 0.0,
    "messages": [
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": "Review this PR."},
    ],
    "reasoning_effort": "none",
    "service_tier": "default",
    # Digest prefix re-pinned for the SHAPER v3 + token_limit_param rotation.
    "prompt_cache_key": "outrider:e397406ea91794b0:analyze@1.0.0",
    "response_format": {"type": "json_object"},
}
# §8a writes-reported: prompt(100) INCLUDES cached(30) → input=70; writes(20)
# are their own billed class, NOT subtracted from input. Nonzero cached AND
# nonzero writes so both branches of the accounting are exercised at once.
GOLDEN_RESPONSE_ACCOUNTING: dict[str, int] = {
    "input_tokens": 70,
    "cache_read_tokens": 30,
    "cache_write_tokens": 20,
    "output_tokens": 50,
}
# cost_usd LITERALLY pinned (never recomputed via compute_cost_outcome, so a
# rate re-key can't move provider and expectation together): Sol short-context,
# default tier ×1 — 70·$5 + 20·$6.25 + 30·$0.50 + 50·$30 per MTok = $0.00199.
GOLDEN_COST_USD: float = 0.00199
GOLDEN_PRICING_VERSION = "v7"

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
    assert event.input_tokens == 70
    assert event.output_tokens == 50
    assert event.cached_tokens == 30
    assert event.cache_hit is True
    assert event.billed_prompt_tokens == 100
    assert event.cache_write_tokens == 20
    assert event.service_tier == "default"
    assert event.cost_unpriced_reason is None
    assert event.pricing_version == GOLDEN_PRICING_VERSION
    assert event.model == "gpt-5.6-sol"
    assert event.cost_usd == pytest.approx(GOLDEN_COST_USD), "openai cost_usd drifted"
