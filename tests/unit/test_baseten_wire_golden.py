"""Check the saved Baseten request and accounting output byte for byte.

Compare the full analyze envelope, token/cache counts, pricing version, and
persisted cost against fixed expected values. This checks request compatibility,
not model quality. See DECISIONS.md#056."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from openai.resources.chat.completions.completions import AsyncCompletions
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
from pydantic import SecretStr

from outrider.audit.events import ContextManifestEntry, LLMCallEvent
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.llm.glm_provider import GLM_MODEL_ID, GLMProvider
from outrider.llm.host_profiles import resolve_host_identity, resolve_host_profile
from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider, _build_sdk_kwargs
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA_JSON

# Frozen Baseten request wire contract for the PRODUCTION analyze path — the post-rename
# provider MUST reproduce this. The `response_format` envelope (json_schema name + strict)
# is the production analyze path; freezing it catches envelope drift the bare-kwargs golden
# would miss (audit-9 #1). The schema body itself is owned/digest-pinned by the analyze
# constrained-decoding spec, so the golden freezes the ENVELOPE + that the schema rides it.
GOLDEN_KWARGS: dict[str, Any] = {
    "model": "zai-org/GLM-5.2",
    "max_tokens": 100,
    "temperature": 0.0,
    "messages": [
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": "Review this PR."},
    ],
    "extra_body": {"chat_template_args": {"enable_thinking": False}},
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "outrider_analyze",
            "strict": True,
            "schema": json.loads(ANALYZE_RESPONSE_SCHEMA_JSON),
        },
    },
}
# §8a: prompt_tokens(100) INCLUDES cached(30) -> input=70, cache_read=30, cache_write=0.
# NONZERO cache hit so the includes-vs-excludes subtraction is actually exercised
# (audit-7 #8 / audit-8 #2: a zero-cache response can't prove §8a semantics).
GOLDEN_RESPONSE_ACCOUNTING: dict[str, int] = {
    "input_tokens": 70,
    "cache_read_tokens": 30,
    "cache_write_tokens": 0,
    "output_tokens": 50,
}
# Persisted-event identity: the wrapper maps response.cache_read_tokens -> event.cached_tokens
# and stamps cache_hit + pricing_version + cost_usd (audit-9 #2).
GOLDEN_EVENT_TOKENS: dict[str, int] = {"input_tokens": 70, "output_tokens": 50, "cached_tokens": 30}
# cost_usd is LITERALLY pinned (NOT recomputed via compute_cost_usd) so a pricing re-key
# regression can't move both the provider and the expected together and stay green (audit-10).
# 70·$1.40 + 30·$0.14 + 50·$4.40 per MTok = $0.0003222 (GLM-5.2 RATE_TABLE). An intentional
# rate change must update this literal explicitly.
GOLDEN_COST_USD: float = 0.0003222
# Pin the table generation independently of the live PRICING_VERSION constant.
GOLDEN_PRICING_VERSION = "v8"

_FIXED_REVIEW_ID = UUID("00000000-0000-0000-0000-0000000000aa")


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
        model=GLM_MODEL_ID,
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


def _chat_completion_with_cache() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-golden",
        created=0,
        model="",  # Baseten echoes an empty model string.
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="ok"),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=30),
        ),
    )


@pytest.mark.asyncio
async def test_baseten_wire_golden_request_accounting_and_event() -> None:
    """Baseten request envelope (incl. the analyze `response_format`), §8a response
    accounting, AND the persisted `LLMCallEvent` cost/token/cache fields are frozen —
    the `OpenAICompatibleProvider(baseten)` rename must change none of them."""
    persister = _RecordingPersister()
    provider = GLMProvider(api_key=SecretStr("baseten-test-key"), persister=persister)
    mock = AsyncMock(return_value=_chat_completion_with_cache())
    try:
        with patch.object(AsyncCompletions, "create", mock):
            response = await provider.complete(_golden_request())
    finally:
        await provider.aclose()

    # 1) request wire shape — incl. the production analyze response_format envelope.
    assert mock.call_args.kwargs == GOLDEN_KWARGS, "Baseten request wire shape drifted"

    # 2) §8a accounting on the returned response.
    assert {
        "input_tokens": response.input_tokens,
        "cache_read_tokens": response.cache_read_tokens,
        "cache_write_tokens": response.cache_write_tokens,
        "output_tokens": response.output_tokens,
    } == GOLDEN_RESPONSE_ACCOUNTING, "Baseten §8a response accounting drifted"

    # 3) the persisted LLMCallEvent's cost/token/cache identity.
    assert len(persister.calls) == 1, "expected exactly one persisted LLMCallEvent"
    event, _req, _resp = persister.calls[0]
    assert {
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "cached_tokens": event.cached_tokens,
    } == GOLDEN_EVENT_TOKENS, "LLMCallEvent token accounting drifted"
    assert event.cache_hit is True, "cache_hit should be True for a nonzero cache hit"
    assert event.pricing_version == GOLDEN_PRICING_VERSION
    assert event.model == GLM_MODEL_ID
    # The literal also catches incorrect host/model rate lookup.
    assert event.cost_usd == pytest.approx(GOLDEN_COST_USD), "Baseten cost_usd drifted"


@pytest.mark.parametrize(
    ("host", "model", "effort", "digest", "cost"),
    [
        (
            "baseten",
            "zai-org/GLM-5.3",
            "low",
            "83d226389f0375698ec3bd014fed0924318b33ad4d8867eed516f4bad42b31ab",
            0.0003222,
        ),
        (
            "baseten",
            "zai-org/GLM-5.3-Flash",
            "low",
            "83d226389f0375698ec3bd014fed0924318b33ad4d8867eed516f4bad42b31ab",
            0.0000364,
        ),
        (
            "fireworks",
            "accounts/fireworks/models/glm-5p3",
            None,
            "e8afeb4b346b09ebb3a2e5bb937b7248447cb0a22ddaebec4b8c9dd75daa288e",
            0.0003258,
        ),
        (
            "fireworks",
            "accounts/fireworks/models/glm-5p3-flash",
            None,
            "e8afeb4b346b09ebb3a2e5bb937b7248447cb0a22ddaebec4b8c9dd75daa288e",
            0.0000364,
        ),
    ],
)
@pytest.mark.asyncio
async def test_current_glm_models_request_and_reasoning_accounting(
    host: str, model: str, effort: str | None, digest: str, cost: float
) -> None:
    request = _golden_request().model_copy(update={"model": model})
    expected = GOLDEN_KWARGS | {"model": model}
    del expected["extra_body"]
    if effort is not None:
        expected["reasoning_effort"] = effort
    profile = resolve_host_profile(host)
    assert _build_sdk_kwargs(request, profile=profile) == expected
    persister = _RecordingPersister()
    provider = OpenAICompatibleProvider(
        api_key=SecretStr("host-test-key"),
        profile=profile,
        persister=persister,
        models=(model,),
    )
    completion = _chat_completion_with_cache()
    assert completion.usage is not None
    completion.usage.completion_tokens_details = CompletionTokensDetails(reasoning_tokens=40)
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
    ) == (70, 30, 0, 50)
    event, _, persisted_response = persister.calls[0]
    assert persisted_response is response
    assert event.output_tokens == 50  # All completion tokens are charged, including reasoning.
    assert event.cost_usd == pytest.approx(cost)
    assert event.reasoning_enabled is True
    assert (response.profile_id, response.reasoning_enabled, response.profile_contract_digest) == (
        host,
        True,
        digest,
    )
    assert resolve_host_identity(host, reasoning=False, models=(model,)) == (host, True, digest)
    assert event.profile_contract_digest == digest
