"""OpenAICompatibleProvider tests — mirror of AnthropicProvider coverage for the
openai wire shape, exercised on `BASETEN_PROFILE` (+ a synthetic profile proving
the per-host axes are profile-driven).

Strategy: mock the SDK at `AsyncCompletions.create` so no real HTTP fires.
The focus is the wire deltas the wrapper must get right:
  - §8a: `input_tokens = prompt_tokens - cached_tokens` (Baseten prompt_tokens
    INCLUDES cached; Anthropic's input_tokens excludes it).
  - usage null-guards: the `*_details` objects (and `usage` itself) can be None.
  - request-side model keying: Baseten echoes `response.model=""`, so cost +
    the audit event key on `request.model`, never `response.model`.
  - no `cache_control` marker (automatic caching) → `cache_write_tokens=0`.
  - `response_format` envelope (name + strict) vs Anthropic's output_config.
  - openai → LLMProviderError mapping with the introspected isinstance order.
  - per-host axes (base_url, §8a mode, reasoning shaper) follow the HostProfile.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import openai
import pytest
from openai.resources.chat.completions.completions import AsyncCompletions
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
from pydantic import SecretStr

from outrider.audit.events import ContextManifestEntry, LLMCallEvent
from outrider.llm.base import (
    LLMAuthError,
    LLMConflictError,
    LLMInvalidRequestError,
    LLMInvalidResponseError,
    LLMMissingAPIKeyError,
    LLMPersisterError,
    LLMPersisterNotWiredError,
    LLMPricingMissingError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMUnexpectedContentBlocksError,
    LLMUnknownError,
    LLMUpstreamError,
)
from outrider.llm.host_profiles import (
    BASETEN_PROFILE,
    HostPrivacy,
    HostProfile,
    JsonMode,
    ReasoningMechanism,
    TokenAccounting,
)
from outrider.llm.openai_compatible_provider import (
    BASETEN_BASE_URL,
    GLM_MODEL_ID,
    GLMProvider,
    OpenAICompatibleProvider,
)
from outrider.llm.pricing import PRICING_VERSION

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

_UNSET = object()


def _entry() -> ContextManifestEntry:
    return ContextManifestEntry(
        file_path="src/foo.py",
        scope_unit_name="Foo.bar",
        line_start=1,
        line_end=10,
        inclusion_reason="changed_scope",
    )


def _request(**overrides: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "system_prompt": "You are a code reviewer.",
        "user_prompt": "Review this PR.",
        "model": GLM_MODEL_ID,
        "max_tokens": 100,
        "temperature": 0.0,
        "review_id": uuid4(),
        "node_id": "analyze",
        "prompt_template_version": "analyze@1.0.0",
        "degraded_mode": False,
        "context_summary": (_entry(),),
    }
    base.update(overrides)
    return LLMRequest(**base)


def _usage(
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cached: int | None = None,
    reasoning: int | None = None,
) -> CompletionUsage:
    """Build a CompletionUsage. `cached`/`reasoning` None → the *_details
    object is omitted (None), mirroring the GLM-5.2 non-streaming example
    where both details objects are null."""
    ptd = PromptTokensDetails(cached_tokens=cached) if cached is not None else None
    ctd = CompletionTokensDetails(reasoning_tokens=reasoning) if reasoning is not None else None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=ptd,
        completion_tokens_details=ctd,
    )


def _chat_completion(
    text: str = "model output",
    *,
    model: str = "",  # Baseten echoes empty string — the default reproduces that.
    finish_reason: str = "stop",
    usage: Any = _UNSET,
    choices: list[Any] | None = None,
    reasoning_content: str | None = None,
) -> ChatCompletion:
    """Realistic openai ChatCompletion. `model=""` by default reproduces
    Baseten's empty-model echo."""
    if choices is None:
        msg_kwargs: dict[str, Any] = {"role": "assistant", "content": text}
        if reasoning_content is not None:
            msg_kwargs["reasoning_content"] = reasoning_content
        choices = [
            Choice(
                index=0,
                finish_reason=finish_reason,  # type: ignore[arg-type]
                message=ChatCompletionMessage(**msg_kwargs),
            )
        ]
    return ChatCompletion(
        id="chatcmpl-test-001",
        created=0,
        model=model,
        object="chat.completion",
        choices=choices,
        usage=_usage() if usage is _UNSET else usage,
    )


@dataclass
class _RecordingPersister:
    """Captures persist() args so tests can inspect what the provider built."""

    raise_with: Exception | None = None
    calls: list[tuple[LLMCallEvent, LLMRequest, LLMResponse]] = field(default_factory=list)

    async def persist(
        self,
        event: LLMCallEvent,
        request: LLMRequest,
        response: LLMResponse,
    ) -> None:
        self.calls.append((event, request, response))
        if self.raise_with is not None:
            raise self.raise_with


def _api_key() -> SecretStr:
    return SecretStr("baseten-test-key")


def _provider(
    persister: _RecordingPersister | None = None, **kwargs: Any
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key=_api_key(),
        profile=BASETEN_PROFILE,
        persister=persister if persister is not None else _RecordingPersister(),
        models=(GLM_MODEL_ID,),
        **kwargs,
    )


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://inference.baseten.co/v1/chat/completions")


def _fake_response(status_code: int = 500) -> httpx.Response:
    return httpx.Response(status_code=status_code, request=_fake_request())


@contextmanager
def _patched_create(
    return_value: ChatCompletion | None = None,
    raise_with: Exception | None = None,
) -> Iterator[AsyncMock]:
    """Patch `AsyncCompletions.create` to return `return_value` OR raise
    `raise_with`. Yields the mock so tests can inspect call kwargs."""
    if return_value is not None and raise_with is not None:
        raise ValueError("specify either return_value OR raise_with")
    mock = AsyncMock()
    if raise_with is not None:
        mock.side_effect = raise_with
    else:
        mock.return_value = return_value if return_value is not None else _chat_completion()
    with patch.object(AsyncCompletions, "create", mock):
        yield mock


# ---------------------------------------------------------------------------
# Constructor — eager validation.
# ---------------------------------------------------------------------------


def test_constructor_empty_api_key_raises_missing() -> None:
    with pytest.raises(LLMMissingAPIKeyError):
        GLMProvider(api_key=SecretStr(""), persister=_RecordingPersister())


def test_constructor_unknown_model_raises_pricing_missing() -> None:
    with pytest.raises(LLMPricingMissingError, match="zai-org/GLM-9.9"):
        GLMProvider(
            api_key=_api_key(),
            persister=_RecordingPersister(),
            models=("zai-org/GLM-9.9",),
        )


def test_constructor_non_glm_model_raises() -> None:
    """A priced-but-non-GLM model (the Anthropic models are in RATE_TABLE) is
    rejected at construction — pricing coverage is NOT 'servable by GLM'. Closes
    the hole where a claude-* slug could be configured and then routed to Baseten
    by the per-call guard."""
    with pytest.raises(LLMInvalidRequestError, match="does not match host 'baseten'"):
        GLMProvider(
            api_key=_api_key(),
            persister=_RecordingPersister(),
            models=("claude-sonnet-4-6",),
        )


def test_constructor_default_succeeds() -> None:
    assert _provider() is not None


def test_constructor_disables_sdk_retry_and_points_at_baseten() -> None:
    provider = _provider()
    assert provider._client.max_retries == 0  # noqa: SLF001
    assert str(provider._client.base_url).rstrip("/") == BASETEN_BASE_URL  # noqa: SLF001
    timeout = provider._client.timeout  # noqa: SLF001
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 300.0
    assert timeout.connect == 5.0


def test_constructor_repr_does_not_leak_api_key() -> None:
    secret = "baseten-VERY-SECRET-xyz-987"  # noqa: S105 — test fixture, not a real secret
    provider = GLMProvider(api_key=SecretStr(secret), persister=_RecordingPersister())
    assert secret not in repr(provider)


def test_constructor_emits_egress_notice(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="outrider.llm.privacy_notice")
    _provider()
    notices = [r for r in caplog.records if r.name == "outrider.llm.privacy_notice"]
    notice = next(
        r for r in notices if getattr(r, "egress_destination", None) == "inference.baseten.co"
    )
    # The notice is the FULL auditable claim: no-training posture + provenance, not just egress.
    assert getattr(notice, "trains_on_inputs", None) is False
    assert getattr(notice, "source_url", "").startswith("https://")
    assert getattr(notice, "verified_date", None) == "2026-06-27"


# ---------------------------------------------------------------------------
# complete() — fail-closed pre-call.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_persister_none_fails_closed_before_sdk() -> None:
    provider = GLMProvider(api_key=_api_key(), persister=None)
    with _patched_create() as mock, pytest.raises(LLMPersisterNotWiredError):
        await provider.complete(_request())
    assert mock.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_model",
    [
        "zai-org/GLM-9.9",  # unpriced AND unconfigured
        "claude-sonnet-4-6",  # PRICED (in RATE_TABLE) but not a GLM model this provider serves
    ],
)
async def test_unconfigured_request_model_refused_before_paid_sdk_call(bad_model: str) -> None:
    """A request.model not in the provider's configured set is refused BEFORE the
    paid SDK call — no billed call, no orphan cost. Being in RATE_TABLE is NOT
    enough: the Anthropic models are priced too, so a claude-* slug must not reach
    the Baseten endpoint."""
    provider = _provider()  # configured for (GLM_MODEL_ID,)
    with (
        _patched_create() as mock,
        pytest.raises(LLMInvalidRequestError, match="configured model set"),
    ):
        await provider.complete(_request(model=bad_model))
    assert mock.call_count == 0


# ---------------------------------------------------------------------------
# complete() — request translation (openai envelope).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_messages_are_system_then_user() -> None:
    provider = _provider()
    with _patched_create() as mock:
        await provider.complete(_request(system_prompt="SYS", user_prompt="USR"))
    messages = mock.call_args.kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


@pytest.mark.asyncio
async def test_request_passes_model_max_tokens_temperature() -> None:
    provider = _provider()
    with _patched_create() as mock:
        await provider.complete(_request(max_tokens=77, temperature=0.3))
    kwargs = mock.call_args.kwargs
    assert kwargs["model"] == GLM_MODEL_ID
    assert kwargs["max_tokens"] == 77
    assert kwargs["temperature"] == 0.3


@pytest.mark.asyncio
async def test_reasoning_off_by_default_via_extra_body() -> None:
    provider = _provider()
    with _patched_create() as mock:
        await provider.complete(_request())
    extra_body = mock.call_args.kwargs["extra_body"]
    assert extra_body == {"chat_template_args": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_no_cache_control_marker_and_stream_omitted() -> None:
    """GLM uses automatic prefix caching — no cache_control anywhere — and
    the non-streaming path omits `stream`."""
    provider = _provider()
    with _patched_create() as mock:
        await provider.complete(_request(cache_control=True))
    kwargs = mock.call_args.kwargs
    assert "cache_control" not in kwargs
    assert "stream" not in kwargs


@pytest.mark.asyncio
async def test_response_schema_translates_to_response_format() -> None:
    import json as _json

    provider = _provider()
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    with _patched_create() as mock:
        await provider.complete(
            _request(response_schema_json=_json.dumps(schema, separators=(",", ":")))
        )
    rf = mock.call_args.kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "outrider_analyze"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == schema


@pytest.mark.asyncio
async def test_no_response_schema_omits_response_format() -> None:
    provider = _provider()
    with _patched_create() as mock:
        await provider.complete(_request())
    assert "response_format" not in mock.call_args.kwargs


# ---------------------------------------------------------------------------
# complete() — §8a usage normalization + null guards.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cached_subtracted_from_prompt_tokens() -> None:
    """§8a: input_tokens = prompt_tokens - cached; cache_read = cached;
    cache_write = 0; output = completion_tokens."""
    provider = _provider()
    usage = _usage(prompt_tokens=1000, completion_tokens=500, cached=200)
    with _patched_create(return_value=_chat_completion(usage=usage)):
        resp = await provider.complete(_request())
    assert resp.input_tokens == 800
    assert resp.cache_read_tokens == 200
    assert resp.cache_write_tokens == 0
    assert resp.output_tokens == 500


@pytest.mark.asyncio
async def test_prompt_tokens_details_none_means_zero_cached() -> None:
    """The whole prompt_tokens_details object can be null (GLM-5.2 example
    shows it). cached → 0, input_tokens == prompt_tokens."""
    provider = _provider()
    usage = _usage(prompt_tokens=1000, completion_tokens=500, cached=None)
    assert usage.prompt_tokens_details is None
    with _patched_create(return_value=_chat_completion(usage=usage)):
        resp = await provider.complete(_request())
    assert resp.input_tokens == 1000
    assert resp.cache_read_tokens == 0


@pytest.mark.asyncio
async def test_cached_greater_than_prompt_caps_cache_read_to_prompt() -> None:
    """Defensive: a malformed cached > prompt is capped at prompt_tokens, so
    input_tokens stays 0 AND cache_read never exceeds the prompt — input +
    cache_read == prompt_tokens (self-consistent audited counts, no over-bill)."""
    provider = _provider()
    usage = _usage(prompt_tokens=100, completion_tokens=10, cached=250)
    with _patched_create(return_value=_chat_completion(usage=usage)):
        resp = await provider.complete(_request())
    assert resp.input_tokens == 0
    assert resp.cache_read_tokens == 100  # capped at prompt_tokens, not 250
    assert resp.input_tokens + resp.cache_read_tokens == 100


@pytest.mark.asyncio
async def test_usage_none_raises_invalid_response() -> None:
    provider = _provider()
    with (
        _patched_create(return_value=_chat_completion(usage=None)),
        pytest.raises(LLMInvalidResponseError),
    ):
        await provider.complete(_request())


# ---------------------------------------------------------------------------
# complete() — request-side model keying (response.model is "").
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_model_empty_keys_on_request_model() -> None:
    """Baseten echoes response.model="" — LLMResponse.model, cost, and the
    audit event must all key on request.model (the GLM slug), not the empty
    echo."""
    persister = _RecordingPersister()
    provider = _provider(persister)
    with _patched_create(return_value=_chat_completion(model="")):
        resp = await provider.complete(_request())
    assert resp.model == GLM_MODEL_ID
    event, _, _ = persister.calls[0]
    assert event.model == GLM_MODEL_ID
    assert event.cost_usd > 0  # cost computed from the request-keyed rate
    assert event.pricing_version == PRICING_VERSION


# ---------------------------------------------------------------------------
# complete() — content extraction + reasoning strip.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_content_degrades_not_raises() -> None:
    """Empty content (truncation) is coalesced to "" and returned with the
    normalized finish_reason — the downstream parser degrades that file (matching
    AnthropicProvider's empty-block path) rather than aborting the whole review
    with a non-retryable error. "length"→"max_tokens" is what lets the analyze
    truncation diagnostic fire downstream."""
    provider = _provider()
    with _patched_create(return_value=_chat_completion(text="", finish_reason="length")):
        resp = await provider.complete(_request())
    assert resp.text == ""
    assert resp.finish_reason == "max_tokens"


@pytest.mark.asyncio
async def test_none_content_degrades_not_raises() -> None:
    """message.content=None (a structured refusal / content_filter) coalesces to
    "" too — degrade, never raise."""
    provider = _provider()
    none_choice = [
        Choice(
            index=0,
            finish_reason="content_filter",
            message=ChatCompletionMessage(role="assistant", content=None),
        )
    ]
    with _patched_create(return_value=_chat_completion(choices=none_choice)):
        resp = await provider.complete(_request())
    assert resp.text == ""
    assert resp.finish_reason == "refusal"


@pytest.mark.asyncio
async def test_multiple_choices_raises_unexpected_blocks() -> None:
    provider = _provider()
    two = [
        Choice(
            index=0,
            finish_reason="stop",
            message=ChatCompletionMessage(role="assistant", content="a"),
        ),
        Choice(
            index=1,
            finish_reason="stop",
            message=ChatCompletionMessage(role="assistant", content="b"),
        ),
    ]
    with (
        _patched_create(return_value=_chat_completion(choices=two)),
        pytest.raises(LLMUnexpectedContentBlocksError),
    ):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_reasoning_content_is_stripped_from_text() -> None:
    """When reasoning is on, only message.content becomes LLMResponse.text;
    message.reasoning_content is never concatenated in."""
    provider = _provider()
    cc = _chat_completion(text="final answer", reasoning_content="chain of thought")
    with _patched_create(return_value=cc):
        resp = await provider.complete(_request())
    assert resp.text == "final answer"
    assert "chain of thought" not in resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire_value,canonical",
    [
        ("length", "max_tokens"),  # load-bearing: the truncation sentinel
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("content_filter", "refusal"),
    ],
)
async def test_finish_reason_normalized_to_canonical_vocab(wire_value: str, canonical: str) -> None:
    """The wrapper normalizes openai's finish_reason to Outrider's canonical
    (Anthropic) vocabulary so the downstream truncation guard + analyze
    cache-write gate (which key on "max_tokens") fire provider-neutrally.
    openai "length" MUST become "max_tokens", or a truncated GLM analyze
    response is silently cached and served incomplete."""
    provider = _provider()
    with _patched_create(return_value=_chat_completion(finish_reason=wire_value)):
        resp = await provider.complete(_request())
    assert resp.finish_reason == canonical


def test_normalize_finish_reason_unit() -> None:
    """Direct unit on the normalizer: mapping, None/empty → 'unknown', and an
    unmapped (future) value passes through unmasked."""
    from outrider.llm.openai_compatible_provider import _normalize_finish_reason

    assert _normalize_finish_reason("length") == "max_tokens"
    assert _normalize_finish_reason("novel_future_value") == "novel_future_value"
    assert _normalize_finish_reason(None) == "unknown"
    assert _normalize_finish_reason("") == "unknown"


# ---------------------------------------------------------------------------
# complete() — openai exception mapping (isinstance order verified vs SDK MRO).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sdk_exc_factory,expected",
    [
        (lambda: openai.APITimeoutError(request=_fake_request()), LLMTimeoutError),
        (lambda: openai.APIConnectionError(request=_fake_request()), LLMUpstreamError),
        # 408 has no dedicated openai subclass (bare APIStatusError); must map to a
        # retryable timeout, not fall through to LLMUnknownError (terminal).
        (
            lambda: openai.APIStatusError("rt", response=_fake_response(408), body=None),
            LLMTimeoutError,
        ),
        (
            lambda: openai.RateLimitError("rl", response=_fake_response(429), body=None),
            LLMRateLimitError,
        ),
        (
            lambda: openai.AuthenticationError("auth", response=_fake_response(401), body=None),
            LLMAuthError,
        ),
        (
            lambda: openai.PermissionDeniedError("perm", response=_fake_response(403), body=None),
            LLMAuthError,
        ),
        (
            lambda: openai.ConflictError("conflict", response=_fake_response(409), body=None),
            LLMConflictError,
        ),
        (
            lambda: openai.BadRequestError("bad", response=_fake_response(400), body=None),
            LLMInvalidRequestError,
        ),
        (
            lambda: openai.UnprocessableEntityError("unp", response=_fake_response(422), body=None),
            LLMInvalidRequestError,
        ),
        (
            lambda: openai.NotFoundError("nf", response=_fake_response(404), body=None),
            LLMInvalidRequestError,
        ),
        (
            lambda: openai.InternalServerError("5xx", response=_fake_response(500), body=None),
            LLMUpstreamError,
        ),
        (
            lambda: openai.APIResponseValidationError(response=_fake_response(200), body=None),
            LLMInvalidResponseError,
        ),
    ],
)
async def test_openai_exception_translation(
    sdk_exc_factory: Any, expected: type[LLMProviderError]
) -> None:
    provider = _provider()
    with _patched_create(raise_with=sdk_exc_factory()), pytest.raises(expected):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_timeout_wins_over_connection_error() -> None:
    """APITimeoutError IS a subclass of APIConnectionError (openai==2.44.0);
    the ladder must test it first → LLMTimeoutError, not LLMUpstreamError."""
    assert issubclass(openai.APITimeoutError, openai.APIConnectionError)
    provider = _provider()
    with (
        _patched_create(raise_with=openai.APITimeoutError(request=_fake_request())),
        pytest.raises(LLMTimeoutError),
    ):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_unmapped_openai_error_translates_to_unknown() -> None:
    provider = _provider()

    class _UnknownAPIError(openai.APIError):
        def __init__(self) -> None:
            super().__init__("weird new error", request=_fake_request(), body=None)

    with _patched_create(raise_with=_UnknownAPIError()), pytest.raises(LLMUnknownError):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_sdk_error_body_not_leaked_into_wrapper() -> None:
    """openai error str() renders the response body, which can echo prompt
    fragments. The wrapper must not carry it; `from None` drops the cause."""
    provider = _provider()
    sentinel = "secret_prompt_fragment_zzz9876"  # noqa: S105 — test fixture
    sdk_exc = openai.RateLimitError(sentinel, response=_fake_response(429), body=None)
    assert sentinel in str(sdk_exc)
    with _patched_create(raise_with=sdk_exc), pytest.raises(LLMRateLimitError) as exc_info:
        await provider.complete(_request())
    wrapper = exc_info.value
    assert sentinel not in str(wrapper)
    assert sentinel not in repr(wrapper)
    for arg in wrapper.args:
        assert sentinel not in str(arg)
    assert wrapper.__cause__ is None
    assert wrapper.__suppress_context__ is True


# ---------------------------------------------------------------------------
# complete() — persister contract.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persister_receives_complete_event() -> None:
    persister = _RecordingPersister()
    provider = _provider(persister)
    usage = _usage(prompt_tokens=1000, completion_tokens=500, cached=200)
    with _patched_create(return_value=_chat_completion(usage=usage)):
        await provider.complete(_request())
    assert len(persister.calls) == 1
    event, request, response = persister.calls[0]
    assert event.cost_usd > 0
    assert event.input_tokens == 800
    assert event.output_tokens == 500
    assert event.cached_tokens == 200
    assert event.cache_hit is True
    assert event.model == GLM_MODEL_ID


@pytest.mark.asyncio
async def test_no_cache_hit_when_cached_zero() -> None:
    persister = _RecordingPersister()
    provider = _provider(persister)
    usage = _usage(prompt_tokens=1000, completion_tokens=500, cached=None)
    with _patched_create(return_value=_chat_completion(usage=usage)):
        await provider.complete(_request())
    event, _, _ = persister.calls[0]
    assert event.cache_hit is False
    assert event.cached_tokens == 0


# ---------------------------------------------------------------------------
# aclose() — idempotency.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_is_idempotent() -> None:
    provider = _provider()
    await provider.aclose()
    await provider.aclose()  # second call is a no-op via the _closed guard
    assert provider._closed is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_complete_after_aclose_raises() -> None:
    provider = _provider()
    await provider.aclose()
    with pytest.raises(LLMUnknownError, match="closed"):
        await provider.complete(_request())


# ---------------------------------------------------------------------------
# complete() — persister-failure handling (metadata-only discipline).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persister_failure_wraps_as_persister_error() -> None:
    """Post-SDK persistence failure → LLMPersisterError (terminal); the SDK
    call had already succeeded."""
    persister = _RecordingPersister(raise_with=RuntimeError("DB blip"))
    provider = _provider(persister)
    with _patched_create(), pytest.raises(LLMPersisterError):
        await provider.complete(_request())
    assert len(persister.calls) == 1


@pytest.mark.asyncio
async def test_persister_unknown_exception_drops_cause_chain() -> None:
    """An unknown persister exception type is wrapped with `from None` and
    rendered as `<TypeName>` only — no content-bearing repr leaks past the
    wrapper, and the cause chain (incl. the rendered traceback) is dropped."""
    import traceback

    secret = "SECRET_LEAK_SENTINEL_xyz"  # noqa: S105 — test fixture
    persister = _RecordingPersister(raise_with=ValueError(secret))
    provider = _provider(persister)
    with _patched_create(), pytest.raises(LLMPersisterError) as exc_info:
        await provider.complete(_request())
    exc = exc_info.value
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
    assert "<ValueError>" in str(exc)
    assert secret not in str(exc)
    assert secret not in repr(exc)
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert secret not in rendered


@pytest.mark.asyncio
async def test_persister_metadata_only_exception_preserves_cause_chain() -> None:
    """A METADATA_ONLY_EXCEPTION_TYPES member preserves the cause chain via
    `from exc` (the chain is metadata-only by contract) and renders str(exc)."""
    from outrider.audit.persister import AuditPersisterIdempotencyConflict, FieldDigest

    conflict = AuditPersisterIdempotencyConflict(
        event_id=uuid4(),
        mismatched_fields=("cost_usd",),
        field_digests={"cost_usd": FieldDigest("a" * 64, "b" * 64, 10, 12)},
    )
    persister = _RecordingPersister(raise_with=conflict)
    provider = _provider(persister)
    with _patched_create(), pytest.raises(LLMPersisterError) as exc_info:
        await provider.complete(_request())
    exc = exc_info.value
    assert exc.__cause__ is conflict
    assert "cost_usd" in str(exc)


# ---------------------------------------------------------------------------
# complete() — non-openai exception + close-race (best-effort _closed guard).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_openai_exception_translates_to_unknown() -> None:
    """A non-openai exception leaking from create() (e.g. an httpx error)
    translates to LLMUnknownError with the type name only — no args leak —
    and the cause chain is dropped."""
    provider = _provider()
    boom = RuntimeError("httpx_internal_url_secret")
    with _patched_create(raise_with=boom), pytest.raises(LLMUnknownError) as exc_info:
        await provider.complete(_request())
    exc = exc_info.value
    assert "RuntimeError" in str(exc)
    assert "httpx_internal_url_secret" not in str(exc)
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True


@pytest.mark.asyncio
async def test_close_race_translates_to_unknown_with_aclose_message() -> None:
    """If aclose() flips _closed between the Step-0 check and the SDK call,
    the non-openai branch names the close-race specifically."""
    provider = _provider()

    def _flip_then_raise(*_args: object, **_kwargs: object) -> None:
        provider._closed = True  # noqa: SLF001
        raise RuntimeError("client has been closed")

    mock = AsyncMock(side_effect=_flip_then_raise)
    with (
        patch.object(AsyncCompletions, "create", mock),
        pytest.raises(LLMUnknownError, match="raced with aclose"),
    ):
        await provider.complete(_request())


# ---------------------------------------------------------------------------
# complete() — reasoning tokens are never double-counted into output/cost.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_tokens_not_added_to_output_or_cost() -> None:
    """reasoning_tokens are a SUBSET of completion_tokens — output_tokens and
    cost must equal the completion-token count, never completion + reasoning."""
    persister = _RecordingPersister()
    provider = _provider(persister)
    usage = _usage(prompt_tokens=1000, completion_tokens=500, reasoning=120)
    with _patched_create(return_value=_chat_completion(usage=usage)):
        resp = await provider.complete(_request())
    assert resp.output_tokens == 500  # NOT 620
    event, _, _ = persister.calls[0]
    assert event.output_tokens == 500
    from outrider.llm.pricing import RATE_TABLE, pricing_key

    glm = RATE_TABLE[pricing_key("baseten", GLM_MODEL_ID)]
    # input=1000 (cached=0), output=500; cost must use output=500, not 620.
    expected = float(glm.in_per_token * 1000 + glm.out_per_token * 500)
    assert abs(event.cost_usd - expected) < 1e-12


# ---------------------------------------------------------------------------
# Per-host axes are profile-driven (the #056 two-hosts-one-slug separation).
# ---------------------------------------------------------------------------


def _synthetic_profile() -> HostProfile:
    """A second profile serving the SAME priced slug, but with EXCLUDES-cached
    accounting, a different base_url, and the reasoning_effort_none shaper — proves
    the per-host *wire axes* (base_url, §8a mode, reasoning shaper) are profile-driven.
    host_id stays "baseten" here only so the (profile_id, model) pricing key resolves
    against the production RATE_TABLE without injection; the two-host cost + audit
    SEPARATION (#056's distinct-host claim) is proven separately in
    test_two_hosts_one_slug_get_separate_cost_and_audit, which injects a synthetic
    host_id + rate (the production table never carries synthetic rows)."""
    return HostProfile(
        host_id="baseten",
        base_url="https://synthetic.example/v1",
        api_key_env="SYNTHETIC_API_KEY",
        model_slug_pattern=r"^zai-org/GLM-\d+(\.\d+)?$",
        json_mode=JsonMode.STRICT_JSON_SCHEMA,
        token_accounting=TokenAccounting.PROMPT_EXCLUDES_CACHED,
        reasoning_mechanism=ReasoningMechanism.REASONING_EFFORT_NONE,
        privacy=HostPrivacy(
            egress_host="synthetic.example",
            model_origin="synthetic",
            direct_hosted=True,
            trains_on_inputs=False,
            retention="test profile — no real host",
            source_url="https://synthetic.example/security",
            verified_date="2026-06-27",
        ),
    )


def _synthetic_provider(persister: _RecordingPersister | None = None) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key=_api_key(),
        profile=_synthetic_profile(),
        persister=persister if persister is not None else _RecordingPersister(),
        models=(GLM_MODEL_ID,),
    )


def test_constructor_base_url_follows_profile() -> None:
    provider = _synthetic_provider()
    assert str(provider._client.base_url).rstrip("/") == "https://synthetic.example/v1"  # noqa: SLF001


@pytest.mark.asyncio
async def test_excludes_cached_profile_does_not_subtract() -> None:
    """A PROMPT_EXCLUDES_CACHED host treats prompt_tokens as already-uncached: the
    same usage Baseten subtracts to input=800 passes through to input=1000 here.
    Proves §8a is profile-driven, not a Baseten constant."""
    provider = _synthetic_provider()
    usage = _usage(prompt_tokens=1000, completion_tokens=500, cached=200)
    with _patched_create(return_value=_chat_completion(usage=usage)):
        resp = await provider.complete(_request())
    assert resp.input_tokens == 1000  # NOT 800 — excludes-cached passes through
    assert resp.cache_read_tokens == 200


@pytest.mark.asyncio
async def test_reasoning_shaper_follows_profile() -> None:
    """The reasoning-off wire is the host's shaper: reasoning_effort_none emits a
    top-level `reasoning_effort="none"`, not Baseten's extra_body.chat_template_args."""
    provider = _synthetic_provider()
    with _patched_create() as mock:
        await provider.complete(_request())
    kwargs = mock.call_args.kwargs
    assert kwargs["reasoning_effort"] == "none"
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_trains_on_inputs_profile_fails_closed() -> None:
    """DECISIONS.md#056: a host declaring trains_on_inputs=True is a construction
    hard-fail — no blanket override."""
    base = _synthetic_profile()
    training = base.model_copy(
        update={"privacy": base.privacy.model_copy(update={"trains_on_inputs": True})}
    )
    with pytest.raises(LLMInvalidRequestError, match="trains_on_inputs"):
        OpenAICompatibleProvider(
            api_key=_api_key(),
            profile=training,
            persister=_RecordingPersister(),
            models=(GLM_MODEL_ID,),
        )


def test_json_object_profile_rejected_at_construction() -> None:
    """A JSON_OBJECT host needs a `response_format` wire that isn't built yet — construction
    fails closed rather than send a json_schema envelope to it (DECISIONS.md#056)."""
    json_object_host = _synthetic_profile().model_copy(update={"json_mode": JsonMode.JSON_OBJECT})
    with pytest.raises(LLMInvalidRequestError, match="json_mode='json_object'"):
        OpenAICompatibleProvider(
            api_key=_api_key(),
            profile=json_object_host,
            persister=_RecordingPersister(),
            models=(GLM_MODEL_ID,),
        )


def test_glm_alias_binds_baseten_profile() -> None:
    """The transitional GLMProvider alias constructs via BASETEN_PROFILE; the wire
    golden pins byte-equivalence with the pre-rename spike."""
    provider = GLMProvider(api_key=_api_key(), persister=_RecordingPersister())
    assert isinstance(provider, OpenAICompatibleProvider)
    assert str(provider._client.base_url).rstrip("/") == BASETEN_BASE_URL  # noqa: SLF001


@pytest.mark.asyncio
async def test_two_hosts_one_slug_get_separate_cost_and_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DECISIONS.md#056 two-host separation: the SAME slug served by two hosts bills
    at two rates and audits under two profile_ids. A test-injected synthetic host is
    priced distinctly from Baseten on `GLM_MODEL_ID` (the production RATE_TABLE never
    carries synthetic rows — #056), proving host-qualified (profile_id, model) COST +
    AUDIT identity, not merely profile-driven wire axes. Cache separation by profile_id
    is pinned in test_analyze_cache_key's per-component sensitivity."""
    from decimal import Decimal
    from types import MappingProxyType

    import outrider.llm.openai_compatible_provider as provider_mod
    import outrider.llm.pricing as pricing_mod
    from outrider.llm.pricing import ModelPricing

    # A synthetic host on the SAME slug, priced exactly 2x Baseten (test-only
    # injection — patched into BOTH the pricing module and the provider's imported
    # reference so the construction coverage check and compute_cost_usd agree).
    synthetic_rates = ModelPricing(
        in_per_token=Decimal("0.0000028"),  # 2x Baseten's 1.40/MTok
        cache_write_per_token=Decimal("0"),
        cache_read_per_token=Decimal("0.00000052"),
        out_per_token=Decimal("0.0000088"),  # 2x Baseten's 4.40/MTok
    )
    patched = MappingProxyType(
        {**pricing_mod.RATE_TABLE, ("synthetic", GLM_MODEL_ID): synthetic_rates}
    )
    monkeypatch.setattr(pricing_mod, "RATE_TABLE", patched)
    monkeypatch.setattr(provider_mod, "RATE_TABLE", patched)

    synthetic_host = _synthetic_profile().model_copy(update={"host_id": "synthetic"})
    baseten_persister = _RecordingPersister()
    synthetic_persister = _RecordingPersister()
    baseten = _provider(baseten_persister)
    synthetic = OpenAICompatibleProvider(
        api_key=_api_key(),
        profile=synthetic_host,
        persister=synthetic_persister,
        models=(GLM_MODEL_ID,),
    )
    # cached=0 neutralizes the §8a accounting difference, so cost differs by RATE only.
    usage = _usage(prompt_tokens=1000, completion_tokens=500, cached=0)
    with _patched_create(return_value=_chat_completion(usage=usage)):
        await baseten.complete(_request())
    with _patched_create(return_value=_chat_completion(usage=usage)):
        await synthetic.complete(_request())

    (b_event, _, _) = baseten_persister.calls[0]
    (s_event, _, _) = synthetic_persister.calls[0]
    # Same slug, two hosts → distinct COST (the (profile_id, model) pricing key)...
    assert b_event.model == s_event.model == GLM_MODEL_ID
    assert s_event.cost_usd != b_event.cost_usd
    assert s_event.cost_usd == pytest.approx(b_event.cost_usd * 2, rel=1e-9)
    # ...and distinct AUDIT identity (the triad's profile_id).
    assert b_event.profile_id == "baseten"
    assert s_event.profile_id == "synthetic"


# ---------------------------------------------------------------------------
# Identity-triad stamping (DECISIONS.md#056, step 4).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_stamps_triad_on_response_and_event() -> None:
    """The provider stamps `(profile_id, reasoning_enabled, profile_contract_digest)` from the
    profile on BOTH the LLMResponse and the persisted LLMCallEvent (event mirrors response)."""
    persister = _RecordingPersister()
    provider = _provider(persister)
    with _patched_create():
        resp = await provider.complete(_request())
    assert resp.profile_id == "baseten"
    assert resp.reasoning_enabled is False
    assert resp.profile_contract_digest == BASETEN_PROFILE.profile_contract_digest
    event, _req, event_resp = persister.calls[0]
    assert (event.profile_id, event.reasoning_enabled, event.profile_contract_digest) == (
        "baseten",
        False,
        BASETEN_PROFILE.profile_contract_digest,
    )
    # Event triad is sourced from the response (single source) — they match exactly.
    assert event.profile_id == event_resp.profile_id
    assert event.profile_contract_digest == event_resp.profile_contract_digest


def test_reasoning_true_on_off_switch_host_fails_closed() -> None:
    """V1 has no verified reasoning-ON wire for an off-switch host (Baseten=CHAT_TEMPLATE_ARGS):
    requesting it fails closed rather than stamp reasoning_enabled=True while the wire still
    sends the off directive (a lie the persister cross-check can't catch)."""
    with pytest.raises(LLMInvalidRequestError, match="no verified reasoning-ON wire"):
        OpenAICompatibleProvider(
            api_key=_api_key(),
            profile=BASETEN_PROFILE,
            persister=_RecordingPersister(),
            models=(GLM_MODEL_ID,),
            reasoning=True,
        )


@pytest.mark.asyncio
async def test_none_host_reasoning_on_wire_matches_stamp() -> None:
    """A NONE-mechanism host forces reasoning on: the wire sends NO reasoning-off directive
    (apply_reasoning_off is a no-op) AND the stamp is reasoning_enabled=True — wire matches
    stamp, so the triad never claims an on state the request contradicts."""
    none_profile = _synthetic_profile().model_copy(
        update={"reasoning_mechanism": ReasoningMechanism.NONE}
    )
    provider = OpenAICompatibleProvider(
        api_key=_api_key(),
        profile=none_profile,
        persister=_RecordingPersister(),
        models=(GLM_MODEL_ID,),
    )
    with _patched_create() as mock:
        resp = await provider.complete(_request())
    kwargs = mock.call_args.kwargs
    assert "extra_body" not in kwargs  # no chat_template_args off directive
    assert "reasoning_effort" not in kwargs
    assert resp.reasoning_enabled is True


def test_none_mechanism_host_forces_reasoning_enabled_true() -> None:
    """A host with no off-switch (reasoning_forced_on) stamps reasoning_enabled=True even when
    reasoning is NOT requested — the audit never claims a silent off."""
    none_profile = _synthetic_profile().model_copy(
        update={"reasoning_mechanism": ReasoningMechanism.NONE}
    )
    provider = OpenAICompatibleProvider(
        api_key=_api_key(),
        profile=none_profile,
        persister=_RecordingPersister(),
        models=(GLM_MODEL_ID,),
        reasoning=False,
    )
    assert provider._reasoning_enabled is True  # noqa: SLF001
