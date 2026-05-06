"""AnthropicProvider tests — covers AC#3, #4, #5, #6, #9, #10, #11, #12,
#13, #14, #18, #19, #20, #24.

Strategy: mock the SDK client at the AsyncAnthropic level so we don't
issue real HTTP calls. The wrapper's contract — translation, mapping,
fail-closed, error taxonomy, persister contract, audit-event population —
is testable without a live Anthropic endpoint.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import anthropic
import httpx
import pytest
from anthropic.types import (
    Message,
    TextBlock,
    Usage,
)
from pydantic import SecretStr

from outrider.audit.events import ContextManifestEntry, LLMCallEvent
from outrider.llm import (
    PRICING_VERSION,
    AnthropicProvider,
    LLMAuthError,
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
    ModelConfig,
)

# ---------------------------------------------------------------------------
# Test fixtures.
# ---------------------------------------------------------------------------


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
        "model": "claude-sonnet-4-7",
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


def _sdk_message(
    text: str = "model output",
    *,
    model: str = "claude-sonnet-4-7",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    stop_reason: str = "end_turn",
    content_blocks: list[Any] | None = None,
) -> Message:
    """Construct a realistic SDK Message instance for mocking."""
    if content_blocks is None:
        content_blocks = [TextBlock(citations=None, text=text, type="text")]
    return Message(
        id="msg_test_001",
        content=content_blocks,
        model=model,
        role="assistant",
        type="message",
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )


@dataclass
class _RecordingPersister:
    """Captures the exact arguments passed to persist() so tests can
    inspect what the provider built."""

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


def _model_config() -> ModelConfig:
    return ModelConfig()


def _api_key() -> SecretStr:
    return SecretStr("sk-test-key")


@contextmanager
def _patched_create(
    return_value: Message | None = None,
    raise_with: Exception | None = None,
) -> Iterator[AsyncMock]:
    """Patch `AsyncAnthropic.messages.create` to return `return_value`
    OR raise `raise_with`. Yields the mock so tests can inspect call args."""
    if return_value is not None and raise_with is not None:
        raise ValueError("specify either return_value OR raise_with")
    mock = AsyncMock(spec=lambda **kw: None)
    if raise_with is not None:
        mock.side_effect = raise_with
    else:
        mock.return_value = return_value if return_value is not None else _sdk_message()
    with patch.object(
        anthropic.resources.messages.AsyncMessages,
        "create",
        mock,
    ):
        yield mock


# ---------------------------------------------------------------------------
# Constructor — eager validation.
# ---------------------------------------------------------------------------


def test_constructor_with_empty_api_key_raises_missing_api_key() -> None:
    """AC#13: eager api_key validation."""
    with pytest.raises(LLMMissingAPIKeyError):
        AnthropicProvider(
            api_key=SecretStr(""),
            model_config=_model_config(),
            persister=_RecordingPersister(),
        )


def test_constructor_with_unknown_model_raises_pricing_missing() -> None:
    """AC#24: eager pricing-coverage validation."""
    cfg = ModelConfig(triage_model="claude-haiku-99-99")  # not in RATE_TABLE
    with pytest.raises(LLMPricingMissingError, match="claude-haiku-99-99"):
        AnthropicProvider(
            api_key=_api_key(),
            model_config=cfg,
            persister=_RecordingPersister(),
        )


def test_constructor_with_default_model_config_succeeds() -> None:
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=_RecordingPersister(),
    )
    assert provider is not None


def test_constructor_emits_privacy_notice(caplog: pytest.LogCaptureFixture) -> None:
    """AC#3: startup notice on `outrider.llm.privacy_notice` logger."""
    caplog.set_level(logging.INFO, logger="outrider.llm.privacy_notice")
    AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=_RecordingPersister(),
    )
    notice_records = [r for r in caplog.records if r.name == "outrider.llm.privacy_notice"]
    assert len(notice_records) >= 1
    assert any(getattr(r, "privacy_notice", False) is True for r in notice_records)


def test_constructor_zdr_kwarg_overrides_env() -> None:
    """AC#4: ZDR is operator attestation. Constructor kwarg wins; env var
    is the fallback."""
    saved = os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
    try:
        os.environ["ANTHROPIC_ZDR_ENABLED"] = "true"
        provider = AnthropicProvider(
            api_key=_api_key(),
            model_config=_model_config(),
            persister=_RecordingPersister(),
            zdr_enabled=False,
        )
        # Reach into the private attribute to verify resolution
        assert provider._zdr_enabled is False  # noqa: SLF001
    finally:
        if saved is None:
            os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
        else:
            os.environ["ANTHROPIC_ZDR_ENABLED"] = saved


def test_constructor_zdr_env_truthy_attestation() -> None:
    """ANTHROPIC_ZDR_ENABLED truthy values: '1', 'true', 'yes'."""
    saved = os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
    try:
        for raw in ("1", "true", "TRUE", "yes"):
            os.environ["ANTHROPIC_ZDR_ENABLED"] = raw
            provider = AnthropicProvider(
                api_key=_api_key(),
                model_config=_model_config(),
                persister=_RecordingPersister(),
            )
            assert provider._zdr_enabled is True, f"failed for {raw!r}"  # noqa: SLF001
    finally:
        if saved is None:
            os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
        else:
            os.environ["ANTHROPIC_ZDR_ENABLED"] = saved


@pytest.mark.parametrize("raw", ["", "0", "false", "FALSE", "no", "No"])
def test_constructor_zdr_env_falsy_attestation(raw: str) -> None:
    """Round-16 sharp-edges M1 fold: falsy values resolve to False AND
    do NOT emit a warning."""
    saved = os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
    try:
        if raw:
            os.environ["ANTHROPIC_ZDR_ENABLED"] = raw
        provider = AnthropicProvider(
            api_key=_api_key(),
            model_config=_model_config(),
            persister=_RecordingPersister(),
        )
        assert provider._zdr_enabled is False, f"failed for {raw!r}"  # noqa: SLF001
    finally:
        os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
        if saved is not None:
            os.environ["ANTHROPIC_ZDR_ENABLED"] = saved


@pytest.mark.parametrize("raw", ["maybe", "trrue", "enabled", "kinda", "garbage"])
def test_constructor_zdr_env_unrecognized_fails_closed_with_warning(
    raw: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Round-16 sharp-edges M1 fold: unrecognized ZDR env values fail
    CLOSED (no attestation) AND emit a WARNING on the privacy-notice
    logger so the operator sees the misconfiguration. Silent fail-open
    or silent fail-closed both fail the operator who *thought* they
    enabled ZDR."""
    saved = os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
    try:
        os.environ["ANTHROPIC_ZDR_ENABLED"] = raw
        caplog.set_level(logging.WARNING, logger="outrider.llm.privacy_notice")
        provider = AnthropicProvider(
            api_key=_api_key(),
            model_config=_model_config(),
            persister=_RecordingPersister(),
        )
        # Fail closed
        assert provider._zdr_enabled is False, f"failed-closed expected for {raw!r}"  # noqa: SLF001
        # And warning emitted
        warning_records = [
            r
            for r in caplog.records
            if r.name == "outrider.llm.privacy_notice" and r.levelno == logging.WARNING
        ]
        observed_raw = [str(getattr(r, "anthropic_zdr_enabled_raw", "")) for r in warning_records]
        assert any(raw in seen for seen in observed_raw), (
            f"expected WARNING with anthropic_zdr_enabled_raw={raw!r}; got: {observed_raw!r}"
        )
    finally:
        os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
        if saved is not None:
            os.environ["ANTHROPIC_ZDR_ENABLED"] = saved


def test_constructor_repr_does_not_leak_api_key() -> None:
    """AC#14: provider's __repr__ never embeds the secret."""
    test_key = "sk-VERY-SECRET-KEY-ABC-123"
    provider = AnthropicProvider(
        api_key=SecretStr(test_key),
        model_config=_model_config(),
        persister=_RecordingPersister(),
    )
    rendered = repr(provider)
    assert test_key not in rendered
    assert str(provider) == repr(provider) or test_key not in str(provider)


def test_constructor_disables_sdk_internal_retry() -> None:
    """AC#12: max_retries=0; retry policy lives in the agent layer."""
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=_RecordingPersister(),
    )
    assert provider._client.max_retries == 0  # noqa: SLF001


def test_constructor_uses_30s_read_timeout() -> None:
    """AC#12: 30s read timeout fits inside the 60-120s webhook budget."""
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=_RecordingPersister(),
    )
    timeout = provider._client.timeout  # noqa: SLF001
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 30.0
    assert timeout.connect == 5.0


# ---------------------------------------------------------------------------
# complete() — fail-closed pre-call (AC#5).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_persister_not_wired_when_persister_none() -> None:
    """AC#5: persister=None → fail-closed BEFORE SDK call."""
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=None,
    )
    with _patched_create() as mock_create, pytest.raises(LLMPersisterNotWiredError):
        await provider.complete(_request())
    assert mock_create.call_count == 0, "AC#5: SDK must NOT be called when persister=None"


# ---------------------------------------------------------------------------
# complete() — SDK kwarg translation (AC#9).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_uses_system_kwarg_not_system_prompt() -> None:
    """AC#9: SDK kwarg name is `system`, not `system_prompt`."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create() as mock_create:
        await provider.complete(_request())
    sdk_kwargs = mock_create.call_args.kwargs
    assert "system" in sdk_kwargs
    assert "system_prompt" not in sdk_kwargs


@pytest.mark.asyncio
async def test_complete_passes_user_prompt_as_user_message() -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create() as mock_create:
        await provider.complete(_request(user_prompt="Specific user prompt"))
    messages = mock_create.call_args.kwargs["messages"]
    assert messages == [{"role": "user", "content": "Specific user prompt"}]


@pytest.mark.asyncio
async def test_complete_omits_stream_kwarg() -> None:
    """AC#11 (implicit): wrapper does not pass `stream=...`; SDK returns
    a single Message, never an AsyncStream."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create() as mock_create:
        await provider.complete(_request())
    assert "stream" not in mock_create.call_args.kwargs


@pytest.mark.asyncio
async def test_complete_uses_request_model() -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create() as mock_create:
        await provider.complete(_request(model="claude-haiku-4-5"))
    assert mock_create.call_args.kwargs["model"] == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# complete() — cache_control translation (AC#6).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_control_true_attaches_ephemeral_to_system_block() -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create() as mock_create:
        await provider.complete(_request(cache_control=True))
    system = mock_create.call_args.kwargs["system"]
    # When cache_control=True, system is a list of TextBlockParam-shaped dicts
    assert isinstance(system, list)
    assert len(system) == 1
    block = system[0]
    assert block["type"] == "text"
    assert block["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_cache_control_false_passes_system_as_string() -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create() as mock_create:
        await provider.complete(_request(cache_control=False))
    system = mock_create.call_args.kwargs["system"]
    # When cache_control=False, system is the bare string
    assert isinstance(system, str)


@pytest.mark.asyncio
async def test_cache_read_tokens_recorded_on_response() -> None:
    """Prompt cache validation per AC#6: response must carry cache_read_tokens
    from the SDK's Usage.cache_read_input_tokens."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create(return_value=_sdk_message(cache_read_input_tokens=500)):
        response = await provider.complete(_request())
    assert response.cache_read_tokens == 500


# ---------------------------------------------------------------------------
# complete() — multi-block fail-loud (AC#10).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_block_response_raises_unexpected_content_blocks() -> None:
    """Two TextBlocks → fail-loud."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    blocks = [
        TextBlock(citations=None, text="block1", type="text"),
        TextBlock(citations=None, text="block2", type="text"),
    ]
    with (
        _patched_create(return_value=_sdk_message(content_blocks=blocks)),
        pytest.raises(LLMUnexpectedContentBlocksError),
    ):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_zero_block_response_raises_unexpected_content_blocks() -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with (
        _patched_create(return_value=_sdk_message(content_blocks=[])),
        pytest.raises(LLMUnexpectedContentBlocksError),
    ):
        await provider.complete(_request())


# ---------------------------------------------------------------------------
# complete() — Anthropic exception mapping.
# ---------------------------------------------------------------------------


def _fake_response(status_code: int = 500) -> httpx.Response:
    """Build a minimal httpx.Response for SDK exception construction."""
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sdk_exc_factory,expected",
    [
        (
            lambda: anthropic.APITimeoutError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            LLMTimeoutError,
        ),
        (
            lambda: anthropic.RateLimitError("rate limit", response=_fake_response(429), body=None),
            LLMRateLimitError,
        ),
        (
            lambda: anthropic.AuthenticationError(
                "auth failed", response=_fake_response(401), body=None
            ),
            LLMAuthError,
        ),
        (
            lambda: anthropic.PermissionDeniedError(
                "perm denied", response=_fake_response(403), body=None
            ),
            LLMAuthError,
        ),
        (
            lambda: anthropic.BadRequestError(
                "bad request", response=_fake_response(400), body=None
            ),
            LLMInvalidRequestError,
        ),
        (
            lambda: anthropic.UnprocessableEntityError(
                "unprocessable", response=_fake_response(422), body=None
            ),
            LLMInvalidRequestError,
        ),
        (
            lambda: anthropic.InternalServerError("5xx", response=_fake_response(500), body=None),
            LLMUpstreamError,
        ),
        # Round-16 fold: previously-untested branches (coverage-audit M2).
        (
            lambda: anthropic.APIResponseValidationError(
                response=_fake_response(200),
                body=None,
            ),
            LLMInvalidResponseError,
        ),
        (
            lambda: anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            ),
            LLMUpstreamError,
        ),
    ],
)
async def test_anthropic_exception_translation(
    sdk_exc_factory: Any, expected: type[LLMProviderError]
) -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create(raise_with=sdk_exc_factory()), pytest.raises(expected):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_unmapped_apierror_translates_to_unknown() -> None:
    """Fall-through: an unmapped APIError becomes LLMUnknownError, not a
    bare LLMProviderError (which is abstract)."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )

    class _UnknownAPIError(anthropic.APIError):
        def __init__(self) -> None:
            super().__init__(
                "weird new error",
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
                body=None,
            )

    with _patched_create(raise_with=_UnknownAPIError()), pytest.raises(LLMUnknownError):
        await provider.complete(_request())


# ---------------------------------------------------------------------------
# complete() — persister contract + AC#11 + AC#18 + AC#19 + AC#20.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persister_receives_complete_llm_call_event() -> None:
    """AC#18: cost_usd is computed provider-side and present on the event
    BEFORE the persister sees it. AC#19: four-class formula."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create(
        return_value=_sdk_message(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=100,
        )
    ):
        await provider.complete(_request())
    assert len(persister.calls) == 1
    event, request, response = persister.calls[0]
    # AC#18: cost_usd populated on event already
    assert event.cost_usd > 0
    # AC#19: four-class computation
    from outrider.llm.pricing import RATE_TABLE

    expected_decimal = (
        RATE_TABLE["claude-sonnet-4-7"].in_per_token * 1000
        + RATE_TABLE["claude-sonnet-4-7"].cache_write_per_token * 100
        + RATE_TABLE["claude-sonnet-4-7"].cache_read_per_token * 200
        + RATE_TABLE["claude-sonnet-4-7"].out_per_token * 500
    )
    # Provider casts Decimal to float; allow tiny float-precision tolerance
    assert abs(event.cost_usd - float(expected_decimal)) < 1e-9


@pytest.mark.asyncio
async def test_event_carries_pricing_version() -> None:
    """LLMCallEvent.pricing_version comes from llm.pricing.PRICING_VERSION."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create():
        await provider.complete(_request())
    event = persister.calls[0][0]
    assert event.pricing_version == PRICING_VERSION


@pytest.mark.asyncio
async def test_audit_context_fields_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC#20: provider passes audit-context fields through unchanged."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    review_id = uuid4()
    request = _request(
        review_id=review_id,
        node_id="synthesize",
        is_eval=True,
        prompt_template_version="synth@2.0.0",
        degraded_mode=True,
    )
    with _patched_create():
        await provider.complete(request)
    event = persister.calls[0][0]
    assert event.review_id == review_id
    assert event.node_id == "synthesize"
    assert event.is_eval is True
    assert event.prompt_template_version == "synth@2.0.0"
    assert event.degraded_mode is True


@pytest.mark.asyncio
async def test_persister_failure_wraps_as_persister_error() -> None:
    """AC#11: post-SDK persistence-failure → LLMPersisterError, terminal."""
    persister = _RecordingPersister(raise_with=RuntimeError("DB blip"))
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create(), pytest.raises(LLMPersisterError):
        await provider.complete(_request())
    # SDK call DID succeed before the persister raised.
    assert len(persister.calls) == 1


@pytest.mark.asyncio
async def test_provider_does_not_import_agent_state() -> None:
    """AC#20 paired source-scan: anthropic_provider.py must NOT import
    from outrider.agent.* or outrider.schemas.review_state."""
    import outrider.llm.anthropic_provider as ap_module

    src_path = ap_module.__file__
    assert src_path is not None
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    # Reject any line that imports agent state / review_state
    forbidden = ["from outrider.agent", "from outrider.schemas.review_state"]
    for line in src.split("\n"):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern in forbidden:
            assert pattern not in line, (
                f"AC#20 violation: anthropic_provider.py imports from {pattern!r} on line: {line!r}"
            )


# ---------------------------------------------------------------------------
# complete() — return value shape.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_llm_response_with_text_extracted() -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create(return_value=_sdk_message(text="the response text")):
        response = await provider.complete(_request())
    assert response.text == "the response text"
    assert response.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_complete_measures_latency() -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create():
        response = await provider.complete(_request())
    assert response.latency_ms >= 0


# ---------------------------------------------------------------------------
# complete() — null cache token fields coalesce to 0.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_cache_tokens_coalesce_to_zero() -> None:
    """Anthropic's Usage.cache_*_input_tokens are Optional[int] = None when
    no caching occurred; round-15 fold added explicit `or 0` coalesce."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create(
        return_value=_sdk_message(
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        )
    ):
        response = await provider.complete(_request())
    assert response.cache_read_tokens == 0
    assert response.cache_write_tokens == 0


@pytest.mark.asyncio
async def test_null_finish_reason_coalesces_to_unknown() -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create(return_value=_sdk_message(stop_reason=None)):
        response = await provider.complete(_request())
    assert response.finish_reason == "unknown"
