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
        "model": "claude-sonnet-4-6",
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
    model: str = "claude-sonnet-4-6",
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


def test_constructor_accepts_dated_model_pin_via_pricing_normalization() -> None:
    """Round-27 fold (Copilot): dated SDK-catalog pins (e.g.,
    `claude-haiku-4-5-20251001`) accepted by ModelConfig must normalize
    to their undated alias for pricing-coverage validation. Without
    normalization, every dated env pin would fail this check despite
    RATE_TABLE carrying the correct alias."""
    cfg = ModelConfig(
        triage_model="claude-haiku-4-5-20251001",
        analyze_model="claude-sonnet-4-6-20251015",
        synthesize_model="claude-sonnet-4-6-20251015",
        trace_model="claude-haiku-4-5-20251001",
    )
    # Constructor must NOT raise — dated pins normalize for pricing lookup.
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=cfg,
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


def test_constructor_privacy_notice_text_zdr_not_attested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Round-20 fold per Codex finding: DECISIONS#015 point 4 specifies
    the EXACT message text. Without ZDR, the notice must name the
    contract-arrangement requirement and the 2y/7y retention exceptions."""
    saved = os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
    try:
        caplog.set_level(logging.INFO, logger="outrider.llm.privacy_notice")
        AnthropicProvider(
            api_key=_api_key(),
            model_config=_model_config(),
            persister=_RecordingPersister(),
            zdr_enabled=False,
        )
        msgs = [
            r.getMessage()
            for r in caplog.records
            if r.name == "outrider.llm.privacy_notice" and r.levelno == logging.INFO
        ]
        # Exactly one INFO with the no-ZDR shape
        assert any("anthropic_retention=30d zdr=not_attested" in m for m in msgs)
        assert any("contract arrangement" in m for m in msgs)
        assert any("2 years content" in m and "7 years classification" in m for m in msgs)
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_ZDR_ENABLED"] = saved


def test_constructor_privacy_notice_text_zdr_attested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Round-20 fold: ZDR-attested notice still names the policy-violation
    retention exceptions per #015 point 4 ("ZDR narrows standard retention
    ... but policy-violation retention still applies")."""
    saved = os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
    try:
        caplog.set_level(logging.INFO, logger="outrider.llm.privacy_notice")
        AnthropicProvider(
            api_key=_api_key(),
            model_config=_model_config(),
            persister=_RecordingPersister(),
            zdr_enabled=True,
        )
        msgs = [
            r.getMessage()
            for r in caplog.records
            if r.name == "outrider.llm.privacy_notice" and r.levelno == logging.INFO
        ]
        assert any("anthropic_retention=zdr_attested" in m for m in msgs)
        assert any("operator attestation" in m for m in msgs)
        assert any("2 years content" in m and "7 years classification" in m for m in msgs)
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_ZDR_ENABLED"] = saved


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


@pytest.fixture
def _reset_zdr_warned_set() -> Iterator[None]:
    """Round-17: ZDR warning is rate-limited via a process-local
    `_WARNED_RAW_VALUES` set. Reset between tests so once-per-process
    behavior doesn't break test isolation when the same raw value
    happens to be used twice."""
    from outrider.llm.anthropic_provider import _WARNED_RAW_VALUES

    saved = _WARNED_RAW_VALUES.copy()
    _WARNED_RAW_VALUES.clear()
    try:
        yield
    finally:
        _WARNED_RAW_VALUES.clear()
        _WARNED_RAW_VALUES.update(saved)


def test_zdr_warning_fires_only_once_per_distinct_raw_value(
    caplog: pytest.LogCaptureFixture,
    _reset_zdr_warned_set: None,
) -> None:
    """Round-17 audit fold (M2): under V1.5 parallel-analyze, N providers
    per review constructed with the same misconfigured env would spam
    thousands of WARNINGs/day. Once-per-distinct-raw-value guard caps
    the spam while preserving the diagnostic signal."""
    saved = os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
    try:
        os.environ["ANTHROPIC_ZDR_ENABLED"] = "garbage"
        caplog.set_level(logging.WARNING, logger="outrider.llm.privacy_notice")
        # Construct 5 providers with the same misconfigured env
        for _ in range(5):
            AnthropicProvider(
                api_key=_api_key(),
                model_config=_model_config(),
                persister=_RecordingPersister(),
            )
        # Expect exactly one WARNING (the others suppressed by the guard)
        warning_records = [
            r
            for r in caplog.records
            if r.name == "outrider.llm.privacy_notice"
            and r.levelno == logging.WARNING
            and getattr(r, "anthropic_zdr_enabled_raw", "") == "garbage"
        ]
        assert len(warning_records) == 1, (
            f"expected exactly 1 WARNING for repeated 'garbage' env value; "
            f"got {len(warning_records)} (without the guard, would be 5)"
        )
    finally:
        os.environ.pop("ANTHROPIC_ZDR_ENABLED", None)
        if saved is not None:
            os.environ["ANTHROPIC_ZDR_ENABLED"] = saved


@pytest.mark.parametrize("raw", ["maybe", "trrue", "enabled", "kinda", "weird-value"])
def test_constructor_zdr_env_unrecognized_fails_closed_with_warning(
    raw: str,
    caplog: pytest.LogCaptureFixture,
    _reset_zdr_warned_set: None,
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


def test_constructor_uses_generation_sized_read_timeout() -> None:
    """AC#12 (revised 2026-06-10): the read timeout must cover the worst
    LEGITIMATE non-streaming generation the wrapper permits (MAX_TOKENS=8192
    ≈ 234s at loaded-Sonnet throughput, plus TTFT tail) — the original 30s
    capped legitimate work and a TTFT spike killed a paid eval run. Pinned
    at 300s (half the SDK's 600s default); connect stays tight at 5s."""
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=_RecordingPersister(),
    )
    timeout = provider._client.timeout  # noqa: SLF001
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 300.0
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
# complete() — constrained decoding translation (FUP-096).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_schema_translates_to_output_config() -> None:
    """`response_schema_json` set → the SDK call carries the documented
    `output_config.format` envelope with the schema round-tripped
    losslessly from the compact order-preserving JSON string (the
    production serialization — property order is part of the format)."""
    import json as _json

    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    with _patched_create() as mock_create:
        await provider.complete(
            _request(response_schema_json=_json.dumps(schema, separators=(",", ":")))
        )
    assert mock_create.call_args.kwargs["output_config"] == {
        "format": {"type": "json_schema", "schema": schema}
    }


@pytest.mark.asyncio
async def test_no_response_schema_omits_output_config() -> None:
    """Free-form requests (triage, synthesize, trace) must not carry the
    kwarg at all — absent, not None."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create() as mock_create:
        await provider.complete(_request())
    assert "output_config" not in mock_create.call_args.kwargs


@pytest.mark.asyncio
async def test_persisted_event_carries_response_format_digest() -> None:
    """The LLMCallEvent the provider persists records the request's
    derived digest — populated when a schema rode the call, None when
    not — so replay/ops can split the two output populations."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create():
        await provider.complete(_request(response_schema_json='{"type":"object"}'))
        await provider.complete(_request())
    (with_schema, req_with, _), (without_schema, _, _) = persister.calls
    assert with_schema.response_format_digest == req_with.response_format_digest
    assert with_schema.response_format_digest is not None
    assert without_schema.response_format_digest is None


# ---------------------------------------------------------------------------
# complete() — cache_control translation (AC#6).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_control_true_attaches_ephemeral_to_system_block() -> None:
    """Round-21 fold per Codex finding: V1 single-turn shape needs
    per-block cache_control on the SYSTEM block (stable across calls).
    Top-level "Automatic Caching" targets the last cacheable block,
    which in V1's `system + [user]` shape is the volatile user message —
    defeats the cache. Per spec.md §9.5 (Prompt caching for cost
    reduction) the system prompt is the cache boundary; the volatile
    user/diff content stays outside."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create() as mock_create:
        await provider.complete(_request(cache_control=True))
    sdk_kwargs = mock_create.call_args.kwargs
    # No top-level cache_control kwarg (round-21 reverted that).
    assert "cache_control" not in sdk_kwargs
    # System is a list with one TextBlockParam carrying ephemeral cache_control.
    system = sdk_kwargs["system"]
    assert isinstance(system, list)
    assert len(system) == 1
    block = system[0]
    assert block["type"] == "text"
    assert block["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_cache_control_false_passes_system_as_bare_string() -> None:
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create() as mock_create:
        await provider.complete(_request(cache_control=False))
    sdk_kwargs = mock_create.call_args.kwargs
    # No top-level cache_control kwarg AND system is a bare string
    assert "cache_control" not in sdk_kwargs
    assert isinstance(sdk_kwargs["system"], str)


@pytest.mark.asyncio
async def test_cache_control_default_is_true() -> None:
    """Round-20 fold per DECISIONS#013 point 4 + spec §9.5
    "prompt-caching-always-on" — default must be True."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    # _request() doesn't pass cache_control — should default to True
    with _patched_create() as mock_create:
        await provider.complete(_request())
    sdk_kwargs = mock_create.call_args.kwargs
    # Round-21: per-block placement on system, not top-level kwarg
    assert "cache_control" not in sdk_kwargs
    system = sdk_kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


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
# complete() — prompt-caching silently-disabled diagnostic.
# Per Anthropic SDK 0.100 prompt-caching docs, prompts shorter than the
# model's min-cacheable threshold (authoritative per-model values:
# `pricing.MIN_CACHEABLE_TOKENS`) are processed without caching with NO error. Detection:
# cache_control=True request with both cache_creation_input_tokens=0
# AND cache_read_input_tokens=0 in the response.
# ---------------------------------------------------------------------------


@pytest.fixture
def _reset_noncacheable_warned_set() -> Iterator[None]:
    """Round-22: cache-silently-disabled warning is rate-limited via a
    process-local `_WARNED_NONCACHEABLE` set keyed by (model,
    system_prompt_hash). Reset between tests so once-per-key behavior
    doesn't break test isolation when the same prompt happens to be
    used twice."""
    from outrider.llm.anthropic_provider import _WARNED_NONCACHEABLE

    saved = _WARNED_NONCACHEABLE.copy()
    _WARNED_NONCACHEABLE.clear()
    try:
        yield
    finally:
        _WARNED_NONCACHEABLE.clear()
        _WARNED_NONCACHEABLE.update(saved)


@pytest.mark.asyncio
async def test_cache_silently_disabled_warns_when_both_cache_token_fields_zero(
    caplog: pytest.LogCaptureFixture,
    _reset_noncacheable_warned_set: None,
) -> None:
    """Round-22 fold: when cache_control=True but the SDK reports zero
    cache_creation AND zero cache_read tokens, the prompt was likely
    below the model's min-cacheable threshold. Surface as WARNING so
    the operator sees the misconfiguration without aggregating audit
    events."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    caplog.set_level(logging.WARNING, logger="outrider.llm.anthropic_provider")
    with _patched_create(
        return_value=_sdk_message(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    ):
        await provider.complete(_request(cache_control=True))
    warning_records = [
        r
        for r in caplog.records
        if r.name == "outrider.llm.anthropic_provider"
        and r.levelno == logging.WARNING
        and "min-cacheable threshold" in r.getMessage()
    ]
    assert len(warning_records) == 1
    # Metadata-only — no prompt content in the extras
    rec = warning_records[0]
    assert rec.model == "claude-sonnet-4-6"  # type: ignore[attr-defined]
    assert isinstance(rec.system_prompt_hash, str)  # type: ignore[attr-defined]
    assert rec.node_id == "analyze"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cache_silently_disabled_warns_only_once_per_model_and_prompt(
    caplog: pytest.LogCaptureFixture,
    _reset_noncacheable_warned_set: None,
) -> None:
    """Mirror of round-17's ZDR warn-once pattern: under V1.5 parallel-
    analyze, N providers calling with the same too-short prompt would
    spam thousands of WARNINGs/day. The (model, system_prompt_hash)
    key bounds spam while preserving the diagnostic signal."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    caplog.set_level(logging.WARNING, logger="outrider.llm.anthropic_provider")
    with _patched_create(
        return_value=_sdk_message(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    ):
        for _ in range(5):
            await provider.complete(_request(cache_control=True))
    warning_records = [
        r
        for r in caplog.records
        if r.name == "outrider.llm.anthropic_provider"
        and r.levelno == logging.WARNING
        and "min-cacheable threshold" in r.getMessage()
    ]
    assert len(warning_records) == 1, (
        f"expected exactly 1 WARNING for repeated calls with the same "
        f"(model, system_prompt_hash); got {len(warning_records)} "
        f"(without the guard, would be 5)"
    )


@pytest.mark.asyncio
async def test_cache_silently_disabled_does_not_warn_when_cache_engaged(
    caplog: pytest.LogCaptureFixture,
    _reset_noncacheable_warned_set: None,
) -> None:
    """When cache_creation_input_tokens > 0 (first-call cache write),
    caching IS engaged — no diagnostic warning fires. Same for cache
    eviction-and-rewrite cycles, which also produce non-zero
    cache_creation."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    caplog.set_level(logging.WARNING, logger="outrider.llm.anthropic_provider")
    # First call: cache write, no read — caching engaged
    with _patched_create(
        return_value=_sdk_message(
            cache_creation_input_tokens=2500,
            cache_read_input_tokens=0,
        )
    ):
        await provider.complete(_request(cache_control=True))
    warning_records = [r for r in caplog.records if "min-cacheable threshold" in r.getMessage()]
    assert warning_records == []


@pytest.mark.asyncio
async def test_cache_silently_disabled_does_not_warn_when_cache_control_false(
    caplog: pytest.LogCaptureFixture,
    _reset_noncacheable_warned_set: None,
) -> None:
    """Diagnostic only fires when caching was OPTED INTO. cache_control=False
    with both cache token fields at 0 is the expected no-cache case and
    must not trip the warning."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    caplog.set_level(logging.WARNING, logger="outrider.llm.anthropic_provider")
    with _patched_create(
        return_value=_sdk_message(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    ):
        await provider.complete(_request(cache_control=False))
    warning_records = [r for r in caplog.records if "min-cacheable threshold" in r.getMessage()]
    assert warning_records == []


@pytest.mark.asyncio
async def test_cache_silently_disabled_warns_separately_per_model(
    caplog: pytest.LogCaptureFixture,
    _reset_noncacheable_warned_set: None,
) -> None:
    """The (response.model, system_prompt_hash) key correctly distinguishes
    models — the same prompt can clear one model's min-cacheable floor and
    miss another's (per-model values: `pricing.MIN_CACHEABLE_TOKENS`), so
    each model deserves its own warn-once budget. The dedup key uses
    `response.model` (the executed model, which determines the threshold),
    not `request.model` — the SDK could substitute via alias resolution
    or deprecation routing."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    caplog.set_level(logging.WARNING, logger="outrider.llm.anthropic_provider")
    # Each call must have a DIFFERENT response.model since the dedup key
    # is keyed off response.model. Patch _patched_create twice with
    # different sdk_message defaults — re-entering the patch context per
    # call lets each call return a distinct fixture.
    sonnet_msg = _sdk_message(
        model="claude-sonnet-4-6",
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    with _patched_create(return_value=sonnet_msg):
        await provider.complete(_request(cache_control=True, model="claude-sonnet-4-6"))
    haiku_msg = _sdk_message(
        model="claude-haiku-4-5",
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    with _patched_create(return_value=haiku_msg):
        await provider.complete(_request(cache_control=True, model="claude-haiku-4-5"))
    warning_records = [r for r in caplog.records if "min-cacheable threshold" in r.getMessage()]
    assert len(warning_records) == 2
    seen_models = {getattr(r, "model", None) for r in warning_records}
    assert seen_models == {"claude-sonnet-4-6", "claude-haiku-4-5"}
    # Each warning's `request_model` extra is also populated for
    # operator debugging when SDK substitution makes response.model
    # differ from request.model.
    seen_request_models = {getattr(r, "request_model", None) for r in warning_records}
    assert seen_request_models == {"claude-sonnet-4-6", "claude-haiku-4-5"}


@pytest.mark.asyncio
async def test_cache_silently_disabled_dedup_normalizes_dated_aliases(
    caplog: pytest.LogCaptureFixture,
    _reset_noncacheable_warned_set: None,
) -> None:
    """Two calls whose response.model values are dated/undated aliases of
    the SAME base model (claude-haiku-4-5 vs claude-haiku-4-5-20251001)
    should share a warn-once budget — they're the same model family for
    cache-threshold purposes. The dedup key passes response.model
    through normalize_to_pricing_key, so the second call is suppressed.
    The literal response.model still appears in the extras for operator
    visibility into what actually executed each time."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    caplog.set_level(logging.WARNING, logger="outrider.llm.anthropic_provider")
    # First call: undated alias.
    undated_msg = _sdk_message(
        model="claude-haiku-4-5",
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    with _patched_create(return_value=undated_msg):
        await provider.complete(_request(cache_control=True, model="claude-haiku-4-5"))
    # Second call: dated alias of the same base model.
    dated_msg = _sdk_message(
        model="claude-haiku-4-5-20251001",
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    with _patched_create(return_value=dated_msg):
        await provider.complete(_request(cache_control=True, model="claude-haiku-4-5-20251001"))
    warning_records = [r for r in caplog.records if "min-cacheable threshold" in r.getMessage()]
    # Exactly ONE warn fires across both calls — dated and undated dedup
    # together because they share the same cache threshold.
    assert len(warning_records) == 1, (
        f"expected 1 warn (dated/undated should share dedup budget); got "
        f"{len(warning_records)} — dedup is incorrectly distinguishing aliases"
    )
    # The single warning's `model` extra carries the literal response.model
    # of whichever call fired first (the undated alias here).
    assert getattr(warning_records[0], "model", None) == "claude-haiku-4-5"


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
        # Round-20 fold: 404 NotFoundError mapped to terminal LLMInvalidRequestError.
        (
            lambda: anthropic.NotFoundError(
                "model not found", response=_fake_response(404), body=None
            ),
            LLMInvalidRequestError,
        ),
        # Round-21 correction per Codex finding: 409 ConflictError is in
        # the Anthropic SDK's default-retry set (alongside 408/429/5xx),
        # so route to LLMConflictError with retry_at_layer="node" rather
        # than terminal LLMInvalidRequestError.
        (
            lambda: anthropic.ConflictError("conflict", response=_fake_response(409), body=None),
            LLMConflictError,
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


@pytest.mark.asyncio
async def test_translate_anthropic_error_does_not_leak_sdk_text_into_wrapper() -> None:
    """Pins the round-26 codex fold: `_translate_anthropic_error()` MUST
    NOT pass `str(exc)` (or any SDK exception body text) to the wrapper
    class constructor, AND the `raise ... from None` at the wrapper site
    MUST drop the SDK exception via `__suppress_context__`.

    The leak vector: Anthropic SDK error messages render the underlying
    httpx response body via `str(exc)`. The body can echo prompt
    fragments from the failing request (most concretely:
    context-length-exceeded errors quote the offending text). If the
    wrapper passed `str(exc)` to e.g. `LLMRateLimitError(str(exc))`,
    that text would land in `Exception.args[0]` and render in
    `repr(wrapper)`, `str(wrapper)`, and traceback formatting by any
    log handler using `exc_info=True`.

    Test: mock an `anthropic.RateLimitError` whose `str()` contains a
    distinctive sentinel; trigger `provider.complete()`; verify the
    wrapper `LLMRateLimitError` does NOT carry the sentinel in any of
    `str()`, `repr()`, `args`, or via the cause chain
    (`exc.__cause__`/`__context__` with `__suppress_context__` set).
    """
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )

    sentinel = "secret_prompt_fragment_zzz9876_in_sdk_error_body"  # noqa: S105 — test fixture
    sdk_exc = anthropic.RateLimitError(
        sentinel,
        response=_fake_response(429),
        body=None,
    )
    # Sanity: the SDK exception DOES carry the sentinel via str().
    assert sentinel in str(sdk_exc)

    with _patched_create(raise_with=sdk_exc), pytest.raises(LLMRateLimitError) as exc_info:
        await provider.complete(_request())

    wrapper = exc_info.value
    # Wrapper's own rendering surfaces — none carry SDK text.
    assert sentinel not in str(wrapper), "wrapper str() leaks SDK body text"
    assert sentinel not in repr(wrapper), "wrapper repr() leaks SDK body text"
    for arg in wrapper.args:
        assert sentinel not in str(arg), "wrapper args[] leaks SDK body text"

    # `from None` suppresses cause-chain rendering. __cause__ must be
    # None and __suppress_context__ must be True so traceback formatters
    # don't walk __context__ either.
    assert wrapper.__cause__ is None, "raise-from-None failed to drop __cause__"
    assert wrapper.__suppress_context__ is True, (
        "raise-from-None failed to set __suppress_context__; "
        "traceback formatter would still render __context__"
    )


@pytest.mark.asyncio
async def test_non_apierror_anthropic_subclass_does_not_escape() -> None:
    """The SDK's exception root is `anthropic.AnthropicError`, not
    `APIError`. `WorkloadIdentityError` is a real example that inherits
    from `AnthropicError` directly (not via `APIError`); it would have
    escaped a narrower `except APIError` block and broken the
    'no vendor SDK exception escapes complete()' contract. The wrapper
    catches `AnthropicError` to cover this and any future non-APIError
    additions; unmapped subclasses translate to LLMUnknownError."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )

    # Construct a non-APIError but AnthropicError. WorkloadIdentityError
    # is the concrete real-world case but takes auth-config args we
    # don't want to fake; subclassing AnthropicError directly proves
    # the catch shape works for any future addition to the hierarchy.
    class _NonAPIError(anthropic.AnthropicError):
        pass

    with (
        _patched_create(raise_with=_NonAPIError("synthetic non-API error")),
        pytest.raises(LLMUnknownError, match="unmapped AnthropicError"),
    ):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_non_anthropic_exception_from_sdk_translates_to_llm_unknown_error() -> None:
    """The pre-call `_closed` check is best-effort, not atomic vs.
    `aclose()`. A close that lands between the check and the awaited
    SDK call surfaces a `RuntimeError("Cannot send a request, as the
    client has been closed.")` from httpx — NOT an
    `anthropic.AnthropicError`. Without translation, that would escape
    the typed `LLMProviderError` contract.

    Pin: any non-Anthropic Exception raised by the SDK call translates
    to `LLMUnknownError`. The exception type name appears in the
    wrapper message (class-level identifier, safe to render);
    `from None` is used to drop the cause chain so SDK exception args
    don't propagate.
    """
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )

    runtime_error = RuntimeError("Cannot send a request, as the client has been closed.")

    with (
        _patched_create(raise_with=runtime_error),
        pytest.raises(LLMUnknownError) as exc_info,
    ):
        await provider.complete(_request())

    # Message identifies the class but does NOT echo the SDK exception's args.
    rendered = str(exc_info.value)
    assert "RuntimeError" in rendered
    assert "Cannot send a request" not in rendered
    # Cause chain is dropped (`from None`).
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


@pytest.mark.asyncio
async def test_close_race_translates_to_llm_unknown_error_with_aclose_message() -> None:
    """When the SDK raises during a close-race AND `_closed` is True,
    the wrapper message names the close-race specifically rather than
    the generic "non-Anthropic SDK failure" path. Operators reading
    the log can distinguish a real SDK failure from a graceful-shutdown
    request-after-close.
    """
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )

    # Flip the closed flag DIRECTLY (simulating a race where aclose()
    # set _closed=True between the Step 0 check and the SDK call).
    # The Step 0 check sees False at request-start because we haven't
    # called aclose() yet at that point — but by the time the SDK
    # invocation runs, it sees True. The mock's side_effect lets us
    # interleave: when SDK is called, flip the flag THEN raise.
    def _flip_then_raise(*_args: object, **_kwargs: object) -> None:
        provider._closed = True
        raise RuntimeError("client has been closed")

    mock = AsyncMock(side_effect=_flip_then_raise)
    with (
        patch.object(anthropic.resources.messages.AsyncMessages, "create", mock),
        pytest.raises(LLMUnknownError, match="raced with aclose"),
    ):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_step8_keyerror_fallback_on_unknown_response_model() -> None:
    """After the response.model fix, step-8 cost lookup uses
    response.model (not request.model). If the SDK substitutes a model
    not in RATE_TABLE (alias resolution, deprecation routing), the
    constructor's eager pricing-coverage check (which only validates
    configured models) won't catch it — but the step-8 KeyError
    fallback raises LLMPricingMissingError loudly with both
    response.model and request.model in the message."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    # SDK responds with a model that's NOT in RATE_TABLE. This response
    # model has no dated suffix, so normalize_to_pricing_key is a no-op
    # and the literal response.model and the normalized pricing key are
    # identical.
    sdk_msg = _sdk_message(model="claude-haiku-99-99-substituted")
    with (
        _patched_create(return_value=sdk_msg),
        pytest.raises(LLMPricingMissingError) as exc_info,
    ):
        await provider.complete(_request(model="claude-sonnet-4-6"))
    err = exc_info.value
    # Error message names BOTH response.model and request.model for debug
    assert "claude-haiku-99-99-substituted" in str(err)
    assert "claude-sonnet-4-6" in str(err)
    # Structured attribute carries the host-qualified pricing key str
    # (DECISIONS.md#056) — `str((profile_id, normalized_model))`. For an
    # un-dated model the normalized model equals the literal response.model.
    assert err.missing_models == (str(("anthropic", "claude-haiku-99-99-substituted")),)


@pytest.mark.asyncio
async def test_step8_keyerror_message_names_normalized_pricing_key() -> None:
    """Copilot follow-on: when the SDK substitutes a DATED model that
    normalizes to a pricing key not in RATE_TABLE, the error message
    AND `missing_models` must name the normalized key, not the literal
    response.model. Otherwise an operator reading the error and adding
    the literal dated string to RATE_TABLE would NOT fix the lookup —
    `compute_cost_usd` would still resolve via `normalize_to_pricing_key`
    to the undated key and miss the new dated entry."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    # Dated form that normalizes via -YYYYMMDD strip to a base alias
    # NOT in RATE_TABLE. `normalize_to_pricing_key` strips the trailing
    # 8-digit date suffix → "claude-fake-99-99" (not in RATE_TABLE).
    sdk_msg = _sdk_message(model="claude-fake-99-99-20251020")
    with (
        _patched_create(return_value=sdk_msg),
        pytest.raises(LLMPricingMissingError) as exc_info,
    ):
        await provider.complete(_request(model="claude-sonnet-4-6"))
    err = exc_info.value
    msg = str(err)
    # Both the literal response.model AND the normalized pricing key
    # appear in the message so an operator updating RATE_TABLE fixes
    # the actual missing entry, not the un-normalized dated literal.
    assert "claude-fake-99-99-20251020" in msg, (
        f"error message must name the literal response.model; got: {msg}"
    )
    assert "claude-fake-99-99" in msg, (
        f"error message must name the normalized pricing key; got: {msg}"
    )
    # Structured attribute carries the host-qualified pricing key str
    # (DECISIONS.md#056) — `str((profile_id, normalized_model))`, where the
    # dated literal normalizes to the undated base, NOT the dated literal.
    assert err.missing_models == (str(("anthropic", "claude-fake-99-99")),), (
        f"missing_models must hold the host-qualified pricing key str for an "
        f"operator to add the right RATE_TABLE entry; got: "
        f"{err.missing_models}"
    )


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
    from outrider.llm.pricing import RATE_TABLE, pricing_key

    sonnet_key = pricing_key("anthropic", "claude-sonnet-4-6")
    expected_decimal = (
        RATE_TABLE[sonnet_key].in_per_token * 1000
        + RATE_TABLE[sonnet_key].cache_write_per_token * 100
        + RATE_TABLE[sonnet_key].cache_read_per_token * 200
        + RATE_TABLE[sonnet_key].out_per_token * 500
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
async def test_cost_computed_against_response_model_not_request_model() -> None:
    """Audit-fidelity: when the SDK echoes back a different model than was
    requested (alias resolution, deprecation routing), cost_usd must be
    computed against `response.model` so the persisted `LLMCallEvent.model`
    matches the rate-table key used to compute `LLMCallEvent.cost_usd`.
    Otherwise replay reconstruction would see event.model paired with a
    cost computed at a different model's rate."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    # Request claude-sonnet-4-6 (in pricing); SDK responds with
    # claude-haiku-4-5 (also in pricing, different rates). Cost must
    # match the haiku rate, not the sonnet rate.
    sdk_msg = _sdk_message(
        model="claude-haiku-4-5",
        input_tokens=1000,
        output_tokens=500,
    )
    with _patched_create(return_value=sdk_msg):
        await provider.complete(_request(model="claude-sonnet-4-6"))
    event, _, response = persister.calls[0]
    # event.model echoes response (haiku)
    assert event.model == "claude-haiku-4-5"
    assert response.model == "claude-haiku-4-5"
    # cost computed at HAIKU rates, not SONNET rates
    from outrider.llm.pricing import RATE_TABLE, pricing_key

    haiku_rates = RATE_TABLE[pricing_key("anthropic", "claude-haiku-4-5")]
    expected_haiku_cost = float(haiku_rates.in_per_token * 1000 + haiku_rates.out_per_token * 500)
    assert abs(event.cost_usd - expected_haiku_cost) < 1e-9, (
        f"cost should match response.model (haiku) rates, got {event.cost_usd}, "
        f"expected {expected_haiku_cost}"
    )
    # Sanity: sonnet rates would have produced a meaningfully different cost
    sonnet_rates = RATE_TABLE[pricing_key("anthropic", "claude-sonnet-4-6")]
    sonnet_cost = float(sonnet_rates.in_per_token * 1000 + sonnet_rates.out_per_token * 500)
    assert abs(event.cost_usd - sonnet_cost) > 1e-6, (
        "test fixture invariant: haiku and sonnet rates must differ enough "
        "for the assertion above to be meaningful"
    )


@pytest.mark.asyncio
async def test_audit_context_fields_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC#20: provider passes audit-context fields through unchanged.

    Per §0b: `degraded_mode=True` is analyze-only in V1 — the previous
    framing of this test pinned it on synthesize, which the §0b
    provenance validator now correctly rejects. Switched to analyze and
    paired with the required `degradation_reason` so the pass-through
    contract is exercised under a valid configuration.
    """
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    review_id = uuid4()
    request = _request(
        review_id=review_id,
        node_id="analyze",
        is_eval=True,
        prompt_template_version="analyze@2.0.0",
        degraded_mode=True,
        degradation_reason="parse_failed",
    )
    with _patched_create():
        await provider.complete(request)
    event = persister.calls[0][0]
    assert event.review_id == review_id
    assert event.node_id == "analyze"
    assert event.is_eval is True
    assert event.prompt_template_version == "analyze@2.0.0"
    assert event.degraded_mode is True
    # Post-PR review fold: the request carries degradation_reason and the
    # wrapper must pass it through to the audit event. Without this
    # assertion a regression that drops the field mid-pipeline would
    # silently pass the pass-through contract test.
    assert event.degradation_reason == "parse_failed"


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
async def test_persister_unknown_exception_drops_cause_chain() -> None:
    """Round-9 regression for DECISIONS#016 logs-stay-metadata-only.

    Unknown persister exception types (not in `METADATA_ONLY_EXCEPTION_TYPES`)
    must be wrapped with `raise ... from None`. The wrapper's message is
    sanitized to `<TypeName>`, but without `from None`, Python's traceback
    formatter would render `__cause__` (the underlying exception's
    `args` / `str()`), leaking raw content past the wrapper's sanitization.

    Sentinel-string approach: raise `ValueError("SECRET_LEAK_SENTINEL")`
    from the persister; catch the LLMPersisterError; verify:
    - `__cause__ is None` (cause chain dropped)
    - `__suppress_context__ is True` (implicit context also hidden)
    - the sentinel string does NOT appear in `str(exc)` or `repr(exc)`
    - the sentinel does NOT appear in the rendered traceback
    """
    import traceback

    secret = "SECRET_LEAK_SENTINEL_DO_NOT_LEAK_xyz"  # noqa: S105 — test fixture
    persister = _RecordingPersister(raise_with=ValueError(secret))
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create(), pytest.raises(LLMPersisterError) as exc_info:
        await provider.complete(_request())

    exc = exc_info.value
    # Cause chain dropped — the round-9 fix.
    assert exc.__cause__ is None, (
        f"unknown persister exception must use `from None` to drop the "
        f"cause chain; got __cause__={exc.__cause__!r}"
    )
    assert exc.__suppress_context__ is True, (
        "from None should also set __suppress_context__=True to hide the implicit __context__"
    )
    # Wrapper message uses sanitized type name only.
    assert "<ValueError>" in str(exc)
    assert secret not in str(exc)
    assert secret not in repr(exc)
    # Rendered traceback does NOT carry the sentinel.
    rendered_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert secret not in rendered_tb, (
        "rendered traceback leaked the sentinel string; `from None` did "
        "not actually suppress the cause chain in the formatted output"
    )


@pytest.mark.asyncio
async def test_persister_metadata_only_exception_preserves_cause_chain() -> None:
    """Round-9 regression: metadata-only persister exception types preserve
    the cause chain via `from exc` (the chain is also metadata-only, by
    contract). Useful for operator debugging — the LLMPersisterError carries
    diagnostic context, but only metadata-only content.
    """
    from outrider.audit.persister import AuditPersisterIdempotencyConflict, FieldDigest

    conflict = AuditPersisterIdempotencyConflict(
        event_id=uuid4(),
        mismatched_fields=("cost_usd",),
        field_digests={"cost_usd": FieldDigest("a" * 64, "b" * 64, 10, 12)},
    )
    persister = _RecordingPersister(raise_with=conflict)
    provider = AnthropicProvider(
        api_key=_api_key(),
        model_config=_model_config(),
        persister=persister,
    )
    with _patched_create(), pytest.raises(LLMPersisterError) as exc_info:
        await provider.complete(_request())

    exc = exc_info.value
    # Metadata-only types: cause chain IS preserved for debug context.
    assert exc.__cause__ is conflict
    # Wrapper message renders the metadata-only `str(conflict)`.
    assert "idempotency conflict" in str(exc)
    assert "cost_usd" in str(exc)


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


# ---------------------------------------------------------------------------
# Identity-triad stamping (DECISIONS.md#056, step 4).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_stamps_native_triad() -> None:
    """The native path stamps a constant triad — profile_id='anthropic', reasoning off, the
    constant `_ANTHROPIC_CONTRACT_DIGEST` — on BOTH the response and the persisted event."""
    from outrider.llm.anthropic_provider import _ANTHROPIC_CONTRACT_DIGEST

    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(), model_config=_model_config(), persister=persister
    )
    with _patched_create():
        resp = await provider.complete(_request())
    assert (resp.profile_id, resp.reasoning_enabled, resp.profile_contract_digest) == (
        "anthropic",
        False,
        _ANTHROPIC_CONTRACT_DIGEST,
    )
    event, _req, _resp = persister.calls[0]
    assert (event.profile_id, event.reasoning_enabled, event.profile_contract_digest) == (
        "anthropic",
        False,
        _ANTHROPIC_CONTRACT_DIGEST,
    )


async def test_stamped_triad_matches_resolve_host_identity() -> None:
    """Drift guard (Codex): the triad AnthropicProvider stamps MUST equal what the lifespan
    resolves via resolve_host_identity('anthropic', reasoning=False) — the single source
    build_graph closes into the completion events. Divergence would split host identity
    between the per-call LLMCallEvents and the per-node completion events."""
    from outrider.llm.host_profiles import resolve_host_identity

    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(), model_config=_model_config(), persister=persister
    )
    with _patched_create():
        resp = await provider.complete(_request())
    assert (
        resp.profile_id,
        resp.reasoning_enabled,
        resp.profile_contract_digest,
    ) == resolve_host_identity("anthropic", reasoning=False)


async def test_unpriced_request_model_rejected_before_sdk_call() -> None:
    """FUP-197: a request.model not priced under the anthropic host is rejected BEFORE the
    paid SDK call (parity with OpenAICompatibleProvider) — no billed call, no orphan row.
    The constructor validates CONFIGURED models; this guards a request.model that bypassed
    config (a routing bug)."""
    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(), model_config=_model_config(), persister=persister
    )
    with (
        _patched_create() as mock_create,
        pytest.raises(LLMInvalidRequestError, match="not in RATE_TABLE"),
    ):
        await provider.complete(_request(model="claude-opus-4-7"))
    mock_create.assert_not_called()


def test_anthropic_reasoning_requested_fails_loud() -> None:
    """OUTRIDER_LLM_REASONING=true under anthropic must fail loud — the native path has no
    reasoning toggle, so it can't be silently no-op'd into a stamped reasoning_enabled."""
    with pytest.raises(LLMInvalidRequestError, match="does not support reasoning"):
        AnthropicProvider(
            api_key=_api_key(),
            model_config=_model_config(),
            persister=_RecordingPersister(),
            reasoning=True,
        )


# ---------------------------------------------------------------------------
# Sonnet 5 migration — adaptive-thinking request shape + refusal gate.
# ---------------------------------------------------------------------------


def test_adaptive_thinking_model_omits_temperature_and_disables_thinking() -> None:
    """Sonnet 5+ (adaptive-thinking generation): non-default sampling params 400,
    so `temperature` is OMITTED; adaptive thinking is disabled so the response
    stays a single text block (`_extract_single_text_block`)."""
    from outrider.llm.anthropic_provider import _build_sdk_kwargs

    kwargs = _build_sdk_kwargs(_request(model="claude-sonnet-5", temperature=0.0))
    assert "temperature" not in kwargs
    assert kwargs["thinking"] == {"type": "disabled"}


def test_legacy_model_sends_temperature_and_no_thinking_kwarg() -> None:
    """Current-generation models keep the legacy shape: `temperature` is sent and
    no `thinking` kwarg is passed (thinking off by default)."""
    from outrider.llm.anthropic_provider import _build_sdk_kwargs

    for model in ("claude-haiku-4-5", "claude-sonnet-4-6"):
        kwargs = _build_sdk_kwargs(_request(model=model, temperature=0.3))
        assert kwargs["temperature"] == 0.3
        assert "thinking" not in kwargs


def test_adaptive_thinking_model_keeps_structured_output() -> None:
    """The disabled-thinking branch composes with constrained decoding: an
    adaptive-thinking model with a response schema still gets `output_config`
    AND disabled thinking AND no temperature."""
    import json as _json

    from outrider.llm.anthropic_provider import _build_sdk_kwargs

    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    kwargs = _build_sdk_kwargs(
        _request(
            model="claude-sonnet-5",
            response_schema_json=_json.dumps(schema, separators=(",", ":")),
        )
    )
    assert kwargs["thinking"] == {"type": "disabled"}
    assert "temperature" not in kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_complete_raises_llm_refusal_on_refusal_stop_reason() -> None:
    """A safety-classifier decline (HTTP 200, stop_reason='refusal', empty
    content) halts the review as `LLMRefusalError` — not a misleading
    unexpected-content-blocks shape error, and not a silent empty result.
    Outrider asks the model to find vulnerabilities; a decline must fail loud."""
    from outrider.llm.base import LLMRefusalError

    persister = _RecordingPersister()
    provider = AnthropicProvider(
        api_key=_api_key(), model_config=_model_config(), persister=persister
    )
    refused = _sdk_message(stop_reason="refusal", content_blocks=[])
    with (
        _patched_create(return_value=refused),
        pytest.raises(LLMRefusalError, match="refusal") as exc_info,
    ):
        # A priced legacy model — the refusal gate fires on stop_reason,
        # independent of which model was used.
        await provider.complete(_request(model="claude-sonnet-4-6"))
    # No stop_details on the stub → category is None; the error still raises.
    assert exc_info.value.category is None
    # Raised before persist (consistent with the content-shape fail-loud path).
    assert persister.calls == []


@pytest.mark.asyncio
async def test_complete_refusal_propagates_stop_details_category() -> None:
    """When the SDK populates `stop_details.category` on a refusal, the gate
    surfaces it on `LLMRefusalError.category` for operator inspection — the real
    populated-category path that the empty-stub refusal test cannot exercise."""
    from types import SimpleNamespace

    from outrider.llm.base import LLMRefusalError

    provider = AnthropicProvider(
        api_key=_api_key(), model_config=_model_config(), persister=_RecordingPersister()
    )
    refused = _sdk_message(stop_reason="refusal", content_blocks=[])
    # pydantic v2 does not validate on assignment by default; attach a populated
    # stop_details the way a real RefusalStopDetails carries `.category`.
    refused.stop_details = SimpleNamespace(category="cyber")  # type: ignore[attr-defined]
    with (
        _patched_create(return_value=refused),
        pytest.raises(LLMRefusalError) as exc_info,
    ):
        await provider.complete(_request(model="claude-sonnet-4-6"))
    assert exc_info.value.category == "cyber"


@pytest.mark.asyncio
async def test_cache_silently_disabled_suppressed_for_unknown_floor_model(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    _reset_noncacheable_warned_set: None,
) -> None:
    """A None min-cacheable floor (the DECISIONS.md#056 unknown-floor sentinel — a
    host with an undocumented threshold) suppresses the silently-disabled-cache
    diagnostic: a cache miss can't be attributed to a below-floor prompt when the
    floor is undocumented, so the provider stays silent (per the
    `min_cacheable_tokens` contract) rather than logging a confusing 'None tokens'
    warning. No current Anthropic model returns None (Sonnet 5's floor is the
    documented 1024), so the None return is forced here to cover the defensive
    branch that protects any future undocumented-floor host."""
    import outrider.llm.anthropic_provider as ap

    monkeypatch.setattr(ap, "min_cacheable_tokens", lambda profile_id, model: None)
    provider = AnthropicProvider(
        api_key=_api_key(), model_config=_model_config(), persister=_RecordingPersister()
    )
    caplog.set_level(logging.WARNING, logger="outrider.llm.anthropic_provider")
    with _patched_create(
        return_value=_sdk_message(
            model="claude-sonnet-4-6",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    ):
        await provider.complete(_request(cache_control=True, model="claude-sonnet-4-6"))
    warning_records = [
        r
        for r in caplog.records
        if r.name == "outrider.llm.anthropic_provider"
        and r.levelno == logging.WARNING
        and "min-cacheable threshold" in r.getMessage()
    ]
    assert warning_records == [], "unknown-floor (None) model must suppress the cache diagnostic"
