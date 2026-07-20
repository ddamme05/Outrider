"""`RawOpenAICaptureClient` — the spike wire-capture transport under `llm/`.

Pins that the helper OWNS the openai SDK end-to-end: it constructs the client, and
`capture()` normalizes the response into the frozen project-owned `RawCapture` DTO
(preserving the raw wire as `raw_json`) while a vendor error becomes the project-owned
`RawOpenAICaptureError` — so no SDK object or exception class escapes the boundary
(`vendor-payloads-normalized-at-boundary`, trust boundary #8). See
`src/outrider/llm/raw_openai_capture.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from outrider.llm.raw_openai_capture import (
    RawCapture,
    RawCaptureShapeError,
    RawOpenAICaptureClient,
    RawOpenAICaptureError,
    RawUsage,
)


async def _capture_with(response: object) -> RawCapture:
    fake_sdk = MagicMock()
    fake_sdk.chat.completions.create = AsyncMock(return_value=response)
    with patch("openai.AsyncOpenAI", return_value=fake_sdk):
        client = RawOpenAICaptureClient(api_key="k", base_url="https://api.openai.com/v1")
        return await client.capture(model="gpt-5.6-sol", messages=[])


def _sdk_response() -> SimpleNamespace:
    """A stand-in shaped like the openai SDK response object the spikes read."""
    message = SimpleNamespace(content="  {}  ", refusal="I can't help with that.")
    choice = SimpleNamespace(message=message, finish_reason="stop")
    ptd = SimpleNamespace(cached_tokens=1500, cache_write_tokens=400)
    usage = SimpleNamespace(
        prompt_tokens=2000,
        completion_tokens=50,
        total_tokens=2050,
        prompt_tokens_details=ptd,
    )
    return SimpleNamespace(
        id="chatcmpl-abc123",
        created=1_700_000_000,
        service_tier="default",
        choices=[choice],
        usage=usage,
        model_dump_json=lambda **_: '{"id": "chatcmpl-abc123"}',
    )


def test_construction_mirrors_the_spikes_prior_direct_usage() -> None:
    """max_retries=0 and NO custom timeout — the spikes' provenance-preserving shape
    (deliberately not the production provider's httpx.Timeout, which would alter wire)."""
    with patch("openai.AsyncOpenAI") as mk:
        RawOpenAICaptureClient(api_key="k", base_url="https://api.openai.com/v1")
    mk.assert_called_once_with(api_key="k", base_url="https://api.openai.com/v1", max_retries=0)
    assert "timeout" not in mk.call_args.kwargs  # SDK default, unlike OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_capture_normalizes_the_response_into_a_project_dto() -> None:
    """capture() returns a frozen RawCapture — NOT the SDK object — with the raw wire
    preserved as raw_json and every field the spikes read typed."""
    fake_sdk = MagicMock()
    fake_sdk.chat.completions.create = AsyncMock(return_value=_sdk_response())
    with patch("openai.AsyncOpenAI", return_value=fake_sdk):
        client = RawOpenAICaptureClient(api_key="k", base_url="https://api.openai.com/v1")
        capture = await client.capture(model="gpt-5.6-sol", messages=[])
    assert isinstance(capture, RawCapture)
    assert capture.raw_json == '{"id": "chatcmpl-abc123"}'  # untouched wire preserved
    assert capture.response_id == "chatcmpl-abc123"
    assert capture.created == 1_700_000_000
    assert capture.content == "  {}  "
    assert capture.refusal == "I can't help with that."
    assert capture.finish_reason == "stop"
    assert capture.service_tier == "default"
    assert capture.usage == RawUsage(
        prompt_tokens=2000,
        completion_tokens=50,
        total_tokens=2050,
        cached_tokens=1500,
        cache_write_tokens=400,
    )
    fake_sdk.chat.completions.create.assert_awaited_once_with(model="gpt-5.6-sol", messages=[])


@pytest.mark.asyncio
async def test_missing_prompt_details_normalize_to_none() -> None:
    """A response without prompt_tokens_details yields None cache counts, not a crash."""
    resp = _sdk_response()
    resp.usage.prompt_tokens_details = None
    capture = await _capture_with(resp)
    assert capture.usage.cached_tokens is None
    assert capture.usage.cache_write_tokens is None


@pytest.mark.asyncio
async def test_missing_choices_raises_shape_error_preserving_raw_evidence() -> None:
    """An empty `choices` (malformed wire) becomes a project shape error — not an
    IndexError — and the error CARRIES the raw serialized payload, so a novel wire shape
    stays inspectable as evidence. `capture()` raises (returns no RawCapture), so a
    malformed shape can never reach the admission grader."""
    resp = _sdk_response()
    resp.choices = []
    with pytest.raises(RawCaptureShapeError, match="malformed openai response shape") as excinfo:
        await _capture_with(resp)
    assert excinfo.value.raw_json == '{"id": "chatcmpl-abc123"}'  # raw wire preserved
    assert excinfo.value.reason  # human-readable reason retained


@pytest.mark.asyncio
async def test_missing_usage_raises_shape_error() -> None:
    """A response with no usage object becomes a project shape error, not AttributeError."""
    resp = _sdk_response()
    resp.usage = None
    with pytest.raises(RawCaptureShapeError):
        await _capture_with(resp)


@pytest.mark.asyncio
async def test_negative_token_count_raises_shape_error() -> None:
    """Strict validation: a negative token count is rejected (NonNegativeInt), surfaced as
    a project shape error rather than silently entering the DTO."""
    resp = _sdk_response()
    resp.usage.prompt_tokens = -5
    with pytest.raises(RawCaptureShapeError):
        await _capture_with(resp)


@pytest.mark.asyncio
async def test_non_string_content_raises_shape_error() -> None:
    """Strict validation: a non-string message.content is rejected, not coerced."""
    resp = _sdk_response()
    resp.choices[0].message.content = 123
    with pytest.raises(RawCaptureShapeError):
        await _capture_with(resp)


@pytest.mark.asyncio
async def test_invalid_error_status_is_dropped_to_none() -> None:
    """An out-of-range HTTP status on the vendor error is validated away to None
    (_valid_http_status), never entering the project error as an arbitrary int."""

    class _FakeAPIError(openai.OpenAIError):
        def __init__(self) -> None:
            super().__init__("boom")
            self.status_code = 999  # out of the 100..599 HTTP range
            self.request_id = "not a valid id !!!"  # not id-shaped

    fake_sdk = MagicMock()
    fake_sdk.chat.completions.create = AsyncMock(side_effect=_FakeAPIError())
    with patch("openai.AsyncOpenAI", return_value=fake_sdk):
        client = RawOpenAICaptureClient(api_key="k", base_url="https://api.openai.com/v1")
        with pytest.raises(RawOpenAICaptureError) as excinfo:
            await client.capture(model="gpt-5.6-sol", messages=[])
    assert excinfo.value.status is None  # 999 rejected
    assert excinfo.value.request_id is None  # non-id-shaped rejected


@pytest.mark.asyncio
async def test_vendor_error_becomes_project_error_without_leaking_the_sdk_exception() -> None:
    """A vendor openai.OpenAIError is translated to RawOpenAICaptureError with validated
    status / validated request_id / raw bounded message excerpt, and the SDK exception
    does NOT escape via the cause chain (raised `from None`)."""

    class _FakeAPIError(openai.OpenAIError):
        def __init__(self) -> None:
            super().__init__("boom: sensitive detail")
            self.status_code = 401
            self.request_id = "req_" + "a" * 20

    fake_sdk = MagicMock()
    fake_sdk.chat.completions.create = AsyncMock(side_effect=_FakeAPIError())
    with patch("openai.AsyncOpenAI", return_value=fake_sdk):
        client = RawOpenAICaptureClient(api_key="k", base_url="https://api.openai.com/v1")
        with pytest.raises(RawOpenAICaptureError) as excinfo:
            await client.capture(model="gpt-5.6-sol", messages=[])
    err = excinfo.value
    assert err.status == 401
    assert err.request_id == "req_" + "a" * 20  # id-shaped, passed the validator
    assert "boom" in err.message
    # The vendor exception is not chained out (from None): no SDK type in the cause.
    assert err.__cause__ is None
    assert not isinstance(err, openai.OpenAIError)


@pytest.mark.asyncio
async def test_close_delegates_to_the_sdk_client() -> None:
    fake_sdk = MagicMock()
    fake_sdk.close = AsyncMock(return_value=None)
    with patch("openai.AsyncOpenAI", return_value=fake_sdk):
        client = RawOpenAICaptureClient(api_key="k", base_url="https://api.openai.com/v1")
        await client.close()
    fake_sdk.close.assert_awaited_once_with()
