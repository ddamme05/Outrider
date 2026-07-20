"""Raw OpenAI wire-capture transport for evidence spikes — NOT a production provider.

This module exists for one reason: the `spikes/openai/` research probes need to make
raw `chat.completions.create` calls and keep the response as evidence, yet the
trust-boundary `#8` invariant forbids `import openai` outside `src/outrider/llm/` AND
`vendor-payloads-normalized-at-boundary` forbids raw vendor SDK objects/exceptions from
escaping the wrapper. So the openai SDK — its client, its response objects, AND its
exception classes — is owned ENTIRELY here: `capture()` VALIDATES + normalizes the
response into the frozen, strict `RawCapture` DTO (which retains the SDK's
reserialization of the response as `sdk_response_json`), a malformed response shape
becomes `RawCaptureShapeError`, and a vendor transport error becomes `RawOpenAICaptureError`.
After this, no spike file references `openai` — not the import, not a response object, not
an exception type, and no unvalidated SDK value enters a project DTO.

It is deliberately the OPPOSITE of `OpenAICompatibleProvider` in every way except the
boundary discipline: NOT part of the `LLMProvider` Protocol, NO `LLMResponse`, NO cost
path, NO persister — but it DOES validate + normalize the vendor payload at the boundary
(strict Pydantic models; `_valid_http_status`/`_valid_request_id` on the error fields).
Construction reproduces the spikes' prior client shape (`max_retries=0`, SDK-default
timeout) so evidence provenance is unchanged — deliberately NOT the production provider's
custom `httpx.Timeout`, which would alter the observed wire.

`tests/unit/test_raw_openai_capture.py` pins the normalization + validation + boundary
contract (malformed choices/usage, invalid counts/status, non-string content, no-leak).
"""

from __future__ import annotations

from typing import Any

import openai
from pydantic import BaseModel, ConfigDict, NonNegativeInt, ValidationError

from outrider.llm.base import _valid_http_status, _valid_request_id

__all__ = [
    "RawCapture",
    "RawCaptureShapeError",
    "RawOpenAICaptureClient",
    "RawOpenAICaptureError",
    "RawUsage",
]


class RawUsage(BaseModel):
    """Strict, frozen project-owned view of the SDK's `usage` object. Counts are
    non-negative or absent (the wire omits fields; the spikes record `None` faithfully).
    Strict mode rejects non-int / bool / negative values rather than coercing them."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    prompt_tokens: NonNegativeInt | None
    completion_tokens: NonNegativeInt | None
    total_tokens: NonNegativeInt | None
    cached_tokens: NonNegativeInt | None
    cache_write_tokens: NonNegativeInt | None


class RawCapture(BaseModel):
    """Strict, frozen, project-owned normalization of one `chat.completions.create`
    response. Every field is a validated projection of what the spikes read, so no raw
    SDK object crosses the `llm/` boundary.

    `sdk_response_json` is `response.model_dump_json()` — the SDK's RESERIALIZATION of
    its parsed model, NOT the bytes off the wire. Measured against openai 2.44.0: unknown
    vendor fields survive, but key order is rewritten to the SDK's field-declaration order.
    An earlier docstring called this "the exact wire ... preserved verbatim", which is
    false on precisely the property a probe measuring generative key order depends on.
    Read it as a faithful record of the response's VALUES, never of its byte layout.

    `response_model` is the model the API said answered — distinct from the model that
    was requested, and the only field that can tell them apart."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sdk_response_json: str
    response_id: str | None
    response_model: str | None
    created: int | None
    content: str | None
    refusal: str | None
    finish_reason: str | None
    service_tier: str | None
    usage: RawUsage


class RawCaptureShapeError(Exception):
    """The openai response object did not have the shape the capture projects — missing
    `choices`/`usage`, a non-scalar count, non-string content, etc. Project-owned so no
    builtin or SDK exception escapes the boundary on a malformed response.

    Carries `sdk_response_json`: the response serialized BEFORE projection, so a novel
    malformed shape stays INSPECTABLE as evidence (the spikes persist it) even though it
    cannot be projected into a `RawCapture` or satisfy admission. `None` only if the
    object was not even serializable. Same caveat as `RawCapture.sdk_response_json`:
    values are faithful, byte layout is the SDK's, not the wire's."""

    def __init__(self, *, reason: str, sdk_response_json: str | None) -> None:
        self.reason = reason
        self.sdk_response_json = sdk_response_json
        super().__init__(reason)


class RawOpenAICaptureError(Exception):
    """Project-owned translation of an `openai.OpenAIError` (a transport/API failure).

    `status` is validated to the HTTP range, `request_id` to the id-shaped token regex.
    `message` is a deliberately RAW, bounded excerpt of the vendor error string — NOT
    sanitized: it may include vendor/request detail, so it is fit only for the spikes'
    gitignored, operator-local evidence rows, never for a user-facing surface. The vendor
    exception object itself never escapes (raised `from None`)."""

    def __init__(self, *, status: int | None, request_id: str | None, message: str) -> None:
        self.status = status
        self.request_id = request_id
        self.message = message
        super().__init__(f"openai capture failed (status={status}, request_id={request_id})")


def _normalize(response: Any) -> RawCapture:
    """Validate + project the raw SDK response into `RawCapture` — the sole place SDK
    fields are dereferenced. A shape mismatch (missing field, bad count, non-string
    content, not exactly one choice) becomes `RawCaptureShapeError`; no builtin/SDK
    exception escapes. The response is serialized FIRST so a malformed shape still
    preserves inspectable evidence."""
    try:
        sdk_response_json: str | None = response.model_dump_json(indent=2)
    except Exception:  # noqa: BLE001 — a non-serializable object leaves no evidence to keep
        sdk_response_json = None
    try:
        # EXACTLY one choice. The spikes never send `n>1`, so a multi-choice response
        # means the request was not the one under study — and silently grading
        # `choices[0]` while discarding the rest would hide that. Zero choices is
        # likewise unprojectable rather than an empty completion.
        choices = response.choices
        if len(choices) != 1:
            msg = f"expected exactly one choice, got {len(choices)}"
            raise TypeError(msg)
        choice = choices[0]
        message = choice.message
        usage = response.usage
        ptd = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(ptd, "cached_tokens", None) if ptd is not None else None
        cache_write = getattr(ptd, "cache_write_tokens", None) if ptd is not None else None
        if sdk_response_json is None:
            raise TypeError("response was not serializable to JSON")
        return RawCapture(
            sdk_response_json=sdk_response_json,
            response_id=getattr(response, "id", None),
            response_model=getattr(response, "model", None),
            created=getattr(response, "created", None),
            content=message.content,
            refusal=getattr(message, "refusal", None),
            finish_reason=choice.finish_reason,
            service_tier=getattr(response, "service_tier", None),
            usage=RawUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=getattr(usage, "total_tokens", None),
                cached_tokens=cached,
                cache_write_tokens=cache_write,
            ),
        )
    except (AttributeError, IndexError, TypeError, ValidationError) as exc:
        raise RawCaptureShapeError(
            reason=f"malformed openai response shape: {type(exc).__name__}: {str(exc)[:500]}",
            sdk_response_json=sdk_response_json,
        ) from None


class RawOpenAICaptureClient:
    """Owns an `openai.AsyncOpenAI` client for spike wire capture; returns project DTOs.

    `capture()` validates + normalizes into a `RawCapture` (never the SDK object); a
    vendor error becomes `RawOpenAICaptureError`, a malformed shape `RawCaptureShapeError`.
    The openai SDK is fully contained in this module."""

    def __init__(self, *, api_key: str, base_url: str, max_retries: int = 0) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

    async def capture(self, **kwargs: Any) -> RawCapture:
        """Raw `chat.completions.create`, validated + normalized to `RawCapture`.

        A vendor `openai.OpenAIError` is caught here and re-raised as the project-owned
        `RawOpenAICaptureError` (validated status/request_id, raw bounded message),
        `from None` so the vendor exception does not escape via the chain."""
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise RawOpenAICaptureError(
                status=_valid_http_status(getattr(exc, "status_code", None)),
                request_id=_valid_request_id(getattr(exc, "request_id", None)),
                message=str(exc)[:2000],
            ) from None
        return _normalize(response)

    async def close(self) -> None:
        """Passthrough to the SDK client's `close()` (spikes call this in a finally block)."""
        await self._client.close()
