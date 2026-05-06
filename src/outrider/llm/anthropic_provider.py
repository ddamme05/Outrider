# AnthropicProvider — concrete LLMProvider implementation.
# Sole `import anthropic` per `vendor-sdks-only-in-wrappers` invariant.
# See specs/2026-05-05-llm-provider-wrapper.md and DECISIONS.md #013/#015/#016.
"""Anthropic concrete provider.

Constructor performs eager validation:
  - Empty `api_key` raises `LLMMissingAPIKeyError` (eager — SDK does NOT
    error on construction with a missing key; deferring would surface
    mid-review as an opaque 401).
  - Any model in `model_config` not in `pricing.RATE_TABLE` raises
    `LLMPricingMissingError` (eager — eliminates step 8's `KeyError`
    failure path between SDK success and persister write).
  - Privacy startup notice emits to logger `outrider.llm.privacy_notice`
    on every construction (per DECISIONS#015 point 4).

`complete()` step ordering (spec §Implementation sketch):

  1. Fail-closed pre-call: persister=None → raise immediately.
  2. Translate `LLMRequest` to SDK kwargs (system=request.system_prompt,
     messages=[...], stream omitted).
  3. cache_control translation: bool → ephemeral block on system prompt.
  4. await client.messages.create(...); catch APIError → typed subclass.
  5. Validate response shape: exactly one TextBlock or fail-loud.
  6. Construct LLMResponse + measure latency_ms.
  7. Compute prompt_hash + system_prompt_hash.
  8. Compute cost_usd via pricing.compute_cost_usd().
  9. Build LLMCallEvent; await persister.persist(...); wrap failures as
     LLMPersisterError.
 10. Return LLMResponse.
"""

import logging
import os
import time
from datetime import UTC, datetime
from decimal import Decimal  # noqa: TC003 — runtime use in step 8 cost computation
from typing import Any

import anthropic
import httpx
from anthropic.types import Message, TextBlock, TextBlockParam
from pydantic import SecretStr

from outrider.audit.events import LLMCallEvent
from outrider.llm.base import (
    LLMAuthError,
    LLMExchangePersister,
    LLMInvalidRequestError,
    LLMInvalidResponseError,
    LLMMissingAPIKeyError,
    LLMPersisterError,
    LLMPersisterNotWiredError,
    LLMPricingMissingError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMUnexpectedContentBlocksError,
    LLMUnknownError,
    LLMUpstreamError,
    _canonical_prompt_hash,
    _canonical_system_prompt_hash,
)
from outrider.llm.config import ModelConfig
from outrider.llm.pricing import (
    PRICING_VERSION,
    RATE_TABLE,
    compute_cost_usd,
)

__all__ = ["AnthropicProvider"]


_PRIVACY_NOTICE_LOGGER = logging.getLogger("outrider.llm.privacy_notice")
_LOGGER = logging.getLogger("outrider.llm.anthropic_provider")


def _resolve_zdr_attestation(zdr_enabled: bool | None) -> bool:
    """Read ZDR attestation per DECISIONS#015 — operator-attestation only,
    NEVER a per-request header. Constructor kwarg wins; falls back to
    `ANTHROPIC_ZDR_ENABLED` env var (truthy values: "1", "true", "True").
    """
    if zdr_enabled is not None:
        return zdr_enabled
    raw = os.environ.get("ANTHROPIC_ZDR_ENABLED", "")
    return raw.strip().lower() in {"1", "true", "yes"}


class AnthropicProvider:
    """Concrete `LLMProvider` for Anthropic. Sole `import anthropic` site
    in the codebase per `vendor-sdks-only-in-wrappers`.

    Constructor enforces eager validation (api_key + pricing-coverage)
    and emits the DECISIONS#015 privacy notice on every construction.
    """

    def __init__(
        self,
        api_key: SecretStr,
        *,
        model_config: ModelConfig,
        persister: LLMExchangePersister | None = None,
        zdr_enabled: bool | None = None,
    ) -> None:
        # Eager api_key validation — SDK doesn't error on missing key,
        # so we surface eagerly per AC#13.
        if not api_key.get_secret_value():
            raise LLMMissingAPIKeyError(
                "AnthropicProvider requires a non-empty api_key; "
                "the Anthropic SDK does not error on missing keys at "
                "construction, so the wrapper validates eagerly."
            )

        # Eager pricing-coverage validation — eliminates step 8's
        # KeyError path between SDK success and persister write per AC#24.
        configured_models = {
            model_config.triage_model,
            model_config.analyze_model,
            model_config.synthesize_model,
            model_config.trace_model,
        }
        missing = sorted(configured_models - set(RATE_TABLE.keys()))
        if missing:
            raise LLMPricingMissingError(
                f"AnthropicProvider construction: configured model(s) "
                f"{missing!r} have no entry in llm.pricing.RATE_TABLE. "
                f"Update RATE_TABLE + bump PRICING_VERSION before using "
                f"these models, or correct OUTRIDER_MODEL_* env vars."
            )

        self._api_key = api_key
        self._model_config = model_config
        self._persister = persister
        self._zdr_enabled = _resolve_zdr_attestation(zdr_enabled)

        # SDK client — DefaultAsyncHttpxClient preserves SDK defaults
        # (headers, retry hooks, etc.) while we customize limits/timeout.
        # max_retries=0: retry policy lives in the agent layer per spec.
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key.get_secret_value(),
            max_retries=0,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=10.0),
            http_client=anthropic.DefaultAsyncHttpxClient(
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            ),
        )

        # Privacy startup notice (DECISIONS#015 point 4).
        # Operator-attestation only; never a per-request header.
        _PRIVACY_NOTICE_LOGGER.info(
            "anthropic_provider startup",
            extra={
                "privacy_notice": True,
                "zdr_attested": self._zdr_enabled,
                "egress_destination": "api.anthropic.com",
                "retention_policy": (
                    "anthropic_default_30_days"
                    if not self._zdr_enabled
                    else "operator_attested_zdr"
                ),
            },
        )

    def __repr__(self) -> str:
        # Elide api_key (SecretStr also redacts but explicit is safer);
        # surface the persister-wired status so debug tooling can see it.
        persister_status = "wired" if self._persister is not None else "none"
        return (
            f"<AnthropicProvider model_config={self._model_config!r} "
            f"persister={persister_status} zdr={self._zdr_enabled}>"
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send `request` to the Anthropic API; return `LLMResponse`.

        Strict 10-step ordering — see module docstring. Failures surface
        as typed `LLMProviderError` subclasses; the calling node reads
        `error.retry_at_layer` to decide retry behavior.
        """
        # Step 1: fail-closed pre-call.
        # If persister is None, raise BEFORE the SDK call so the SDK
        # never sees a request from a misconfigured provider.
        if self._persister is None:
            raise LLMPersisterNotWiredError(
                "AnthropicProvider.complete() called with persister=None; "
                "production deployments must wire a real LLMExchangePersister "
                "per DECISIONS#016 single-transaction-insert contract."
            )

        # Step 2 + 3: translate request to SDK shape.
        sdk_kwargs = _build_sdk_kwargs(request)

        # Step 4: SDK call + exception translation.
        # `time.perf_counter_ns()` for monotonic, high-res latency measurement.
        t_start_ns = time.perf_counter_ns()
        try:
            sdk_response: Message = await self._client.messages.create(**sdk_kwargs)
        except anthropic.APIError as exc:
            raise _translate_anthropic_error(exc) from exc
        latency_ms = (time.perf_counter_ns() - t_start_ns) // 1_000_000

        # Step 5: validate response shape.
        text = _extract_single_text_block(sdk_response)

        # Step 6: construct LLMResponse from SDK response.
        usage = sdk_response.usage
        response = LLMResponse(
            text=text,
            model=sdk_response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=(usage.cache_read_input_tokens or 0),
            cache_write_tokens=(usage.cache_creation_input_tokens or 0),
            finish_reason=sdk_response.stop_reason or "unknown",
            latency_ms=int(latency_ms),
        )

        # Step 7: hash the prompts.
        prompt_hash = _canonical_prompt_hash(request.system_prompt, request.user_prompt)
        system_prompt_hash = _canonical_system_prompt_hash(request.system_prompt)

        # Step 8: compute cost_usd from pricing table.
        # KeyError unreachable in production thanks to constructor's eager
        # pricing-coverage check; the defensive try/except below catches
        # the case where RATE_TABLE mutates between construction and call
        # (shouldn't happen — module-level Final dict — but typed for safety).
        try:
            cost_decimal: Decimal = compute_cost_usd(
                model=request.model,
                input_tokens=response.input_tokens,
                cache_write_tokens=response.cache_write_tokens,
                cache_read_tokens=response.cache_read_tokens,
                output_tokens=response.output_tokens,
            )
        except KeyError as exc:
            raise LLMPricingMissingError(
                f"Model {request.model!r} not in RATE_TABLE at "
                f"complete() step 8; constructor's eager validation "
                f"should have caught this — pricing table mutation?"
            ) from exc

        # Step 9: build LLMCallEvent + await persister.persist().
        event = LLMCallEvent(
            review_id=request.review_id,
            timestamp=datetime.now(UTC),
            is_eval=request.is_eval,
            # Canonical metadata-only fields per spec.md §8.3.
            model=response.model,
            node_id=request.node_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cache_read_tokens,
            cost_usd=float(cost_decimal),
            pricing_version=PRICING_VERSION,
            latency_ms=response.latency_ms,
            prompt_hash=prompt_hash,
            cache_hit=(response.cache_read_tokens > 0),
            context_summary=request.context_summary,
            prompt_template_version=request.prompt_template_version,
            system_prompt_hash=system_prompt_hash,
            degraded_mode=request.degraded_mode,
        )
        try:
            await self._persister.persist(event, request, response)
        except Exception as exc:
            # Wrap any persister failure as LLMPersisterError so callers
            # have one error class to pattern-match for the post-SDK-
            # failure path. SDK call has succeeded (billing accounted);
            # no audit row landed → calling node halts the review.
            raise LLMPersisterError(
                f"Persister failed after successful SDK call: {exc!r}. "
                f"The audit row did not land; calling node halts the review."
            ) from exc

        # Step 10: return response.
        return response


def _build_sdk_kwargs(request: LLMRequest) -> dict[str, Any]:
    """Translate `LLMRequest` to Anthropic SDK `messages.create()` kwargs.

    Key mappings (round 12 + 14 corrections):
      - `request.system_prompt` → SDK kwarg `system` (NOT `system_prompt`)
      - `request.user_prompt` → single user-role message
      - `request.cache_control=True` → ephemeral cache_control on system block
      - `stream` omitted → returns `Message`, not `AsyncStream`
      - `request.messages` is V1.5+; rejected at LLMRequest construction
        (validator); never reaches here in V1.
    """
    if request.cache_control:
        # Per-block ephemeral cache_control on system prompt block.
        # SDK has no boolean cache toggle; this is the documented surface.
        system_param: str | list[TextBlockParam] = [
            TextBlockParam(
                type="text",
                text=request.system_prompt,
                cache_control={"type": "ephemeral"},
            )
        ]
    else:
        system_param = request.system_prompt

    return {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "system": system_param,
        "messages": [{"role": "user", "content": request.user_prompt}],
    }


def _extract_single_text_block(message: Message) -> str:
    """Validate response is exactly one `TextBlock`; return its text.

    Per AC#10, multi-block responses (extended thinking, tool use, etc.)
    fail loud rather than silently flatten or drop. V1's single-text-block
    assumption is robust as long as the wrapper never passes the
    `thinking` kwarg (which we don't).
    """
    if len(message.content) != 1 or not isinstance(message.content[0], TextBlock):
        actual_types = [type(b).__name__ for b in message.content]
        raise LLMUnexpectedContentBlocksError(
            f"Anthropic response has {len(message.content)} content block(s) "
            f"of types {actual_types}; V1 wrapper expects exactly one TextBlock. "
            f"This may indicate extended-thinking or tool-use responses (not "
            f"supported in V1) or an SDK shape change."
        )
    return message.content[0].text


def _translate_anthropic_error(exc: anthropic.APIError) -> Exception:
    """Map an Anthropic SDK exception to the typed `LLMProviderError`
    subclass per the round-13 mapping table.

    Order matters: more-specific subclasses must come before
    APIStatusError fallback (e.g., `RateLimitError` before
    `APIStatusError`). The fall-through is `LLMUnknownError` (per
    round-13/14 abstract-base redesign — bare `LLMProviderError` is no
    longer raisable).
    """
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError(str(exc))
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError(str(exc))
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return LLMAuthError(str(exc))
    if isinstance(exc, (anthropic.BadRequestError, anthropic.UnprocessableEntityError)):
        return LLMInvalidRequestError(str(exc))
    if isinstance(exc, anthropic.APIResponseValidationError):
        return LLMInvalidResponseError(str(exc))
    if isinstance(exc, (anthropic.InternalServerError, anthropic.APIConnectionError)):
        return LLMUpstreamError(str(exc))
    # Fall-through for unmapped APIError subclasses.
    return LLMUnknownError(f"unmapped APIError: {type(exc).__name__}: {exc}")
