# AnthropicProvider — concrete LLMProvider implementation.
# Owns the Anthropic transport surface; vendor SDK imports stay inside
# `src/outrider/llm/` per the folder-scoped `vendor-sdks-only-in-wrappers`
# invariant (sibling modules within `llm/` may import SDK metadata).
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
  7a. Round-22 diagnostic: if `cache_control=True` AND both
      cache_creation_input_tokens=0 AND cache_read_input_tokens=0, log a
      WARN once per (model, system_prompt_hash) per process — the SDK
      silently rejected caching, typically because the prompt is below
      the model's min-cacheable threshold (Sonnet 4.6: 2048 tokens;
      Haiku 4.5: 4096 tokens). Diagnostic only; does not raise.
  8. Compute cost_usd via pricing.compute_cost_usd() (uses
     `normalize_to_pricing_key` so dated SDK-catalog model pins resolve
     to their undated alias for RATE_TABLE lookup, per round-27).
  9. Build LLMCallEvent; await persister.persist(...); wrap failures as
     LLMPersisterError.
 10. Return LLMResponse.
"""

import logging
import os
import time
from datetime import UTC, datetime
from decimal import Decimal  # noqa: TC003 — runtime use in step 8 cost computation
from typing import Any, Final

import anthropic
import httpx
from anthropic.types import Message, TextBlock, TextBlockParam
from pydantic import SecretStr

from outrider.audit.events import LLMCallEvent
from outrider.llm.base import (
    LLMAuthError,
    LLMConflictError,
    LLMExchangePersister,
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
    _canonical_prompt_hash,
    _canonical_system_prompt_hash,
)
from outrider.llm.config import ModelConfig
from outrider.llm.pricing import (
    PRICING_VERSION,
    RATE_TABLE,
    compute_cost_usd,
    normalize_to_pricing_key,
)

__all__ = ["AnthropicProvider"]


_PRIVACY_NOTICE_LOGGER = logging.getLogger("outrider.llm.privacy_notice")
_LOGGER = logging.getLogger("outrider.llm.anthropic_provider")


_ZDR_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes"})
_ZDR_FALSY: Final[frozenset[str]] = frozenset({"", "0", "false", "no"})

# Round-17 fold per audit-agent finding M2: warn once per misconfigured
# raw value per process. V1.5 parallel-analyze constructs N providers per
# review — without this guard, a typo'd env var spams thousands of WARNING
# records per day. The set is process-local so each worker still warns
# once, which is the diagnostic signal we want without the spam.
_WARNED_RAW_VALUES: set[str] = set()

# Round-22 fold per Anthropic SDK 0.100 prompt-caching docs: silently-
# disabled-cache diagnostic. Per the docs, prompts shorter than the
# model's min-cacheable threshold (Sonnet 4.6: 2048 tokens; Haiku 4.5:
# 4096 tokens) are processed without caching, with NO error returned.
# Detection: `cache_control=True` request with both
# `cache_creation_input_tokens=0` AND `cache_read_input_tokens=0` in the
# response. Without this signal, an operator who opted into caching sees
# no cost reduction and has no way to discover the root cause without
# aggregating across many events. Process-local set keyed by
# (model, system_prompt_hash) bounds spam under V1.5 parallel-analyze
# fanout (same shape as `_WARNED_RAW_VALUES` above).
_WARNED_NONCACHEABLE: set[tuple[str, str]] = set()


def _resolve_zdr_attestation(zdr_enabled: bool | None) -> bool:
    """Read ZDR attestation per DECISIONS#015 — operator-attestation only,
    NEVER a per-request header. Constructor kwarg wins; falls back to
    `ANTHROPIC_ZDR_ENABLED` env var.

    Truthy values (case-insensitive): `"1"`, `"true"`, `"yes"`.
    Falsy values (case-insensitive): `""`, `"0"`, `"false"`, `"no"`.
    Unrecognized values fail closed (no ZDR attestation) AND log a
    WARNING on `outrider.llm.privacy_notice` so the operator sees the
    misconfiguration at construction time (round-16 sharp-edges M1
    fold — silent fail-closed-on-typo means the operator who *thought*
    they enabled ZDR ships with retention claims they didn't intend).
    The warning fires once per distinct raw value per process to avoid
    log spam under V1.5's parallel-analyze fanout (round-17 audit fold).
    """
    if zdr_enabled is not None:
        return zdr_enabled
    raw = os.environ.get("ANTHROPIC_ZDR_ENABLED", "").strip().lower()
    if raw in _ZDR_TRUTHY:
        return True
    if raw in _ZDR_FALSY:
        return False
    # Unrecognized — fail closed AND warn (once per distinct raw value).
    if raw not in _WARNED_RAW_VALUES:
        _WARNED_RAW_VALUES.add(raw)
        _PRIVACY_NOTICE_LOGGER.warning(
            "anthropic_provider zdr-attestation env-var unrecognized; falling back to ZDR=False",
            extra={
                "privacy_notice": True,
                "zdr_attested": False,
                "anthropic_zdr_enabled_raw": raw,
                "expected_truthy": sorted(_ZDR_TRUTHY),
                "expected_falsy": sorted(_ZDR_FALSY),
            },
        )
    return False


class AnthropicProvider:
    """Concrete `LLMProvider` for Anthropic.

    Owns the Anthropic transport (`messages.create` calls). The
    `vendor-sdks-only-in-wrappers` invariant is folder-scoped: vendor
    SDK imports are confined to `src/outrider/llm/`, not exclusive to
    this file — sibling modules within the wrapper folder may
    legitimately import SDK metadata too (e.g., `config.py` imports
    `anthropic.resources.messages.DEPRECATED_MODELS` for eager
    deprecation validation). Tests under `tests/unit/` import SDK types
    for fixture construction, which is also outside the invariant's
    folder scope but inside the test surface.

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
        # Round-27 fold (Copilot): dated model IDs (e.g.,
        # `claude-haiku-4-5-20251001`) accepted by ModelConfig must
        # normalize to their undated alias for pricing lookup; otherwise
        # every dated env pin would fail this check despite RATE_TABLE
        # carrying the correct alias.
        configured_models = {
            model_config.triage_model,
            model_config.analyze_model,
            model_config.synthesize_model,
            model_config.trace_model,
        }
        missing = sorted(
            m for m in configured_models if normalize_to_pricing_key(m) not in RATE_TABLE
        )
        if missing:
            raise LLMPricingMissingError(
                f"AnthropicProvider construction: configured model(s) "
                f"{missing!r} have no entry in llm.pricing.RATE_TABLE "
                f"(checked after dated-suffix normalization). "
                f"Update RATE_TABLE + bump PRICING_VERSION before using "
                f"these models, or correct OUTRIDER_MODEL_* env vars.",
                missing_models=missing,
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

        # Privacy startup notice — canonical text per DECISIONS#015
        # point 4. Operator-attestation only; never a per-request header.
        # Round-20 audit fold per Codex: previous emit had only structured
        # fields, missing the mandatory message text that names the
        # 30-day default / 2-year content / 7-year classification
        # retention exceptions and the contract-arrangement requirement.
        if self._zdr_enabled:
            notice_message = (
                "privacy_notice anthropic_retention=zdr_attested; "
                "ZDR arrangement assumed per operator attestation "
                "(Outrider does not verify). Policy-violation retention "
                "up to 2 years content / 7 years classification still "
                "applies per Anthropic's ZDR terms."
            )
        else:
            notice_message = (
                "privacy_notice anthropic_retention=30d zdr=not_attested; "
                "ZDR cannot be enabled by this flag alone — it requires "
                "a contract arrangement with Anthropic (contact sales). "
                "Set ANTHROPIC_ZDR_ENABLED=true if your organization has "
                "ZDR arranged. Policy-violation retention up to 2 years "
                "content / 7 years classification applies regardless."
            )
        _PRIVACY_NOTICE_LOGGER.info(
            notice_message,
            extra={
                "privacy_notice": True,
                "zdr_attested": self._zdr_enabled,
                "egress_destination": "api.anthropic.com",
                "retention_default_days": 30,
                "retention_policy_violation_content_years": 2,
                "retention_policy_violation_classification_years": 7,
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

        # Round-22 fold: prompt-caching silently-disabled diagnostic.
        # Per Anthropic SDK 0.100 prompt-caching docs, prompts shorter than
        # the model's min-cacheable threshold (Sonnet 4.6: 2048 tokens;
        # Haiku 4.5: 4096 tokens) are processed without caching with NO
        # error returned. The unambiguous signal is `cache_control=True`
        # AND both cache_creation/read tokens at 0. The first-ever cache
        # write has cache_creation > 0 (so doesn't trigger); cache eviction
        # also rewrites with cache_creation > 0 (so doesn't trigger). Only
        # the genuine "SDK refused to cache this prompt" condition matches.
        # Warn once per (model, system_prompt_hash) per process.
        if (
            request.cache_control
            and response.cache_read_tokens == 0
            and response.cache_write_tokens == 0
        ):
            cache_warn_key = (request.model, system_prompt_hash)
            if cache_warn_key not in _WARNED_NONCACHEABLE:
                _WARNED_NONCACHEABLE.add(cache_warn_key)
                _LOGGER.warning(
                    "anthropic_provider cache_control=True but neither "
                    "cache_creation_input_tokens nor cache_read_input_tokens "
                    "fired; system prompt likely below the model's "
                    "min-cacheable threshold (Sonnet 4.6: 2048 tokens; "
                    "Haiku 4.5: 4096 tokens — see Anthropic prompt-caching "
                    "docs). cache_control=True will produce no cost savings "
                    "on this prompt until it grows past the threshold or "
                    "cache_control is removed.",
                    extra={
                        "model": request.model,
                        "system_prompt_hash": system_prompt_hash,
                        "review_id": str(request.review_id),
                        "node_id": request.node_id,
                    },
                )

        # Step 8: compute cost_usd from pricing table.
        # KeyError unreachable in production thanks to constructor's eager
        # pricing-coverage check; the defensive try/except below catches
        # the case where RATE_TABLE mutates between construction and call
        # (shouldn't happen — module-level Final dict — but typed for safety).
        #
        # Audit-fidelity fix (deep self-audit): the cost lookup uses
        # `response.model` rather than `request.model` so the rate-table
        # key used to compute `cost_usd` matches the `LLMCallEvent.model`
        # value that's persisted on the audit row (line below). Currently
        # identical in practice — Anthropic SDK 0.100 echoes back the
        # request model exactly — but if a future SDK ever substitutes
        # (alias resolution, deprecation routing), audit replay would
        # otherwise see `event.model = response.model` paired with a cost
        # computed at `request.model`'s rate. The eager pricing-coverage
        # check still validates configured models; if the SDK substitutes
        # to an unknown model, the KeyError fallback below surfaces it
        # loudly as `LLMPricingMissingError`.
        try:
            cost_decimal: Decimal = compute_cost_usd(
                model=response.model,
                input_tokens=response.input_tokens,
                cache_write_tokens=response.cache_write_tokens,
                cache_read_tokens=response.cache_read_tokens,
                output_tokens=response.output_tokens,
            )
        except KeyError as exc:
            raise LLMPricingMissingError(
                f"Model {response.model!r} not in RATE_TABLE at "
                f"complete() step 8 (request.model={request.model!r}); "
                f"constructor's eager validation covers configured models "
                f"only — this can fire if the SDK substitutes the model "
                f"in its response (alias resolution, deprecation routing).",
                missing_models=[response.model],
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

    Key mappings (round 12 + 14 + 21 corrections):
      - `request.system_prompt` → SDK kwarg `system` (NOT `system_prompt`)
      - `request.user_prompt` → single user-role message
      - `request.cache_control=True` → **per-block** `cache_control` on
        the system block. Round-21 fold per Codex finding: round-20's
        top-level "Automatic Caching" kwarg was a regression for V1's
        single-turn shape. Per spec.md §1476-1478, the system prompt is
        the cache boundary; the volatile user/diff content stays outside
        the cache. Top-level automatic caching applies the breakpoint to
        the LAST cacheable block — in V1's `system + [user]` shape that's
        the user message, which changes per call. Per-block on system is
        what produces measurable hits.
      - `stream` omitted → returns `Message`, not `AsyncStream`
      - `request.messages` is V1.5+; rejected at LLMRequest construction
        (validator); never reaches here in V1.
    """
    if request.cache_control:
        # Per-block ephemeral cache_control on the stable system block.
        # Volatile user/diff content stays outside the cache boundary
        # per spec.md §1476-1478.
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
            f"supported in V1) or an SDK shape change.",
            actual_block_types=actual_types,
        )
    return message.content[0].text


def _translate_anthropic_error(exc: anthropic.APIError) -> LLMProviderError:
    """Map an Anthropic SDK exception to the typed `LLMProviderError`
    subclass per the round-13 mapping table.

    **Order matters** — `isinstance` checks fall through to broader
    parent classes. Two specific orderings are load-bearing:

      - `APITimeoutError` ⊂ `APIConnectionError` in the SDK hierarchy
        (timeouts ARE-A connection errors). The `APITimeoutError`
        check MUST come before `APIConnectionError` or every timeout
        silently routes to `LLMUpstreamError`.
      - `RateLimitError` and other status-code subclasses both inherit
        `APIStatusError` directly; their order among themselves
        doesn't matter, but they must all precede any `APIStatusError`
        fallback (none exists today, but flagged for future).

    The fall-through is `LLMUnknownError` (per round-13/14 abstract-base
    redesign — bare `LLMProviderError` is no longer raisable).
    """
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError(str(exc))
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError(str(exc))
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return LLMAuthError(str(exc))
    if isinstance(exc, anthropic.ConflictError):
        # Round-21 fold per Codex finding: 409 is in Anthropic SDK's
        # default-retry set (alongside 408/429/5xx), so the right
        # taxonomy is `retry_at_layer="node"`, not terminal. Round-20
        # incorrectly bucketed it with 404 as terminal.
        return LLMConflictError(str(exc))
    if isinstance(
        exc,
        (
            anthropic.BadRequestError,
            anthropic.UnprocessableEntityError,
            anthropic.NotFoundError,
        ),
    ):
        # Round-20 fold per Codex finding: 404 (NotFoundError — e.g., a
        # configured model id that the Anthropic catalog doesn't know)
        # is a documented terminal request/config error. 400/422 are
        # also terminal (request shape errors). Mapping all three to
        # LLMInvalidRequestError gives them the right
        # `retry_at_layer="none"` semantics.
        return LLMInvalidRequestError(str(exc))
    if isinstance(exc, anthropic.APIResponseValidationError):
        return LLMInvalidResponseError(str(exc))
    if isinstance(exc, (anthropic.InternalServerError, anthropic.APIConnectionError)):
        return LLMUpstreamError(str(exc))
    # Fall-through for unmapped APIError subclasses.
    return LLMUnknownError(f"unmapped APIError: {type(exc).__name__}: {exc}")
