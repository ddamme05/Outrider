# AnthropicProvider — concrete LLMProvider implementation.
# Owns the Anthropic transport surface; vendor SDK imports stay inside
# `src/outrider/llm/` per the folder-scoped `vendor-sdks-only-in-wrappers`
# invariant (sibling modules within `llm/` may import SDK metadata).
# See specs/2026-05-05-llm-provider-wrapper.md and DECISIONS.md #013/#015/#016.
# DECISIONS.md#056: native host identity + FUP-197 pre-call pricing guard.
"""Anthropic transport: validate, call, normalize, price, and persist.

Construction rejects missing credentials and configured models without rates,
then logs the privacy notice (`DECISIONS.md#015`). `complete()` guards the
closed state, missing persister, and request pricing before the paid call.

The call translates SDK failures into typed, content-clean errors, normalizes
the response, and hashes the prompts. Refusals become empty completions with
a pending refusal error. Run the cache diagnostic before pricing, then price
the returned model; an upstream substitution can still lack a rate.

Persist the event and content together before returning or raising a pending
refusal. Persistence failure halts the review and supersedes that refusal
(`DECISIONS.md#016`). The body's step numbers remain cross-reference markers.
"""

import asyncio
import json
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
    LLMRefusalError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMUnexpectedContentBlocksError,
    LLMUnknownError,
    LLMUpstreamError,
    _canonical_prompt_hash,
    _canonical_system_prompt_hash,
)
from outrider.llm.config import ModelConfig, model_uses_adaptive_thinking
from outrider.llm.host_profiles import ANTHROPIC_CONTRACT_DIGEST, ANTHROPIC_PROFILE_ID
from outrider.llm.pricing import (
    PRICING_VERSION,
    RATE_TABLE,
    compute_cost_usd,
    min_cacheable_tokens,
    normalize_to_pricing_key,
    pricing_key,
)

__all__ = ["AnthropicProvider"]


_PRIVACY_NOTICE_LOGGER = logging.getLogger("outrider.llm.privacy_notice")
_LOGGER = logging.getLogger("outrider.llm.anthropic_provider")


_ZDR_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes"})
_ZDR_FALSY: Final[frozenset[str]] = frozenset({"", "0", "false", "no"})

# Warn once per distinct misconfigured raw value per process. Read only by
# `resolve_zdr_attestation`, which runs on provider construction and on every
# `/privacy` read — so repeated resolution of one invalid env value would
# otherwise warn every time. Process-local, so each worker still warns once.
_WARNED_RAW_VALUES: set[str] = set()

# per Anthropic SDK 0.100 prompt-caching docs: silently-
# disabled-cache diagnostic. Per the docs, prompts shorter than the
# model's min-cacheable threshold (authoritative per-model values:
# `pricing.MIN_CACHEABLE_TOKENS`) are processed without caching, with
# NO error returned.
# Detection: `cache_control=True` request with both
# `cache_creation_input_tokens=0` AND `cache_read_input_tokens=0` in the
# response. Without this signal, an operator who opted into caching sees
# no cost reduction and has no way to discover the root cause without
# aggregating across many events. Process-local set keyed by
# (model, system_prompt_hash) bounds the volume when the parallel-analyze
# fan-out drives many calls with the same prompt (same shape as
# `_WARNED_RAW_VALUES` above).
_WARNED_NONCACHEABLE: set[tuple[str, str]] = set()


# Read timeout for the SDK's httpx client. `complete()` is NON-streaming,
# so no bytes arrive until generation finishes — the read timeout bounds
# the WHOLE generation, not gaps between chunks. It must therefore cover
# the worst LEGITIMATE response the wrapper itself permits: MAX_TOKENS=8192
# at a loaded-Sonnet ~35 tok/s ≈ 234s, plus a TTFT/queuing tail (spikes
# past 30s observed 2026-06-10 — one killed a full eval evidence run).
# 300s covers that arithmetic and stays at half the SDK's own 600s
# default. The original 30s (provider-wrapper spec AC#12, sized to the
# "60-120s typical review" figure) conflated hung-connection fast-fail
# with slow-but-healthy generation; fast hung-call detection properly
# arrives with streaming or the FUP-025 node-retry layer, not by capping
# legitimate work.
_READ_TIMEOUT_SECONDS: Final[float] = 300.0

# Bounded teardown for `AnthropicProvider.aclose()`. 10s is twice the
# default httpx pool-timeout (5s) so a legitimately-slow drain still
# completes; a hung close (e.g., an in-flight request still waiting out
# the read timeout) exceeds this and the wrapper releases without waiting.
_ACLOSE_TIMEOUT_SECONDS: Final[float] = 10.0

# Host-identity triad constants for the anthropic native path (DECISIONS.md#056) are
# centralized in `host_profiles` (the single identity source the lifespan's
# resolve_host_identity + this provider's stamp share, so they can't drift — Codex
# guardrail). Re-bound under these private names so this module's stamping and the
# downstream `from anthropic_provider import _ANTHROPIC_*` importers are unchanged (an
# assignment, not an import-alias, so the names are explicitly defined/exportable).
_ANTHROPIC_PROFILE_ID: Final[str] = ANTHROPIC_PROFILE_ID
_ANTHROPIC_CONTRACT_DIGEST: Final[str] = ANTHROPIC_CONTRACT_DIGEST


def resolve_zdr_attestation(zdr_enabled: bool | None) -> bool:
    """Read ZDR attestation per DECISIONS#015 — operator-attestation only,
    NEVER a per-request header. Constructor kwarg wins; falls back to
    `ANTHROPIC_ZDR_ENABLED` env var.

    Public (was `_resolve_zdr_attestation`) so the `/privacy` disclosure page
    (`api/privacy.py`) reads ZDR through THIS logic — the same truthy/falsy/
    warn-once path the provider uses — so the rendered statement can't drift
    from runtime attestation. Call `resolve_zdr_attestation(None)` for the
    env-only resolution.

    Truthy values (case-insensitive): `"1"`, `"true"`, `"yes"`.
    Falsy values (case-insensitive): `""`, `"0"`, `"false"`, `"no"`.
    Unrecognized values fail closed (no ZDR attestation) AND log a
    WARNING on `outrider.llm.privacy_notice` so the operator sees the
    misconfiguration at construction time .
    The warning fires once per distinct raw value per process, so repeated
    resolution of one invalid value does not warn repeatedly.
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
        reasoning: bool = False,
    ) -> None:
        # Eager api_key validation — SDK doesn't error on missing key,
        # so we surface eagerly per AC#13.
        if not api_key.get_secret_value():
            raise LLMMissingAPIKeyError(
                "AnthropicProvider requires a non-empty api_key; "
                "the Anthropic SDK does not error on missing keys at "
                "construction, so the wrapper validates eagerly."
            )

        # Reasoning fails loud on the anthropic native path (DECISIONS.md#056): V1 has no
        # reasoning toggle here, so OUTRIDER_LLM_REASONING=true must NOT silently no-op into a
        # stamped reasoning_enabled=True. It is meaningful only for a host whose HostProfile
        # declares a reasoning mechanism.
        if reasoning:
            raise LLMInvalidRequestError(
                "AnthropicProvider does not support reasoning (OUTRIDER_LLM_REASONING=true) in "
                "V1 — the native path has no reasoning toggle. Set it only for an "
                "OpenAI-compatible host whose HostProfile declares a reasoning mechanism."
            )

        # Reject missing configured rates before any paid call (AC#24). Normalize
        # dated model IDs to their aliases; returned substitutions are checked later.
        configured_models = {
            model_config.triage_model,
            model_config.analyze_model,
            model_config.standard_analyze_model,
            model_config.synthesize_model,
            model_config.trace_model,
            model_config.patch_model,
        }
        missing = sorted(
            m for m in configured_models if pricing_key(_ANTHROPIC_PROFILE_ID, m) not in RATE_TABLE
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
        self._zdr_enabled = resolve_zdr_attestation(zdr_enabled)
        # Always False on the native path (the fail-loud guard above rejects a True request);
        # stamped on every LLMResponse + LLMCallEvent as the triad's reasoning_enabled.
        self._reasoning_enabled = False
        # Teardown state; `aclose` documents what each of the three is for.
        # Constructing the Lock here is safe because it binds to the running loop
        # on first acquire and `aclose` is its only awaiter, in the same loop.
        self._closed: bool = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

        # SDK client — DefaultAsyncHttpxClient preserves SDK defaults
        # (headers, retry hooks, etc.) while we customize limits/timeout.
        # max_retries=0: retry policy lives in the agent layer per spec.
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key.get_secret_value(),
            max_retries=0,
            timeout=httpx.Timeout(connect=5.0, read=_READ_TIMEOUT_SECONDS, write=30.0, pool=10.0),
            http_client=anthropic.DefaultAsyncHttpxClient(
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            ),
        )

        # No observability-SDK import in core transport (vendor-sdks-only-in-wrappers).
        # LLM-call tracing was removed in DECISIONS.md#058 (LangSmith dropped as a direct
        # dep); `langsmith` is now forbidden in all project code by the import-lint.

        # Privacy startup notice — canonical text per DECISIONS#015
        # point 4. Operator-attestation only; never a per-request header.
        # previous emit had only structured
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

        Fixed lifecycle ordering — see the module docstring. Failures
        surface as typed `LLMProviderError` subclasses; the calling node
        reads `error.retry_at_layer` to decide retry behavior.
        """
        # Step 0: post-teardown guard. `app.state.provider` survives
        # lifespan teardown — a request handler queued by uvicorn's
        # graceful-shutdown sequence (in-flight requests finish after
        # the lifespan yields back) could call `complete()` on a closed
        # client; without this guard, the underlying httpx call surfaces
        # an obscure `RuntimeError("Cannot send a request, as the client
        # has been closed.")` deep in the SDK. Loud-fail at the wrapper
        # boundary is the right shape.
        if self._closed:
            raise LLMUnknownError(
                "AnthropicProvider.complete() called after aclose(); "
                "provider is closed and cannot accept new requests"
            )
        # Step 1: fail-closed pre-call.
        # If persister is None, raise BEFORE the SDK call so the SDK
        # never sees a request from a misconfigured provider.
        if self._persister is None:
            raise LLMPersisterNotWiredError(
                "AnthropicProvider.complete() called with persister=None; "
                "production deployments must wire a real LLMExchangePersister "
                "per DECISIONS#016 single-transaction-insert contract."
            )

        # Step 1b (FUP-197): pre-flight model check BEFORE the paid SDK call — parity
        # with OpenAICompatibleProvider. A request.model not priced under the anthropic
        # host would otherwise reach the SDK and only fail at the step-8 cost lookup
        # AFTER the billed call (orphan/unpriced-cost row). The constructor validates
        # CONFIGURED models; this guards a request.model that bypassed config (a routing
        # bug). RATE_TABLE membership under "anthropic" is sufficient here — anthropic
        # serves exactly its priced models (unlike the OpenAI-compatible provider, whose
        # configured set is narrower than RATE_TABLE).
        if pricing_key(_ANTHROPIC_PROFILE_ID, request.model) not in RATE_TABLE:
            raise LLMInvalidRequestError(
                f"AnthropicProvider.complete(): request.model={request.model!r} maps to "
                f"pricing key {pricing_key(_ANTHROPIC_PROFILE_ID, request.model)!r}, which is "
                f"not in RATE_TABLE — refusing the paid SDK call before an unpriced/orphan "
                f"cost row results. Add the model to RATE_TABLE + the provider's ModelConfig."
            )

        # Step 2 + 3: translate request to SDK shape.
        sdk_kwargs = _build_sdk_kwargs(request)

        sdk_response, latency_ms = await self._call_sdk(sdk_kwargs)

        # Step 5: refusal gate (capture → persist → raise), then shape validation.
        # A safety-classifier decline returns HTTP 200 with stop_reason="refusal"
        # (GA on the adaptive-thinking generation) — a COMPLETED exchange (the model
        # responded), so per DECISIONS.md#016 Amended 2026-06-30 it is persisted and
        # costed like any other call, and the terminal `LLMRefusalError` is raised AFTER
        # the step-9 persist (vs transport errors, which raise at step 4 with no completed
        # exchange to record). `completion=""` — a pre-output refusal carries an empty
        # content array and a mid-stream refusal's partial text is NOT extracted; the
        # refusal is carried durably by `finish_reason="refusal"`. Outrider asks the model
        # to find vulnerabilities, so a decline must halt the review loudly (output-
        # boundary #6: a decline is not a zero-finding result). `stop_details` is populated
        # only on a refusal; guard the attribute access.
        pending_refusal: LLMRefusalError | None = None
        if sdk_response.stop_reason == "refusal":
            stop_details = getattr(sdk_response, "stop_details", None)
            category = getattr(stop_details, "category", None)
            pending_refusal = LLMRefusalError(
                f"Anthropic declined the request (stop_reason='refusal', "
                f"category={category!r}) for model {request.model!r}; review halts. "
                f"A refusal is terminal — retrying the same prompt will not help.",
                category=category,
            )
            text = ""
        else:
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
            # Host-identity triad (DECISIONS.md#056): the anthropic native path is a constant
            # (profile_id="anthropic", reasoning off, constant digest). Stamped together so the
            # coherence envelope holds; the LLMCallEvent below mirrors these from the response.
            profile_id=_ANTHROPIC_PROFILE_ID,
            reasoning_enabled=self._reasoning_enabled,
            profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
        )

        # Step 7: hash the prompts.
        prompt_hash = _canonical_prompt_hash(
            system_prompt=request.system_prompt, user_prompt=request.user_prompt
        )
        system_prompt_hash = _canonical_system_prompt_hash(request.system_prompt)

        _warn_if_cache_silently_disabled(
            request=request,
            response=response,
            system_prompt_hash=system_prompt_hash,
            refused=pending_refusal is not None,
        )

        cost_decimal = _compute_response_cost(request, response)

        await _persist_exchange(
            persister=self._persister,
            request=request,
            response=response,
            prompt_hash=prompt_hash,
            system_prompt_hash=system_prompt_hash,
            cost_decimal=cost_decimal,
        )

        # Step 10: a captured refusal raises AFTER the exchange is persisted (the
        # completed-but-declined exchange is now in the audit + cost); else return.
        if pending_refusal is not None:
            raise pending_refusal
        return response

    async def _call_sdk(self, sdk_kwargs: dict[str, Any]) -> tuple[Message, int]:
        """Make the paid call and return the SDK response with its latency in ms.

        Anthropic errors and any other ordinary `Exception` from the SDK path leave
        as typed `LLMProviderError` subclasses. Cancellation, `SystemExit` and
        `KeyboardInterrupt` are not `Exception` subclasses and propagate untouched,
        which is what keeps task cancellation and interrupt working.
        """
        # Catch the SDK root: auth failures can be outside its APIError subtree.
        # Measure latency with the monotonic clock.
        t_start_ns = time.perf_counter_ns()
        try:
            sdk_response: Message = await self._client.messages.create(**sdk_kwargs)
        except anthropic.AnthropicError as exc:
            # Vendor exception messages can echo content. Suppress the traceback
            # chain; the typed wrapper preserves the operational failure class.
            raise _translate_anthropic_error(exc) from None
        except Exception as exc:
            # A close race can raise outside the SDK hierarchy. Keep it typed and
            # content-clean; ordinary Exception leaves cancellation untouched.
            if self._closed:
                raise LLMUnknownError(
                    "AnthropicProvider.complete() raced with aclose(); "
                    "provider is closed and cannot accept new requests"
                ) from None
            raise LLMUnknownError(
                f"AnthropicProvider non-Anthropic SDK failure: <{type(exc).__name__}>"
            ) from None
        latency_ms = (time.perf_counter_ns() - t_start_ns) // 1_000_000

        return sdk_response, int(latency_ms)

    async def aclose(self) -> None:
        """Close the shared SDK client with a bounded, shielded drain (FUP-011).

        The lock and closed flag make concurrent/repeated close calls idempotent.
        `wait_for` bounds this caller's wait; `shield` lets the inner drain continue
        after timeout. `_close_task` keeps a strong reference until its callback
        consumes any exception and releases it. Only exception types are logged,
        because vendor messages can contain request or response text.
        """
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._close_task = asyncio.create_task(self._client.close())
            self._close_task.add_done_callback(self._clear_close_task)
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._close_task),
                    timeout=_ACLOSE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                _LOGGER.warning(
                    "AnthropicProvider.aclose() exceeded %.0fs timeout; "
                    "leaking connection pool to OS reaper rather than blocking "
                    "lifespan teardown (the shielded close task continues in "
                    "the background, retained via self._close_task until "
                    "completion or event-loop shutdown cancels it)",
                    _ACLOSE_TIMEOUT_SECONDS,
                )

    def _clear_close_task(self, task: asyncio.Task[None]) -> None:
        """Done-callback: retrieve any exception and release the strong reference.

        See `aclose` for why both matter. Cancellation is the expected outcome when
        the wait timed out and event-loop shutdown collected the shielded task, so
        it is not warned about.
        """
        if task.cancelled():
            # Cancellation by event-loop shutdown is the expected path
            # when the wait_for timed out; no warning needed.
            pass
        else:
            exc = task.exception()
            if exc is not None:
                _LOGGER.warning(
                    "AnthropicProvider close task raised %s after aclose() "
                    "returned; exception consumed to prevent unretrieved-"
                    "exception log spam at task GC (lifespan teardown is "
                    "already complete; this is the leak-on-rare-teardown "
                    "trade-off documented on aclose).",
                    type(exc).__name__,
                )
        if self._close_task is task:
            self._close_task = None


async def _persist_exchange(
    *,
    persister: LLMExchangePersister,
    request: LLMRequest,
    response: LLMResponse,
    prompt_hash: str,
    system_prompt_hash: str,
    cost_decimal: Decimal,
) -> None:
    """Build the `LLMCallEvent` and hand it to the persister.

    A failure here raises `LLMPersisterError`: the SDK call already succeeded and
    was billed, but no audit row landed, so the calling node halts the review.

    The persister arrives as an argument because `complete()`'s fail-closed guard
    has already established it is not None; re-checking here would add a second
    runtime gate for a condition the caller settled.
    """
    # Step 9: build LLMCallEvent + await persister.persist().
    event = LLMCallEvent(
        review_id=request.review_id,
        timestamp=datetime.now(UTC),
        is_eval=request.is_eval,
        # Canonical metadata-only fields per spec.md §8.3.
        model=response.model,
        # Mirror the provider's normalized stop reason onto the event so a refusal is
        # distinguishable from a zero-output success on metadata-only replay
        # (DECISIONS.md#016 Amended 2026-06-30); the persister cross-checks it.
        finish_reason=response.finish_reason,
        node_id=request.node_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached_tokens=response.cache_read_tokens,
        cost_usd=float(cost_decimal),
        pricing_version=PRICING_VERSION,
        # Triad mirrored from the response (single source) so the persister cross-check
        # verifies this identity (DECISIONS.md#056).
        profile_id=response.profile_id,
        reasoning_enabled=response.reasoning_enabled,
        profile_contract_digest=response.profile_contract_digest,
        latency_ms=response.latency_ms,
        prompt_hash=prompt_hash,
        cache_hit=(response.cache_read_tokens > 0),
        context_summary=request.context_summary,
        prompt_template_version=request.prompt_template_version,
        system_prompt_hash=system_prompt_hash,
        degraded_mode=request.degraded_mode,
        # Preserve the typed degradation cause for metadata-only replay.
        degradation_reason=request.degradation_reason,
        phase_key=request.phase_key,
        # Constrained-decoding provenance (FUP-096): identical prompt
        # bytes + template version can produce different output
        # populations once `output_config.format` exists — metadata-only
        # replay/ops need the request-format identity on the stream.
        response_format_digest=request.response_format_digest,
    )
    try:
        await persister.persist(event, request, response)
    except Exception as exc:
        # Persistence failure halts after a billed call and supersedes any pending
        # refusal. Preserve messages/causes only for the metadata-only allowlist;
        # unknown exceptions may contain prompt or completion text, so suppress
        # their traceback chain as well as their message (DECISIONS.md#016).
        from outrider.audit.persister import METADATA_ONLY_EXCEPTION_TYPES

        if isinstance(exc, METADATA_ONLY_EXCEPTION_TYPES):
            raise LLMPersisterError(
                f"Persister failed after successful SDK call: {exc}. "
                f"The audit row did not land; calling node halts the review."
            ) from exc
        raise LLMPersisterError(
            f"Persister failed after successful SDK call: "
            f"<{type(exc).__name__}>. "
            f"The audit row did not land; calling node halts the review."
        ) from None


def _warn_if_cache_silently_disabled(
    *,
    request: LLMRequest,
    response: LLMResponse,
    system_prompt_hash: str,
    refused: bool,
) -> None:
    """Warn once per model/system-prompt hash on zero cache counters.

    Refusals are excluded; an unknown documented floor suppresses the warning.
    This records a candidate cache diagnostic, not proof of the cause.
    """  # Prompt-caching silently-disabled diagnostic.
    # Per Anthropic SDK 0.100 prompt-caching docs, prompts shorter than
    # the model's min-cacheable threshold (authoritative per-model
    # values: `pricing.MIN_CACHEABLE_TOKENS`) are processed without
    # caching with NO error returned. `cache_control=True` AND both
    # cache_creation/read tokens at 0 is the CANDIDATE signal: the
    # first-ever cache write has cache_creation > 0, and eviction
    # rewrites with cache_creation > 0, so neither reaches here. A
    # refusal carrying 0/0 is the known counterexample — those counters
    # arise for an unrelated reason and are excluded below. The floor lookup then
    # decides whether to warn at all: a model with no documented floor
    # cannot have a cache miss attributed to a short prompt.
    #
    # The dedup key uses `normalize_to_pricing_key(response.model)` so
    # dated aliases share a warn-once budget with their undated base
    # (e.g., `claude-haiku-4-5-20251001` and `claude-haiku-4-5` are
    # treated as the same model for warning suppression — they share
    # the same cache threshold by definition). The literal
    # `response.model` is still logged in the extras for operator
    # visibility into what actually executed. Same normalization rule
    # as the cost-computation lookup at step 8.
    if (
        request.cache_control
        and response.cache_read_tokens == 0
        and response.cache_write_tokens == 0
        # When a refusal carries 0/0 cache counters they do not establish a
        # below-floor prompt — the model declined — so skip the spurious warning.
        and not refused
    ):
        cache_warn_key = (
            normalize_to_pricing_key(response.model),
            system_prompt_hash,
        )
        if cache_warn_key not in _WARNED_NONCACHEABLE:
            _WARNED_NONCACHEABLE.add(cache_warn_key)
            # Floor derived from the authoritative table rather than
            # hardcoded prose, so a floor change can't strand a stale
            # number in the operator-facing message. Defensive lookup:
            # `response.model` is SDK-returned and could (in principle)
            # be a substituted id outside the table → KeyError.
            floor_text: str | None
            try:
                floor = min_cacheable_tokens(_ANTHROPIC_PROFILE_ID, response.model)
            except KeyError:
                floor_text = "see llm/pricing.py::MIN_CACHEABLE_TOKENS"
            else:
                # None = the DECISIONS.md#056 unknown-floor sentinel: the model is
                # priced but its min-cacheable threshold is undocumented, so a cache
                # miss can't be attributed to a below-floor prompt. Suppress the
                # diagnostic (per the `min_cacheable_tokens` contract) rather than
                # log a confusing "None tokens" message.
                floor_text = (
                    None
                    if floor is None
                    else f"{floor} tokens for {normalize_to_pricing_key(response.model)}"
                )
            if floor_text is not None:
                _LOGGER.warning(
                    "anthropic_provider cache_control=True but neither "
                    "cache_creation_input_tokens nor cache_read_input_tokens "
                    "fired; system prompt likely below the model's "
                    "min-cacheable threshold (%s — see Anthropic "
                    "prompt-caching docs). cache_control=True will produce "
                    "no cost savings on this prompt until it grows past the "
                    "threshold or cache_control is removed.",
                    floor_text,
                    extra={
                        "model": response.model,
                        "request_model": request.model,
                        "system_prompt_hash": system_prompt_hash,
                        "review_id": str(request.review_id),
                        "node_id": request.node_id,
                    },
                )


def _compute_response_cost(request: LLMRequest, response: LLMResponse) -> Decimal:
    """Compute from the response model; raise when its current pricing key has no rate.

    The response model may differ from the requested one, and the rate table may
    have changed since the constructor's eager check — either reaches this raise.
    """
    # Price the returned model so cost and audit identity agree. Constructor
    # checks cover configured models; an upstream substitution can still miss
    # the rate table, including after dated-slug normalization.
    try:
        cost_decimal: Decimal = compute_cost_usd(
            _ANTHROPIC_PROFILE_ID,
            response.model,
            input_tokens=response.input_tokens,
            cache_write_tokens=response.cache_write_tokens,
            cache_read_tokens=response.cache_read_tokens,
            output_tokens=response.output_tokens,
        )
    except KeyError as exc:
        missing_key = pricing_key(_ANTHROPIC_PROFILE_ID, response.model)
        raise LLMPricingMissingError(
            f"Model {response.model!r} maps to host-qualified pricing key "
            f"{missing_key!r}, which is not in RATE_TABLE at "
            f"complete() step 8 (request.model={request.model!r}). "
            f"Constructor's eager validation covers configured models "
            f"only — this can fire if the SDK substitutes the model "
            f"in its response (alias resolution, deprecation routing). "
            f"Add the key to RATE_TABLE + bump "
            f"PRICING_VERSION to fix.",
            missing_models=[str(missing_key)],
        ) from exc

    return cost_decimal


def _build_sdk_kwargs(request: LLMRequest) -> dict[str, Any]:
    """Shape a non-streaming Anthropic request from the project DTO.

    The system text is the sole cacheable block (`docs/spec.md` §9.5);
    per-call user/diff text stays outside its breakpoint. Adaptive-thinking
    models disable thinking and omit sampling parameters to keep the expected
    text-block shape. A response schema uses `output_config.format`; refusal
    and truncation still need downstream handling (FUP-096).
    """
    if request.cache_control:
        # Per-block ephemeral cache_control on the stable system block.
        # Volatile user/diff content stays outside the cache boundary
        # per spec.md §9.5 (Prompt caching for cost reduction).
        system_param: str | list[TextBlockParam] = [
            TextBlockParam(
                type="text",
                text=request.system_prompt,
                cache_control={"type": "ephemeral"},
            )
        ]
    else:
        system_param = request.system_prompt
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "system": system_param,
        "messages": [{"role": "user", "content": request.user_prompt}],
    }
    if model_uses_adaptive_thinking(request.model):
        # Adaptive-thinking generation (Sonnet 5+, Opus 4.7+): non-default
        # `temperature`/`top_p`/`top_k` are rejected with a 400, so OMIT the
        # sampling param entirely (the model runs at its default sampling).
        # Adaptive thinking is ON by default and would add a `thinking` block,
        # breaking the single-text-block contract — disable it explicitly so the
        # response stays one TextBlock (`_extract_single_text_block`). With
        # structured output (`output_config.format` below) the single block is
        # the schema-valid JSON; disabling thinking keeps that shape.
        kwargs["thinking"] = {"type": "disabled"}
    else:
        # Current-generation models (Haiku 4.5, Sonnet 4.6, Opus 4.6) accept
        # `temperature` and have thinking off by default — keep the legacy shape.
        kwargs["temperature"] = request.temperature
    if request.response_schema_json is not None:
        # Constrained decoding (specs/2026-06-12-constrained-decoding.md,
        # FUP-096): `output_config.format` per the pinned structured-outputs
        # docs — the API guarantees schema-valid JSON in the single text
        # block, eliminating the invalid-JSON rejection class at the source.
        # Two documented escapes remain (stop_reason "refusal" and
        # "max_tokens" may not match the schema), which is why the parser's
        # rejection path and `strip_outer_json_fence` stay as belt-and-
        # suspenders. An unsupported-schema 400 surfaces through the normal
        # APIError → typed-error translation (fail-loud, no fallback).
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": json.loads(request.response_schema_json),
            }
        }
    return kwargs


def _extract_single_text_block(message: Message) -> str:
    """Validate response is exactly one `TextBlock`; return its text.

    Per AC#10, multi-block responses (extended thinking, tool use, etc.)
    fail loud rather than silently flatten or drop. V1's single-text-block
    assumption holds because the wrapper either passes no `thinking` kwarg
    (current-generation models — thinking off by default) or explicitly
    disables it (adaptive-thinking models via `thinking={"type":"disabled"}`
    in `_build_sdk_kwargs`); both yield a single `TextBlock`. A refusal
    (stop_reason="refusal", empty content) is caught upstream as
    `LLMRefusalError` before reaching here.
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


def _safe_anthropic_diagnostics(
    exc: anthropic.AnthropicError,
) -> tuple[object, object, object]:
    """Extract ONLY bounded, non-content fields for `LLMProviderError.attach_diagnostics`
    (which re-validates each). Reads the HTTP `status_code`, the `request_id`, and the error
    `type` — a bounded category enum. NEVER the error message, the body text, or `str(exc)`;
    reads exactly the `body["error"]["type"]` key, not the message. Returns raw values; the
    base-class validator is the single gate that admits or drops each."""
    status = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    category: object = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error_payload = body.get("error")
        if isinstance(error_payload, dict):
            category = error_payload.get("type")  # bounded enum; `attach_diagnostics` allowlists it
    return status, request_id, category


def _translate_anthropic_error(exc: anthropic.AnthropicError) -> LLMProviderError:
    """Map an Anthropic SDK exception to the typed `LLMProviderError` and attach bounded,
    privacy-safe diagnostic metadata (status / request id / error category — never the message
    or body). The typed IDENTITY plus this metadata is the operational signal."""
    provider_error = _classify_anthropic_error(exc)
    status, request_id, category = _safe_anthropic_diagnostics(exc)
    return provider_error.attach_diagnostics(
        http_status=status, request_id=request_id, error_category=category
    )


def _classify_anthropic_error(exc: anthropic.AnthropicError) -> LLMProviderError:
    """Map an Anthropic SDK exception to the typed `LLMProviderError`
    subclass per the mapping table.

    Accepts `anthropic.AnthropicError` (the SDK exception root), not
    just `APIError`. The broader signature catches `WorkloadIdentityError`
    and any future non-APIError subclasses of AnthropicError; they fall
    through to `LLMUnknownError` rather than escaping the wrapper.

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

    The fall-through is `LLMUnknownError` .
    """
    # Metadata-only by contract: NEVER pass `str(exc)` or any other text
    # extracted from the SDK exception body to the wrapper class
    # constructor. Anthropic SDK error messages render the underlying
    # response body, which can echo prompt/completion fragments from the
    # request (e.g., context-length errors that quote the offending text).
    # Passing `str(exc)` would store the body in `Exception.args[0]`,
    # exposing it via `repr(exc)`, `str(exc)`, and traceback rendering.
    # The wrapper class IDENTITY is the operational signal; the SDK
    # exception type name (a safe-by-construction class name, NOT data)
    # is the only attribute we surface, and only for the unknown branch.
    # Defense-in-depth for the persister-side metadata-only contract.
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError()
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError()
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return LLMAuthError()
    if isinstance(exc, anthropic.ConflictError):
        # 409 is in Anthropic SDK's
        # default-retry set (alongside 408/429/5xx), so the right
        # taxonomy is `retry_at_layer="node"`, not terminal.
        # incorrectly bucketed it with 404 as terminal.
        return LLMConflictError()
    if isinstance(
        exc,
        (
            anthropic.BadRequestError,
            anthropic.UnprocessableEntityError,
            anthropic.NotFoundError,
        ),
    ):
        # 404 (NotFoundError — e.g., a
        # configured model id that the Anthropic catalog doesn't know)
        # is a documented terminal request/config error. 400/422 are
        # also terminal (request shape errors). Mapping all three to
        # LLMInvalidRequestError gives them the right
        # `retry_at_layer="none"` semantics.
        return LLMInvalidRequestError()
    if isinstance(exc, anthropic.APIResponseValidationError):
        return LLMInvalidResponseError()
    if isinstance(exc, (anthropic.InternalServerError, anthropic.APIConnectionError)):
        return LLMUpstreamError()
    # Fall-through for unmapped AnthropicError subclasses (covers any
    # non-APIError SDK exception like WorkloadIdentityError, plus future
    # additions to the SDK hierarchy that we haven't mapped yet). The
    # SDK type NAME (a Python class identifier, not data) is preserved
    # here so operators can see which SDK shape was unmapped — `str(exc)`
    # remains forbidden.
    return LLMUnknownError(f"unmapped AnthropicError: {type(exc).__name__}")
