# GLMProvider — concrete LLMProvider for GLM 5.2 on Baseten (OpenAI-compatible).
# Owns the openai transport surface; vendor SDK imports stay inside
# `src/outrider/llm/` per the folder-scoped `vendor-sdks-only-in-wrappers`
# invariant (trust boundary #8). Second deployment mode alongside
# AnthropicProvider — see GLM_PROVIDER_THOUGHTS.md.
"""GLM 5.2 (Baseten) concrete provider.

Mirror of `AnthropicProvider` using the `openai` SDK against Baseten's
OpenAI-compatible Model APIs (`https://inference.baseten.co/v1`, model
`zai-org/GLM-5.2`). Same `LLMProvider` Protocol, same `LLMRequest` /
`LLMResponse` shapes, same metadata-only error discipline, same per-call
`LLMCallEvent` + `compute_cost_usd()` cost path. Only the wire surface
that differs from Anthropic is pinned here; everything else copies the
existing provider's contract.

Wire deltas from Anthropic (verified against the aegis-docs mirror +
introspection of the installed openai==2.44.0):

  - Token accounting: Baseten `usage.prompt_tokens` INCLUDES cached tokens
    (cached is a SUBSET), whereas Anthropic's `input_tokens` EXCLUDES
    cache. `compute_cost_usd` charges `input_tokens` at the full input
    rate, so the uncached portion is `prompt_tokens - cached_tokens`. The
    `*_details` objects are nullable — guard before nested access.
  - Model identity: cost and the audit event key on the REQUEST-side model id,
    not `response.model`. request.model is deterministic + pre-validated
    (constructor GLM-family + per-call configured-set guards), so it's the
    reliable cost key. The live probe returned `response.model` populated with
    the slug, but Baseten's GLM library example shows it can echo `""` — keying
    on the request side is correct regardless of what `response.model` carries.
  - Caching: automatic prefix caching, no `cache_control` marker and no
    cache-write token class → `cache_write_tokens=0`. The Anthropic
    silently-disabled-cache diagnostic does not apply.
  - Structured output: `response_format={"type":"json_schema",
    "json_schema":{"name",strict:true,"schema"}}` (name required), not
    Anthropic's `output_config.format`. NOTE: GLM wraps the JSON in a markdown
    code fence even under strict mode (confirmed live) — the wrapper returns the
    raw text and the node parsers strip it via `strip_outer_json_fence` (same as
    Anthropic's occasional fences). Schema CONFORMANCE inside the fence is a yield
    question the eval scorecard measures (non-conforming → Pydantic rejects →
    fewer findings, never corrupted output).
  - Reasoning: opt-in via `extra_body={"chat_template_args":
    {"enable_thinking": <bool>}}`; off by default. `reasoning_content`
    (when on) is an untyped extra field, stripped — only `message.content`
    becomes `LLMResponse.text`. Reasoning tokens are already inside
    `completion_tokens`; never added to the cost path.

`complete()` step ordering (mirrors AnthropicProvider):
  0. post-teardown guard (`_closed`).
  1. fail-closed pre-call: persister=None → raise.
  2. translate `LLMRequest` → openai `chat.completions.create` kwargs.
  3. await create(stream=False); catch openai.OpenAIError → typed subclass.
  4. extract assistant text (exactly one choice, non-empty content).
  5. normalize usage (§8a cached subtraction) → LLMResponse + latency.
  6. compute prompt/system hashes.
  7. compute cost_usd (keyed on the request model id).
  8. build LLMCallEvent; await persister.persist(); wrap failures.
  9. return LLMResponse.
"""

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any, Final

import httpx
import openai
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
from outrider.llm.pricing import (
    PRICING_VERSION,
    RATE_TABLE,
    compute_cost_usd,
    normalize_to_pricing_key,
)

__all__ = ["BASETEN_BASE_URL", "GLM_MODEL_ID", "GLMProvider"]


# Baseten hosted Model APIs (OpenAI-compatible). GLM 5.2 is a hosted model,
# so this shared endpoint is correct (NOT the per-deployment surface).
BASETEN_BASE_URL: Final[str] = "https://inference.baseten.co/v1"
GLM_MODEL_ID: Final[str] = "zai-org/GLM-5.2"

# GLM/Baseten model family. The constructor restricts configured models to this
# pattern so a priced-but-non-GLM slug can't be configured and then routed to the
# Baseten endpoint — RATE_TABLE holds the Anthropic models too, so pricing coverage
# alone is NOT "servable by GLMProvider". The host-aware ModelConfig refactor
# (build phase) generalizes per-host validation; this is the spike floor.
_VALID_GLM_MODEL_PATTERN: Final = re.compile(r"^zai-org/GLM-\d+(\.\d+)?$")


_PRIVACY_NOTICE_LOGGER = logging.getLogger("outrider.llm.privacy_notice")
_LOGGER = logging.getLogger("outrider.llm.glm_provider")


# Read timeout sized to the worst legitimate non-streaming generation the
# wrapper permits (MAX_TOKENS=8192), matching AnthropicProvider's 300s.
_READ_TIMEOUT_SECONDS: Final[float] = 300.0
# Bounded teardown for aclose() — twice httpx's default pool-timeout.
_ACLOSE_TIMEOUT_SECONDS: Final[float] = 10.0


class GLMProvider:
    """Concrete `LLMProvider` for GLM 5.2 on Baseten via the `openai` SDK.

    Constructor enforces eager validation (non-empty api_key + pricing
    coverage for every configured model) and emits an egress privacy
    notice. The thin V1 spike form takes the model id(s) explicitly
    rather than the Anthropic-locked `ModelConfig` (whose regex rejects
    the `zai-org/GLM-5.2` slug); the host-aware `ModelConfig` refactor is
    deferred to the build phase.
    """

    def __init__(
        self,
        api_key: SecretStr,
        *,
        persister: LLMExchangePersister | None = None,
        models: tuple[str, ...] = (GLM_MODEL_ID,),
        base_url: str = BASETEN_BASE_URL,
        enable_thinking: bool = False,
    ) -> None:
        # Eager api_key validation — the SDK does not error on missing key
        # at construction, so surface it here rather than mid-review.
        if not api_key.get_secret_value():
            raise LLMMissingAPIKeyError(
                "GLMProvider requires a non-empty api_key (Baseten key); the "
                "openai SDK does not error on missing keys at construction, so "
                "the wrapper validates eagerly. NOTE: the openai SDK defaults "
                "api_key from OPENAI_API_KEY — GLMProvider passes BASETEN_API_KEY "
                "explicitly, never the env default."
            )

        # Restrict configured models to the GLM/Baseten family BEFORE the pricing
        # check. Pricing coverage is necessary but NOT sufficient: RATE_TABLE also
        # holds the Anthropic models, so a claude-* slug would otherwise be accepted
        # here and then routed to Baseten by the per-call guard. The family pattern
        # + the pricing check together mean self._models is always GLM-servable AND
        # priced, which is what the per-call `request.model in self._models` relies on.
        non_glm = sorted(m for m in models if not _VALID_GLM_MODEL_PATTERN.match(m))
        if non_glm:
            raise LLMInvalidRequestError(
                f"GLMProvider configured with non-GLM model(s) {non_glm!r}; "
                f"GLMProvider only serves GLM/Baseten models matching "
                f"{_VALID_GLM_MODEL_PATTERN.pattern!r}. (Being in RATE_TABLE is not "
                f"enough — the Anthropic models are priced too.)"
            )

        # Eager pricing-coverage validation — eliminates a KeyError between
        # SDK success and the persister write. GLM ids carry no dated suffix,
        # so normalization is a no-op, but keep it for symmetry with the table.
        missing = sorted(m for m in models if normalize_to_pricing_key(m) not in RATE_TABLE)
        if missing:
            raise LLMPricingMissingError(
                f"GLMProvider construction: configured model(s) {missing!r} have "
                f"no entry in llm.pricing.RATE_TABLE. Add the row + bump "
                f"PRICING_VERSION before using these models.",
                missing_models=missing,
            )

        self._api_key = api_key
        self._models = tuple(models)
        self._persister = persister
        # NOTE: enable_thinking changes GLM output (reasoning on/off alters the
        # findings) but is a provider-construction flag — it does NOT flow into
        # LLMRequest, so it is NOT part of the analyze file-cache key (which keys
        # on model + prompt + version). Toggling it against a WARM cache serves
        # stale opposite-mode findings on every hit, silently. Productionizing GLM
        # mode must fold reasoning state into the cache key (or run reasoning
        # experiments against a cold/shadow cache).
        self._enable_thinking = enable_thinking
        self._base_url = base_url

        # aclose() idempotency machinery (mirror of AnthropicProvider).
        self._closed: bool = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

        # max_retries=0: retry policy lives in the agent/node layer, same as
        # the Anthropic provider. The openai SDK would otherwise auto-retry
        # connection/408/409/429/5xx, double-handling the node-layer retry.
        self._client = openai.AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            base_url=base_url,
            max_retries=0,
            timeout=httpx.Timeout(connect=5.0, read=_READ_TIMEOUT_SECONDS, write=30.0, pool=10.0),
        )

        # Egress privacy notice — GLM is a Zhipu-origin model hosted on
        # Baseten US infra; code/prompts egress to Baseten, not Zhipu. The
        # model-provenance vs data-residency distinction is an operator
        # concern (GLM_PROVIDER_THOUGHTS.md §18).
        _PRIVACY_NOTICE_LOGGER.info(
            "privacy_notice glm_provider egress=inference.baseten.co model=%s "
            "host=baseten; GLM is a Zhipu-origin model served on Baseten US "
            "infrastructure (prompts egress to Baseten, not Zhipu).",
            GLM_MODEL_ID,
            extra={
                "privacy_notice": True,
                "egress_destination": "inference.baseten.co",
                "model_origin": "zhipu",
                "host": "baseten",
            },
        )

    def __repr__(self) -> str:
        persister_status = "wired" if self._persister is not None else "none"
        return (
            f"<GLMProvider models={self._models!r} base_url={self._base_url!r} "
            f"persister={persister_status} enable_thinking={self._enable_thinking}>"
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send `request` to Baseten/GLM; return `LLMResponse`.

        Failures surface as typed `LLMProviderError` subclasses; the calling
        node reads `error.retry_at_layer` to decide retry behavior.
        """
        # Step 0: post-teardown guard.
        if self._closed:
            raise LLMUnknownError(
                "GLMProvider.complete() called after aclose(); provider is "
                "closed and cannot accept new requests"
            )
        # Step 1: fail-closed pre-call.
        if self._persister is None:
            raise LLMPersisterNotWiredError(
                "GLMProvider.complete() called with persister=None; production "
                "deployments must wire a real LLMExchangePersister per "
                "DECISIONS#016 single-transaction-insert contract."
            )

        # Step 1b: pre-flight model check — BEFORE the paid SDK call. The request
        # model must be one this provider is CONFIGURED to serve (self._models),
        # not merely priced: RATE_TABLE ALSO holds the Anthropic models, so a
        # priced-but-unconfigured slug (e.g. a claude-* model) would otherwise sail
        # through and hit the Baseten endpoint — a billed call for a model this
        # provider doesn't serve, with the wrong-rate cost / orphan audit row that
        # follows. The constructor's eager check guarantees every configured model
        # is priced, so passing this implies the step-7 cost lookup succeeds.
        if request.model not in self._models:
            raise LLMInvalidRequestError(
                f"GLMProvider.complete(): request.model={request.model!r} is not in "
                f"this provider's configured model set {self._models!r}; refusing "
                f"the paid SDK call. (Being in RATE_TABLE is not enough — the "
                f"Anthropic models are priced too.) Construct GLMProvider with this "
                f"model in `models`."
            )

        # Step 2: translate request → SDK kwargs.
        sdk_kwargs = _build_sdk_kwargs(request, enable_thinking=self._enable_thinking)

        # Step 3: SDK call + exception translation. Catch the SDK exception
        # root `openai.OpenAIError` (broader than APIError, mirroring the
        # AnthropicError catch) so no vendor exception escapes complete().
        # `from None`: openai error str() renders the response body, which
        # can echo prompt fragments — drop it (metadata-only contract).
        t_start_ns = time.perf_counter_ns()
        try:
            sdk_response = await self._client.chat.completions.create(**sdk_kwargs)
        except openai.OpenAIError as exc:
            raise _translate_openai_error(exc) from None
        except Exception as exc:
            # Non-openai exception leaking from the SDK call (e.g. an httpx
            # RuntimeError from a close-race). The Step 0 check is best-effort,
            # not atomic vs aclose().
            if self._closed:
                raise LLMUnknownError(
                    "GLMProvider.complete() raced with aclose(); provider is "
                    "closed and cannot accept new requests"
                ) from None
            raise LLMUnknownError(
                f"GLMProvider non-openai SDK failure: <{type(exc).__name__}>"
            ) from None
        latency_ms = (time.perf_counter_ns() - t_start_ns) // 1_000_000

        # Step 4: extract assistant text (fail loud on unexpected shape).
        text, finish_reason = _extract_assistant_text(sdk_response)

        # Step 5: usage normalization (§8a). The whole usage object can be
        # None on the SDK type; non-streaming GLM populates it, but fail loud
        # if absent rather than silently zero the token contract.
        usage = sdk_response.usage
        if usage is None:
            raise LLMInvalidResponseError()
        ptd = usage.prompt_tokens_details
        raw_cached = (ptd.cached_tokens or 0) if ptd is not None else 0
        # §8a: prompt_tokens INCLUDES cached (cached is a SUBSET); subtract to get
        # the uncached input the cost table charges at the full input rate. Cap
        # cached at prompt_tokens so a malformed cached > prompt can't drive
        # input_tokens negative OR make cache_read exceed the prompt — input +
        # cache_read must stay == prompt_tokens (self-consistent audited counts).
        cached_tokens = min(raw_cached, usage.prompt_tokens)
        input_tokens = usage.prompt_tokens - cached_tokens
        output_tokens = usage.completion_tokens

        # Model identity: key cost + the audit event on the REQUEST model id, not
        # response.model. request.model is deterministic + pre-validated, so it's
        # the reliable cost key. The live probe returned response.model populated
        # with the slug, but Baseten's GLM library example shows it can echo "" —
        # we never depend on response.model regardless of what it carries.
        model_id = request.model

        response = LLMResponse(
            text=text,
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached_tokens,
            cache_write_tokens=0,  # GLM/Baseten has no cache-write token class.
            finish_reason=finish_reason,
            latency_ms=int(latency_ms),
        )

        # Step 6: hash the prompts.
        prompt_hash = _canonical_prompt_hash(
            system_prompt=request.system_prompt, user_prompt=request.user_prompt
        )
        system_prompt_hash = _canonical_system_prompt_hash(request.system_prompt)

        # Step 7: compute cost_usd. KeyError is unreachable given the eager
        # constructor check (request.model is one of the validated models in
        # the eval/spike path), but keep the loud fallback for safety.
        try:
            # Single source of truth: cost reads the same token counts the
            # LLMResponse + the audit event carry (response.*), not the raw
            # locals — so the billed counts can't drift from the audited ones.
            cost_decimal = compute_cost_usd(
                model=model_id,
                input_tokens=response.input_tokens,
                cache_write_tokens=response.cache_write_tokens,
                cache_read_tokens=response.cache_read_tokens,
                output_tokens=response.output_tokens,
            )
        except KeyError as exc:
            pricing_key = normalize_to_pricing_key(model_id)
            raise LLMPricingMissingError(
                f"Model {model_id!r} normalizes to pricing key {pricing_key!r}, "
                f"which is not in RATE_TABLE at complete() step 7. Add the key "
                f"to RATE_TABLE + bump PRICING_VERSION to fix.",
                missing_models=[pricing_key],
            ) from exc

        # Step 8: build LLMCallEvent + persist.
        event = LLMCallEvent(
            review_id=request.review_id,
            timestamp=datetime.now(UTC),
            is_eval=request.is_eval,
            model=model_id,
            node_id=request.node_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=float(cost_decimal),
            pricing_version=PRICING_VERSION,
            latency_ms=response.latency_ms,
            prompt_hash=prompt_hash,
            cache_hit=(cached_tokens > 0),
            context_summary=request.context_summary,
            prompt_template_version=request.prompt_template_version,
            system_prompt_hash=system_prompt_hash,
            degraded_mode=request.degraded_mode,
            degradation_reason=request.degradation_reason,
            response_format_digest=request.response_format_digest,
        )
        try:
            await self._persister.persist(event, request, response)
        except Exception as exc:
            # Mirror AnthropicProvider: SDK call already succeeded (billed),
            # but no audit row landed → the calling node halts the review.
            # Known metadata-only persister exceptions keep str(exc) + the
            # cause chain; unknown types render only <TypeName> with from None
            # so no content-bearing repr leaks past the wrapper.
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

        # Step 9: return.
        return response

    async def aclose(self) -> None:
        """Close the underlying openai SDK client and drain its pool.

        Wired into the FastAPI lifespan teardown. Idempotent via the
        `_closed` guard + `_close_lock`; bounded by `asyncio.wait_for` so a
        hung close doesn't block teardown; `asyncio.shield` keeps the inner
        close from being cancelled mid-drain. `AsyncOpenAI` exposes async
        teardown as `.close()` (a coroutine; there is no `.aclose()` on the
        SDK — verified against openai==2.44.0).
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
                    "GLMProvider.aclose() exceeded %.0fs timeout; leaking the "
                    "connection pool to the OS reaper rather than blocking "
                    "lifespan teardown (the shielded close task continues in "
                    "the background until completion or event-loop shutdown).",
                    _ACLOSE_TIMEOUT_SECONDS,
                )

    def _clear_close_task(self, task: asyncio.Task[None]) -> None:
        """Done-callback: consume the close task's exception (if any) and
        release the strong reference. Metadata-only: log the exception type
        name, never repr/str (SDK exceptions may carry response-body text).
        """
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                _LOGGER.warning(
                    "GLMProvider close task raised %s after aclose() returned; "
                    "exception consumed to prevent unretrieved-exception log spam.",
                    type(exc).__name__,
                )
        if self._close_task is task:
            self._close_task = None


def _build_sdk_kwargs(request: LLMRequest, *, enable_thinking: bool) -> dict[str, Any]:
    """Translate `LLMRequest` to openai `chat.completions.create()` kwargs.

    Deltas from the Anthropic translation:
      - `system_prompt` → a `{"role":"system"}` message (not the SDK
        `system` kwarg); `user_prompt` → a `{"role":"user"}` message.
      - NO `cache_control` marker — Baseten caches automatically.
      - reasoning off by default via `extra_body.chat_template_args`.
      - `response_schema_json` → `response_format.json_schema` (name
        required, strict const-true), not `output_config.format`.
      - `stream` omitted → non-streaming single response (usage is present).
    """
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ],
        # Reasoning is opt-in (off by default). `enable_thinking` is exposed
        # as an eval-tunable; reasoning_content is stripped regardless.
        "extra_body": {"chat_template_args": {"enable_thinking": enable_thinking}},
    }
    if request.response_schema_json is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                # Stable per-node name (required by the json_schema surface).
                "name": f"outrider_{request.node_id}",
                # const-true on Baseten's JsonSchema; the only accepted value.
                "strict": True,
                "schema": json.loads(request.response_schema_json),
            },
        }
    return kwargs


# openai finish_reason → Outrider's canonical (Anthropic stop_reason) vocabulary.
# Downstream guards key on the Anthropic words: analyze_parser.py raises on
# `finish_reason == "max_tokens"` and analyze.py caches only when
# `finish_reason != "max_tokens"`. Passing openai's "length" through verbatim
# would silently dodge BOTH — a truncated analyze response would be cached and
# served incomplete with no diagnostic. Normalizing at the wrapper boundary keeps
# every downstream guard provider-neutral (vendor-payloads-normalized-at-boundary).
_FINISH_REASON_MAP: Final[dict[str, str]] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
    "function_call": "tool_use",  # deprecated openai alias
}


def _normalize_finish_reason(value: str | None) -> str:
    """Map the openai finish_reason to Outrider's canonical (Anthropic
    stop_reason) vocabulary. None/empty → "unknown"; an unmapped value passes
    through unchanged so a novel reason stays visible rather than masked."""
    if not value:
        return "unknown"
    return _FINISH_REASON_MAP.get(value, value)


def _extract_assistant_text(response: Any) -> tuple[str, str]:
    """Return `(content, finish_reason)` from a single-choice response.

    `reasoning_content` (present only with reasoning on) is an untyped extra
    field and is intentionally ignored — only `message.content` is the
    schema-conforming payload. `finish_reason` is normalized to Outrider's
    canonical vocabulary (see `_normalize_finish_reason`).

    `message.content` is `Optional[str]` — None/empty on a refusal or a
    truncated (`length`) response. It is coalesced to "" (NOT fail-loud) so the
    downstream parser + the normalized finish_reason (`max_tokens` → the analyze
    truncation diagnostic) degrade that file gracefully, exactly like
    AnthropicProvider's empty-TextBlock path — rather than aborting the whole
    review with a non-retryable error. Only an unexpected CHOICE count fails loud.
    """
    choices = response.choices
    if len(choices) != 1:
        raise LLMUnexpectedContentBlocksError(
            f"GLM response has {len(choices)} choice(s); V1 wrapper expects "
            f"exactly one. This may indicate a streaming/tool-use response "
            f"(not supported in V1) or an SDK shape change.",
            actual_block_types=[f"choices={len(choices)}"],
        )
    content = choices[0].message.content or ""
    finish_reason = _normalize_finish_reason(choices[0].finish_reason)
    return content, finish_reason


def _translate_openai_error(exc: openai.OpenAIError) -> LLMProviderError:
    """Map an openai SDK exception to the typed `LLMProviderError` subclass.

    Order is load-bearing (subclass before parent, Python `except`
    semantics). `APITimeoutError` IS a subclass of `APIConnectionError`
    (verified against openai==2.44.0), so it must be tested first or every
    timeout would route to `LLMUpstreamError`. `APIResponseValidationError`
    subclasses `APIError` directly (not `APIStatusError`), so it precedes
    the fallback. The retry layers match AnthropicProvider's set exactly —
    only the SDK class names differ.

    Metadata-only: never pass `str(exc)` to the wrapper constructor; openai
    error messages render the response body, which can echo prompt
    fragments. Only the SDK class name (a safe identifier) is surfaced, and
    only for the unmapped fallback.
    """
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError()
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError()
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMAuthError()
    if isinstance(exc, openai.ConflictError):
        return LLMConflictError()
    if isinstance(
        exc,
        (openai.BadRequestError, openai.UnprocessableEntityError, openai.NotFoundError),
    ):
        return LLMInvalidRequestError()
    if isinstance(exc, openai.APIResponseValidationError):
        return LLMInvalidResponseError()
    if isinstance(exc, (openai.InternalServerError, openai.APIConnectionError)):
        return LLMUpstreamError()
    return LLMUnknownError(f"unmapped OpenAIError: {type(exc).__name__}")
