# LLM provider wrapper — typed call surface + Protocols.
# See specs/2026-05-05-llm-provider-wrapper.md and DECISIONS.md #013/#014/#015/#016/#019.
"""Provider-neutral requests, responses, errors, and persistence contracts.

Agent nodes call `LLMProvider`; concrete wrappers own vendor SDK access
under `vendor-sdks-only-in-wrappers`.
Requests carry transport inputs and audit context, while responses carry
normalized text, usage, and host identity. Providers persist a completed
exchange before returning it.

Content serializers redact by default. `INCLUDE_TEXT_OPT_IN` enables raw
serialization through an identity check; `AuditPersister` instead reads raw
attributes directly into the retained content table (`DECISIONS.md#016`).

`LLMProviderError` enforces a retry layer on every concrete subclass.
Its docstring records the complete taxonomy and construction rules.
"""

import hashlib
import json
import re
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    Protocol,
    Self,
    get_args,
    runtime_checkable,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import SerializationInfo

from outrider.audit.events import ContextManifestEntry, DegradationReason, LLMCallEvent
from outrider.policy.canonical import SHA256_HEX_PATTERN

__all__ = [
    "INCLUDE_TEXT_OPT_IN",
    "LLMAuthError",
    "LLMConflictError",
    "LLMExchangePersister",
    "LLMInvalidRequestError",
    "LLMInvalidResponseError",
    "LLMMessage",
    "LLMMissingAPIKeyError",
    "LLMPersisterError",
    "LLMPersisterNotWiredError",
    "LLMPricingContractError",
    "LLMPricingMissingError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRefusalError",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMUnexpectedContentBlocksError",
    "LLMUnknownError",
    "LLMUpstreamError",
    "RetryLayer",
    "_canonical_prompt_hash",
    "_canonical_system_prompt_hash",
]


# ---------------------------------------------------------------------------
# Typed sentinel for content serialization opt-in.
# ---------------------------------------------------------------------------


class _IncludeTextOptIn:
    """Singleton token for explicitly serializing raw model-call content.

    Only the module-held construction token can create it. Serializers check
    `info.context is INCLUDE_TEXT_OPT_IN`, so a string or dictionary cannot
    accidentally opt in. Persistence uses direct attributes instead; see the
    audit-persister spec's Actual Outcome.
    """

    _CONSTRUCT_TOKEN: ClassVar[object] = object()

    def __init__(self, _token: object) -> None:
        if _token is not _IncludeTextOptIn._CONSTRUCT_TOKEN:
            raise TypeError("Use llm.base.INCLUDE_TEXT_OPT_IN; do not construct directly.")

    def __repr__(self) -> str:
        return "<INCLUDE_TEXT_OPT_IN>"


INCLUDE_TEXT_OPT_IN: Final[_IncludeTextOptIn] = _IncludeTextOptIn(
    _IncludeTextOptIn._CONSTRUCT_TOKEN
)


def _redact_text(value: str, info: SerializationInfo) -> str:
    """`field_serializer` helper. Returns the literal value only when the
    serialization context is exactly `INCLUDE_TEXT_OPT_IN` (identity check);
    otherwise returns a redacted placeholder. Used by every content-bearing
    string field on `LLMRequest`/`LLMResponse`/`LLMMessage`.
    """
    if info.context is INCLUDE_TEXT_OPT_IN:
        return value
    return f"<redacted, {len(value)} chars>"


# ---------------------------------------------------------------------------
# Typed exception hierarchy.
# ---------------------------------------------------------------------------

RetryLayer = Literal["wrapper", "node", "graph", "none"]

# Diagnostic-metadata validation (see `LLMProviderError.attach_diagnostics`). These gate what a
# provider may attach to a terminal error, so no vendor content leaks through the
# metadata channel.
# `error_category` is the vendor error TYPE — a bounded category enum, NOT the free-text message.
# The set covers Anthropic's documented error types; other providers reusing the taxonomy
# add theirs.
_ALLOWED_ERROR_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "request_too_large",
        "rate_limit_error",
        "api_error",
        "overloaded_error",
        "billing_error",
        "timeout_error",
    }
)
# A request id is an opaque, id-shaped token (e.g. `req_011C...`) — never content. Bound its shape
# and length so an unexpected value can't carry arbitrary text.
_PROVIDER_REQUEST_ID_RE: Final[re.Pattern[str]] = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


def _valid_http_status(value: object) -> int | None:
    """An int in the HTTP status range; anything else (incl. bool) → None."""
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return None


def _valid_request_id(value: object) -> str | None:
    """An id-shaped token per `_PROVIDER_REQUEST_ID_RE`; anything else → None."""
    if isinstance(value, str) and _PROVIDER_REQUEST_ID_RE.match(value):
        return value
    return None


def _valid_error_category(value: object) -> str | None:
    """A member of the bounded `_ALLOWED_ERROR_CATEGORIES` enum; anything else → None."""
    if isinstance(value, str) and value in _ALLOWED_ERROR_CATEGORIES:
        return value
    return None


class LLMProviderError(Exception):
    """Abstract-by-construction base.

    Two enforcement layers:

      (a) `__init__` type-guard: raises `TypeError` if instantiated as
          the base class itself. Allows subclasses through.
      (b) `__init_subclass__` enforcement at class-definition time:
          presence — every concrete subclass must set `retry_at_layer`
          ClassVar in its OWN body (the check uses `cls.__dict__`, not
          inheritance lookup);
          value — must be one of `RetryLayer`'s allowed literals.

    `cls.__dict__` is intentional and
    stricter than `getattr(cls, ...)`. A sub-subclass like
    `class TimeoutWithRetryAfter(LLMTimeoutError): pass` MUST also set
    `retry_at_layer` in its own body, even though inheritance would
    otherwise resolve. The strictness is a feature: every concrete
    error class self-documents its retry layer in source. If a
    sub-subclass legitimately wants to inherit, it can declare
    `retry_at_layer = LLMTimeoutError.retry_at_layer` explicitly.

    `Exception.__new__` bypasses ABC's `__abstractmethods__` check, so
    `class Foo(Exception, ABC)` with or without `@abstractmethod` does NOT
    prevent instantiation. The pattern below is what works on Python 3.x.

    `retry_at_layer` semantics:
      - `"node"`: the calling agent node should retry (used for ALL FOUR
        retry-eligible classes — `LLMTimeoutError` / `LLMRateLimitError`
        / `LLMConflictError` / `LLMUpstreamError`. The 4-class set
        mirrors Anthropic SDK 0.100's default-retry set 408/429/409/5xx.
        Omitting any of the four here would silently invite
        the class-omission bug pattern FUP-025 has been defending
        against; pinned by both
        `tests/unit/test_llm_error_taxonomy.py::test_recoverable_subclasses_are_node_layer`
        (every named class IS `"node"`) and
        `::test_provider_error_docstring_names_every_node_layer_class`
        (every `"node"`-layer class IS named in THIS docstring).
      - `"graph"`: LangGraph-level retry policy handles it (unused in V1).
      - `"wrapper"`: reserved for future use (currently the wrapper sets
        `max_retries=0` on the SDK so this is unused in V1).
      - `"none"`: terminal; calling node halts the review.
    """

    retry_at_layer: ClassVar[RetryLayer]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "retry_at_layer" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must set retry_at_layer ClassVar (one of {get_args(RetryLayer)})"
            )
        if cls.retry_at_layer not in get_args(RetryLayer):
            raise TypeError(
                f"{cls.__name__}.retry_at_layer = "
                f"{cls.retry_at_layer!r} not in {get_args(RetryLayer)}"
            )

    # Bounded, privacy-safe diagnostic metadata about a terminal provider failure. Stored as
    # instance attributes SEPARATE from `Exception.args`, so `str(exc)`, `repr(exc)`, and traceback
    # rendering never surface them — and, by construction, never surface the vendor message/body the
    # translator deliberately drops. Populated ONLY through `attach_diagnostics`, which validates
    # every field. Default `None` = not populated. NOT auto-logged or persisted — exposed only for
    # explicit structured inspection via `diagnostic_metadata()`.
    diagnostic_http_status: int | None = None
    diagnostic_request_id: str | None = None
    diagnostic_error_category: str | None = None

    def __init__(self, *args: object) -> None:
        if type(self) is LLMProviderError:
            raise TypeError("LLMProviderError is abstract; raise a concrete subclass.")
        super().__init__(*args)

    def attach_diagnostics(
        self,
        *,
        http_status: object = None,
        request_id: object = None,
        error_category: object = None,
    ) -> Self:
        """Attach bounded, validated diagnostic metadata; returns `self` so a caller can
        `raise err.attach_diagnostics(...)`.

        Every field is validated and kept ONLY if it is a bounded, non-content value:
          - `http_status`: an int in the HTTP range (100-599); anything else is dropped.
          - `request_id`: matches `_PROVIDER_REQUEST_ID_RE` (short id-shaped token); else dropped.
          - `error_category`: a member of `_ALLOWED_ERROR_CATEGORIES` (a fixed enum of
            vendor error
            TYPES, never the free-text message); anything else is dropped.
        A provider extracts raw values from its SDK exception and calls this; validation runs here
        AND again at output (`diagnostic_metadata`), so no provider can smuggle content in and no
        direct mutation of the public attrs can leak text either. Never accepts a message or body.
        """
        self.diagnostic_http_status = _valid_http_status(http_status)
        self.diagnostic_request_id = _valid_request_id(request_id)
        self.diagnostic_error_category = _valid_error_category(error_category)
        return self

    def diagnostic_metadata(self) -> dict[str, object | None]:
        """The validated, privacy-safe diagnostic fields as a dict, for explicit inspection
        (structured logging, a diagnostic harness). Contains ONLY the bounded fields — never the
        vendor message, body, or `str(exc)`.

        Re-validates on output: the backing attributes are public and mutable, so this re-applies
        the same bounds — direct assignment of arbitrary text to `diagnostic_error_category` (etc.)
        is filtered back to `None` here rather than surfaced.
        """
        return {
            "http_status": _valid_http_status(self.diagnostic_http_status),
            "request_id": _valid_request_id(self.diagnostic_request_id),
            "error_category": _valid_error_category(self.diagnostic_error_category),
        }


class LLMUnknownError(LLMProviderError):
    """Fall-through for unmapped Anthropic `APIError` subclasses."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMTimeoutError(LLMProviderError):
    """SDK timeout (httpx connect/read/pool); translated from
    `anthropic.APITimeoutError`."""

    retry_at_layer: ClassVar[RetryLayer] = "node"


class LLMRateLimitError(LLMProviderError):
    """Rate limit (HTTP 429); translated from `anthropic.RateLimitError`."""

    retry_at_layer: ClassVar[RetryLayer] = "node"


class LLMConflictError(LLMProviderError):
    """Resource conflict (HTTP 409); translated from `anthropic.ConflictError`.

    Per the Anthropic SDK 0.100 docs, 409 is in the SDK's default-retry set
    alongside 408/429/5xx. We disable SDK retries (`max_retries=0`), so the
    calling node owns retry — same layer as Timeout/RateLimit/Upstream.
    """

    retry_at_layer: ClassVar[RetryLayer] = "node"


class LLMUpstreamError(LLMProviderError):
    """Upstream failure: server 5xx OR connection-level failure.

    Translated from BOTH `anthropic.InternalServerError` (5xx with
    HTTP response) AND `anthropic.APIConnectionError` (no HTTP
    response — connect refused, DNS, SSL handshake). Per the Anthropic SDK
    0.100 docs, connection errors are in its documented retry-eligible set
    alongside 5xx. SDK auto-retries are disabled in the wrapper
    (`max_retries=0`), so the calling node owns retry for both cases — the
    same `retry_at_layer="node"` semantic regardless of whether an HTTP
    response was received.
    """

    retry_at_layer: ClassVar[RetryLayer] = "node"


class LLMAuthError(LLMProviderError):
    """Auth failure (401/403); translated from `AuthenticationError` or
    `PermissionDeniedError`."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMInvalidRequestError(LLMProviderError):
    """Malformed request (400/422); translated from `BadRequestError` or
    `UnprocessableEntityError`."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMInvalidResponseError(LLMProviderError):
    """Response shape failed Pydantic validation; translated from
    `APIResponseValidationError`."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMUnexpectedContentBlocksError(LLMProviderError):
    """Multi-block fail-loud: `response.content` has anything other than
    exactly one `TextBlock` (V1 single-text-block assumption).

    Carries `actual_block_types: tuple[str, ...]` for structured caller
    inspection (e.g., metrics on which extended-thinking shapes appeared).
    """

    retry_at_layer: ClassVar[RetryLayer] = "none"

    def __init__(
        self,
        *args: object,
        actual_block_types: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.actual_block_types: tuple[str, ...] = tuple(actual_block_types or ())
        super().__init__(*args)


class LLMRefusalError(LLMProviderError):
    """The model declined the request: a successful HTTP 200 carrying
    `stop_reason="refusal"` (a safety-classifier decline, GA on the
    adaptive-thinking generation — Opus 4.7+ / Sonnet 5+). A pre-output
    refusal has an empty `content` array; a mid-stream refusal may carry
    partial text, which the wrapper discards.

    Outrider asks the model to find vulnerabilities, so a refusal MUST halt
    the review loudly rather than silently read as an empty/no-findings
    response (output-boundary #6: the model proposes, deterministic systems
    dispose — a decline is not a finding-set of zero). Caught at the wrapper
    BEFORE `_extract_single_text_block` so the empty content array surfaces
    as a refusal, not a misleading "unexpected content blocks" shape error.
    Terminal — retrying the same prompt will not help.

    Carries `category: str | None` (the `stop_details` refusal category,
    e.g. `"cyber"`) for structured caller inspection.
    """

    retry_at_layer: ClassVar[RetryLayer] = "none"

    def __init__(self, *args: object, category: str | None = None) -> None:
        self.category: str | None = category
        super().__init__(*args)


class LLMMissingAPIKeyError(LLMProviderError):
    """Eager construction-time check: `api_key` is empty."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMPersisterNotWiredError(LLMProviderError):
    """Fail-closed pre-call: `persister=None` on `AnthropicProvider`
    construction (fail-closed-not-stubbed design)."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMPersisterError(LLMProviderError):
    """Post-SDK persistence failure: SDK call succeeded but `persist()`
    raised. The audit row is the only intended record of the call; its
    absence halts the review."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMPricingMissingError(LLMProviderError):
    """Eager construction-time check: a configured model is not in
    `llm.pricing.RATE_TABLE`.

    Carries `missing_models: tuple[str, ...]` so callers can structurally
    enumerate which model id(s) need a pricing-table entry, rather than
    parsing the message string.
    """

    retry_at_layer: ClassVar[RetryLayer] = "none"

    def __init__(
        self,
        *args: object,
        missing_models: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.missing_models: tuple[str, ...] = tuple(missing_models or ())
        super().__init__(*args)


class LLMPricingContractError(LLMProviderError):
    """A COMPLETED, billed exchange deviated from the flat-rate pricing
    contract (specs/2026-07-18-openai-native-host.md): billed prompt tokens
    over the profile's flat-rate ceiling, an echoed service tier differing
    from the requested one, or an unpriceable echoed tier.

    Raised AFTER the exchange is persisted (correctly costed or
    provenance-marked — the FUP-205 persist-before-raise shape under #016).
    Terminal by contract: retrying would bill a SECOND paid request for a
    deviation that will recur, so exactly one SDK invocation and one
    persisted `LLMCallEvent` exist per outcome. Distinct from
    `LLMInvalidResponseError`, which covers genuinely malformed wire and
    aborts BEFORE persist.
    """

    retry_at_layer: ClassVar[RetryLayer] = "none"


# ---------------------------------------------------------------------------
# Pydantic schemas — typed call surface.
# ---------------------------------------------------------------------------


class LLMMessage(BaseModel):
    """Provider-neutral message; reserved for V1.5+ multi-turn extension.

    `role` does NOT include `"system"` — Anthropic's `MessageParam.role`
    accepts only `user`/`assistant`. System content goes to the top-level
    `LLMRequest.system_prompt` field, which the wrapper translates to the
    SDK kwarg `system` (NOT `system_prompt`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str

    @field_serializer("content")
    def _redact_content(self, value: str, info: SerializationInfo) -> str:
        return _redact_text(value, info)

    def __repr_args__(self) -> list[tuple[str, Any]]:
        return [
            ("role", self.role),
            ("content", f"<redacted, {len(self.content)} chars>"),
        ]


class LLMRequest(BaseModel):
    """Frozen provider input: transport settings plus metadata-only audit context.

    System/user text, model, token ceiling, caching, temperature, and optional
    response schema shape the SDK request; each provider maps them to its API.
    `messages` is reserved and must remain None. Review/node IDs, eval scope,
    context manifest, template version, degradation provenance, and phase key
    pass through to the audit event without provider reinterpretation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Transport fields.
    #
    # See DECISIONS.md#042-analyze-prompt-cache-packs-a-cross-file-invariant-prefix.
    # V1 packing convention: V1 has exactly ONE cacheable block — `system_prompt`.
    # Calling nodes pack CROSS-CALL-STABLE content into `system_prompt`
    # and everything per-call into `user_prompt`. For analyze (the
    # analyze-v4 cache-packing repartition), stable means CROSS-FILE
    # stable: the invariant prefix (`SYSTEM_PROMPT_STABLE_PREFIX`) is
    # byte-identical for every file in a review, so the cache hits
    # across the whole per-file fan-out; per-file scope context + the
    # diff travel in `user_prompt`. The wrapper marks `system_prompt`
    # with `cache_control: ephemeral`, so reusing the same
    # `system_prompt` across calls produces cache hits; `user_prompt`
    # stays outside the cache boundary by design. NOTE: prompts below a
    # model's min-cacheable floor (`pricing.MIN_CACHEABLE_TOKENS`) are
    # silently processed uncached. Canonical `docs/spec.md` §9.5
    # explicitly stages this V1 single-block packing vs the V1.5+
    # multi-block extension (deferred until `LLMRequest.messages`
    # becomes supported, which lets stable file-context blocks live in
    # the user message with their own per-block `cache_control`
    # markers — recovering same-file scope-context caching too).
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    messages: list[LLMMessage] | None = None
    model: str
    max_tokens: int = Field(gt=0, le=8192)
    # `cache_control` defaults to True per DECISIONS#013 point 4 + spec
    # §9.5 ("prompt-caching-always-on" convention).
    cache_control: bool = True
    temperature: float = Field(ge=0.0, le=1.0)
    # Constrained-decoding schema (specs/2026-06-12-constrained-decoding.md,
    # FUP-096): the COMPACT ORDER-PRESERVING JSON string form of a pinned
    # response schema (e.g. `ANALYZE_RESPONSE_SCHEMA_JSON`) — a string, not
    # a dict, so nothing mutable lives inside this frozen request and the
    # digest below is derivable from these exact bytes. Property order is
    # part of the format's identity (the API emits properties in schema
    # order — FUP-169), so the validator below requires a compact
    # round-trip but never sorts. `None` = free-form call (today's
    # behavior); the provider translates presence into the API's
    # `output_config.format` constrained decoding.
    response_schema_json: str | None = Field(default=None, min_length=2)

    # Audit-context fields (pass-through to LLMCallEvent)
    review_id: UUID
    node_id: Literal["triage", "analyze", "synthesize", "trace"]
    is_eval: bool = False
    context_summary: tuple[ContextManifestEntry, ...] = ()
    prompt_template_version: str = Field(min_length=1)
    degraded_mode: bool
    # Pins the provenance of `degraded_mode=True` so the bool can't be
    # set without naming a documented degradation cause. The Literal is
    # narrow on purpose: new reasons require expanding it (deliberate
    # friction so degraded-mode causes stay enumerable). V1 reasons
    # match the parser's parse-failure / has_error_in_changed_regions
    # branches.
    #
    # Typed by the shared `DegradationReason` enum (audit/events.py) — the
    # single source; the old three-Literal lockstep is gone. Adding a member
    # there propagates here by the type system; the branch that MAPS a new
    # cause to its reason (a parser branch for parse defects, a
    # `decide_degradation` routing branch otherwise) and the degraded
    # prompt's provenance sentence (totality-tested) are the two companions
    # a new member still needs.
    degradation_reason: DegradationReason | None = None
    # V1.5 parallel-analyze phase attribution: the emitting phase's key,
    # mirrored verbatim onto `LLMCallEvent.phase_key` by every provider —
    # the same request→event ride as `degradation_reason`, because the
    # PROVIDER constructs that event (DECISIONS.md#064; grouping contract
    # in specs/2026-07-05-parallel-analyze.md). None outside analyze
    # fan-out phases.
    phase_key: str | None = None

    @field_validator("response_schema_json")
    @classmethod
    def _enforce_compact_schema_json(cls, value: str | None) -> str | None:
        """The string must be the COMPACT, ORDER-PRESERVING serialization
        of a JSON object: round-trip through `json.dumps(json.loads(v),
        separators=(",", ":"))` to exactly these bytes. Two failure modes
        this closes at construction instead of downstream: a malformed
        string would raise raw `json.JSONDecodeError` inside the
        provider's kwargs-building (outside the typed SDK error
        translation), and a whitespace-variant string would mint a
        different `response_format_digest` for byte-identical wire intent.

        Key order is deliberately NOT normalized (no sorting): the API
        emits object properties in the schema's defined order, so
        property order is generatively load-bearing — two orders ARE two
        formats and must carry two digests (FUP-169: a key-sorted schema
        forced prose fields to be generated before the model's own
        classification tokens existed, and they came back empty).
        """
        if value is None:
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            msg = "response_schema_json must be valid JSON (compact serialization)"
            raise ValueError(msg) from exc
        if not isinstance(parsed, dict):
            msg = "response_schema_json must be a JSON object, not a scalar or array"
            raise ValueError(msg)
        if json.dumps(parsed, separators=(",", ":"), ensure_ascii=False, allow_nan=False) != value:
            msg = (
                "response_schema_json is not in compact order-preserving form; "
                'serialize the schema dict via json.dumps(..., separators=(",", ":"), '
                "ensure_ascii=False)"
            )
            raise ValueError(msg)
        return value

    @property
    def response_format_digest(self) -> str | None:
        """SHA-256 hex of `response_schema_json`'s exact bytes; `None` for
        free-form calls. Derived, never caller-supplied — the digest and
        the schema bytes sent to the API cannot drift. The string is the
        compact order-preserving serialization (enforced at construction
        by `_enforce_compact_schema_json`), and property order is part of
        the format identity, so two key orders carry two digests — by
        design. Provenance on `LLMCallEvent` and the analyze cache key
        both read THIS value.
        """
        if self.response_schema_json is None:
            return None
        return hashlib.sha256(self.response_schema_json.encode("utf-8")).hexdigest()

    @field_serializer("system_prompt")
    def _redact_system_prompt(self, value: str, info: SerializationInfo) -> str:
        return _redact_text(value, info)

    @field_serializer("user_prompt")
    def _redact_user_prompt(self, value: str, info: SerializationInfo) -> str:
        return _redact_text(value, info)

    @model_validator(mode="after")
    def _enforce_v1_messages_unset(self) -> Self:
        """V1 uses `system_prompt` + `user_prompt` only; `messages` is
        reserved for V1.5+ multi-turn extension. The validator rejects any
        non-None messages until that lands."""
        if self.messages is not None:
            raise ValueError(
                "LLMRequest.messages is reserved for V1.5+ multi-turn extension; "
                "V1 uses system_prompt + user_prompt only"
            )
        return self

    # Provenance validator runs FIRST (declared before
    # `_enforce_context_for_scope_nodes`) so its more-informative error
    # fires before the context validator on conflicting requests: a
    # request with `degraded_mode=True` on a non-analyze node AND empty
    # context_summary should report the analyze-only scoping violation,
    # not the context-required violation. Pydantic runs model validators
    # in declaration order.
    @model_validator(mode="after")
    def _enforce_degradation_provenance(self) -> Self:
        """Keep degradation mode and its reason coupled, and analyze-only.

        Phase keys also belong only to analyze fan-out. This validator runs
        before the context requirement so provenance errors surface first.
        """
        # Rule 1: only analyze can be degraded in V1. Other nodes have
        # no degraded-mode contract; allowing degraded_mode=True
        # elsewhere would be silent contract drift.
        if self.degraded_mode and self.node_id != "analyze":
            raise ValueError(
                f"degraded_mode=True only valid for node_id='analyze' in V1; "
                f"got node_id={self.node_id!r}. Synthesize/trace/triage have "
                f"no degraded-mode contract."
            )
        if self.degradation_reason is not None and self.node_id != "analyze":
            raise ValueError(
                f"degradation_reason is only valid for node_id='analyze' in V1; "
                f"got node_id={self.node_id!r}."
            )
        # Rule 2: bool ↔ reason bidirectional coupling (within analyze).
        if self.degraded_mode and self.degradation_reason is None:
            raise ValueError(
                "degraded_mode=True requires degradation_reason; "
                "naked degraded_mode is a silent context-validator bypass"
            )
        if (not self.degraded_mode) and self.degradation_reason is not None:
            raise ValueError(
                "degradation_reason requires degraded_mode=True; "
                "reason without mode is inconsistent"
            )
        # Rule 3: phase attribution is analyze-fan-out-only (DECISIONS.md#064).
        # A triage/trace/synthesize request carrying a worker key would create
        # phase ownership for a phase that cannot exist — the same
        # analyze-scoping contract as degraded_mode, mirrored on LLMCallEvent.
        if self.phase_key is not None and self.node_id != "analyze":
            raise ValueError(
                f"phase_key is only valid for node_id='analyze' (parallel-analyze "
                f"fan-out phases); got node_id={self.node_id!r}."
            )
        return self

    @model_validator(mode="after")
    def _enforce_context_for_scope_nodes(self) -> Self:
        """`analyze` packs per-file scope context; an empty
        `context_summary` from analyze is a node-side bug worth catching
        at request construction.

        Per §0b: analyze admits empty `context_summary` ONLY when
        `degraded_mode=True` AND a typed `degradation_reason` is
        supplied — the provenance validator above already enforces that
        the two are coupled, so we only need to check one.

        **Synthesize is NOT in this allowlist** (corrected per the
        synthesize-node spec audit). Synthesize aggregates already-
        produced findings into a `ReviewReport` and runs ONE summary
        call (config-routed; Haiku default per DECISIONS.md#043) for
        free-form prose; it does NOT walk per-file scope, so it does
        not pack a `context_summary` manifest.
        Triage + trace also legitimately omit the manifest.
        """
        nodes_requiring_context = frozenset({"analyze"})
        if self.node_id in nodes_requiring_context and len(self.context_summary) == 0:
            # Degraded analyze admits empty context, but the provenance
            # validator above already required degradation_reason to be
            # set when degraded_mode is True — so checking either flag
            # is sufficient.
            if self.node_id == "analyze" and self.degraded_mode:
                return self
            raise ValueError(
                f"node_id={self.node_id!r} requires non-empty context_summary; "
                f"the analyze node always packs per-file scope context"
            )
        return self

    @model_validator(mode="after")
    def _enforce_context_summary_unique(self) -> Self:
        """`context_summary` is set-semantic by `(file_path, scope_unit_name)`.
        The same scope unit shouldn't appear twice in one prompt's manifest.

        Mirror of `LLMCallEvent._enforce_context_summary_unique`. Validating
        here means a producer bug surfaces at request construction —
        BEFORE the paid SDK call — instead of after the side effect when
        `LLMCallEvent` rejects the duplicate. The wrapper passes
        `request.context_summary` through to the event verbatim, so
        an audit-shadow-only check would always fire too late.
        """
        keys = [(e.file_path, e.scope_unit_name) for e in self.context_summary]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"LLMRequest.context_summary contains duplicate "
                f"(file_path, scope_unit_name) entries: {sorted(keys)!r}"
            )
        return self

    def __repr_args__(self) -> list[tuple[str, Any]]:
        return [
            ("system_prompt", f"<redacted, {len(self.system_prompt)} chars>"),
            ("user_prompt", f"<redacted, {len(self.user_prompt)} chars>"),
            ("messages", None if self.messages is None else "<redacted>"),
            ("model", self.model),
            ("max_tokens", self.max_tokens),
            ("cache_control", self.cache_control),
            ("temperature", self.temperature),
            ("review_id", self.review_id),
            ("node_id", self.node_id),
            ("is_eval", self.is_eval),
            ("context_summary_count", len(self.context_summary)),
            ("prompt_template_version", self.prompt_template_version),
            ("degraded_mode", self.degraded_mode),
            ("degradation_reason", self.degradation_reason),
        ]


class LLMResponse(BaseModel):
    """Frozen normalized completion and usage, with no policy or cost fields.

    Nodes enforce finding policy; providers calculate cost for `LLMCallEvent`.
    Default serialization redacts `text`. Raw serialized text requires the
    identity-checked `INCLUDE_TEXT_OPT_IN`; `AuditPersister` instead reads the
    attribute directly for the separate, expiring completion-content row.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    finish_reason: str
    latency_ms: int = Field(ge=0)
    # Host-identity triad (DECISIONS.md#056): providers stamp all three together;
    # the persister cross-checks them against LLMCallEvent. None is read-compatible.
    profile_id: str | None = None
    reasoning_enabled: bool | None = None
    # The canonical sha256-hex shape (policy.canonical.SHA256_HEX_PATTERN — already a transitive
    # dep here via audit.events, so reusing it is no new import edge) so the provider boundary
    # rejects a malformed digest before the persister cross-checks response-vs-event, and a future
    # tightening of the canonical pattern can't drift LLMResponse from the audit events.
    profile_contract_digest: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)
    # Pricing context (specs/2026-07-18-openai-native-host.md): the raw billed prompt-token
    # wire count and the echoed service tier. NOT reconstructible from the normalized
    # input/cache_read split once a host reports cache writes as a distinct class, so the
    # provider carries both here for the persister's recomputation; `pricing.py` derives
    # the >272K long-context determination from `billed_prompt_tokens` internally.
    # None on hosts without the concept (Anthropic, GLM hosts). The tier is a BOUNDED raw
    # string for forensic fidelity — a novel echo is preserved verbatim, never collapsed
    # into `CostUnpricedReason`.
    billed_prompt_tokens: int | None = Field(default=None, ge=0)
    service_tier_actual: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _enforce_triad_coherence(self) -> Self:
        """The triad are peers (DECISIONS.md#056) — all-present or all-None; the provider
        stamps all three together. A partial triad is a provider/wrapper bug."""
        present = (
            self.profile_id is not None,
            self.reasoning_enabled is not None,
            self.profile_contract_digest is not None,
        )
        if not (all(present) or not any(present)):
            raise ValueError(
                "host-identity triad (DECISIONS.md#056) must be all-present or all-None on "
                "LLMResponse; the provider stamps all three together."
            )
        return self

    @field_serializer("text")
    def _redact_text_field(self, value: str, info: SerializationInfo) -> str:
        return _redact_text(value, info)

    def __repr_args__(self) -> list[tuple[str, Any]]:
        return [
            ("text", f"<redacted, {len(self.text)} chars>"),
            ("model", self.model),
            ("finish_reason", self.finish_reason),
            ("tokens_in_out", f"{self.input_tokens}/{self.output_tokens}"),
            ("cache_read", self.cache_read_tokens),
            ("cache_write", self.cache_write_tokens),
            ("latency_ms", self.latency_ms),
        ]


# ---------------------------------------------------------------------------
# Protocols.
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMExchangePersister(Protocol):
    """Single-transaction-insert contract per DECISIONS#016.

    Implementations MUST:
      - Insert `LLMCallEvent` + `llm_call_content` rows in one DB
        transaction (or fail closed — never one without the other).
      - Acquire a fresh `AsyncSession` per call (SQLAlchemy
        `AsyncSession` is not concurrent-safe across coroutines; V1.5
        parallel-analyze fanout will issue concurrent `persist()` calls).
      - Be idempotent on `event.event_id` (UUID4 — collision-free in
        practice; the wrapper pre-mints via `default_factory=uuid4` on
        `AuditEventBase`).
    """

    async def persist(
        self,
        event: LLMCallEvent,
        request: LLMRequest,
        response: LLMResponse,
    ) -> None:
        """Persist the audit event + content row in one transaction."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Transport-layer Protocol; agent nodes consume this, never concrete
    SDKs. Two concrete providers ship behind it: `AnthropicProvider` (native)
    and `OpenAICompatibleProvider`, parameterized by a `HostProfile` per
    DECISIONS.md#056 — which superseded the earlier V1.5 `OpenAIProvider`
    naming; api.openai.com is the `openai` host profile, not a third class."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send `request` to the LLM, return the typed response.

        Persistence (audit row + content row) happens internally before
        return. Failure paths surface as `LLMProviderError` subclasses;
        the calling node reads `error.retry_at_layer` to decide retry.
        """
        ...

    async def aclose(self) -> None:
        """Release transport resources (e.g. drain the connection pool).

        Wired into the FastAPI lifespan teardown; idempotent. Formalized on the
        Protocol per DECISIONS.md#035 (retained under #058 after the tracing
        decorator that originally motivated it was removed) — the lifespan has
        always depended on this method (`api/lifespan.py`
        push_async_callback(provider.aclose)); the Protocol declares the contract
        it relied on.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers — exposed at module level so tests can verify hash stability
# independently of any provider (AC#15).
# ---------------------------------------------------------------------------


def _canonical_prompt_hash(*, system_prompt: str, user_prompt: str) -> str:
    """Hash the exact UTF-8 bytes of both prompts, each prefixed by its byte length.

    Length framing keeps attacker-controlled delimiters from moving the
    system/user boundary. No trimming, Unicode normalization, or line-ending
    conversion occurs. The keyword-only arguments prevent an unnoticed swap.
    """
    system_prompt_bytes = system_prompt.encode("utf-8")
    user_prompt_bytes = user_prompt.encode("utf-8")
    payload = (
        f"{len(system_prompt_bytes)}:".encode("ascii")
        + system_prompt_bytes
        + f"{len(user_prompt_bytes)}:".encode("ascii")
        + user_prompt_bytes
    )
    return hashlib.sha256(payload).hexdigest()


def _canonical_system_prompt_hash(system_prompt: str) -> str:
    """SHA-256 of the system prompt alone — different cache lifecycle from
    per-request prompt content. Used for `LLMCallEvent.system_prompt_hash`.
    """
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
