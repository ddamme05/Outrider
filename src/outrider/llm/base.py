# LLM provider wrapper — typed call surface + Protocols.
# See specs/2026-05-05-llm-provider-wrapper.md and DECISIONS.md #013/#014/#015/#016/#019.
"""LLM provider wrapper foundation.

This module is the boundary between agent nodes and concrete LLM SDKs:
- Agent nodes consume `LLMProvider` (Protocol), never the SDK directly.
- Concrete providers (`AnthropicProvider`, in `anthropic_provider.py`)
  implement the Protocol and are the only modules importing vendor SDKs
  per `vendor-sdks-only-in-wrappers`.

Round 13 design + round 14/15 corrections; see spec for the full audit
chain. Two abstract-base enforcement notes worth pinning here:

  - `LLMProviderError(Exception, ABC)` does NOT prevent instantiation
    because `Exception.__new__` bypasses ABC's `__abstractmethods__`
    check. We use `__init__` type-guard + `__init_subclass__`
    presence + value-membership enforcement instead (rounds 13–15).
  - `INCLUDE_TEXT_OPT_IN` is a typed sentinel, not a string key. The
    persister opts into content serialization via
    `model_dump(context=INCLUDE_TEXT_OPT_IN)`; identity check, not dict
    lookup, so typos like `"INCLUDE_TEXT"` cannot accidentally pass.
"""

import hashlib
from typing import (
    Any,
    ClassVar,
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
    model_validator,
)
from pydantic_core.core_schema import SerializationInfo

from outrider.audit.events import ContextManifestEntry, LLMCallEvent

__all__ = [
    "INCLUDE_TEXT_OPT_IN",
    "LLMAuthError",
    "LLMExchangePersister",
    "LLMInvalidRequestError",
    "LLMInvalidResponseError",
    "LLMMessage",
    "LLMMissingAPIKeyError",
    "LLMPersisterError",
    "LLMPersisterNotWiredError",
    "LLMPricingMissingError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
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
    """Sentinel; only constructable via the module-level `INCLUDE_TEXT_OPT_IN`
    singleton. Direct construction outside this module raises `TypeError`.

    The serializer uses identity (`info.context is INCLUDE_TEXT_OPT_IN`),
    NOT dict-key lookup — there is no string to typo. The persister imports
    the singleton and passes it as `context=` on `model_dump()` to retrieve
    full content for `llm_call_content` storage.
    """

    _CONSTRUCT_TOKEN: ClassVar[object] = object()

    def __init__(self, _token: object) -> None:
        if _token is not _IncludeTextOptIn._CONSTRUCT_TOKEN:
            raise TypeError("Use llm.base.INCLUDE_TEXT_OPT_IN; do not construct directly.")

    def __repr__(self) -> str:
        return "<INCLUDE_TEXT_OPT_IN>"


INCLUDE_TEXT_OPT_IN: _IncludeTextOptIn = _IncludeTextOptIn(_IncludeTextOptIn._CONSTRUCT_TOKEN)


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

    Round-18 audit clarification: `cls.__dict__` is intentional and
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
      - `"node"`: the calling agent node should retry (used for
        `LLMTimeoutError`/`LLMRateLimitError`/`LLMUpstreamError`).
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

    def __init__(self, *args: object) -> None:
        if type(self) is LLMProviderError:
            raise TypeError("LLMProviderError is abstract; raise a concrete subclass.")
        super().__init__(*args)


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


class LLMUpstreamError(LLMProviderError):
    """5xx after SDK retries; translated from
    `anthropic.InternalServerError` or `APIConnectionError`."""

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
    exactly one `TextBlock` (V1 single-text-block assumption)."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMMissingAPIKeyError(LLMProviderError):
    """Eager construction-time check: `api_key` is empty."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMPersisterNotWiredError(LLMProviderError):
    """Fail-closed pre-call: `persister=None` on `AnthropicProvider`
    construction (round 13 fail-closed-not-stubbed design)."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMPersisterError(LLMProviderError):
    """Post-SDK persistence failure: SDK call succeeded but `persist()`
    raised. The audit row is the only intended record of the call; its
    absence halts the review."""

    retry_at_layer: ClassVar[RetryLayer] = "none"


class LLMPricingMissingError(LLMProviderError):
    """Eager construction-time check: a configured model is not in
    `llm.pricing.RATE_TABLE`."""

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
    """Wrapper input. Two field groups:

    - **Transport fields** the wrapper sends to the SDK:
      `system_prompt`, `user_prompt`, `messages` (V1.5+ only),
      `model`, `max_tokens`, `cache_control`, `temperature`.
    - **Audit-context fields** the wrapper passes opaquely through to
      `LLMCallEvent` at `complete()` step 9: `review_id`, `node_id`,
      `is_eval`, `context_summary`, `prompt_template_version`,
      `degraded_mode`. Provider does NOT modify or re-derive these.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Transport fields
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    messages: list[LLMMessage] | None = None
    model: str
    max_tokens: int = Field(gt=0, le=8192)
    cache_control: bool = False
    temperature: float = Field(ge=0.0, le=1.0)

    # Audit-context fields (pass-through to LLMCallEvent)
    review_id: UUID
    node_id: Literal["triage", "analyze", "synthesize", "trace"]
    is_eval: bool = False
    context_summary: tuple[ContextManifestEntry, ...] = ()
    prompt_template_version: str = Field(min_length=1)
    degraded_mode: bool

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

    @model_validator(mode="after")
    def _enforce_context_for_scope_nodes(self) -> Self:
        """`analyze` and `synthesize` always pack scope context; an empty
        `context_summary` from those nodes is a node-side bug worth
        catching at request construction (round 11 sharp-edges H1)."""
        nodes_requiring_context = frozenset({"analyze", "synthesize"})
        if self.node_id in nodes_requiring_context and len(self.context_summary) == 0:
            raise ValueError(
                f"node_id={self.node_id!r} requires non-empty context_summary; "
                f"the analyze/synthesize node always packs scope context"
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
        ]


class LLMResponse(BaseModel):
    """Wrapper output.

    Per AC#7, has NO `severity`/`evidence_tier`/`confidence`/`cost_usd`
    fields — schema layer enforces severity/tier/confidence; cost is
    computed by the provider in `complete()` step 8 from token counts ×
    `llm.pricing.RATE_TABLE` and lands on `LLMCallEvent` (NOT here).

    `text` is completion content; default `model_dump()` redacts via
    `field_serializer`. Persister opts in via
    `model_dump(context=INCLUDE_TEXT_OPT_IN)`.
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
    SDKs. V1 ships `AnthropicProvider`; V1.5 adds `OpenAIProvider` behind
    the same Protocol."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send `request` to the LLM, return the typed response.

        Persistence (audit row + content row) happens internally before
        return. Failure paths surface as `LLMProviderError` subclasses;
        the calling node reads `error.retry_at_layer` to decide retry.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers — exposed at module level so tests can verify hash stability
# independently of any provider (AC#15).
# ---------------------------------------------------------------------------


# `\x1e` = ASCII Information Separator Two; fixed delimiter so prompts that
# happen to contain "system_prompt"-like substrings cannot collide with the
# delimiter sequence.
_PROMPT_HASH_DELIMITER = b"\x1e"


def _canonical_prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """Replay-equivalence canonicalization for `LLMCallEvent.prompt_hash`.

    SHA-256 over `system_prompt.encode("utf-8") + b"\\x1e" +
    user_prompt.encode("utf-8")`. No Unicode normalization, no whitespace
    trimming, no line-ending conversion. Pinned by AC#15: a known
    (system_prompt, user_prompt) pair produces a known hex digest.

    Drift in this function silently breaks replay reconstruction — the
    paired test must fail loud if anyone changes the canonicalization.
    """
    return hashlib.sha256(
        system_prompt.encode("utf-8") + _PROMPT_HASH_DELIMITER + user_prompt.encode("utf-8")
    ).hexdigest()


def _canonical_system_prompt_hash(system_prompt: str) -> str:
    """SHA-256 of the system prompt alone — different cache lifecycle from
    per-request prompt content. Used for `LLMCallEvent.system_prompt_hash`.
    """
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
