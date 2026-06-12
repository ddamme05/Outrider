# LLM provider wrapper — typed call surface + Protocols.
# See specs/2026-05-05-llm-provider-wrapper.md and DECISIONS.md #013/#014/#015/#016/#019.
"""LLM provider wrapper foundation.

This module is the boundary between agent nodes and concrete LLM SDKs:
- Agent nodes consume `LLMProvider` (Protocol), never the SDK directly.
- Concrete providers (`AnthropicProvider`, in `anthropic_provider.py`)
  implement the Protocol and import the vendor SDK at the call surface.
  The `vendor-sdks-only-in-wrappers` invariant is folder-scoped (vendor
  imports confined to `src/outrider/llm/`), not file-scoped — supporting
  modules within `llm/` may legitimately import SDK metadata too (e.g.,
  `config.py` imports `anthropic.resources.messages.DEPRECATED_MODELS`
  for eager deprecation validation; cleanup per Copilot).

See spec for the full design chain. Two abstract-base enforcement
notes worth pinning here:

  - `LLMProviderError(Exception, ABC)` does NOT prevent instantiation
    because `Exception.__new__` bypasses ABC's `__abstractmethods__`
    check. We use `__init__` type-guard + `__init_subclass__`
    presence + value-membership enforcement instead.
  - `INCLUDE_TEXT_OPT_IN` is a typed sentinel, not a string key. ANY
    caller that intentionally needs to serialize `LLMRequest` or
    `LLMResponse` with raw content (rather than the default redacted
    form) passes it as the `model_dump()` context — identity check
    (`info.context is INCLUDE_TEXT_OPT_IN`), not dict lookup, so typos
    like `"INCLUDE_TEXT"` cannot accidentally pass. NOTE: the shipped
    `AuditPersister` does NOT take this path — it persists raw
    `prompt`/`completion` via direct attribute access
    (`request.user_prompt`, `response.text`) into the `llm_call_content`
    side-table, bypassing both the redaction serializer AND the audit
    payload entirely. The sentinel remains as a utility for any future
    caller that genuinely needs serialized-with-content form; today no
    production code path uses it.
"""

import hashlib
import json
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

from outrider.audit.events import ContextManifestEntry, LLMCallEvent
from outrider.policy.canonical import canonicalize_for_hash

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
    NOT dict-key lookup — there is no string to typo.

    **reconciliation:** the older provider-wrapper spec at
    `specs/2026-05-05-llm-provider-wrapper.md` described this sentinel
    as the ONLY way to retrieve raw content (e.g., persister calling
    `request.model_dump(context=INCLUDE_TEXT_OPT_IN)`). That was the
    at-approval plan; the audit-persister spec
    (`specs/2026-05-16-audit-persister.md`) deliberately chose
    **direct attribute access** instead (`request.user_prompt`,
    `response.text`) because the persister stores raw content in the
    `llm_call_content` side-table and there's no reason to round-trip
    through `model_dump()` for the same result. The sentinel remains
    as a utility for any future caller that genuinely needs the
    serialized-WITH-content form; no production code path uses it
    today. Both retrieval paths are content-clean by their own
    discipline (the `field_serializer` redaction guards the
    `model_dump()` path; the direct-attribute-access path bypasses the
    serializer entirely and writes only to the dedicated content
    side-table). See `specs/2026-05-16-audit-persister.md` Actual
    Outcome for the contract history.
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


class LLMConflictError(LLMProviderError):
    """Resource conflict (HTTP 409); translated from `anthropic.ConflictError`.

    Per Anthropic SDK 0.100 docs (): 409 is in the
    SDK's default-retry set alongside 408/429/5xx. We disable SDK
    retries (`max_retries=0`), so the calling node owns retry — same
    layer as Timeout/RateLimit/Upstream.
    """

    retry_at_layer: ClassVar[RetryLayer] = "node"


class LLMUpstreamError(LLMProviderError):
    """Upstream failure: server 5xx OR connection-level failure.

    Translated from BOTH `anthropic.InternalServerError` (5xx with
    HTTP response) AND `anthropic.APIConnectionError` (no HTTP
    response — connect refused, DNS, SSL handshake). Per Anthropic SDK
    0.100 docs , connection errors are in the
    SDK's documented retry-eligible set alongside 5xx. SDK auto-retries
    are disabled in the wrapper (`max_retries=0`), so the calling node
    owns retry for both cases — same `retry_at_layer="node"` semantic
    regardless of whether an HTTP response was received.

    The earlier docstring "5xx after SDK retries" was doubly wrong:
    (a) SDK retries are not enabled, and (b) it omitted the
    connection-error branch. Both corrected in
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
    # FUP-096): the CANONICAL-JSON string form of a pinned response schema
    # (e.g. `ANALYZE_RESPONSE_SCHEMA_JSON`), produced by
    # `policy/canonical.canonicalize_for_hash` — a string, not a dict, so
    # nothing mutable lives inside this frozen request and the digest below
    # is derivable from these exact bytes. `None` = free-form call (today's
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
    # Sibling-sweep checklist when adding a new value: (1) extend this
    # Literal; (2) extend `LLMCallEvent.degradation_reason` Literal in
    # lockstep at `outrider.audit.events`; (3) add a parser branch
    # mapping the new ast_facts outcome to this reason; (4) extend
    # `tests/unit/test_llm_request_schema.py`'s truth-table tests.
    # `"tree_has_error_no_scope"` added per DECISIONS.md#033 (no-scope syntax error:
    # changed addable line intersects a tree error with no recovered scope).
    degradation_reason: (
        Literal[
            "parse_failed",
            "tree_has_error_in_changed_regions",
            "tree_has_error_no_scope",
        ]
        | None
    ) = None

    @field_validator("response_schema_json")
    @classmethod
    def _enforce_canonical_schema_json(cls, value: str | None) -> str | None:
        """The string must BE `canonicalize_for_hash` output: a JSON object
        that round-trips to exactly these bytes. Two failure modes this
        closes at construction instead of downstream: a malformed string
        would raise raw `json.JSONDecodeError` inside the provider's
        kwargs-building (outside the typed SDK error translation), and a
        valid-but-noncanonical string (whitespace, key order) would mint a
        different `response_format_digest` for the same parsed schema —
        fragmenting the one request-format identity the audit stream and
        the analyze cache key are supposed to share.
        """
        if value is None:
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            msg = "response_schema_json must be valid JSON (canonicalize_for_hash output)"
            raise ValueError(msg) from exc
        if not isinstance(parsed, dict):
            msg = "response_schema_json must be a JSON object, not a scalar or array"
            raise ValueError(msg)
        if canonicalize_for_hash(parsed).decode("utf-8") != value:
            msg = (
                "response_schema_json is not in canonical form; serialize the "
                "schema dict via policy/canonical.canonicalize_for_hash"
            )
            raise ValueError(msg)
        return value

    @property
    def response_format_digest(self) -> str | None:
        """SHA-256 hex of `response_schema_json`'s exact bytes; `None` for
        free-form calls. Derived, never caller-supplied — the digest and
        the schema bytes sent to the API cannot drift. Because the string
        is `canonicalize_for_hash` output (enforced at construction by
        `_enforce_canonical_schema_json`), this equals
        `policy/canonical.compute_identity_hash(schema_dict)` for the same
        schema — one recipe, provenance on `LLMCallEvent` and the analyze
        cache key both read THIS value.
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
        """`degraded_mode` requires `node_id == "analyze"` AND a typed
        `degradation_reason`, bidirectionally.

        Three-way coupling prevents silent bypass:
          (a) a buggy caller cannot set `degraded_mode=True` without
              also naming a documented degradation cause;
          (b) a non-analyze request (trace/synthesize/triage) cannot
              carry analyze-specific degradation semantics at all
              .
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
    """Wrapper output.

    Per AC#7, has NO `severity`/`evidence_tier`/`confidence`/`cost_usd`
    fields — schema layer enforces severity/tier/confidence; cost is
    computed by the provider in `complete()` step 8 from token counts ×
    `llm.pricing.RATE_TABLE` and lands on `LLMCallEvent` (NOT here).

    `text` is completion content; default `model_dump()` redacts via
    `field_serializer` (renders `"<redacted, N chars>"`). The redacted
    form is what flows through any log/serialization path that uses
    `model_dump()` without the opt-in sentinel. The shipped
    `AuditPersister` does NOT use `model_dump()` for content
    persistence — it reads `response.text` via direct attribute access
    so the raw content lands in `llm_call_content.completion`. Callers
    that genuinely want the model-dumped form WITH raw text pass
    `INCLUDE_TEXT_OPT_IN` as the context (identity-checked); no
    production code path does so today.
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

    async def aclose(self) -> None:
        """Release transport resources (e.g. drain the connection pool).

        Wired into the FastAPI lifespan teardown; idempotent. Formalized on
        the Protocol per DECISIONS.md#035 so the composition root can
        `aclose()` whatever the provider factory returns — including a
        `TracingLLMProvider` decorator, which forwards this to the provider it
        wraps. The lifespan has always depended on this method
        (`api/lifespan.py` push_async_callback(provider.aclose)); the Protocol
        now declares the contract it relied on.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers — exposed at module level so tests can verify hash stability
# independently of any provider (AC#15).
# ---------------------------------------------------------------------------


def _canonical_prompt_hash(*, system_prompt: str, user_prompt: str) -> str:
    """Replay-equivalence canonicalization for `LLMCallEvent.prompt_hash`.

    Length-prefixed SHA-256 — the input bytes are
    `f"{len(sp_bytes)}:".encode() + sp_bytes + f"{len(up_bytes)}:".encode() + up_bytes`,
    where `sp_bytes = system_prompt.encode("utf-8")` and similar for user.
    The length prefix makes the prompt-boundary unambiguous regardless of
    delimiter characters appearing inside either string. A fixed-delimiter
    recipe collides whenever the delimiter character can appear in the
    prompt body — PR content (which can flow into either prompt via
    template substitution) is attacker-controlled, so a `\\x1e`-bearing
    payload could move the boundary across two distinct (system, user)
    pairs that share a digest. Length-prefix encoding is collision-resistant
    by structure.

    No Unicode normalization, no whitespace trimming, no line-ending
    conversion — the hash is over the BYTES the LLM provider received.

    Keyword-only because both args are `str` and adjacent; positional
    swap would silently produce a different valid SHA-256 with no type
    signal. Sibling pattern matches `compute_finding_content_hash` in
    `audit/events.py`.
    """
    sp_bytes = system_prompt.encode("utf-8")
    up_bytes = user_prompt.encode("utf-8")
    payload = (
        f"{len(sp_bytes)}:".encode("ascii")
        + sp_bytes
        + f"{len(up_bytes)}:".encode("ascii")
        + up_bytes
    )
    return hashlib.sha256(payload).hexdigest()


def _canonical_system_prompt_hash(system_prompt: str) -> str:
    """SHA-256 of the system prompt alone — different cache lifecycle from
    per-request prompt content. Used for `LLMCallEvent.system_prompt_hash`.
    """
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
