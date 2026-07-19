# Host profiles for the OpenAI-compatible provider, per DECISIONS.md#056.
# `import openai` stays out of this module (re/hashlib/enum/pydantic + llm.base only),
# so it is import-lint-clean under trust boundary #8.
"""Per-host profiles for `OpenAICompatibleProvider` (DECISIONS.md#056).

A `HostProfile` is per-host DATA (base_url, slug pattern, json mode, token-accounting
mode, privacy posture) plus a single closed code axis — the reasoning-off shaper,
resolved through `_SHAPER_REGISTRY` (a frozen-model can't hold a callable). Identity is
`(profile_id, model)`; `profile_contract_digest` covers the wire-affecting fields PLUS
`SHAPER_CONTRACT_VERSION`, so a shaper/accounting *function* change rotates the digest
even when the enum is unchanged (audit-7 #3).

Arc 1a ships only `BASETEN_PROFILE` (byte-identical to the merged GLM spike) +
`HOST_DEFAULT_MODELS["anthropic"]` (the native path's per-node defaults — Anthropic is
selected by the `OUTRIDER_LLM_HOST` string, NOT a profile). DeepInfra/Fireworks/custom
are later arcs, each gated on a captured wire fixture + a scorecard pass.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from outrider.llm.base import LLMInvalidResponseError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# Bump on ANY shaper/accounting FUNCTION-body change (audit-7 #3). It is folded into
# `profile_contract_digest`, so a behavior change to a shaper invalidates warm cache rows
# even though no profile DATA field changed.
# v2 (specs/2026-07-18-openai-native-host.md): the openai-native-host provider-boundary
# extension — GPT-5.6's novel usage layout (cache writes as a distinct billed class →
# read_usage widened to a 4-tuple), the JsonMode-aware kwargs builder branch
# (JSON_OBJECT), message.refusal normalization, and three new digest-folded profile
# fields (flat-rate ceiling, prompt_cache_key, requested_service_tier). Rotates every
# profile's contract digest; GLM-host analyze cache rows re-key by design (#056).
# v3: `token_limit_param` — the kwargs builder sends the completion-token ceiling under
# a profile-declared parameter name. Wire-driven (#056 captured-evidence rule): the paid
# probe's 13-row capture returned HTTP 400 "Unsupported parameter: 'max_tokens' ... Use
# 'max_completion_tokens' instead" on every GPT-5.6 call; GLM hosts keep the verified
# `max_tokens` wire via the field default.
SHAPER_CONTRACT_VERSION: Final[str] = "v3"


class ReasoningMechanism(StrEnum):
    """How a host disables reasoning — the four observed wire shapes + a sentinel for
    hosts with no documented off-switch."""

    CHAT_TEMPLATE_ARGS = (
        "chat_template_args"  # Baseten/Telnyx: extra_body.chat_template_args.enable_thinking=False
    )
    REASONING_EFFORT_NONE = (
        "reasoning_effort_none"  # Fireworks/DeepInfra/OpenAI: top-level reasoning_effort="none"
    )
    REASONING_ENABLED_FALSE = (
        "reasoning_enabled_false"  # Together: extra_body.reasoning={"enabled": False}
    )
    THINKING_DISABLED = "thinking_disabled"  # Z.ai: extra_body.thinking={"type": "disabled"}
    NONE = "none"  # no documented off-switch (Cloudflare); reasoning stays on


class TokenAccounting(StrEnum):
    """Whether the host's `prompt_tokens` includes the cached subset (§8a)."""

    PROMPT_INCLUDES_CACHED = "prompt_includes_cached"  # Baseten/DeepInfra
    PROMPT_EXCLUDES_CACHED = "prompt_excludes_cached"  # Anthropic-like
    # OpenAI GPT-5.6+: cached reads ⊂ prompt_tokens AND cache writes reported as a
    # DISTINCT billed class (usage.prompt_tokens_details.cache_write_tokens, 1.25×
    # input). Whether writes are ALSO inside prompt_tokens is the conservation
    # equation the cold-write paid fixture pins (openai-native-host spec).
    PROMPT_INCLUDES_CACHED_WRITES_REPORTED = "prompt_includes_cached_writes_reported"
    UNVERIFIED = "unverified"  # never assumed — fail loud if cached actually fires


class JsonMode(StrEnum):
    """Structured-output capability the host's `response_format` honors."""

    STRICT_JSON_SCHEMA = "strict_json_schema"  # Fireworks/DeepInfra: constrained decoding
    SOFT_FENCED = "soft_fenced"  # Baseten: soft/fenced (FUP-196; fence-strip backstop)
    JSON_OBJECT = "json_object"  # Z.ai: json_object only, no schema


class TokenLimitParam(StrEnum):
    """Which chat-completions kwarg carries the completion-token ceiling. The values ARE
    the wire parameter names — the kwargs builder sends `request.max_tokens` under this
    key verbatim."""

    MAX_TOKENS = "max_tokens"  # legacy chat-completions param; GLM hosts' verified wire
    # Native OpenAI: the GPT-5.6 family 400s on `max_tokens` ("Unsupported parameter:
    # ... Use 'max_completion_tokens' instead" — paid probe capture, 13/13 rows).
    MAX_COMPLETION_TOKENS = "max_completion_tokens"


# --- reasoning-off shapers (the one procedural axis; mutate create-kwargs in place) ---


def _shape_chat_template_args(kwargs: dict[str, Any]) -> None:
    kwargs.setdefault("extra_body", {})["chat_template_args"] = {"enable_thinking": False}


def _shape_reasoning_effort_none(kwargs: dict[str, Any]) -> None:
    kwargs["reasoning_effort"] = "none"


def _shape_reasoning_enabled_false(kwargs: dict[str, Any]) -> None:
    kwargs.setdefault("extra_body", {})["reasoning"] = {"enabled": False}


def _shape_thinking_disabled(kwargs: dict[str, Any]) -> None:
    kwargs.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}


def _shape_none(_kwargs: dict[str, Any]) -> None:
    """No off-switch — reasoning stays on. `HostProfile.reasoning_forced_on` is True here, so
    the provider's `reasoning_enabled = requested or forced_on` audits this host as
    reasoning-on instead of a silent off. Carry this only when always-on cost is
    acknowledged."""


_SHAPER_REGISTRY: Final[Mapping[ReasoningMechanism, Callable[[dict[str, Any]], None]]] = (
    MappingProxyType(
        {
            ReasoningMechanism.CHAT_TEMPLATE_ARGS: _shape_chat_template_args,
            ReasoningMechanism.REASONING_EFFORT_NONE: _shape_reasoning_effort_none,
            ReasoningMechanism.REASONING_ENABLED_FALSE: _shape_reasoning_enabled_false,
            ReasoningMechanism.THINKING_DISABLED: _shape_thinking_disabled,
            ReasoningMechanism.NONE: _shape_none,
        }
    )
)


def read_usage(
    *,
    prompt_tokens: int,
    raw_cached_tokens: int,
    completion_tokens: int,
    accounting: TokenAccounting,
    raw_cache_write_tokens: int = 0,
) -> tuple[int, int, int, int]:
    """§8a normalization. Returns
    `(input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)`.

    `prompt_includes_cached` (Baseten/DeepInfra): cached is a SUBSET of prompt_tokens, so
    subtract — capping cached at prompt_tokens keeps `input + cache_read == prompt_tokens`
    self-consistent; no write class exists (`cache_write=0`).
    `prompt_includes_cached_writes_reported` (OpenAI 5.6+): reads as above, writes carried
    through as their own billed class. The write-vs-prompt conservation equation is
    pinned by the cold-write paid fixture (openai-native-host spec); until then writes
    are NOT subtracted from input, and a write count exceeding `prompt_tokens` is
    rejected as malformed. `prompt_excludes_cached`: prompt_tokens is already the
    uncached input. `unverified`: NEVER guess — raise if cached or write tokens fire.

    A negative usage component is a malformed wire payload (normalized at this boundary per
    trust-boundaries §5 sub-rule 6) — reject it before it drives a negative token or cost.
    """
    if (
        prompt_tokens < 0
        or raw_cached_tokens < 0
        or completion_tokens < 0
        or raw_cache_write_tokens < 0
    ):
        raise LLMInvalidResponseError(
            f"negative usage component: prompt={prompt_tokens} "
            f"cached={raw_cached_tokens} completion={completion_tokens} "
            f"cache_write={raw_cache_write_tokens}"
        )
    if accounting is TokenAccounting.PROMPT_INCLUDES_CACHED:
        cache_read = min(raw_cached_tokens, prompt_tokens)
        return prompt_tokens - cache_read, cache_read, 0, completion_tokens
    if accounting is TokenAccounting.PROMPT_INCLUDES_CACHED_WRITES_REPORTED:
        cache_read = min(raw_cached_tokens, prompt_tokens)
        if raw_cache_write_tokens > prompt_tokens:
            raise LLMInvalidResponseError(
                f"cache_write_tokens={raw_cache_write_tokens} exceeds "
                f"prompt_tokens={prompt_tokens}: malformed usage payload"
            )
        return prompt_tokens - cache_read, cache_read, raw_cache_write_tokens, completion_tokens
    if accounting is TokenAccounting.PROMPT_EXCLUDES_CACHED:
        return prompt_tokens, raw_cached_tokens, 0, completion_tokens
    if raw_cached_tokens > 0 or raw_cache_write_tokens > 0:
        raise LLMInvalidResponseError()
    return prompt_tokens, 0, 0, completion_tokens


class HostPrivacy(BaseModel):
    """Per-host privacy posture — SURFACED at construction (#013/#015), not enforced.

    Carries retention + the no-training stance, not just egress, so the construction
    notice can satisfy the #013 privacy contract (audit-8 #3). `trains_on_inputs=True` is
    a construction hard-fail in the provider (no blanket override; a future training host
    needs a per-host opt-out attestation). `source_url`/`verified_date` are provenance so a
    claim is auditable and stale-detectable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    egress_host: str
    model_origin: str
    direct_hosted: bool
    trains_on_inputs: bool
    retention: str
    source_url: str
    verified_date: str  # YYYY-MM-DD

    @field_validator("egress_host", "model_origin", "retention")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("privacy provenance field must be non-empty")
        return v

    @field_validator("source_url")
    @classmethod
    def _https_source(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError(f"source_url must be an https:// URL, got {v!r}")
        return v

    @field_validator("verified_date")
    @classmethod
    def _iso_date(cls, v: str) -> str:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) is None:
            raise ValueError(f"verified_date must be YYYY-MM-DD, got {v!r}")
        return v


class HostProfile(BaseModel):
    """A validated OpenAI-compatible host. Data + a `reasoning_mechanism` enum resolved
    through `_SHAPER_REGISTRY`. `AnthropicProvider` is NOT a profile (native SDK)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host_id: str
    base_url: str
    api_key_env: str
    model_slug_pattern: str
    json_mode: JsonMode
    token_accounting: TokenAccounting
    reasoning_mechanism: ReasoningMechanism
    privacy: HostPrivacy
    # openai-native-host spec (2026-07-18) — four digest-folded behaviors:
    # (1) flat-rate input ceiling: billed prompt tokens above this reprice the FULL
    #     request (pricing.LONG_CONTEXT_POLICY), so the provider rejects pre-flight on a
    #     conservative byte bound and post-checks the billed count. None = no documented
    #     repricing boundary (every pre-5.6 host).
    flat_rate_input_ceiling_tokens: int | None = Field(default=None, gt=0)
    # (2) prompt_cache_key: GPT-5.6+ requires the key for reliable cache matching; the
    #     provider sends a stable key derived from (profile_contract_digest, prompt
    #     VERSION) so the #042 stable-prefix packing keeps paying. ~15 RPM per key.
    sends_prompt_cache_key: bool = False
    # (3) requested tier: sent verbatim on every request (host DATA, never a builder
    #     constant). Declaring it means an echo is EXPECTED — an absent echoed tier
    #     becomes Unpriced(absent_tier); tier-less hosts (None) can never produce it.
    requested_service_tier: Literal["default"] | None = None
    # (4) completion-token-ceiling kwarg name (SHAPER v3, wire-driven): GPT-5.6 rejects
    #     `max_tokens` outright; the default keeps every GLM host's verified wire.
    token_limit_param: TokenLimitParam = TokenLimitParam.MAX_TOKENS

    @property
    def profile_contract_digest(self) -> str:
        """sha256 over the wire-affecting fields + `SHAPER_CONTRACT_VERSION`. A shaping
        change (or a shaper-function change, via the version) rotates it so a warm analyze
        cache invalidates correctly (DECISIONS.md#056)."""
        payload = "\n".join(
            (
                self.base_url,
                self.model_slug_pattern,
                self.json_mode.value,
                self.token_accounting.value,
                self.reasoning_mechanism.value,
                str(self.flat_rate_input_ceiling_tokens),
                str(self.sends_prompt_cache_key),
                str(self.requested_service_tier),
                self.token_limit_param.value,
                SHAPER_CONTRACT_VERSION,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_model_slug(self, slug: str) -> None:
        # fullmatch, not match: `re.match` against a `$`-anchored pattern still admits a
        # trailing newline (`$` matches before a final `\n`), so a slug like
        # "zai-org/GLM-5.2\n" would slip the typo gate.
        if re.fullmatch(self.model_slug_pattern, slug) is None:
            raise ValueError(
                f"model {slug!r} does not match host {self.host_id!r} slug pattern "
                f"{self.model_slug_pattern!r}"
            )

    def apply_reasoning_off(self, kwargs: dict[str, Any]) -> None:
        """Mutate `kwargs` to disable reasoning per this host's mechanism."""
        _SHAPER_REGISTRY[self.reasoning_mechanism](kwargs)

    @property
    def reasoning_forced_on(self) -> bool:
        """True iff this host has no off-switch (`NONE`), so reasoning runs regardless of the
        requested flag. This is the PROFILE's contribution to the triad's `reasoning_enabled`,
        NOT the stamped value: per DECISIONS.md#056 `reasoning_enabled` is `profile +
        OUTRIDER_LLM_REASONING`, so the provider/factory computes
        `reasoning_enabled = requested or profile.reasoning_forced_on`. Stamping this
        predicate directly would audit an operator-enabled run on an off-switch host as
        `False` while reasoning is actually on. The mechanism is folded into
        `profile_contract_digest`, so cache never colludes a forced-on host with an
        off-switch host."""
        return self.reasoning_mechanism is ReasoningMechanism.NONE


# Baseten — byte-identical to the merged GLM spike (glm_provider.py constants + §8a).
BASETEN_PROFILE: Final[HostProfile] = HostProfile(
    host_id="baseten",
    base_url="https://inference.baseten.co/v1",
    api_key_env="BASETEN_API_KEY",
    model_slug_pattern=r"^zai-org/GLM-\d+(\.\d+)?$",
    json_mode=JsonMode.SOFT_FENCED,
    token_accounting=TokenAccounting.PROMPT_INCLUDES_CACHED,
    reasoning_mechanism=ReasoningMechanism.CHAT_TEMPLATE_ARGS,
    privacy=HostPrivacy(
        egress_host="inference.baseten.co",
        model_origin="zhipu",
        direct_hosted=True,
        trains_on_inputs=False,
        retention=(
            "Baseten does not store inputs or outputs for synchronous inference by default "
            "(SOC 2 Type II, HIPAA); async inference has temporary input storage, which "
            "Outrider does not use. A DPA is available via the Trust Center."
        ),
        source_url="https://docs.baseten.co/observability/security",
        verified_date="2026-06-27",
    ),
)

# Fireworks — arc 1b (DECISIONS.md#056 amendment 2026-07-06). Ships on the captured paid
# wire (`spikes/fireworks/fixtures/`): strict json_schema ACCEPTS the raw
# ANALYZE_RESPONSE_SCHEMA verbatim (nullable anyOf honored) → STRICT_JSON_SCHEMA, NO adapter;
# usage is OpenAI-shape (cached_tokens ⊂ prompt_tokens) → PROMPT_INCLUDES_CACHED;
# reasoning_effort="none" accepted → REASONING_EFFORT_NONE. GLM-5.2 is an OPEN model on
# Fireworks, served on the chat-completions path this provider uses — Fireworks' zero-data-
# retention policy applies (the Responses-API store=True/30-day retention does NOT — a
# different endpoint we never call).
FIREWORKS_PROFILE: Final[HostProfile] = HostProfile(
    host_id="fireworks",
    base_url="https://api.fireworks.ai/inference/v1",
    api_key_env="FIREWORKS_API_KEY",
    # Fireworks renders the model version's dot as `p` (glm-5p2). Anchor both segments so a
    # bare `zai-org/GLM-5.2` (Baseten's shape) can't cross-validate against this host.
    model_slug_pattern=r"^accounts/fireworks/models/glm-\d+p\d+$",
    json_mode=JsonMode.STRICT_JSON_SCHEMA,
    token_accounting=TokenAccounting.PROMPT_INCLUDES_CACHED,
    reasoning_mechanism=ReasoningMechanism.REASONING_EFFORT_NONE,
    privacy=HostPrivacy(
        egress_host="api.fireworks.ai",
        model_origin="zhipu",  # GLM-5.2 is Zhipu/z.ai's model; Fireworks hosts it directly.
        direct_hosted=True,  # Fireworks runs model-library requests on its own infra.
        trains_on_inputs=False,  # "We do not use your prompts ... to train or improve our
        # AI models without your explicit opt-in" (privacy notice); Outrider does not opt in.
        retention=(
            "Fireworks does not log or store prompt or generation data for open models "
            "(GLM-5.2 is an open model) on the chat-completions inference path Outrider uses; "
            "prompt-cache KV may reside in volatile memory for several minutes, never "
            "persisted. The Responses-API store=True 30-day retention is a DIFFERENT endpoint "
            "Outrider never calls."
        ),
        source_url="https://docs.fireworks.ai/guides/security_compliance/data_handling",
        verified_date="2026-07-06",
    ),
)

# OpenAI native (api.openai.com), GPT-5.6 family — specs/2026-07-18-openai-native-host.md.
# WIRE-PENDING (#056 admission): registered + selectable so the paid probe can exercise
# the real code path, but NOT production-admitted until the captured wire fixture, the
# scorecard, and the node-admission instruments land — the Fireworks precedent shipped
# its profile WITH the fixtures; this one ships ahead of them by design (spec gates).
# JSON_OBJECT, not strict: OpenAI's strict json_schema REQUIRES every property in
# `required` + additionalProperties:false (structured-outputs guide, mirror 2026-07-18) —
# the required-completion shape #056 amendment (b) rejected as harmful to the proof
# boundary — and unlike the GLM hosts OpenAI genuinely ENFORCES strict, so the raw
# partial-required analyze schema would 400. json_object gives a syntactic-JSON guarantee
# with the soft-path backstops unchanged. Explicit model slugs only: the `gpt-5.6` alias
# routes to Sol server-side and would desync the request-side pricing key.
OPENAI_PROFILE: Final[HostProfile] = HostProfile(
    host_id="openai",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    model_slug_pattern=r"^gpt-5\.6-(sol|terra|luna)$",
    json_mode=JsonMode.JSON_OBJECT,
    token_accounting=TokenAccounting.PROMPT_INCLUDES_CACHED_WRITES_REPORTED,
    reasoning_mechanism=ReasoningMechanism.REASONING_EFFORT_NONE,
    flat_rate_input_ceiling_tokens=272_000,
    sends_prompt_cache_key=True,
    requested_service_tier="default",
    token_limit_param=TokenLimitParam.MAX_COMPLETION_TOKENS,
    privacy=HostPrivacy(
        egress_host="api.openai.com",
        model_origin="openai",
        direct_hosted=True,  # first-party API; no reseller in the path.
        trains_on_inputs=False,  # "data sent to the OpenAI API is not used to train or
        # improve OpenAI models (unless you explicitly opt in)" — Outrider does not opt in.
        retention=(
            "Abuse-monitoring logs may retain prompts/responses up to 30 days by default; "
            "Zero Data Retention / Modified Abuse Monitoring are approval-only controls "
            "Outrider does not assume. Prompt-cache prefixes: prompt_cache_options.ttl "
            "sets a 30-minute MINIMUM lifetime (the only supported value and the default) "
            "and OpenAI may retain eligibility longer, while encrypted KV cache "
            "application state is not retained past its 24-hour expiration (the your-data "
            "guide names that maximum explicitly, distinct from the TTL minimum). The "
            "pre-5.6 prompt_cache_retention selector is deprecated for this family. "
            "Chat Completions store defaults to false and is never set true."
        ),
        source_url="https://developers.openai.com/api/docs/guides/your-data",
        verified_date="2026-07-18",
    ),
)

# ADD-A-HOST CHECKLIST (e.g. DeepInfra — arc 2). The registries below
# (HOST_PROFILES, HOST_DEFAULT_MODELS) + pricing's RATE_TABLE / MIN_CACHEABLE_TOKENS are
# the ONLY host enumerations in src/, and the error strings auto-derive their host lists
# from them, so this is genuinely "add-a-profile":
#   1. Here: define a frozen HostProfile (host_id, base_url, api_key_env, model_slug_pattern,
#      json_mode, token_accounting, reasoning_mechanism, privacy w/ source_url + verified_date)
#      and register it in HOST_PROFILES; add a HOST_DEFAULT_MODELS[host_id] row (all six
#      _MODEL_FIELDS). If the host needs a reasoning-off shape not already in
#      ReasoningMechanism, add the enum value + a shaper + a _SHAPER_REGISTRY entry + bump
#      SHAPER_CONTRACT_VERSION. DECIDE the four optional digest-folded wire behaviors
#      EXPLICITLY — never inherit them silently: flat_rate_input_ceiling_tokens (documented
#      repricing boundary?), sends_prompt_cache_key (host needs a cache key?),
#      requested_service_tier (tiered billing? declaring one makes the tier echo REQUIRED),
#      and token_limit_param (which kwarg carries the completion ceiling — the GPT-5.6 wire
#      400s on the `max_tokens` default, and a wrong inherit surfaces only on the first
#      paid probe row).
#   2. pricing.py: add the RATE_TABLE (host_id, model) row + a MIN_CACHEABLE_TOKENS row
#      (None = unknown floor, 0 = documented no-floor), bump PRICING_VERSION, update the
#      pricing-digest test.
#   3. .env: point OUTRIDER_LLM_HOST at the new host_id + set its api_key_env.
#   4. STRICT-JSON hosts (DeepInfra constrained decoding, and any future strict host):
#      ANALYZE_RESPONSE_SCHEMA is hand-trimmed to Anthropic's subset (nullable `anyOf`,
#      partial `required`) and sent verbatim with strict:True. CONFIRM on a CAPTURED WIRE
#      whether the host's strict compiler accepts that shape (#056: a new host ships only on
#      captured wire evidence + the scorecard). FIREWORKS (2026-07-06 wire) ACCEPTS the raw
#      schema verbatim — nullable `anyOf` honored, direct conforming JSON — so it needs NO
#      adapter. The prototyped `_fireworks_adapt` (nullable→type-array + required-completion)
#      is REJECTED as harmful: required-completion induced fabricated proof-boundary metadata
#      (a `query_match_id`/`trace_path` on a JUDGED finding). Do NOT force-required a strict
#      host's optional proof fields without re-examining this. Baseten's SOFT_FENCED path
#      needs no adapter either.
HOST_PROFILES: Final[Mapping[str, HostProfile]] = MappingProxyType(
    {
        BASETEN_PROFILE.host_id: BASETEN_PROFILE,
        FIREWORKS_PROFILE.host_id: FIREWORKS_PROFILE,
        OPENAI_PROFILE.host_id: OPENAI_PROFILE,
    }
)

# Per-host per-node default model slugs (NOT HostProfiles). Anthropic has an entry but no
# profile (native path). `ModelConfig.for_host` (step 4) merges these BELOW env so
# `OUTRIDER_MODEL_*` still wins. The anthropic row reproduces config.py's current ModelConfig
# field defaults; GLM hosts collapse to the single GLM slug (no Haiku/Sonnet tiering).
_MODEL_FIELDS: Final[tuple[str, ...]] = (
    "triage_model",
    "analyze_model",
    "standard_analyze_model",
    "synthesize_model",
    "trace_model",
    "patch_model",
)
_ANTHROPIC_DEFAULT_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "triage_model": "claude-haiku-4-5",
        "analyze_model": "claude-sonnet-5",
        "standard_analyze_model": "claude-haiku-4-5",
        "synthesize_model": "claude-haiku-4-5",
        "trace_model": "claude-haiku-4-5",
        "patch_model": "claude-haiku-4-5",
    }
)
_BASETEN_DEFAULT_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {field: "zai-org/GLM-5.2" for field in _MODEL_FIELDS}
)
_FIREWORKS_DEFAULT_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {field: "accounts/fireworks/models/glm-5p2" for field in _MODEL_FIELDS}
)
# GPT-5.6 tiering mirrors the anthropic shape (big model for DEEP analyze, small for the
# five cheap nodes) rather than the GLM single-slug collapse. PROVISIONAL per the
# openai-native-host spec: canonized per-field by the scorecard / node-specific
# instruments; a miss swaps that field to gpt-5.6-terra and reruns.
_OPENAI_DEFAULT_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "triage_model": "gpt-5.6-luna",
        "analyze_model": "gpt-5.6-sol",
        "standard_analyze_model": "gpt-5.6-luna",
        "synthesize_model": "gpt-5.6-luna",
        "trace_model": "gpt-5.6-luna",
        "patch_model": "gpt-5.6-luna",
    }
)
HOST_DEFAULT_MODELS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "anthropic": _ANTHROPIC_DEFAULT_MODELS,
        "baseten": _BASETEN_DEFAULT_MODELS,
        "fireworks": _FIREWORKS_DEFAULT_MODELS,
        "openai": _OPENAI_DEFAULT_MODELS,
    }
)


def resolve_host_profile(host_id: str) -> HostProfile:
    """Resolve a built-in OpenAI-compatible host profile. (Custom-from-env is arc 2; the
    native `anthropic` path is selected by string upstream, never resolved here.)"""
    try:
        return HOST_PROFILES[host_id]
    except KeyError:
        raise ValueError(
            f"unknown OpenAI-compatible host {host_id!r}; known hosts: {sorted(HOST_PROFILES)}"
        ) from None


# The native Anthropic host's identity. Anthropic stays OUTSIDE the HostProfile registry
# (#056, string-selected), so its identity lives here as constants, not a HostProfile.
# Centralized so the provider's stamp AND the lifespan's build_graph completion-event
# closure share ONE source (no drift — Codex guardrail). The digest is distinct from any
# HostProfile digest so cache + replay separate anthropic from GLM-host calls; computed
# from a fixed string at load (self-documenting; a value change is an intentional bump).
ANTHROPIC_PROFILE_ID: Final[str] = "anthropic"
ANTHROPIC_CONTRACT_DIGEST: Final[str] = hashlib.sha256(b"outrider:anthropic-native:v1").hexdigest()


def resolve_host_identity(host: str, *, reasoning: bool) -> tuple[str, bool, str]:
    """The host-identity triad `(profile_id, reasoning_enabled, profile_contract_digest)`
    for a host (DECISIONS.md#056). The SINGLE source for both a provider's stamp and the
    lifespan's `build_graph` completion-event closure, so the two cannot drift —
    `test_provider_identity` pins provider-stamped == this.

    `anthropic` is the native path (no profile, no reasoning toggle in V1, so a requested
    `reasoning` fails closed). Every other host resolves through `resolve_host_profile`,
    combining the operator's `reasoning` flag with the profile's `reasoning_forced_on`
    exactly as the provider does (`requested or forced_on`).
    """
    if host == ANTHROPIC_PROFILE_ID:
        if reasoning:
            raise ValueError(
                "OUTRIDER_LLM_REASONING is on but OUTRIDER_LLM_HOST is 'anthropic', which "
                "has no reasoning toggle in V1 — unset it or select a reasoning-capable host."
            )
        return (ANTHROPIC_PROFILE_ID, False, ANTHROPIC_CONTRACT_DIGEST)
    profile = resolve_host_profile(host)
    return (
        profile.host_id,
        reasoning or profile.reasoning_forced_on,
        profile.profile_contract_digest,
    )
