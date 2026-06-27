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
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, field_validator

from outrider.llm.base import LLMInvalidResponseError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# Bump on ANY shaper/accounting FUNCTION-body change (audit-7 #3). It is folded into
# `profile_contract_digest`, so a behavior change to a shaper invalidates warm cache rows
# even though no profile DATA field changed.
SHAPER_CONTRACT_VERSION: Final[str] = "v1"


class ReasoningMechanism(StrEnum):
    """How a host disables reasoning — the four observed wire shapes + a sentinel for
    hosts with no documented off-switch."""

    CHAT_TEMPLATE_ARGS = (
        "chat_template_args"  # Baseten/Telnyx: extra_body.chat_template_args.enable_thinking=False
    )
    REASONING_EFFORT_NONE = "reasoning_effort_none"  # Fireworks/DeepInfra: reasoning_effort="none"
    REASONING_ENABLED_FALSE = (
        "reasoning_enabled_false"  # Together: extra_body.reasoning={"enabled": False}
    )
    THINKING_DISABLED = "thinking_disabled"  # Z.ai: extra_body.thinking={"type": "disabled"}
    NONE = "none"  # no documented off-switch (Cloudflare); reasoning stays on


class TokenAccounting(StrEnum):
    """Whether the host's `prompt_tokens` includes the cached subset (§8a)."""

    PROMPT_INCLUDES_CACHED = "prompt_includes_cached"  # Baseten/DeepInfra
    PROMPT_EXCLUDES_CACHED = "prompt_excludes_cached"  # Anthropic-like
    UNVERIFIED = "unverified"  # never assumed — fail loud if cached actually fires


class JsonMode(StrEnum):
    """Structured-output capability the host's `response_format` honors."""

    STRICT_JSON_SCHEMA = "strict_json_schema"  # Fireworks/DeepInfra: constrained decoding
    SOFT_FENCED = "soft_fenced"  # Baseten: soft/fenced (FUP-196; fence-strip backstop)
    JSON_OBJECT = "json_object"  # Z.ai: json_object only, no schema


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
    """No off-switch — reasoning stays on. The provider stamps the EFFECTIVE state via
    `HostProfile.reasoning_enabled_effective` (True here), so a NONE host audits as
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
) -> tuple[int, int, int]:
    """§8a normalization. Returns `(input_tokens, cache_read_tokens, output_tokens)`.

    `prompt_includes_cached` (Baseten/DeepInfra): cached is a SUBSET of prompt_tokens, so
    subtract — capping cached at prompt_tokens keeps `input + cache_read == prompt_tokens`
    self-consistent. `prompt_excludes_cached`: prompt_tokens is already the uncached input.
    `unverified`: NEVER guess — raise if the response actually reports cached tokens.

    A negative usage component is a malformed wire payload (normalized at this boundary per
    trust-boundaries §5 sub-rule 6) — reject it before it drives a negative token or cost.
    """
    if prompt_tokens < 0 or raw_cached_tokens < 0 or completion_tokens < 0:
        raise LLMInvalidResponseError(
            f"negative usage component: prompt={prompt_tokens} "
            f"cached={raw_cached_tokens} completion={completion_tokens}"
        )
    if accounting is TokenAccounting.PROMPT_INCLUDES_CACHED:
        cache_read = min(raw_cached_tokens, prompt_tokens)
        return prompt_tokens - cache_read, cache_read, completion_tokens
    if accounting is TokenAccounting.PROMPT_EXCLUDES_CACHED:
        return prompt_tokens, raw_cached_tokens, completion_tokens
    if raw_cached_tokens > 0:
        raise LLMInvalidResponseError()
    return prompt_tokens, 0, completion_tokens


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
                SHAPER_CONTRACT_VERSION,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_model_slug(self, slug: str) -> None:
        if re.match(self.model_slug_pattern, slug) is None:
            raise ValueError(
                f"model {slug!r} does not match host {self.host_id!r} slug pattern "
                f"{self.model_slug_pattern!r}"
            )

    def apply_reasoning_off(self, kwargs: dict[str, Any]) -> None:
        """Mutate `kwargs` to disable reasoning per this host's mechanism."""
        _SHAPER_REGISTRY[self.reasoning_mechanism](kwargs)

    @property
    def reasoning_enabled_effective(self) -> bool:
        """The `reasoning_enabled` value the provider MUST stamp on the event/response.
        Outrider always requests reasoning OFF, so every mechanism with a real off-switch is
        effectively False; `NONE` has no off-switch (reasoning stays on) and is True, so the
        audit flag reflects reality rather than a silent `False`. The mechanism is folded
        into `profile_contract_digest`, so cache never colludes a NONE host with an
        off-switch host (audit: reasoning/cache identity)."""
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

HOST_PROFILES: Final[Mapping[str, HostProfile]] = MappingProxyType(
    {BASETEN_PROFILE.host_id: BASETEN_PROFILE}
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
        "analyze_model": "claude-sonnet-4-6",
        "standard_analyze_model": "claude-haiku-4-5",
        "synthesize_model": "claude-haiku-4-5",
        "trace_model": "claude-haiku-4-5",
        "patch_model": "claude-haiku-4-5",
    }
)
_BASETEN_DEFAULT_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {field: "zai-org/GLM-5.2" for field in _MODEL_FIELDS}
)
HOST_DEFAULT_MODELS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {"anthropic": _ANTHROPIC_DEFAULT_MODELS, "baseten": _BASETEN_DEFAULT_MODELS}
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
