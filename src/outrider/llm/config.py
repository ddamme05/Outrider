"""Choose models and reasoning settings for each review step.

Startup settings select the host and any per-step model overrides. Call sites
use the resulting configuration rather than embedding model names.
See DECISIONS.md#041, #043, #056.
"""

import re
from typing import Final, Literal

from anthropic.resources.messages import DEPRECATED_MODELS
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from outrider.llm.host_profiles import HOST_DEFAULT_MODELS

__all__ = [
    "ModelConfig",
    "is_anthropic_family_model",
    "model_text_only_thinking",
    "model_uses_adaptive_thinking",
]

# V1 Anthropic family pattern.
# Three accepted Anthropic model-ID shapes:
#   - `claude-{haiku,sonnet,opus}-{major}` (forward-compat with future
#     major-only releases)
#   - `claude-{haiku,sonnet,opus}-{major}-{minor}` (the canonical alias
#     form, e.g., `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`)
#   - `claude-{haiku,sonnet,opus}-{major}-{minor}-{YYYYMMDD}` (the dated
#     "exact pin" form, e.g., `claude-haiku-4-5-20251001`)
# The dated form accepts exact model-version IDs such as `claude-haiku-4-5-20251001`.
_VALID_MODEL_PATTERN: Final = re.compile(r"^claude-(haiku|sonnet|opus)-\d+(-\d+)?(-\d{8})?$")
# A dated slug suffix is YYYYMMDD, so exactly this many digits.
_DATED_SUFFIX_LEN: Final[int] = 8


def is_anthropic_family_model(slug: str) -> bool:
    """True iff `slug` is an Anthropic claude-family model id (the native host's
    slug shape). The canonical predicate over `_VALID_MODEL_PATTERN`, reused by
    `build_graph` to detect a non-anthropic `model_config` that must carry the
    host-identity triad (DECISIONS.md#056)."""
    return bool(_VALID_MODEL_PATTERN.fullmatch(slug))


def model_uses_adaptive_thinking(slug: str) -> bool:
    """Whether the model needs explicit thinking control and no sampling parameters."""
    return model_text_only_thinking(slug) is not None


def model_text_only_thinking(slug: str) -> Literal["disabled", "between_tools"] | None:
    """Choose the thinking setting for a single text response without tools.

    Sonnet 5.5 uses ``between_tools``; Sonnet 5, Opus 4.7+ and Haiku 5.5
    use ``disabled``. Earlier models need no setting. Model-name knowledge
    stays here rather than in the provider (`model-strings-from-config-not-hardcoded`).
    """
    if not is_anthropic_family_model(slug):
        return None
    # claude-{family}-{major}[-{minor}][-{YYYYMMDD}]
    _, family, major_text, *rest = slug.split("-")
    major = int(major_text)
    # The first trailing segment is either the minor OR a bare dated suffix with no
    # minor (`claude-sonnet-5-20260615`) — the greedy `(-\d+)?` group in
    # `_VALID_MODEL_PATTERN` captures a date into the minor slot when no minor is
    # given. A date is exactly `_DATED_SUFFIX_LEN` digits, so a shorter run is the
    # minor, which keeps a three-digit minor from reading as a date.
    minor = int(rest[0]) if rest and len(rest[0]) < _DATED_SUFFIX_LEN else 0
    if family == "sonnet":
        if (major, minor) >= (5, 5):
            return "between_tools"
        return "disabled" if major >= 5 else None
    if (family == "opus" and (major, minor) >= (4, 7)) or (
        family == "haiku" and (major, minor) >= (5, 5)
    ):
        return "disabled"
    return None


class _EnvModelOverrides(BaseSettings):
    """Operator-set `OUTRIDER_MODEL_*` overrides ONLY — each unset field is `None`.

    Separate from `ModelConfig` so the host-aware merge in `ModelConfig.for_host` can tell a
    field the operator set (a value) from one that falls back to the host default (`None`);
    an unset var must be `None`, not a required-field construction error (DECISIONS.md#056).
    `settings_customise_sources` is class-level and cannot take a caller-seeded instance, so
    the merge is a pure two-step in `for_host`, not a sources hook.
    """

    model_config = SettingsConfigDict(env_prefix="OUTRIDER_MODEL_", extra="forbid", frozen=True)

    triage_model: str | None = None
    analyze_model: str | None = None
    standard_analyze_model: str | None = None
    synthesize_model: str | None = None
    trace_model: str | None = None
    patch_model: str | None = None


class ModelConfig(BaseSettings):
    """Per-node model selection. Reads `OUTRIDER_MODEL_*` env vars; falls
    back to the canonical spec §4.2 defaults.

    `frozen=True` means construction-time-only configuration; per-tier
    runtime overrides go via `BaseSettings`-style re-construction with
    explicit kwargs in tests, NOT mutation.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_MODEL_",
        extra="forbid",
        frozen=True,
    )

    # DEEP-tier and trace-fetched file analysis use analyze_model.
    triage_model: str = "claude-haiku-5-5"
    analyze_model: str = "claude-sonnet-5-5"
    # STANDARD-tier files route separately (DECISIONS.md#041).
    standard_analyze_model: str = "claude-haiku-5-5"
    # Summary prose only; findings, severity and dedup are separate (DECISIONS.md#043).
    synthesize_model: str = "claude-haiku-5-5"
    # Trace ranks candidates; fetched-file analysis still uses analyze_model.
    trace_model: str = "claude-haiku-5-5"
    # Suggested replacement text is validated by patch_generation (DECISIONS.md#040).
    patch_model: str = "claude-haiku-5-5"

    def models(self) -> tuple[str, ...]:
        """Return the six role models in declaration order, preserving duplicates."""
        return (
            self.triage_model,
            self.analyze_model,
            self.standard_analyze_model,
            self.synthesize_model,
            self.trace_model,
            self.patch_model,
        )

    @field_validator(
        "triage_model",
        "analyze_model",
        "standard_analyze_model",
        "synthesize_model",
        "trace_model",
        "patch_model",
    )
    @classmethod
    def _validate_model_string(cls, value: str) -> str:
        # fullmatch, not match: `.match` against a `$`-anchored pattern still admits a
        # trailing newline (`$` matches before a final `\n`).
        if not _VALID_MODEL_PATTERN.fullmatch(value):
            raise ValueError(
                f"Model string {value!r} does not match V1 Anthropic family "
                f"pattern {_VALID_MODEL_PATTERN.pattern!r}"
            )
        if value in DEPRECATED_MODELS:
            deprecation_date = DEPRECATED_MODELS[value]
            raise ValueError(
                f"Model {value!r} is deprecated by Anthropic "
                f"({deprecation_date}); update OUTRIDER_MODEL_* env to a "
                f"current model"
            )
        return value

    @classmethod
    def for_host(cls, host: str) -> "ModelConfig":
        """Build the per-node `ModelConfig` for `host` via a pure two-step merge: the env
        override if the operator set it, else `HOST_DEFAULT_MODELS[host]`, field by field
        (DECISIONS.md#056). Env wins; host defaults fill; this never reads `OUTRIDER_LLM_HOST`.

        The `anthropic` host keeps the claude-family regex + `DEPRECATED_MODELS` validation
        byte-identical (`model_validate` runs the field-validator on the merged dict); every
        other host holds its native slugs, validated downstream by the provider's
        `HostProfile.validate_model_slug` (the claude field-validator here would wrongly
        reject `zai-org/GLM-5.2`).
        """
        try:
            defaults = HOST_DEFAULT_MODELS[host]
        except KeyError:
            raise ValueError(
                f"unknown OUTRIDER_LLM_HOST {host!r}; known hosts: {sorted(HOST_DEFAULT_MODELS)}"
            ) from None
        overrides = _EnvModelOverrides()
        merged: dict[str, str] = {}
        for field in cls.model_fields:
            override = getattr(overrides, field)
            merged[field] = override if override is not None else defaults[field]
        if host == "anthropic":
            # Validating path: model_validate runs the claude regex + DEPRECATED_MODELS
            # field-validator on the merged dict (no env re-read), byte-identical to ModelConfig().
            return cls.model_validate(merged)
        # Native-slug host: the provider validates the slug downstream; the claude
        # field-validator would wrongly reject it, so construct without validation.
        # (model_construct with **dict trips a known pydantic-settings/mypy arg-type quirk.)
        return cls.model_construct(**merged)  # type: ignore[arg-type]
