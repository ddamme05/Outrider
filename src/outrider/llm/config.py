# Per-node LLM model selection, env-backed.
# See specs/2026-05-05-llm-provider-wrapper.md and docs/spec.md §4.2.
# Host-aware selection via `ModelConfig.for_host` (DECISIONS.md#056); the lifespan
# OUTRIDER_LLM_HOST read that drives it lands with the provider-factory wiring.
"""ModelConfig — env-backed per-node LLM model selection.

Backs the `model-strings-from-config-not-hardcoded` invariant: every LLM
call site reads its model from this config, not a hardcoded string.
Defaults match canonical spec §4.2 as amended: Sonnet for DEEP-tier
analyze only; Haiku for triage/trace/patch, STANDARD-tier analyze
(DECISIONS.md#041), and synthesize (DECISIONS.md#043). Each field is overridable via
`OUTRIDER_MODEL_<FIELD>` env var (e.g., `OUTRIDER_MODEL_TRIAGE_MODEL`)
so eval runs can swap models per-tier without code changes.

Validators:
  - regex: model strings must match the V1 Anthropic family pattern
    `^claude-(haiku|sonnet|opus)-\\d+(-\\d+)?(-\\d{8})?$`. Catches typos
    at construction (e.g., `OUTRIDER_MODEL_ANALYZE_MODEL=gpt-4`). The
    optional 8-digit `YYYYMMDD` suffix accepts dated SDK-catalog pins;
    dated forms normalize to their undated alias for pricing lookup
    (see `outrider.llm.pricing.normalize_to_pricing_key`).
  - deprecation: rejects any model string in
    `anthropic.resources.messages.DEPRECATED_MODELS` (a `dict[str, str]`
    of model id → deprecation date). The SDK would otherwise emit a
    `DeprecationWarning` at first call; we surface it eagerly at
    construction.
"""

import re
from typing import Final

from anthropic.resources.messages import DEPRECATED_MODELS
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from outrider.llm.host_profiles import HOST_DEFAULT_MODELS

__all__ = ["ModelConfig"]

# V1 Anthropic family pattern.
# Three accepted shapes per Anthropic SDK 0.100 model catalog:
#   - `claude-{haiku,sonnet,opus}-{major}` (forward-compat with future
#     major-only releases)
#   - `claude-{haiku,sonnet,opus}-{major}-{minor}` (the canonical alias
#     form, e.g., `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`)
#   - `claude-{haiku,sonnet,opus}-{major}-{minor}-{YYYYMMDD}` (the dated
#     "exact pin" form, e.g., `claude-haiku-4-5-20251001`)
# previous regex rejected the dated form
# even though the SDK catalog publishes it as the precise model id.
_VALID_MODEL_PATTERN: Final = re.compile(r"^claude-(haiku|sonnet|opus)-\d+(-\d+)?(-\d{8})?$")


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

    # corrected to current Anthropic
    # model family per SDK 0.100 docs. Previous defaults named
    # `claude-sonnet-4-7` which doesn't exist in the SDK; current
    # active models are Opus 4.7, Sonnet 4.6, Haiku 4.5.
    triage_model: str = "claude-haiku-4-5"
    analyze_model: str = "claude-sonnet-4-6"
    # See DECISIONS.md#041 — the DEEP-tier model is `analyze_model`; STANDARD-tier files
    # route here. Defaults to Haiku (1/3 of Sonnet's per-token price) after the eval
    # quality gate (PR #51) showed Haiku holds STANDARD-tier recall and does not over-flag
    # safe code worse than Sonnet. DEEP-tier files stay on `analyze_model` (Sonnet); so do
    # trace-fetched (no-tier) files. Override per-deployment via
    # OUTRIDER_MODEL_STANDARD_ANALYZE_MODEL.
    standard_analyze_model: str = "claude-haiku-4-5"
    # See DECISIONS.md#043 — synthesize runs Haiku: nothing finding-bearing
    # in the node is model-dependent (findings are analyze output, severity
    # is policy-set, dedup is content-hash); the model writes only the
    # user-facing summary PROSE, the same bounded-generation class as the
    # already-Haiku trace/patch calls. Watch is summary quality, not
    # findings. Override per-deployment via OUTRIDER_MODEL_SYNTHESIZE_MODEL.
    synthesize_model: str = "claude-haiku-4-5"
    trace_model: str = "claude-haiku-4-5"
    # See DECISIONS.md#040 — suggested-patch generation (synthesize) uses Haiku:
    # patch-gen is the kind of bounded generation Haiku handles, and cost is
    # analyze-dominated, so the patch call must not be a second Sonnet call.
    patch_model: str = "claude-haiku-4-5"

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
