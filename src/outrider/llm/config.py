# Per-node LLM model selection, env-backed.
# See specs/2026-05-05-llm-provider-wrapper.md and docs/spec.md §4.2.
"""ModelConfig — env-backed per-node LLM model selection.

Backs the `model-strings-from-config-not-hardcoded` invariant: every LLM
call site reads its model from this config, not a hardcoded string.
Defaults match canonical spec §4.2 (Haiku for triage/trace; Sonnet for
analyze/synthesize). Each field is overridable via
`OUTRIDER_MODEL_<FIELD>` env var (e.g., `OUTRIDER_MODEL_TRIAGE_MODEL`)
so eval runs can swap models per-tier without code changes.

Validators:
  - regex: model strings must match the V1 Anthropic family pattern
    `^claude-(haiku|sonnet|opus)-\\d+(-\\d+)?(-\\d{8})?$`. Catches typos
    at construction (e.g., `OUTRIDER_MODEL_ANALYZE_MODEL=gpt-4`). The
    optional 8-digit `YYYYMMDD` suffix accepts dated SDK-catalog pins
    ; dated forms normalize to their undated alias for
    pricing lookup (see `outrider.llm.pricing.normalize_to_pricing_key`).
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
    synthesize_model: str = "claude-sonnet-4-6"
    trace_model: str = "claude-haiku-4-5"

    @field_validator(
        "triage_model",
        "analyze_model",
        "synthesize_model",
        "trace_model",
    )
    @classmethod
    def _validate_model_string(cls, value: str) -> str:
        if not _VALID_MODEL_PATTERN.match(value):
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
