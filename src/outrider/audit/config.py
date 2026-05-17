# See specs/2026-05-16-audit-persister.md + DECISIONS.md#012 + #016.
"""RetentionSettings — operator-overridable retention TTL for audit content.

Per `DECISIONS.md#012` ("Every review, finding, audit event, and installation
row has a retention TTL, set in configuration and operator-overridable via
`pydantic-settings`"), retention values must be operator-tunable rather than
hard-coded module constants.

This module owns the LLM-content TTL only. Other retention TTLs (`findings`,
`reviews`) belong to their owning subsystems' settings when those land.

Validators:
  - `gt=timedelta(0)`: rejects zero AND negative TTLs at construction. Operator
    setting `OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL=0` raises `ValidationError`
    at startup, not a silent metadata-only-replay slip where the retention sweep
    deletes every content row on its next tick.
"""

from datetime import timedelta
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["RetentionSettings"]


# Default 90 days per DECISIONS.md#016: the architectural anchor is
# "LLM content TTL ≤ findings TTL — most-sensitive content has shortest TTL."
# Specific number lives here as a default; operator override is the
# expected path for compliance/forensics use cases.
_DEFAULT_LLM_CONTENT_RETENTION_TTL: Final[timedelta] = timedelta(days=90)


class RetentionSettings(BaseSettings):
    """Retention TTLs for audit content tables.

    Env-prefix matches the per-subsystem `OUTRIDER_<SUBSYSTEM>_` convention
    established by `ModelConfig` in `llm/config.py`. Full env var name for
    the only field today: `OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL`.

    `frozen=True` means construction-time-only configuration; tests that
    need a non-default value re-construct with explicit kwargs rather
    than mutating an instance. Mirrors the `ModelConfig` pattern.

    `pydantic-settings` 2.13.1 parses `timedelta` env vars as **ISO-8601
    duration strings only** (`P7D`, `PT24H`, `PT3600S`). Bare integers
    (e.g., `604800` for seconds) are NOT accepted — pinned by
    `tests/unit/test_retention_settings.py::test_env_var_bare_integer_seconds_is_rejected`.
    The `gt=timedelta(0)` constraint rejects zero AND negative values at
    construction, surfacing operator misconfiguration before any content
    row lands.

    **Test-construction pattern.** `frozen=True` blocks both attribute
    reassignment AND `model_copy(update={...})` (the copy raises
    ValidationError on the frozen check). Tests that need a non-default
    TTL must construct via explicit kwargs:
    `RetentionSettings(llm_content_retention_ttl=timedelta(days=N))`.
    Reaching for `model_copy` first will produce a confusing error.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_AUDIT_",
        extra="forbid",
        frozen=True,
    )

    llm_content_retention_ttl: timedelta = Field(
        default=_DEFAULT_LLM_CONTENT_RETENTION_TTL,
        gt=timedelta(0),
        description=(
            "TTL for llm_call_content rows. Default 90 days per "
            "DECISIONS.md#016. Overridable via "
            "OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL env var (ISO-8601 "
            "duration string only — e.g., P7D, PT24H, PT3600S; bare "
            "integer seconds are NOT accepted by pydantic-settings 2.13.1)."
        ),
    )
