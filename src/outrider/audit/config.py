# See specs/2026-05-16-audit-persister.md + DECISIONS.md#012 + #016.
"""RetentionSettings — operator-overridable retention TTL for audit content.

Per `DECISIONS.md#012` ("Every review, finding, audit event, and installation
row has a retention TTL, set in configuration and operator-overridable via
`pydantic-settings`"), retention values must be operator-tunable rather than
hard-coded module constants.

This module owns the three audit content-table retention TTLs:
`llm_content_retention_ttl` (llm_call_content rows), `findings_retention_ttl`
(findings rows, written by the findings-content writer), and
`review_retention_ttl` (reviews rows).

Validators:
  - `gt=timedelta(0)` on every TTL field: rejects zero AND negative TTLs at
    construction. Operator setting any TTL env var to `0` raises
    `ValidationError` at startup, not a silent metadata-only-replay slip
    where the retention sweep deletes every content row on its next tick.
  - Ordering check (`_enforce_retention_ordering`): the three TTLs must satisfy
    `llm_content <= findings <= review`. The most-sensitive content (LLM
    prompt/completion text) carries the shortest TTL; findings (structured
    finding content) sit between; the review row outlives both. The ordering
    is the load-bearing replay invariant — `findings.review_id` is
    `ON DELETE CASCADE`, so a findings TTL exceeding the review TTL is
    unreachable (findings cascade-die with the review) AND would let an
    operator manufacture the review-purged/content-survives state that
    `audit.replay._classify_mode` treats as corruption. Fail loud at
    construction instead.
"""

from datetime import timedelta
from typing import Final, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["RetentionSettings"]


# Default 90 days per DECISIONS.md#016: the architectural anchor is
# "LLM content TTL ≤ findings TTL — most-sensitive content has shortest TTL."
# Specific number lives here as a default; operator override is the
# expected path for compliance/forensics use cases.
_DEFAULT_LLM_CONTENT_RETENTION_TTL: Final[timedelta] = timedelta(days=90)

# Default 90 days for the `findings` table per `DECISIONS.md#014`:
# findings are a content-table purge target carrying structured finding
# content (title/description/evidence). Default equals the LLM-content and
# review defaults so the `llm <= findings <= review` ordering holds at the
# defaults; an operator may raise it up to (but not past) the review TTL.
# NOT a longer default than the review TTL — findings.review_id is
# ON DELETE CASCADE, so a longer findings TTL is unreachable.
_DEFAULT_FINDINGS_RETENTION_TTL: Final[timedelta] = timedelta(days=90)

# Default 90 days for the `reviews` table per `DECISIONS.md#012/#014`:
# reviews are a content-table purge target under retention, operator-
# overridable. Mirrors the LLM-content default; reviews are typically
# the less-sensitive surface (no prompt/completion text) so the same
# 90-day floor is the operationally simple default.
_DEFAULT_REVIEW_RETENTION_TTL: Final[timedelta] = timedelta(days=90)


class RetentionSettings(BaseSettings):
    """Retention TTLs for audit content tables.

    Env-prefix matches the per-subsystem `OUTRIDER_<SUBSYSTEM>_` convention
    established by `ModelConfig` in `llm/config.py`. Env var names:

      - `OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL` (llm_call_content rows)
      - `OUTRIDER_AUDIT_FINDINGS_RETENTION_TTL` (findings rows, written by
        the findings-content writer — operator-overridable per
        `DECISIONS.md#014`)
      - `OUTRIDER_AUDIT_REVIEW_RETENTION_TTL` (reviews rows, added by the
        intake-and-webhook spec — operator-overridable per
        `DECISIONS.md#012/#014`)

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

    **Test-construction pattern.** Tests that need a non-default TTL
    MUST construct via explicit kwargs, e.g.
    `RetentionSettings(llm_content_retention_ttl=timedelta(days=N))` or
    `RetentionSettings(review_retention_ttl=timedelta(days=N))`.

    Do NOT use `model_copy(update={...})`. Pydantic v2's `model_copy`
    is permitted on frozen models AND **does not validate the update
    payload** — a copy with `update={"llm_content_retention_ttl":
    timedelta(0)}` (or any other TTL field) silently bypasses the
    `gt=timedelta(0)` constraint AND the ordering validator below. Only the
    explicit-kwarg constructor runs the validators.
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

    findings_retention_ttl: timedelta = Field(
        default=_DEFAULT_FINDINGS_RETENTION_TTL,
        gt=timedelta(0),
        description=(
            "TTL for findings rows. Default 90 days per DECISIONS.md#014: "
            "findings are content-table purge targets carrying structured "
            "finding content. The findings-content writer reads this at "
            "insert time to populate findings.retention_expires_at. Must "
            "satisfy llm_content <= findings <= review (see ordering "
            "validator). Override via OUTRIDER_AUDIT_FINDINGS_RETENTION_TTL "
            "env var (ISO-8601 duration)."
        ),
    )

    review_retention_ttl: timedelta = Field(
        default=_DEFAULT_REVIEW_RETENTION_TTL,
        gt=timedelta(0),
        description=(
            "TTL for reviews rows. Default 90 days per DECISIONS.md#012/#014: "
            "reviews are content-table purge targets carrying operator-"
            "overridable TTL. The webhook handler reads this at insert "
            "time to populate reviews.retention_expires_at. Override via "
            "OUTRIDER_AUDIT_REVIEW_RETENTION_TTL env var (ISO-8601 duration)."
        ),
    )

    @model_validator(mode="after")
    def _enforce_retention_ordering(self) -> Self:
        """Pin `llm_content <= findings <= review` (the retention ordering).

        The most-sensitive content carries the shortest TTL, and content
        purges no later than its review row. Replay's `_classify_mode`
        treats a surviving content row whose review is purged as
        corruption; `findings.review_id` is ON DELETE CASCADE, so a
        findings TTL longer than the review TTL is both unreachable and a
        misconfiguration. Raising here (construction time) keeps the
        ordering an enforced invariant, not just a documented expectation.

        Runs on the explicit-kwarg constructor only; `model_copy(update=...)`
        bypasses it (same caveat as the per-field `gt=timedelta(0)` checks).
        """
        if not (
            self.llm_content_retention_ttl
            <= self.findings_retention_ttl
            <= self.review_retention_ttl
        ):
            raise ValueError(
                "retention TTLs must satisfy "
                "llm_content_retention_ttl <= findings_retention_ttl <= "
                "review_retention_ttl (most-sensitive content purges first, "
                "and content purges no later than its review row; "
                "findings.review_id is ON DELETE CASCADE so a longer findings "
                f"TTL is unreachable). Got llm_content="
                f"{self.llm_content_retention_ttl}, findings="
                f"{self.findings_retention_ttl}, review="
                f"{self.review_retention_ttl}."
            )
        return self
