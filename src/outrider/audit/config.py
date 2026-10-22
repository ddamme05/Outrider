# See specs/2026-05-16-audit-persister.md + DECISIONS.md#012 + #016.
"""Operator-configured retention for LLM content, findings, and reviews.

All three default to 90 days and must be positive, with
`llm_content <= findings <= review` (`DECISIONS.md#012/#014/#016`).
Prompt/completion content expires first; its review outlives the content.
Construction validates the ordering before writers assign row deadlines.
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
    """Frozen retention settings read from `OUTRIDER_AUDIT_*` environment variables.

    The TTL names are `LLM_CONTENT_RETENTION_TTL`, `FINDINGS_RETENTION_TTL`,
    and `REVIEW_RETENTION_TTL`. Environment values use ISO-8601 durations
    such as `P7D` or `PT24H`; bare integer seconds are rejected.

    Construct with explicit keyword arguments for overrides. Do not use
    `model_copy(update=...)`: it bypasses positivity and ordering validation
    even on frozen models. See `test_retention_settings.py`.
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
        """Require positive-field TTLs to follow `llm_content <= findings <= review`.

        Content must purge no later than its review; replay rejects surviving
        content without that row. `findings.review_id` also cascades on review
        deletion. `model_copy(update=...)` bypasses this construction-time check.
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
