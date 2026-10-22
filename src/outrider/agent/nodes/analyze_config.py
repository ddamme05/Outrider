# See specs/2026-06-17-analyze-cost-fairness.md Stage 0 — env-configurable analyze budget.
"""Startup settings for analyze's token budget and worker concurrency.

`OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS` and `OUTRIDER_ANALYZE_MAX_CONCURRENCY`
override the defaults from analyze.py. Lifespan passes them to build_graph's
closures (`nodes-receive-deps-via-closure`); nodes do not read them from state."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from outrider.agent.nodes.analyze import ANALYZE_MAX_CONCURRENCY, DEFAULT_REVIEW_BUDGET_TOKENS

__all__ = ["AnalyzeConfig"]


class AnalyzeConfig(BaseSettings):
    """Frozen analyze settings: a positive review budget and at least one worker.

    Defaults are DEFAULT_REVIEW_BUDGET_TOKENS (200_000) and ANALYZE_MAX_CONCURRENCY
    (4). Explicit construction supports tests and callers that supply settings."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_ANALYZE_",
        extra="forbid",
        frozen=True,
    )

    review_budget_tokens: int = Field(default=DEFAULT_REVIEW_BUDGET_TOKENS, gt=0)
    max_concurrency: int = Field(default=ANALYZE_MAX_CONCURRENCY, ge=1)
