"""Load analysis limits and optional context from startup settings.

OUTRIDER_ANALYZE_ variables override the defaults. Startup passes them through
build_graph rather than storing them in review state (nodes-receive-deps-via-closure).

See specs/2026-06-17-analyze-cost-fairness.md.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from outrider.agent.nodes.analyze import ANALYZE_MAX_CONCURRENCY, DEFAULT_REVIEW_BUDGET_TOKENS

__all__ = ["AnalyzeConfig"]


class AnalyzeConfig(BaseSettings):
    """Frozen analysis limits and an off-by-default sibling-context option.

    Defaults are DEFAULT_REVIEW_BUDGET_TOKENS (200_000) and ANALYZE_MAX_CONCURRENCY
    (4). Explicit construction supports tests and callers that supply settings."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_ANALYZE_",
        extra="forbid",
        frozen=True,
    )

    review_budget_tokens: int = Field(default=DEFAULT_REVIEW_BUDGET_TOKENS, gt=0)
    max_concurrency: int = Field(default=ANALYZE_MAX_CONCURRENCY, ge=1)
    companion_context_enabled: bool = False
