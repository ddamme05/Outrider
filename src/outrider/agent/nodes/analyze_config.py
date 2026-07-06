# See specs/2026-06-17-analyze-cost-fairness.md Stage 0 — env-configurable analyze budget.
"""AnalyzeConfig — the per-review analyze token budget.

`DEFAULT_REVIEW_BUDGET_TOKENS` (defined in `analyze.py`) is the field default;
`OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS` overrides it so operators can tune the
per-review analyze spend ceiling without a code change. Closure-injected at
`build_graph(...)` per `nodes-receive-deps-via-closure`; `api/lifespan.py`
reads it and passes the value as `total_review_budget_tokens`.

Before this config existed, production silently ran on the hardcoded 200k default
because lifespan called `build_graph(...)` without `total_review_budget_tokens`
(the `analyze.py` comment claiming "production wires a tighter value from settings"
was aspirational). This config makes that wiring real and tunable. The token
budget is the live cost-control mechanism; the legacy `PER_REVIEW_BUDGET_USD`
env var in `.env.example` is NOT read by any code.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from outrider.agent.nodes.analyze import ANALYZE_MAX_CONCURRENCY, DEFAULT_REVIEW_BUDGET_TOKENS

__all__ = ["AnalyzeConfig"]


class AnalyzeConfig(BaseSettings):
    """Reads `OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS` (default
    `DEFAULT_REVIEW_BUDGET_TOKENS` = 200_000) and
    `OUTRIDER_ANALYZE_MAX_CONCURRENCY` (default `ANALYZE_MAX_CONCURRENCY`
    = 4, the fan-out's in-flight worker bound). Tests construct
    `AnalyzeConfig(review_budget_tokens=...)` directly and inject through
    `build_graph(...)`. `frozen=True`: construction-time-only config."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_ANALYZE_",
        extra="forbid",
        frozen=True,
    )

    review_budget_tokens: int = Field(default=DEFAULT_REVIEW_BUDGET_TOKENS, gt=0)
    max_concurrency: int = Field(default=ANALYZE_MAX_CONCURRENCY, ge=1)
