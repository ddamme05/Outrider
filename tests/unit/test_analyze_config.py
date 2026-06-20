"""AnalyzeConfig — the per-review analyze token budget (Stage 0).

`DEFAULT_REVIEW_BUDGET_TOKENS` (analyze.py) is the field default;
`OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS` overrides it. Before this config,
production silently used the hardcoded default because lifespan never passed
the value to build_graph.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS
from outrider.agent.nodes.analyze_config import AnalyzeConfig


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS", raising=False)


def test_default_matches_analyze_constant() -> None:
    cfg = AnalyzeConfig()
    assert cfg.review_budget_tokens == DEFAULT_REVIEW_BUDGET_TOKENS == 200_000


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS", "350000")
    cfg = AnalyzeConfig()
    assert cfg.review_budget_tokens == 350_000


def test_direct_construction_overrides() -> None:
    cfg = AnalyzeConfig(review_budget_tokens=50_000)
    assert cfg.review_budget_tokens == 50_000


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_rejects_nonpositive(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS", bad)
    with pytest.raises(ValidationError):
        AnalyzeConfig()


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AnalyzeConfig(unknown_field=1)  # type: ignore[call-arg]
