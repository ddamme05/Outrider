"""Check that startup passes the configured review-token budget to build_graph.

Wrap the real builder and set OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS to verify
configuration reaches the graph. See specs/2026-06-17-analyze-cost-fairness.md."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan

lifespan_module = importlib.import_module("outrider.api.lifespan")


@pytest.fixture(autouse=True)
def _activate_github_app_env(github_app_env: None) -> None:  # noqa: ARG001 — fixture activates env
    """Lifespan hard-requires `GitHubAppSettings()` at startup."""


async def test_lifespan_wires_review_budget_into_build_graph(
    monkeypatch: pytest.MonkeyPatch,
    make_stub_llm_provider: type,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
) -> None:
    """The production `build_graph(...)` call passes the env-configured budget,
    not the hardcoded default — so an operator's tuning actually takes effect."""
    # Lifespan now hard-requires the truncation HMAC secret at startup
    # (require_truncation_secret, landed 2026-06-19); set it so this test is
    # self-contained and does not depend on a sourced .env.
    monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", "test-secret-for-integration-01234")
    monkeypatch.setenv("OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS", "123456")

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    mock_engine.sync_engine.hide_parameters = True

    real_build_graph = lifespan_module.build_graph
    captured: dict[str, Any] = {}

    def spying_build_graph(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_build_graph(**kwargs)

    monkeypatch.setattr(lifespan_module, "build_graph", spying_build_graph)

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister, _model_config, _host, _reasoning: (
            make_stub_llm_provider()
        ),  # noqa: E501
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,
    )

    app = FastAPI()
    async with lifespan(app):
        pass

    assert "total_review_budget_tokens" in captured, (
        "lifespan's build_graph call omits total_review_budget_tokens — "
        "production silently uses the hardcoded DEFAULT_REVIEW_BUDGET_TOKENS "
        "regardless of OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS"
    )
    assert captured["total_review_budget_tokens"] == 123456, (
        f"expected the env-configured 123456, got "
        f"{captured['total_review_budget_tokens']} — AnalyzeConfig is not "
        "feeding build_graph"
    )
