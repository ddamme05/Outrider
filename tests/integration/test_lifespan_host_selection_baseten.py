"""OUTRIDER_LLM_HOST=baseten composition-root wiring — the REAL provider factory.

DECISIONS.md#056 lets an operator run the whole pipeline on GLM-5.2 via Baseten by
setting OUTRIDER_LLM_HOST=baseten. Every OTHER lifespan test injects a stub via the
`provider_factory=` seam (and discards the `_host` arg), so the real
`_default_provider_factory` else-branch — host selection, `resolve_host_profile`,
the BASETEN_API_KEY fail-loud, the configured-model tuple, and the
`OpenAICompatibleProvider` construction — ships with NO coverage. These tests boot
the production lifespan with the DEFAULT factory (no stub) and the baseten host env,
so a regression in that branch fails here instead of in production.

No network mock: the openai client is constructed in `OpenAICompatibleProvider.__init__`
but never called during startup (provider construction validates models + pricing,
not key validity), so a dummy BASETEN_API_KEY boots cleanly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan
from outrider.llm.host_profiles import BASETEN_PROFILE
from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider


def _mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.dispose = AsyncMock(return_value=None)
    engine.url.drivername = "postgresql+psycopg"
    engine.sync_engine.hide_parameters = True
    return engine


def _baseten_production_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The minimal production-boot env with the Baseten host selected. `github_app_env`
    supplies OUTRIDER_GITHUB_* + OUTRIDER_ADMIN_API_KEY; this adds the rest."""
    monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", "test-secret")
    monkeypatch.setenv("OUTRIDER_SWEEP_DISABLED", "1")  # don't spawn the real sweep loop
    monkeypatch.setenv("LANGSMITH_TRACING", "")  # no tracing wrapper around the provider
    monkeypatch.setenv("OUTRIDER_LLM_HOST", "baseten")


async def test_production_lifespan_selects_baseten_provider_from_host_env(
    monkeypatch: pytest.MonkeyPatch,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
    github_app_env: None,  # noqa: ARG001 — sets OUTRIDER_GITHUB_* + OUTRIDER_ADMIN_API_KEY
) -> None:
    """OUTRIDER_LLM_HOST=baseten + BASETEN_API_KEY -> the REAL factory builds an
    OpenAICompatibleProvider bound to BASETEN_PROFILE, with every node on GLM-5.2."""
    _baseten_production_env(monkeypatch)
    monkeypatch.setenv("BASETEN_API_KEY", "fake-baseten-key-not-called-at-startup")
    engine = _mock_engine()

    # NOTE: provider_factory is NOT injected — the default `_default_provider_factory`
    # runs, which is the whole point (every other test stubs it out).
    lifespan = build_lifespan(
        engine_factory=lambda: engine,
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,  # type: ignore[arg-type]
    )

    app = FastAPI()
    async with lifespan(app):
        provider = app.state.provider
        assert isinstance(provider, OpenAICompatibleProvider), (
            f"OUTRIDER_LLM_HOST=baseten must select OpenAICompatibleProvider, got {type(provider)}"
        )
        assert provider._profile is BASETEN_PROFILE  # noqa: SLF001 — test-only profile check
        assert provider._profile.host_id == "baseten"  # noqa: SLF001
        # ModelConfig.for_host('baseten') put GLM-5.2 on every node, so the factory
        # configured the provider to serve exactly that one slug.
        assert provider._models == ("zai-org/GLM-5.2",)  # noqa: SLF001
        # the graph was built (under the baseten provider + host-identity triad).
        assert app.state.run_graph is not None


async def test_production_lifespan_baseten_fails_loud_without_key(
    monkeypatch: pytest.MonkeyPatch,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
    github_app_env: None,  # noqa: ARG001
) -> None:
    """OUTRIDER_LLM_HOST=baseten with BASETEN_API_KEY absent -> the factory fails loud
    at startup (RuntimeError naming the env var), never a silent wrong-provider boot."""
    _baseten_production_env(monkeypatch)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    engine = _mock_engine()

    lifespan = build_lifespan(
        engine_factory=lambda: engine,
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,  # type: ignore[arg-type]
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="BASETEN_API_KEY"):
        async with lifespan(app):
            pass
