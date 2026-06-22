"""Production wiring guard — DEMO_MODE boots keyless (no LLM/GitHub/Slack/graph).

Piece 2 of the public demo deployment. The demo box must hold NO live
credentials: the lifespan's `if demo_mode:` branch (right after the shared
persister step) serves the read-only dashboard over precomputed seed reviews and
skips the entire review/write half. These guards prove:

- the lifespan reaches `yield` with NO truncation secret, NO Anthropic key, and
  NO GitHub App env present (only the public read token);
- the provider factory, checkpointer factory, and `build_graph` are never called
  — nothing that could call out is constructed;
- `app.state` carries the read-side deps (engine / session / persister / admin
  token) and `None` for every review-side dep, with the agent-view surface
  disabled (no agent token);
- the contrast: the SAME minimal env with demo_mode off fails loud at the
  truncation-secret gate — so the branch, not the ambient env, is what boots it.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan

lifespan_module = importlib.import_module("outrider.api.lifespan")

# Every credential the production boot path requires — deleted to prove the demo
# branch never reaches the steps that read them.
_PRODUCTION_CREDENTIAL_ENV = (
    "OUTRIDER_TRUNCATION_HMAC_SECRET",
    "ANTHROPIC_API_KEY",
    "OUTRIDER_GITHUB_APP_ID",
    "OUTRIDER_GITHUB_APP_PRIVATE_KEY",
    "OUTRIDER_GITHUB_WEBHOOK_SECRET",
    "OUTRIDER_AGENT_API_KEY",
)


def _mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.dispose = AsyncMock(return_value=None)
    engine.url.drivername = "postgresql+psycopg"
    engine.sync_engine.hide_parameters = True
    return engine


def _strip_production_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _PRODUCTION_CREDENTIAL_ENV:
        monkeypatch.delenv(var, raising=False)
    # The one credential the demo box DOES hold: the public read token.
    monkeypatch.setenv("OUTRIDER_ADMIN_API_KEY", "demo-read-token")


async def test_demo_mode_boots_keyless_and_constructs_no_review_half(
    monkeypatch: pytest.MonkeyPatch,
    noop_severity_policy_fingerprint_check: object,
) -> None:
    """`demo_mode=True` reaches yield with no LLM/GitHub/truncation env, builds no
    provider/checkpointer/graph, and wires read-side app.state with None review-side."""
    _strip_production_credentials(monkeypatch)
    engine = _mock_engine()

    calls = {"provider": 0, "checkpointer": 0, "build_graph": 0}

    def spy_provider_factory(_persister: object, _model_config: object) -> Any:
        calls["provider"] += 1
        raise AssertionError("provider_factory must not be called in demo mode")

    def spy_checkpointer_factory() -> Any:
        calls["checkpointer"] += 1
        raise AssertionError("checkpointer_factory must not be called in demo mode")

    def spy_build_graph(**_kwargs: Any) -> Any:
        calls["build_graph"] += 1
        raise AssertionError("build_graph must not be called in demo mode")

    monkeypatch.setattr(lifespan_module, "build_graph", spy_build_graph)

    lifespan = build_lifespan(
        engine_factory=lambda: engine,
        provider_factory=spy_provider_factory,
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,
        checkpointer_factory=spy_checkpointer_factory,
    )

    app = FastAPI()
    app.state.demo_mode = True
    async with lifespan(app):
        # Read-side: the dashboard's deps are real.
        assert app.state.engine is engine
        assert app.state.session_factory is not None
        assert app.state.persister is not None
        assert app.state.audit_persister is app.state.persister
        # admin token is stashed as the Pydantic SecretStr (same as production).
        assert app.state.admin_api_key.get_secret_value() == "demo-read-token"
        # Agent token unset → agent-view surface disabled.
        assert app.state.agent_api_key is None
        # Review/write half: absent (None, not missing).
        assert app.state.provider is None
        assert app.state.github_app_settings is None
        assert app.state.github_factory is None
        assert app.state.compiled_graph is None
        assert app.state.run_graph is None
        assert app.state.checkpointer is None
        assert app.state.review_status_reader is None
        assert app.state.slack_oauth_settings is None
        assert app.state.anomaly_sink is None
        assert app.state.sweep_task is None

    # Nothing that could call out (LLM client / checkpointer / graph) was built.
    assert calls == {"provider": 0, "checkpointer": 0, "build_graph": 0}
    # The engine is still disposed on teardown (registered on the AsyncExitStack
    # before the demo branch's early return).
    engine.dispose.assert_awaited_once()


async def test_demo_mode_off_with_same_env_fails_at_truncation_gate(
    monkeypatch: pytest.MonkeyPatch,
    make_stub_llm_provider: type,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
) -> None:
    """Contrast: the SAME credential-stripped env, with demo_mode off, fails loud
    at the Step-0 truncation gate — proving the branch (not the env) boots demo."""
    _strip_production_credentials(monkeypatch)
    engine = _mock_engine()

    lifespan = build_lifespan(
        engine_factory=lambda: engine,
        provider_factory=lambda _p, _m: make_stub_llm_provider(),
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,
        checkpointer_factory=in_memory_checkpointer_factory,
    )

    app = FastAPI()
    # demo_mode NOT set → getattr default False → production path.
    with pytest.raises(RuntimeError, match="OUTRIDER_TRUNCATION_HMAC_SECRET"):
        async with lifespan(app):
            pass
