"""OUTRIDER_LLM_HOST=openai composition-root hard refusal — Arc 0 (openai-native-host).

The `openai` HostProfile is wire-implemented and evaluable, but it CANNOT satisfy the
pre-ship refusal-normalization gate in json_object mode: its refusal probes return
completed empty-result envelopes (`message.refusal=null`) indistinguishable from a clean
zero-vulnerability review, so no refusal discriminator has been demonstrated (spec Actual
Outcome; DECISIONS.md#056). A `WIRE-PENDING` label/comment is not enforcement — so the
production composition root
(`api/lifespan.py`) HARD-REFUSES `OUTRIDER_LLM_HOST=openai` before any key lookup or
provider construction. The eval harness and paid probe build the provider DIRECTLY and
bypass this seam, so they stay usable; this is a production-selection gate only.

The first two boot the real lifespan with the DEFAULT factory (no stub), so a regression
that re-opens production OpenAI selection fails here instead of in production; the third
intentionally uses the `provider_factory=` injection seam with a spy to prove ZERO
construction calls (the gate precedes construction, not just assignment).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan


def _mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.dispose = AsyncMock(return_value=None)
    engine.url.drivername = "postgresql+psycopg"
    engine.sync_engine.hide_parameters = True
    return engine


def _openai_production_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", "test-secret-truncation-hmac-key32")
    monkeypatch.setenv("OUTRIDER_SWEEP_DISABLED", "1")
    monkeypatch.setenv("OUTRIDER_LLM_HOST", "openai")


async def test_production_lifespan_openai_fails_closed_not_admitted(
    monkeypatch: pytest.MonkeyPatch,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
    github_app_env: None,  # noqa: ARG001 — sets OUTRIDER_GITHUB_* + OUTRIDER_ADMIN_API_KEY
) -> None:
    """OUTRIDER_LLM_HOST=openai (even WITH a key present) never boots a provider —
    the composition root raises the not-admitted RuntimeError."""
    _openai_production_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-must-never-be-reached")
    engine = _mock_engine()

    lifespan = build_lifespan(
        engine_factory=lambda: engine,
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,  # type: ignore[arg-type]
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="production-admitted"):
        async with lifespan(app):
            pass
    # never constructed a provider
    assert not hasattr(app.state, "provider")


async def test_openai_gate_fires_before_key_lookup(
    monkeypatch: pytest.MonkeyPatch,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
    github_app_env: None,  # noqa: ARG001
) -> None:
    """Ordering proof: with OPENAI_API_KEY ABSENT, the admission gate still fires —
    the error names the admission failure, NOT a missing key. This pins that the gate
    precedes key lookup and provider construction (so no paid activity is reachable)."""
    _openai_production_env(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = _mock_engine()

    lifespan = build_lifespan(
        engine_factory=lambda: engine,
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,  # type: ignore[arg-type]
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="production-admitted") as excinfo:
        async with lifespan(app):
            pass
    # the ADMISSION failure, not "OPENAI_API_KEY env var is required"
    assert "OPENAI_API_KEY" not in str(excinfo.value)


@pytest.mark.parametrize("spelling", ["OPENAI", "OpenAI", "  openai  "])
async def test_openai_gate_is_case_and_whitespace_normalized(
    spelling: str,
    monkeypatch: pytest.MonkeyPatch,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
    github_app_env: None,  # noqa: ARG001
) -> None:
    """A casing/whitespace variant must hit the ADMISSION refusal, not slip past the
    membership check and die later on an unrelated 'unknown host' ValueError. The host
    id is normalized at the single-authority read, so the gate's promise (refuse BEFORE
    key lookup / ModelConfig / provider construction) holds for every spelling."""
    _openai_production_env(monkeypatch)
    monkeypatch.setenv("OUTRIDER_LLM_HOST", spelling)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-must-never-be-reached")
    engine = _mock_engine()

    lifespan = build_lifespan(
        engine_factory=lambda: engine,
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,  # type: ignore[arg-type]
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="production-admitted") as excinfo:
        async with lifespan(app):
            pass
    # the ADMISSION refusal — never the downstream resolve_host_profile ValueError
    assert "unknown OpenAI-compatible host" not in str(excinfo.value)
    assert not hasattr(app.state, "provider")


async def test_openai_gate_fires_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
    github_app_env: None,  # noqa: ARG001
) -> None:
    """Construction proof (not just assignment): an injected provider_factory spy is
    NEVER called for the un-admitted host — the gate precedes construction, so no
    provider object (and no paid activity) is ever reachable."""
    _openai_production_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-must-never-be-reached")
    engine = _mock_engine()
    calls = {"provider": 0}

    def spy_provider_factory(
        _persister: object, _model_config: object, _host: object, _reasoning: object
    ) -> Any:
        calls["provider"] += 1
        raise AssertionError("provider_factory must not be called for an un-admitted host")

    lifespan = build_lifespan(
        engine_factory=lambda: engine,
        provider_factory=spy_provider_factory,
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,  # type: ignore[arg-type]
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="production-admitted"):
        async with lifespan(app):
            pass
    assert calls["provider"] == 0
