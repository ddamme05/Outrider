# Per DECISIONS.md#070 / spec F6 — setup-only route surface + credential gating.
"""The setup-only route surface (F6): `/setup*` mounts in `database` mode only, and every
credential-dependent router fails closed with 503 while the provider is not `CONFIGURED`.

Two layers, unit-level (no DB, no lifespan):
  1. The `require_credentials_configured` dependency in isolation — 503 for a `None` / unconfigured
     provider, pass-through for a configured one.
  2. The real `create_app()` route surface — `/setup*` present in `database` mode, absent in `env`
     and demo mode; and the three side-effecting production routers carry the gate dependency.

The end-to-end status table under a booted `database`-unconfigured lifespan is the integration
concern; here the gate's return code + the exact gated route set are pinned structurally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from outrider.api.setup.gating import require_credentials_configured
from outrider.main import create_app

if TYPE_CHECKING:
    import pytest
    from fastapi.dependencies.models import Dependant

# The exact credential-dependent / side-effecting routers gated behind CONFIGURED (spec F6).
_GATED_ROUTES = {
    "/webhooks/github",
    "/reviews/{review_id}/decide",
    "/slack/install",
    "/slack/oauth/callback",
}
_SETUP_ROUTES = {"/setup", "/setup/callback", "/setup/status", "/setup/reset"}

# A sufficiently-long, non-placeholder CSRF secret for `database`-mode boot validation.
_STATE_SECRET = "s3tup-state-secret-that-is-long-enough-xyz"  # noqa: S105 — test fixture


class _FakeProvider:
    """Minimal credential provider — only `is_configured()` is read by the gate."""

    def __init__(self, *, configured: bool) -> None:
        self._configured = configured

    async def is_configured(self) -> bool:
        return self._configured

    async def current(self) -> Any:  # pragma: no cover — the gate never reaches current()
        raise AssertionError("gate must not call current()")


def _app_with_gate(provider: object) -> TestClient:
    app = FastAPI()

    @app.get("/gated", dependencies=[Depends(require_credentials_configured)])
    async def _gated() -> dict[str, str]:
        return {"ok": "yes"}

    app.state.credential_provider = provider
    return TestClient(app)


# ── The dependency in isolation ──────────────────────────────────────────────


def test_gate_returns_503_when_provider_absent() -> None:
    with _app_with_gate(None) as client:
        assert client.get("/gated").status_code == 503


def test_gate_returns_503_when_provider_unconfigured() -> None:
    with _app_with_gate(_FakeProvider(configured=False)) as client:
        assert client.get("/gated").status_code == 503


def test_gate_passes_when_provider_configured() -> None:
    with _app_with_gate(_FakeProvider(configured=True)) as client:
        resp = client.get("/gated")
        assert resp.status_code == 200
        assert resp.json() == {"ok": "yes"}


# ── The real create_app() route surface ──────────────────────────────────────


def _database_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_GITHUB_CREDENTIAL_SOURCE", "database")
    monkeypatch.setenv("OUTRIDER_PUBLIC_BASE_URL", "https://outrider.example.com")
    monkeypatch.setenv("OUTRIDER_SETUP_STATE_SECRET", _STATE_SECRET)


def _paths(app: FastAPI) -> set[str]:
    return {r.path for r in app.routes if isinstance(r, APIRoute)}


def test_setup_router_mounted_in_database_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _database_mode_env(monkeypatch)
    assert _paths(create_app(demo_mode=False)) >= _SETUP_ROUTES


def test_setup_router_absent_in_env_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_GITHUB_CREDENTIAL_SOURCE", "env")
    assert not any(p.startswith("/setup") for p in _paths(create_app(demo_mode=False)))


def test_setup_router_absent_in_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even with the `database` source set, demo mode returns before mounting any side-effecting
    # surface — the setup router is a side-effecting admin surface, absent on the keyless box.
    _database_mode_env(monkeypatch)
    assert not any(p.startswith("/setup") for p in _paths(create_app(demo_mode=True)))


def _all_dep_calls(dependant: Dependant) -> set[object]:
    calls: set[object] = set()
    for sub in dependant.dependencies:
        calls.add(sub.call)
        calls |= _all_dep_calls(sub)
    return calls


def test_side_effecting_routers_carry_the_setup_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_GITHUB_CREDENTIAL_SOURCE", "env")
    app = create_app(demo_mode=False)
    seen = set()
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path in _GATED_ROUTES:
            assert require_credentials_configured in _all_dep_calls(route.dependant), (
                f"{route.path} is not behind require_credentials_configured (spec F6)"
            )
            seen.add(route.path)
    assert seen == _GATED_ROUTES, f"missing gated routes: {_GATED_ROUTES - seen}"
