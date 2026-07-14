"""Integration proof for the setup-only serving surface (spec F6, DECISIONS.md#070).

Against a real migrated Postgres, with a FAKE manifest conversion (no GitHub network):

  A. the REAL composition root (`main.create_app` + the production lifespan) boots
     database-unconfigured WITHOUT the GitHub App triad, mounts `/setup`, and returns the exact
     setup-only `(route, method) → status` table;
  B. all four credential consumers fail closed BEFORE configuration;
  C. those same already-constructed consumers work AFTER activation, with **no restart**.

The unit test `tests/unit/test_setup_route_gating.py` pins the gate's return code + the exact gated
set structurally; THIS test pins the booted-app status codes + the
fail-closed→working-without-restart transition against a live DB. Test A drives the actual
`create_app()` + `build_lifespan(...)` (heavy deps stubbed: LLM provider, checkpointer, graph — the
credential provider + setup wiring are real); Tests B/C exercise the four consumers directly over a
real `DatabaseCredentialProvider`.
"""

from __future__ import annotations

import importlib
import json as _json  # avoid shadowing the `json` kwarg in `_FakeAppClient.arequest`
import secrets
import types
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import outrider.github.authz as authz_mod
from outrider.api.lifespan import build_lifespan
from outrider.api.setup.config import SetupSettings
from outrider.api.setup.gating import require_credentials_configured
from outrider.api.setup.router import build_setup_router
from outrider.api.setup.state_machine import SetupStateMachine
from outrider.api.webhooks.router import router as webhook_router
from outrider.github.auth import make_installation_client_factory
from outrider.github.authz import (
    LiveAuthOutcome,
    list_installation_ids,
    make_installation_authorizer,
)
from outrider.github.credential_crypto import CREDENTIAL_ENC_KEY_ENV
from outrider.github.credentials import DatabaseCredentialProvider, GitHubUnconfiguredError
from outrider.github.manifest_conversion import ManifestConversion
from outrider.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

# `outrider.api` re-exports a `lifespan` function that shadows the submodule; resolve the module for
# monkeypatching `build_graph` (mirrors test_lifespan_wires_concurrency_bound).
lifespan_module = importlib.import_module("outrider.api.lifespan")

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_BASE = "https://ci.acme.com"
_INSTALLATION_ID = 12345
_REPO_ID = 999
_APP_ID = 4242
_WEBHOOK_SECRET = "wh-secret-from-onboarding"  # noqa: S105
# GitHub's ACTUAL conversion wire shape — subscribable-only events + implicit metadata:read.
_WIRE_PERMS = {"metadata": "read", "contents": "read", "pull_requests": "write"}
_WIRE_EVENTS = ["pull_request"]


def _valid_rsa_pem() -> str:
    """A real 2048-bit RSA PEM so `make_installation_client_factory`'s client construction succeeds
    after activation (githubkit may parse the key). Generated once at import."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


_TEST_PEM = _valid_rsa_pem()


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_PUBLIC_BASE_URL", _BASE)
    monkeypatch.setenv("OUTRIDER_SETUP_STATE_SECRET", secrets.token_urlsafe(32))
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, Fernet.generate_key().decode("ascii"))


@pytest_asyncio.fixture
async def engine(migrated_db: str) -> AsyncGenerator[AsyncEngine]:
    eng = create_async_engine(migrated_db)
    try:
        yield eng
    finally:
        await eng.dispose()


def _good_conversion() -> Callable[[str], Awaitable[ManifestConversion]]:
    async def _convert(code: str) -> ManifestConversion:  # noqa: ARG001 — fake ignores the code
        return ManifestConversion(
            app_id=_APP_ID,
            slug="acme-outrider",
            client_id="Iv1.dead",
            pem=SecretStr(_TEST_PEM),
            webhook_secret=SecretStr(_WEBHOOK_SECRET),
            owner_login="acme",
            permissions=_WIRE_PERMS,
            events=_WIRE_EVENTS,
        )

    return _convert


def _state_from_target(target_url: str) -> str:
    return parse_qs(urlparse(target_url).query)["state"][0]


# ── (A) the REAL composition root boots database-unconfigured + the exact matrix ──────────────────


@pytest.mark.asyncio
async def test_production_app_boots_database_unconfigured_and_gates(
    monkeypatch: pytest.MonkeyPatch,
    migrated_db: str,
    make_stub_llm_provider: type,
    noop_severity_policy_fingerprint_check: Callable[[object], Awaitable[None]],
    in_memory_checkpointer_factory: Callable[[], Any],
) -> None:
    """`main.create_app(demo_mode=False)` + the production lifespan boot in `database` mode with the
    GitHub App triad DELETED, proving the #070 promise: the real app boots unonboarded and stays
    503-gated. The credential provider + setup router wiring are real; only the LLM provider,
    checkpointer, and graph (irrelevant to F6) are stubbed to keep the boot offline + fast."""
    monkeypatch.setenv("OUTRIDER_GITHUB_CREDENTIAL_SOURCE", "database")
    monkeypatch.setenv("OUTRIDER_ADMIN_API_KEY", _ADMIN_KEY)
    monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", secrets.token_urlsafe(32))
    monkeypatch.setenv("OUTRIDER_SWEEP_DISABLED", "1")
    # The whole point: database mode needs NO GitHub App triad — delete it to prove the boot.
    for var in (
        "OUTRIDER_GITHUB_APP_ID",
        "OUTRIDER_GITHUB_APP_PRIVATE_KEY",
        "OUTRIDER_GITHUB_WEBHOOK_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        lifespan_module, "build_graph", lambda **_: MagicMock(ainvoke=AsyncMock(return_value={}))
    )

    # hide_parameters=True mirrors production (`_default_engine_factory`); the lifespan enforces it
    # (DECISIONS#016 logs-stay-metadata-only) and refuses an engine without it.
    engine = create_async_engine(migrated_db, hide_parameters=True)
    lifespan = build_lifespan(
        engine_factory=lambda: engine,
        provider_factory=lambda *_a, **_kw: make_stub_llm_provider(),
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,
    )
    app = create_app(demo_mode=False)  # THE real composition root (routes + gating + setup mount).
    try:
        async with lifespan(app):
            # Booted database-unconfigured: the provider is the DB-backed fail-closed one, not env.
            provider = app.state.credential_provider
            assert isinstance(provider, DatabaseCredentialProvider)
            assert await provider.is_configured() is False

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://t") as client:
                # AVAILABLE (need no App credentials / reachable, not 503).
                assert (await client.get("/health")).status_code == 200
                assert (await client.get("/privacy")).status_code == 200
                assert (await client.get("/setup/status")).json() == {
                    "status": "UNCONFIGURED",
                    "configured": False,
                }
                assert (await client.post("/setup", json={"org": "acme"})).status_code == 401
                # /setup/callback + /setup/reset are reachable — a bad state / missing admin is a
                # 400 / 401, NOT the credential gate's 503.
                assert (
                    await client.get("/setup/callback", params={"code": "x", "state": "bad.sig"})
                ).status_code == 400
                assert (await client.post("/setup/reset")).status_code == 401
                # Read-only dashboard stays up (admin-authed), not credential-gated.
                assert (await client.get("/api/reviews", headers=_AUTH)).status_code != 503

                # GATED → 503 (fail closed) while not CONFIGURED.
                webhook = await client.post(
                    "/webhooks/github",
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": "sha256=deadbeef",
                    },
                    json={"action": "opened"},
                )
                assert webhook.status_code == 503, webhook.text
                assert (await client.post("/reviews/42/decide", json={})).status_code == 503
                assert (await client.get("/slack/install")).status_code == 503
                assert (
                    await client.get("/slack/oauth/callback", params={"code": "x", "state": "y"})
                ).status_code == 503
    finally:
        await engine.dispose()


# ── (B) all four credential consumers fail closed BEFORE configuration ────────────────────────────


@pytest.mark.asyncio
async def test_four_consumers_fail_closed_before_configured(engine: AsyncEngine) -> None:
    provider = DatabaseCredentialProvider(async_sessionmaker(engine, expire_on_commit=False))

    # Root: the setup-only state — is_configured False, current() raises (the webhook secret
    # source).
    assert await provider.is_configured() is False
    with pytest.raises(GitHubUnconfiguredError):
        await provider.current()

    # Consumer 1 — github_factory: propagates GitHubUnconfiguredError (no client minted).
    factory = make_installation_client_factory(provider)
    with pytest.raises(GitHubUnconfiguredError):
        await factory(_INSTALLATION_ID)

    # Consumer 2 — #065 authorizer: translates to UNCERTAIN (fail closed, no GitHub call).
    authorize = make_installation_authorizer(provider)
    result = await authorize(_INSTALLATION_ID, _REPO_ID)
    assert result.outcome is LiveAuthOutcome.UNCERTAIN
    assert result.authorized is False

    # Consumer 3 — reconcile janitor's list step: propagates GitHubUnconfiguredError.
    with pytest.raises(GitHubUnconfiguredError):
        await list_installation_ids(provider)


# ── (C) all four consumers work AFTER activation — same instances, no restart ─────────────────────


def _resp(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(text=text)


class _FakeAppClient:
    """An `async with`-able githubkit App-client stand-in. GET → an active installation; POST →
    a token; the app-installations list → one installation. Records the `app_id` of the credentials
    `make_app_client` was handed, so a test can prove the consumer resolved the ACTIVATED creds."""

    def __init__(self, seen_app_ids: list[int], credentials: Any) -> None:
        seen_app_ids.append(credentials.app_id)

    async def __aenter__(self) -> _FakeAppClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def arequest(
        self, method: str, path: str, *, json: Any = None, headers: Any = None
    ) -> Any:  # noqa: ARG002
        if method != "GET":  # POST = token mint
            return _resp(_json.dumps({"token": "ghs_x"}))
        # The paginated app-installations LIST carries `per_page=`; the single install-check is
        # `/app/installations/{id}`. One-item page (< page size) ends the list loop after page 1.
        if "per_page=" in path:
            return _resp(_json.dumps([{"id": _INSTALLATION_ID}]))
        return _resp(_json.dumps({"id": _INSTALLATION_ID, "suspended_at": None}))


@pytest.mark.asyncio
async def test_all_four_consumers_work_after_activation_without_restart(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive onboarding to CONFIGURED (fake convert), then prove the SAME already-constructed
    provider + all four consumers resolve the activated credentials with NO restart. The two
    graceful-deny consumers (authorizer, list) reach GitHub via `make_app_client`, monkeypatched to
    an offline fake that records the app_id it was handed — proving credential resolution, not
    just a coincidental fail-closed value."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    provider = DatabaseCredentialProvider(session_factory)
    machine = SetupStateMachine(session_factory)

    app = FastAPI()
    app.include_router(
        build_setup_router(machine=machine, settings=SetupSettings(), convert=_good_conversion())
    )
    gate = [Depends(require_credentials_configured)]
    app.include_router(webhook_router, dependencies=gate)
    app.state.session_factory = session_factory
    app.state.credential_provider = provider
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)

    # Consumers built ONCE, before activation — the no-restart claim is that THESE start working.
    factory = make_installation_client_factory(provider)
    authorize = make_installation_authorizer(provider)
    seen_app_ids: list[int] = []
    monkeypatch.setattr(
        authz_mod, "make_app_client", lambda creds: _FakeAppClient(seen_app_ids, creds)
    )

    with TestClient(app) as client:
        # Before: all four fail closed.
        assert await provider.is_configured() is False
        with pytest.raises(GitHubUnconfiguredError):
            await factory(_INSTALLATION_ID)
        denied = await authorize(_INSTALLATION_ID, _REPO_ID)
        assert denied.outcome is LiveAuthOutcome.UNCERTAIN
        assert denied.authorized is False
        with pytest.raises(GitHubUnconfiguredError):
            await list_installation_ids(provider)
        assert seen_app_ids == []  # current() raised before make_app_client could be reached
        before = client.post(
            "/webhooks/github",
            headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": "sha256=deadbeef"},
            json={"action": "opened"},
        )
        assert before.status_code == 503

        # Activate — same process, no restart.
        start = client.post("/setup", json={"org": "acme"}, headers=_AUTH).json()
        cb = client.get(
            "/setup/callback",
            params={"code": "CODE", "state": _state_from_target(start["target_url"])},
            follow_redirects=False,
        )
        assert cb.status_code == 302, cb.text
        assert client.get("/setup/status").json() == {"status": "CONFIGURED", "configured": True}

        # After — the SAME instances resolve the activated credentials.
        assert await provider.is_configured() is True
        creds = await provider.current()  # (webhook-secret source)
        assert creds.webhook_secret.get_secret_value() == _WEBHOOK_SECRET
        assert creds.app_id == _APP_ID
        # Consumer 1 — github_factory mints a real client now (no raise).
        assert hasattr(await factory(_INSTALLATION_ID), "rest")
        # Consumer 2 — the authorizer resolves creds → reaches GitHub → AUTHORIZED.
        assert (await authorize(_INSTALLATION_ID, _REPO_ID)).outcome is LiveAuthOutcome.AUTHORIZED
        # Consumer 3 — list_installation_ids resolves creds → returns the live set.
        assert await list_installation_ids(provider) == {_INSTALLATION_ID}
        # ...and both graceful-deny consumers received the ACTIVATED app_id, proving resolution.
        assert seen_app_ids == [_APP_ID, _APP_ID]
        # Consumer 4 — the webhook route flips 503 → 4xx (past the gate; handler rejects bad sig).
        after = client.post(
            "/webhooks/github",
            headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": "sha256=deadbeef"},
            json={"action": "opened"},
        )
        assert after.status_code != 503
        assert 400 <= after.status_code < 500
