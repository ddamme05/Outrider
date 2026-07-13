"""Integration tests for `api/setup/router` against a real Postgres (#070).

Drives the four onboarding endpoints through a real state machine (migrated DB) with a FAKE manifest
conversion (no GitHub network): admin gating, the happy path start → callback → CONFIGURED + a
decryptable credential row, the security rejections (bad state, replayed state), and the saga
failures (binding mismatch / conversion error → ORPHANED, never persisted).
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.api.setup.config import SetupSettings
from outrider.api.setup.router import build_setup_router
from outrider.api.setup.state_machine import SetupStateMachine
from outrider.github.credential_crypto import CREDENTIAL_ENC_KEY_ENV, decrypt_credential
from outrider.github.manifest_conversion import ManifestConversion, ManifestConversionError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_BASE = "https://ci.acme.com"
# GitHub's ACTUAL conversion wire shape — subscribable-only events + implicit metadata:read — NOT an
# echo of the EXPECTED_* constants, so a drift in those constants is caught by the happy-path test.
_WIRE_PERMS = {"metadata": "read", "contents": "read", "pull_requests": "write"}
_WIRE_EVENTS = ["pull_request"]


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
            app_id=4242,
            slug="acme-outrider",
            client_id="Iv1.dead",
            pem=SecretStr("-----BEGIN RSA PRIVATE KEY-----\nX\n-----END RSA PRIVATE KEY-----"),
            webhook_secret=SecretStr("wh-secret"),
            owner_login="acme",
            permissions=_WIRE_PERMS,
            events=_WIRE_EVENTS,
        )

    return _convert


def _mount(
    engine: AsyncEngine, *, convert: Callable[[str], Awaitable[ManifestConversion]]
) -> TestClient:
    machine = SetupStateMachine(async_sessionmaker(engine, expire_on_commit=False))
    app = FastAPI()
    app.include_router(
        build_setup_router(machine=machine, settings=SetupSettings(), convert=convert)
    )
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    return TestClient(app)


def _state_from_target(target_url: str) -> str:
    return parse_qs(urlparse(target_url).query)["state"][0]


async def _active_credential_count(engine: AsyncEngine) -> int:
    async with async_sessionmaker(engine)() as session:
        return (
            await session.execute(
                text("SELECT count(*) FROM github_app_credentials WHERE is_active")
            )
        ).scalar_one()


# ── status + admin gating ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_unconfigured(engine: AsyncEngine) -> None:
    client = _mount(engine, convert=_good_conversion())
    resp = client.get("/setup/status")
    assert resp.status_code == 200
    assert resp.json() == {"status": "UNCONFIGURED", "configured": False}


@pytest.mark.asyncio
async def test_start_requires_admin(engine: AsyncEngine) -> None:
    client = _mount(engine, convert=_good_conversion())
    assert client.post("/setup", json={"org": "acme"}).status_code == 401


@pytest.mark.asyncio
async def test_start_returns_manifest_and_target(engine: AsyncEngine) -> None:
    client = _mount(engine, convert=_good_conversion())
    resp = client.post("/setup", json={"org": "acme"}, headers=_AUTH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["target_url"].startswith(
        "https://github.com/organizations/acme/settings/apps/new?state="
    )
    assert body["manifest"]["public"] is False
    assert body["manifest"]["redirect_url"] == f"{_BASE}/setup/callback"
    assert client.get("/setup/status").json()["status"] == "AWAITING_CALLBACK"


@pytest.mark.asyncio
async def test_start_rejects_bad_org(engine: AsyncEngine) -> None:
    client = _mount(engine, convert=_good_conversion())
    assert client.post("/setup", json={"org": "bad/org?x=1"}, headers=_AUTH).status_code == 422


# ── happy path ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_happy_path(engine: AsyncEngine) -> None:
    client = _mount(engine, convert=_good_conversion())
    start = client.post("/setup", json={"org": "acme"}, headers=_AUTH).json()
    state = _state_from_target(start["target_url"])

    cb = client.get(
        "/setup/callback", params={"code": "CODE", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 302
    assert cb.headers["location"] == "https://github.com/apps/acme-outrider/installations/new"
    assert client.get("/setup/status").json() == {"status": "CONFIGURED", "configured": True}

    # a decryptable active credential row landed
    assert await _active_credential_count(engine) == 1
    async with async_sessionmaker(engine)() as session:
        row = (
            await session.execute(
                text("SELECT app_id, slug, pem_ciphertext FROM github_app_credentials")
            )
        ).one()
    assert row[0] == 4242
    assert row[1] == "acme-outrider"
    assert "BEGIN RSA PRIVATE KEY" in decrypt_credential(row[2]).get_secret_value()


# ── security rejections ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_callback_bad_state_rejected(engine: AsyncEngine) -> None:
    client = _mount(engine, convert=_good_conversion())
    resp = client.get(
        "/setup/callback", params={"code": "CODE", "state": "forged.sig"}, follow_redirects=False
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_replayed_state_rejected(engine: AsyncEngine) -> None:
    client = _mount(engine, convert=_good_conversion())
    start = client.post("/setup", json={"org": "acme"}, headers=_AUTH).json()
    state = _state_from_target(start["target_url"])
    first = client.get(
        "/setup/callback", params={"code": "CODE", "state": state}, follow_redirects=False
    )
    assert first.status_code == 302  # consumed
    replay = client.get(
        "/setup/callback", params={"code": "CODE2", "state": state}, follow_redirects=False
    )
    assert replay.status_code == 400  # nonce already spent


# ── saga failures → ORPHANED, never persisted ─────────────────────────────────


@pytest.mark.asyncio
async def test_binding_mismatch_orphans(engine: AsyncEngine) -> None:
    async def _wrong_owner(code: str) -> ManifestConversion:  # noqa: ARG001
        return ManifestConversion(
            app_id=1,
            slug="s",
            client_id=None,
            pem=SecretStr("p"),
            webhook_secret=SecretStr("w"),
            owner_login="attacker-org",  # != bound "acme"
            permissions=_WIRE_PERMS,
            events=_WIRE_EVENTS,
        )

    client = _mount(engine, convert=_wrong_owner)
    state = _state_from_target(
        client.post("/setup", json={"org": "acme"}, headers=_AUTH).json()["target_url"]
    )
    cb = client.get(
        "/setup/callback", params={"code": "CODE", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 302
    assert cb.headers["location"] == f"{_BASE}/setup/status"
    assert client.get("/setup/status").json()["status"] == "ORPHANED"
    assert await _active_credential_count(engine) == 0  # never persisted


@pytest.mark.asyncio
async def test_conversion_error_orphans(engine: AsyncEngine) -> None:
    async def _boom(code: str) -> ManifestConversion:  # noqa: ARG001
        raise ManifestConversionError("conversion 422")

    client = _mount(engine, convert=_boom)
    state = _state_from_target(
        client.post("/setup", json={"org": "acme"}, headers=_AUTH).json()["target_url"]
    )
    cb = client.get(
        "/setup/callback", params={"code": "CODE", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 302
    assert cb.headers["location"] == f"{_BASE}/setup/status"
    assert client.get("/setup/status").json()["status"] == "ORPHANED"
    assert await _active_credential_count(engine) == 0


# ── reset ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_requires_admin_and_orphaned(engine: AsyncEngine) -> None:
    client = _mount(engine, convert=_boom_conversion())
    # drive to ORPHANED
    state = _state_from_target(
        client.post("/setup", json={"org": "acme"}, headers=_AUTH).json()["target_url"]
    )
    client.get("/setup/callback", params={"code": "C", "state": state}, follow_redirects=False)
    assert client.get("/setup/status").json()["status"] == "ORPHANED"

    assert client.post("/setup/reset").status_code == 401  # admin required
    assert client.post("/setup/reset", headers=_AUTH).status_code == 200
    assert client.get("/setup/status").json()["status"] == "UNCONFIGURED"
    # reset again (not ORPHANED now) → 409
    assert client.post("/setup/reset", headers=_AUTH).status_code == 409


def _boom_conversion() -> Callable[[str], Awaitable[ManifestConversion]]:
    async def _boom(code: str) -> ManifestConversion:  # noqa: ARG001
        raise ManifestConversionError("conversion failed")

    return _boom
