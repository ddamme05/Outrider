"""Integration proof for the setup-only serving surface (spec F6, DECISIONS.md#070).

Three claims the spec's `## Test scenarios` (integration) require, proven end-to-end against a real
migrated Postgres with a FAKE manifest conversion (no GitHub network):

  1. the exact setup-only `(route, method) → status` table while NOT `CONFIGURED`;
  2. all four credential consumers fail closed BEFORE configuration;
  3. those SAME consumer/provider instances work AFTER activation, with **no restart**.

The credential provider, gate, and setup state machine are the production objects, wired the way
`main._include_routers` wires them (gate on the side-effecting routers, `/setup` mounted) but with a
fake `convert` injected so the callback reaches `CONFIGURED` offline. The unit test
`tests/unit/test_setup_route_gating.py` pins the gate's return code + the exact gated set
structurally; THIS test pins the booted-app status codes + the fail-closed→working-without-restart
transition against a live DB (the piece the unit test explicitly defers to integration).
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.api.dashboard import hitl_router
from outrider.api.privacy import router as privacy_router
from outrider.api.setup.config import SetupSettings
from outrider.api.setup.gating import require_credentials_configured
from outrider.api.setup.router import build_setup_router
from outrider.api.setup.state_machine import SetupStateMachine
from outrider.api.slack import slack_oauth_router
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

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_BASE = "https://ci.acme.com"
_INSTALLATION_ID = 12345
_REPO_ID = 999
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
_WEBHOOK_SECRET = "wh-secret-from-onboarding"  # noqa: S105


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
            pem=SecretStr(_TEST_PEM),
            webhook_secret=SecretStr(_WEBHOOK_SECRET),
            owner_login="acme",
            permissions=_WIRE_PERMS,
            events=_WIRE_EVENTS,
        )

    return _convert


def _boot(engine: AsyncEngine) -> tuple[TestClient, DatabaseCredentialProvider]:
    """Build the F6 serving surface the way `main._include_routers` does — the `/setup` router (fake
    convert) + the credential-gated side-effecting routers + a couple of always-available reads —
    over a real DatabaseCredentialProvider. Returns the client + the SAME provider so a test can
    assert its before/after-activation behavior without a restart."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    provider = DatabaseCredentialProvider(session_factory)
    machine = SetupStateMachine(session_factory)

    app = FastAPI()
    # Always-available (no App credentials): privacy + a local /health, plus /setup (fake convert).
    app.include_router(privacy_router)
    app.include_router(
        build_setup_router(machine=machine, settings=SetupSettings(), convert=_good_conversion())
    )

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    # Credential-gated side-effecting routers (the F6 → 503 set), gated as main.py wires them.
    gate = [Depends(require_credentials_configured)]
    app.include_router(webhook_router, dependencies=gate)
    app.include_router(hitl_router, dependencies=gate)
    app.include_router(slack_oauth_router, dependencies=gate)

    app.state.session_factory = session_factory
    app.state.credential_provider = provider
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    return TestClient(app), provider


def _state_from_target(target_url: str) -> str:
    return parse_qs(urlparse(target_url).query)["state"][0]


def _drive_to_configured(client: TestClient) -> None:
    """POST /setup (admin) → GET /setup/callback → CONFIGURED (fake convert persists the row)."""
    start = client.post("/setup", json={"org": "acme"}, headers=_AUTH).json()
    state = _state_from_target(start["target_url"])
    cb = client.get(
        "/setup/callback", params={"code": "CODE", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 302, cb.text
    assert client.get("/setup/status").json() == {"status": "CONFIGURED", "configured": True}


# ── (1) the exact setup-only (route, method) → status table while not CONFIGURED ──────────────────


@pytest.mark.asyncio
async def test_setup_only_route_matrix_unconfigured(engine: AsyncEngine) -> None:
    client, _ = _boot(engine)
    with client:
        # AVAILABLE — need no App credentials.
        assert client.get("/health").status_code == 200
        assert client.get("/privacy").status_code == 200
        assert client.get("/setup/status").json() == {"status": "UNCONFIGURED", "configured": False}
        # POST /setup is available (admin-gated, NOT credential-gated): 401 without admin, not 503.
        assert client.post("/setup", json={"org": "acme"}).status_code == 401

        # GATED → 503 (fail closed) while not CONFIGURED. Requests are otherwise well-formed so the
        # ONLY rejection is the credential gate.
        webhook = client.post(
            "/webhooks/github",
            headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": "sha256=deadbeef"},
            json={"action": "opened"},
        )
        assert webhook.status_code == 503, webhook.text
        assert client.post("/reviews/42/decide", json={}).status_code == 503
        assert client.get("/slack/install").status_code == 503
        assert (
            client.get("/slack/oauth/callback", params={"code": "x", "state": "y"}).status_code
            == 503
        )


# ── (2) all four credential consumers fail closed BEFORE configuration ────────────────────────────


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


# ── (3) the SAME consumer/provider instances work AFTER activation — no restart ───────────────────


@pytest.mark.asyncio
async def test_consumers_work_after_activation_without_restart(engine: AsyncEngine) -> None:
    client, provider = _boot(engine)
    # Built ONCE, before activation — the no-restart claim is that this instance starts working.
    factory = make_installation_client_factory(provider)

    with client:
        # Before: fail closed.
        assert await provider.is_configured() is False
        with pytest.raises(GitHubUnconfiguredError):
            await factory(_INSTALLATION_ID)
        before = client.post(
            "/webhooks/github",
            headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": "sha256=deadbeef"},
            json={"action": "opened"},
        )
        assert before.status_code == 503

        # Activate — same process, no restart.
        _drive_to_configured(client)

        # After: the SAME provider + factory now resolve credentials (read per-op from the DB).
        assert await provider.is_configured() is True
        creds = await provider.current()
        assert creds.webhook_secret.get_secret_value() == _WEBHOOK_SECRET
        assert creds.app_id == 4242
        client_obj = await factory(_INSTALLATION_ID)  # no raise — a real client is minted now
        assert hasattr(client_obj, "rest")

        # The webhook route now gets PAST the gate: a bad signature is rejected by the HANDLER
        # (4xx), NOT the credential gate's 503 — proving activation took effect with no restart.
        after = client.post(
            "/webhooks/github",
            headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": "sha256=deadbeef"},
            json={"action": "opened"},
        )
        assert after.status_code != 503
        assert 400 <= after.status_code < 500
