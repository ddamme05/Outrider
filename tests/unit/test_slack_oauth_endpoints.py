"""Slack OAuth install-flow endpoints (commit 6.3e).

Pins the input-boundary behavior of GET /slack/install + GET /slack/oauth/callback:
admin auth on install, opt-in 503 when unconfigured, signed-state round-trip, and —
the load-bearing security property — that the callback reads installation_id/channel
from the VERIFIED STATE (never the query params), encrypts the bot token before
persist, and fails closed on denied/forged/expired/exchange-failure inputs.

`exchange_code` (Slack network) + `set_slack_config` (DB) are mocked; `sign_state` /
`verify_state` / `encrypt_token` run for real against monkeypatched env secrets, so
the CSRF + encryption paths are exercised end-to-end.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from outrider.api.slack import slack_oauth_router
from outrider.notify.config import SlackOAuthSettings
from outrider.notify.oauth_state import STATE_SECRET_ENV, sign_state, verify_state
from outrider.notify.slack_oauth import SlackOAuthError, SlackOAuthResult
from outrider.notify.token_crypto import TOKEN_ENC_KEY_ENV, decrypt_token

# A real (throwaway) Fernet key + state secret so encrypt_token / sign_state run for
# real in these tests. Not live credentials.
_TEST_FERNET_KEY = "5deJqws1Xhk7vjpx0_pKr7GebqDXYiXLSshLmkI5jy0="  # noqa: S105 (test fixture)
_STATE_SECRET = "test-slack-state-secret-value-0123456789"  # noqa: S105 (test fixture, ≥32 chars)
_AUTH = {"Authorization": "Bearer admin-key"}


class _FakeBegin:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    def begin(self) -> _FakeBegin:
        return _FakeBegin()

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _fake_session_factory() -> _FakeSession:
    return _FakeSession()


def _settings() -> SlackOAuthSettings:
    return SlackOAuthSettings(
        client_id="123.456",
        client_secret=SecretStr("app-secret"),
        redirect_uri="https://outrider.example.com/slack/oauth/callback",
    )


def _build_app(*, configured: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(slack_oauth_router)
    app.state.admin_api_key = SecretStr("admin-key")
    app.state.slack_oauth_settings = _settings() if configured else None
    app.state.session_factory = _fake_session_factory
    return app


@pytest.fixture
def secrets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STATE_SECRET_ENV, _STATE_SECRET)
    monkeypatch.setenv(TOKEN_ENC_KEY_ENV, _TEST_FERNET_KEY)


def _mock_exchange(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: SlackOAuthResult | None = None,
    error: Exception | None = None,
) -> None:
    async def _fake(**kwargs: Any) -> SlackOAuthResult:
        if error is not None:
            raise error
        assert result is not None
        return result

    monkeypatch.setattr("outrider.api.slack.oauth.exchange_code", _fake)


def _mock_persist(monkeypatch: pytest.MonkeyPatch, *, returns: bool = True) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    async def _fake(_session: object, **kwargs: Any) -> bool:
        calls.update(kwargs)
        return returns

    monkeypatch.setattr("outrider.api.slack.oauth.set_slack_config", _fake)
    return calls


# ── GET /slack/install ──────────────────────────────────────────────────────


def test_install_requires_admin_auth() -> None:
    client = TestClient(_build_app())
    resp = client.get("/slack/install", params={"installation_id": 42, "channel_id": "C0ABCDE"})
    assert resp.status_code == 401


@pytest.mark.usefixtures("secrets_env")
def test_install_disabled_when_unconfigured() -> None:
    client = TestClient(_build_app(configured=False))
    resp = client.get(
        "/slack/install",
        params={"installation_id": 42, "channel_id": "C0ABCDE"},
        headers=_AUTH,
    )
    assert resp.status_code == 503


@pytest.mark.usefixtures("secrets_env")
def test_install_rejects_bad_channel() -> None:
    client = TestClient(_build_app())
    resp = client.get(
        "/slack/install",
        params={"installation_id": 42, "channel_id": "nope"},
        headers=_AUTH,
    )
    assert resp.status_code == 400


def test_install_500_when_state_secret_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Configured + valid channel, but no OUTRIDER_SLACK_STATE_SECRET → sign_state
    # fails closed; surfaced as a 500 (deploy misconfig), distinct from the 400 above.
    monkeypatch.delenv(STATE_SECRET_ENV, raising=False)
    client = TestClient(_build_app())
    resp = client.get(
        "/slack/install",
        params={"installation_id": 42, "channel_id": "C0ABCDE"},
        headers=_AUTH,
    )
    assert resp.status_code == 500


@pytest.mark.usefixtures("secrets_env")
def test_install_redirects_to_slack_with_signed_state() -> None:
    client = TestClient(_build_app())
    resp = client.get(
        "/slack/install",
        params={"installation_id": 42, "channel_id": "C0ABCDE"},
        headers=_AUTH,
        follow_redirects=False,
    )
    assert resp.status_code == 302
    # State-bearing + content-negotiated → uncacheable, keyed on Accept (RFC 9111).
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["vary"] == "Accept"
    location = resp.headers["location"]
    assert location.startswith("https://slack.com/oauth/v2/authorize?")
    assert "client_id=123.456" in location
    assert "scope=chat%3Awrite" in location
    # The state round-trips through the real verifier to the signed values.
    state = parse_qs(urlparse(location).query)["state"][0]
    verified = verify_state(state)
    assert verified.installation_id == 42
    assert verified.channel_id == "C0ABCDE"


@pytest.mark.usefixtures("secrets_env")
def test_install_returns_json_authorize_url_when_accept_json() -> None:
    # The dashboard "Connect Slack" flow sends Accept: application/json and must READ the
    # URL (a fetch cannot follow a cross-origin 302), so the same authorize URL comes back
    # in a JSON body — carrying the same signed state — instead of a redirect.
    client = TestClient(_build_app())
    resp = client.get(
        "/slack/install",
        params={"installation_id": 42, "channel_id": "C0ABCDE"},
        headers={**_AUTH, "Accept": "application/json"},
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["vary"] == "Accept"
    body = resp.json()
    assert set(body) == {"authorize_url"}
    url = body["authorize_url"]
    assert url.startswith("https://slack.com/oauth/v2/authorize?")
    assert "scope=chat%3Awrite" in url
    state = parse_qs(urlparse(url).query)["state"][0]
    verified = verify_state(state)
    assert verified.installation_id == 42
    assert verified.channel_id == "C0ABCDE"


@pytest.mark.usefixtures("secrets_env")
def test_install_json_branch_still_admin_gated_and_validates_channel() -> None:
    # The content-negotiated branch shares the redirect's guards: no auth → 401, bad
    # channel → 400 (never a 200 URL leak for an unauthenticated or malformed request).
    client = TestClient(_build_app())
    no_auth = client.get(
        "/slack/install",
        params={"installation_id": 42, "channel_id": "C0ABCDE"},
        headers={"Accept": "application/json"},
    )
    assert no_auth.status_code == 401
    bad_channel = client.get(
        "/slack/install",
        params={"installation_id": 42, "channel_id": "nope"},
        headers={**_AUTH, "Accept": "application/json"},
    )
    assert bad_channel.status_code == 400


# ── GET /slack/oauth/callback ───────────────────────────────────────────────


@pytest.mark.usefixtures("secrets_env")
def test_callback_disabled_when_unconfigured() -> None:
    client = TestClient(_build_app(configured=False))
    resp = client.get("/slack/oauth/callback", params={"code": "c", "state": "x"})
    assert resp.status_code == 503


@pytest.mark.usefixtures("secrets_env")
def test_callback_rejects_error_param() -> None:
    client = TestClient(_build_app())
    resp = client.get("/slack/oauth/callback", params={"error": "access_denied", "state": "x"})
    assert resp.status_code == 400


@pytest.mark.usefixtures("secrets_env")
def test_callback_rejects_missing_code() -> None:
    client = TestClient(_build_app())
    resp = client.get("/slack/oauth/callback", params={"state": "x"})  # no code
    assert resp.status_code == 400


@pytest.mark.usefixtures("secrets_env")
def test_callback_rejects_forged_state() -> None:
    client = TestClient(_build_app())
    resp = client.get("/slack/oauth/callback", params={"code": "c", "state": "forged.sig"})
    assert resp.status_code == 400


@pytest.mark.usefixtures("secrets_env")
def test_callback_exchange_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_exchange(monkeypatch, error=SlackOAuthError("invalid_code"))
    client = TestClient(_build_app())
    state = sign_state(installation_id=1, admin_id="admin", channel_id="C0ZZZZZ")
    resp = client.get("/slack/oauth/callback", params={"code": "bad", "state": state})
    assert resp.status_code == 400


@pytest.mark.usefixtures("secrets_env")
def test_callback_404_when_install_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_exchange(
        monkeypatch,
        result=SlackOAuthResult(
            team_id="T", team_name="T", bot_token=SecretStr("xoxb"), bot_user_id="U"
        ),
    )
    _mock_persist(monkeypatch, returns=False)
    client = TestClient(_build_app())
    state = sign_state(installation_id=7, admin_id="admin", channel_id="C0ZZZZZ")
    resp = client.get("/slack/oauth/callback", params={"code": "c", "state": state})
    assert resp.status_code == 404


@pytest.mark.usefixtures("secrets_env")
def test_callback_success_persists_identity_from_verified_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_exchange(
        monkeypatch,
        result=SlackOAuthResult(
            team_id="T0REAL",
            team_name="Acme",
            bot_token=SecretStr("xoxb-real-bot-token"),
            bot_user_id="U0BOT",
        ),
    )
    calls = _mock_persist(monkeypatch, returns=True)
    client = TestClient(_build_app())
    state = sign_state(installation_id=99, admin_id="admin", channel_id="C0ZZZZZ")
    # An attacker-supplied installation_id query param MUST be ignored — identity
    # comes from the verified state only.
    resp = client.get(
        "/slack/oauth/callback",
        params={"code": "good-code", "state": state, "installation_id": 999},
    )
    assert resp.status_code == 200
    assert "Acme" in resp.text
    assert calls["installation_id"] == 99  # from state, NOT the 999 query param
    assert calls["channel_id"] == "C0ZZZZZ"  # from state
    assert calls["team_id"] == "T0REAL"  # from the exchange response
    assert calls["configured_by"] == "admin"
    # The bot token was ENCRYPTED before persist — ciphertext, not plaintext.
    ciphertext = calls["bot_token_ciphertext"]
    assert isinstance(ciphertext, bytes)
    assert b"xoxb-real-bot-token" not in ciphertext
    assert decrypt_token(ciphertext).get_secret_value() == "xoxb-real-bot-token"
