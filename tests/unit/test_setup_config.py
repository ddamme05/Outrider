"""Unit tests for `api/setup/config` — onboarding config (#070).

`SetupSettings.base_url` (env `OUTRIDER_PUBLIC_BASE_URL`) shape validation + `validate_setup_config`
requiring BOTH the public base URL and a valid state secret.
"""

from __future__ import annotations

import secrets

import pytest
from pydantic import ValidationError

from outrider.api.setup.config import SetupSettings, validate_setup_config
from outrider.api.setup.state_token import SETUP_STATE_SECRET_ENV, SetupStateError

_PUBLIC_URL_ENV = "OUTRIDER_PUBLIC_BASE_URL"


def test_valid_https_url_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PUBLIC_URL_ENV, "https://ci.acme.com")
    assert SetupSettings().base_url == "https://ci.acme.com"


def test_trailing_slash_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PUBLIC_URL_ENV, "https://ci.acme.com/")
    assert SetupSettings().base_url == "https://ci.acme.com"


@pytest.mark.parametrize(
    "url",
    ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost", "http://[::1]:8000"],
)
def test_http_permitted_only_for_loopback(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """http:// is the narrow dev exception — allowed ONLY for loopback (GitHub can't reach it,
    so it's simulated-callback local testing)."""
    monkeypatch.setenv(_PUBLIC_URL_ENV, url)
    assert SetupSettings().base_url == url.rstrip("/")


@pytest.mark.parametrize(
    "url", ["http://ci.acme.com", "http://192.168.1.10:8000", "http://example"]
)
def test_http_public_host_rejected(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """A plaintext http:// to a non-loopback host would leak the setup code/state (App private key +
    webhook secret) in transit — rejected (spec §Bootstrap security; #069 TLS)."""
    monkeypatch.setenv(_PUBLIC_URL_ENV, url)
    with pytest.raises(ValidationError, match="must be HTTPS"):
        SetupSettings()


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "ci.acme.com",  # no scheme
        "ftp://ci.acme.com",  # wrong scheme
        "https://",  # no host
        "https://ci.acme.com/setup",  # has a path
        "https://ci.acme.com?x=1",  # has a query
        "https://ci.acme.com#frag",  # has a fragment
    ],
)
def test_malformed_base_url_rejected(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv(_PUBLIC_URL_ENV, bad)
    with pytest.raises(ValidationError):
        SetupSettings()


def test_validate_setup_config_requires_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PUBLIC_URL_ENV, "https://ci.acme.com")
    monkeypatch.setenv(SETUP_STATE_SECRET_ENV, secrets.token_urlsafe(32))
    settings = validate_setup_config()
    assert settings.base_url == "https://ci.acme.com"


def test_validate_setup_config_fails_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PUBLIC_URL_ENV, "https://ci.acme.com")
    monkeypatch.delenv(SETUP_STATE_SECRET_ENV, raising=False)
    with pytest.raises(SetupStateError):
        validate_setup_config()


def test_validate_setup_config_fails_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SETUP_STATE_SECRET_ENV, secrets.token_urlsafe(32))
    monkeypatch.delenv(_PUBLIC_URL_ENV, raising=False)
    with pytest.raises(ValidationError):
        validate_setup_config()
