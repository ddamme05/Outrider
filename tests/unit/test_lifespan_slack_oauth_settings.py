"""`_load_slack_oauth_settings` — Slack OAuth config is opt-in and NON-FATAL.

Slack is an optional integration, so a present-but-invalid OAuth config must DISABLE
Slack (return None + log a loud ERROR), never crash app startup. Regression guard for
the boot-coupling bug where a partial OUTRIDER_SLACK_* config raised at lifespan startup
and took the whole app (PR review included) down with it.
"""

from __future__ import annotations

import logging

import pytest

from outrider.api.lifespan import _SLACK_OAUTH_VARS, _load_slack_oauth_settings
from outrider.notify.config import SlackOAuthSettings


@pytest.fixture(autouse=True)
def _clear_oauth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unit tests run in a clean shell, but be explicit so a stray real env var can't
    # flip a case.
    for var in _SLACK_OAUTH_VARS:
        monkeypatch.delenv(var, raising=False)


def test_none_when_no_oauth_env() -> None:
    """No OAuth vars set → Slack simply not configured; None, no log."""
    assert _load_slack_oauth_settings() is None


def test_valid_config_constructs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_SLACK_CLIENT_ID", "123.456")
    monkeypatch.setenv("OUTRIDER_SLACK_CLIENT_SECRET", "real-app-secret")
    monkeypatch.setenv("OUTRIDER_SLACK_REDIRECT_URI", "https://o.example/slack/oauth/callback")
    settings = _load_slack_oauth_settings()
    assert isinstance(settings, SlackOAuthSettings)
    assert settings.client_id == "123.456"


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("OUTRIDER_SLACK_CLIENT_ID", ""),  # present-but-empty (the reported boot crash)
        ("OUTRIDER_SLACK_CLIENT_SECRET", ""),  # present-but-empty
        ("OUTRIDER_SLACK_REDIRECT_URI", "not-a-url"),  # present-but-malformed
    ],
)
def test_partial_or_invalid_config_disables_not_crashes(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    """One OAuth var present + invalid → returns None WITHOUT raising. The app boots;
    only Slack is disabled."""
    monkeypatch.setenv(var, value)
    assert _load_slack_oauth_settings() is None


def test_invalid_config_logs_loud_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Disabled, but NOT silent — the operator gets a clear ERROR (the 'don't hide a
    typo' intent), just not a crash."""
    monkeypatch.setenv("OUTRIDER_SLACK_CLIENT_ID", "")
    with caplog.at_level(logging.ERROR, logger="outrider.api.lifespan"):
        assert _load_slack_oauth_settings() is None
    assert any("Slack OAuth config is present but invalid" in r.message for r in caplog.records)
