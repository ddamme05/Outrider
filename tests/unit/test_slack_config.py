"""SlackSettings + SlackOAuthSettings (env config) + build_review_deeplink (pure helper).

Pins the SecretStr bot-token guards (empty / placeholder rejected), the
channel-id guard, frozen + extra=forbid, the OAuth client-credential +
redirect-uri guards, and the deep-link shape (/reviews/{id}[?finding={fid}],
trailing-slash tolerant). See specs/2026-06-15-slack-dashboard-in-slack.md.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.notify.config import SlackOAuthSettings, SlackSettings
from outrider.notify.deeplink import build_review_deeplink


def _settings(**overrides: object) -> SlackSettings:
    kwargs: dict[str, object] = {"bot_token": "xoxb-real-token-value", "channel_id": "C0123ABC"}
    kwargs.update(overrides)
    return SlackSettings(**kwargs)  # type: ignore[arg-type]


def test_valid_settings() -> None:
    s = _settings()
    assert s.bot_token.get_secret_value() == "xoxb-real-token-value"
    assert s.channel_id == "C0123ABC"


def test_settings_frozen() -> None:
    s = _settings()
    with pytest.raises(ValidationError):
        s.channel_id = "C999"  # type: ignore[misc]


def test_settings_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        _settings(workspace="acme")  # typo'd env var must fail loudly


@pytest.mark.parametrize("token", ["", "   ", "changeme", "xoxb-your-token"])
def test_empty_or_placeholder_token_rejected(token: str) -> None:
    with pytest.raises(ValidationError):
        _settings(bot_token=token)


def test_empty_channel_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(channel_id="   ")


def _oauth(**overrides: object) -> SlackOAuthSettings:
    kwargs: dict[str, object] = {
        "client_id": "123.456",
        "client_secret": "real-app-client-secret",
        "redirect_uri": "https://outrider.example.com/slack/oauth/callback",
    }
    kwargs.update(overrides)
    return SlackOAuthSettings(**kwargs)  # type: ignore[arg-type]


def test_oauth_valid_settings() -> None:
    s = _oauth()
    assert s.client_id == "123.456"
    assert s.client_secret.get_secret_value() == "real-app-client-secret"
    assert s.redirect_uri == "https://outrider.example.com/slack/oauth/callback"


def test_oauth_frozen() -> None:
    s = _oauth()
    with pytest.raises(ValidationError):
        s.client_id = "999"  # type: ignore[misc]


def test_oauth_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        _oauth(scope="chat:write")  # typo'd / unsupported var must fail loudly


@pytest.mark.parametrize("secret", ["", "   ", "changeme", "your-secret-here"])
def test_oauth_empty_or_placeholder_secret_rejected(secret: str) -> None:
    with pytest.raises(ValidationError):
        _oauth(client_secret=secret)


def test_oauth_empty_client_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _oauth(client_id="   ")


@pytest.mark.parametrize(
    "redirect", ["", "   ", "not-a-url", "ftp://x/cb", "https://", "/relative/cb"]
)
def test_oauth_malformed_redirect_rejected(redirect: str) -> None:
    with pytest.raises(ValidationError):
        _oauth(redirect_uri=redirect)


def test_oauth_redirect_stripped() -> None:
    s = _oauth(redirect_uri="  https://o.example/cb  ")
    assert s.redirect_uri == "https://o.example/cb"


def test_deeplink_review_only() -> None:
    rid = uuid4()
    assert build_review_deeplink("https://outrider.example.com", rid) == (
        f"https://outrider.example.com/reviews/{rid}"
    )


def test_deeplink_with_finding() -> None:
    rid, fid = uuid4(), uuid4()
    assert build_review_deeplink("https://o.example.com", rid, fid) == (
        f"https://o.example.com/reviews/{rid}?finding={fid}"
    )


def test_deeplink_tolerates_trailing_slash() -> None:
    rid = uuid4()
    assert build_review_deeplink("https://o.example.com/", rid) == (
        f"https://o.example.com/reviews/{rid}"
    )


def test_deeplink_rejects_malformed_base_url() -> None:
    # A malformed base URL (per the shared is_safe_link_url gate) -> None, so the
    # caller renders a no-link Slack message instead of a broken mrkdwn link.
    rid = uuid4()
    assert build_review_deeplink("not-a-url", rid) is None
    assert build_review_deeplink("https://", rid) is None  # host-less
    assert build_review_deeplink("https://dash.example/a|b", rid) is None  # pipe breaks <url|text>
