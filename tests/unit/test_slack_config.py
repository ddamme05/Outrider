"""SlackSettings (env config) + build_review_deeplink (pure helper).

Pins the SecretStr bot-token guards (empty / placeholder rejected), the
channel-id guard, frozen + extra=forbid, and the deep-link shape
(/reviews/{id}[?finding={fid}], trailing-slash tolerant). See
specs/2026-06-15-slack-dashboard-in-slack.md.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.notify.config import SlackSettings
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
