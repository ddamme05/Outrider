"""Slack OAuth code-exchange wrapper (commit 6.3b).

Pins the success shape (identity + bot token from the response), SlackApiError →
SlackOAuthError translation (error code surfaced, token never), and malformed-
response rejection. The bot token is returned as a SecretStr. `SlackApiError` here
is the documented surface-tier test exception (wrapper test).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr
from slack_sdk.errors import SlackApiError

from outrider.notify.slack_oauth import SlackOAuthError, exchange_code


class _FakeOAuthClient:
    """Fake AsyncWebClient: returns a canned oauth.v2.access response or raises."""

    def __init__(self, *, response: dict[str, Any] | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def oauth_v2_access(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


@pytest.mark.asyncio
async def test_exchange_code_success() -> None:
    resp = {
        "ok": True,
        "access_token": "xoxb-real-bot-token",
        "team": {"id": "T0X", "name": "Acme"},
        "bot_user_id": "U0BOT",
    }
    client = _FakeOAuthClient(response=resp)
    result = await exchange_code(
        client_id="cid",
        client_secret=SecretStr("csecret"),
        code="code123",
        redirect_uri="https://dash.example/slack/oauth/callback",
        client=client,  # type: ignore[arg-type]
    )
    assert result.team_id == "T0X"
    assert result.team_name == "Acme"
    assert result.bot_user_id == "U0BOT"
    assert isinstance(result.bot_token, SecretStr)
    assert result.bot_token.get_secret_value() == "xoxb-real-bot-token"
    # The code + client_secret are exchanged server-side (sent to the SDK).
    assert client.calls[0]["code"] == "code123"
    assert client.calls[0]["client_secret"] == "csecret"  # noqa: S105  (test fixture)


@pytest.mark.asyncio
async def test_exchange_code_slack_api_error_surfaces_code_not_token() -> None:
    err = SlackApiError("bad", {"ok": False, "error": "invalid_code"})  # type: ignore[arg-type]
    client = _FakeOAuthClient(error=err)
    with pytest.raises(SlackOAuthError, match="invalid_code") as exc_info:
        await exchange_code(
            client_id="cid",
            client_secret=SecretStr("csecret"),
            code="bad",
            redirect_uri="https://dash.example/slack/oauth/callback",
            client=client,  # type: ignore[arg-type]
        )
    assert "csecret" not in str(exc_info.value)  # the secret never leaks into the error


@pytest.mark.asyncio
async def test_exchange_code_malformed_response_rejected() -> None:
    # ok but missing access_token / bot_user_id → fail closed, persist nothing.
    client = _FakeOAuthClient(response={"ok": True, "team": {"id": "T0X"}})
    with pytest.raises(SlackOAuthError, match="missing expected fields"):
        await exchange_code(
            client_id="cid",
            client_secret=SecretStr("csecret"),
            code="x",
            redirect_uri="https://dash.example/slack/oauth/callback",
            client=client,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resp",
    [
        {"ok": True, "access_token": None, "team": {"id": "T0X"}, "bot_user_id": "U0BOT"},  # null
        {"ok": True, "access_token": "xoxb", "team": {"id": None}, "bot_user_id": "U0BOT"},  # null
        {"ok": True, "access_token": "xoxb", "team": {"id": "T0X"}, "bot_user_id": ""},  # empty
        {
            "ok": True,
            "access_token": {"x": 1},
            "team": {"id": "T0X"},
            "bot_user_id": "U0BOT",
        },  # non-str
        {
            "ok": True,
            "access_token": "xoxb",
            "team": {"id": ["T0X"]},
            "bot_user_id": "U0BOT",
        },  # non-str
    ],
)
async def test_exchange_code_rejects_invalid_required_fields(resp: dict[str, Any]) -> None:
    """A present-but-null/empty/non-string required field fails closed — never coerces
    a malformed value (`str(None)`=="None", `str({...})`=="{...}") into a trusted
    token/identity (no bogus team/token/user persisted)."""
    client = _FakeOAuthClient(response=resp)
    with pytest.raises(SlackOAuthError, match="non-empty string"):
        await exchange_code(
            client_id="cid",
            client_secret=SecretStr("csecret"),
            code="x",
            redirect_uri="https://dash.example/slack/oauth/callback",
            client=client,  # type: ignore[arg-type]
        )
