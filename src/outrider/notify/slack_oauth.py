"""Slack OAuth code-exchange (notify/ boundary).

Confines slack_sdk's `oauth_v2_access` to the Slack wrapper boundary
(vendor-sdks-only-in-wrappers, same as `notify/slack.py`). The verified callback
hands this module the OAuth `code`; it exchanges that server-side using the App's
client_id/client_secret for the workspace identity + bot token. The token and
identity come from the EXCHANGE RESPONSE, never from request params (the callback
already refuses to trust query params). The returned bot token is a `SecretStr` —
the caller `encrypt_token`s it (DECISIONS.md#051) before persisting and never
logs/audits it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import SecretStr
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

if TYPE_CHECKING:
    from slack_sdk.web.async_slack_response import AsyncSlackResponse

__all__ = ["SlackOAuthError", "SlackOAuthResult", "exchange_code"]


class SlackOAuthError(RuntimeError):
    """The Slack OAuth code exchange failed — a Slack API error (bad code, bad app
    credentials), a transient transport error, or a malformed response. Fail-closed:
    the callback rejects the install and persists no token."""


@dataclass(frozen=True)
class SlackOAuthResult:
    """Verified workspace identity + bot token from `oauth.v2.access`.

    `bot_token` is a `SecretStr`: encrypt it (`token_crypto.encrypt_token`) before
    persisting; never log or audit it.
    """

    team_id: str
    team_name: str
    bot_token: SecretStr
    bot_user_id: str


async def exchange_code(
    *,
    client_id: str,
    client_secret: SecretStr,
    code: str,
    redirect_uri: str,
    client: AsyncWebClient | None = None,
) -> SlackOAuthResult:
    """Exchange an OAuth `code` for the workspace identity + bot token via
    `oauth.v2.access` (server-side, App credentials). Raises `SlackOAuthError` on a
    Slack API error, transport failure, or malformed response (fail-closed). For
    tests, inject a `client` exposing `oauth_v2_access`."""
    web = client if client is not None else AsyncWebClient()
    try:
        resp: AsyncSlackResponse = await web.oauth_v2_access(
            client_id=client_id,
            client_secret=client_secret.get_secret_value(),
            code=code,
            redirect_uri=redirect_uri,
        )
    except SlackApiError as exc:
        # slack_sdk raises on ok=false; surface the error code, never the token.
        error_code = exc.response.get("error", "unknown") if exc.response is not None else "unknown"
        raise SlackOAuthError(f"Slack oauth.v2.access failed: {error_code}") from exc
    except (TimeoutError, OSError) as exc:  # aiohttp ClientError subclasses OSError
        raise SlackOAuthError(f"Slack oauth.v2.access request failed: {exc}") from exc
    try:
        team = resp["team"]
        return SlackOAuthResult(
            team_id=str(team["id"]),
            team_name=str(team.get("name") or ""),
            bot_token=SecretStr(str(resp["access_token"])),
            bot_user_id=str(resp["bot_user_id"]),
        )
    except (KeyError, TypeError) as exc:
        raise SlackOAuthError("Slack oauth.v2.access response missing expected fields") from exc
