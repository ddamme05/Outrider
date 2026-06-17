"""Slack config: `SlackSettings` (dev bootstrap) + `SlackOAuthSettings` (per-install OAuth).

`SlackSettings` — DEV/LOCAL env bootstrap for the Slack notifier.

**Not the production V1 config authority.** The spec pins V1 Slack config to
**per-installation** storage (the `installations` table, populated by the OAuth
`oauth.v2.access` exchange) — see specs/2026-06-15-slack-dashboard-in-slack.md.
This single-workspace env config exists only to wire + test the notifier locally
before that OAuth/per-install path lands; the composition root must prefer
per-install config and never let this env path become the production posting
authority.

Mirrors `GitHubAppSettings`: `bot_token` routed through `pydantic.SecretStr`,
`.get_secret_value()` called only at the wrapper call site (`notify/slack.py`),
never at log/audit construction.

Env vars (prefix `OUTRIDER_SLACK_`):
  - `OUTRIDER_SLACK_BOT_TOKEN` (SecretStr) — the bot token (`xoxb-…`, `chat:write`).
  - `OUTRIDER_SLACK_CHANNEL_ID` (str) — the channel the bot posts to (e.g. `C0…`);
    the bot must be a member of it (the install precondition).
"""

from urllib.parse import urlparse

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["SlackOAuthSettings", "SlackSettings"]

# Known placeholder values shipped in `.env.example` (+ the usual suspects); reject a
# verbatim copy at startup. Mirrors `github/config.py` / `api/dashboard/config.py`; keep in sync.
_PLACEHOLDER_SECRETS: frozenset[str] = frozenset(
    {
        "replace-me",
        "replace-me-with-a-long-random-secret",
        "change-me",
        "changeme",
        "secret",
        "password",
        "your-secret-here",
        "xoxb-your-token",
    }
)


class SlackSettings(BaseSettings):
    """Slack bot identity, env-backed. `frozen` + `extra="forbid"` per the
    `GitHubAppSettings` precedent (construction-time-only config; typo'd env var
    fails loudly at app start)."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_SLACK_",
        extra="forbid",
        frozen=True,
    )

    bot_token: SecretStr
    channel_id: str

    @field_validator("bot_token", mode="after")
    @classmethod
    def _reject_empty_or_placeholder_token(cls, v: SecretStr) -> SecretStr:
        """An empty / placeholder bot token would fail every post with an opaque
        `invalid_auth` at call time; fail-loud at startup instead. Validation
        only — never rewrite the credential (legitimate tokens are returned as-is).
        """
        stripped = v.get_secret_value().strip()
        if not stripped:
            raise ValueError("OUTRIDER_SLACK_BOT_TOKEN is empty or whitespace-only.")
        if stripped.lower() in _PLACEHOLDER_SECRETS:
            raise ValueError(
                f"OUTRIDER_SLACK_BOT_TOKEN is a known placeholder ({stripped!r}); "
                "set the real bot token (xoxb-…)."
            )
        return v

    @field_validator("channel_id", mode="after")
    @classmethod
    def _reject_empty_channel(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("OUTRIDER_SLACK_CHANNEL_ID is empty or whitespace-only.")
        return v


class SlackOAuthSettings(BaseSettings):
    """Slack App OAuth credentials for the per-install connect flow (commit 6.3c).

    DISTINCT from `SlackSettings` (the dev single-workspace bootstrap): these are the
    App-level client credentials the server uses to exchange the OAuth `code` for a
    per-install bot token (`oauth.v2.access`), then encrypt + persist it
    (DECISIONS.md#051). `frozen` + `extra="forbid"` per the `GitHubAppSettings`
    precedent (construction-time-only config; a missing required var fails loud at
    app start). All three vars share the `OUTRIDER_SLACK_` prefix; pydantic-settings
    ignores prefix-matching vars that map to no field, so this coexists with
    `SlackSettings` even under `extra="forbid"`.

    Env vars (prefix `OUTRIDER_SLACK_`):
      - `OUTRIDER_SLACK_CLIENT_ID` (str) — the Slack App client id (e.g. `123.456`).
      - `OUTRIDER_SLACK_CLIENT_SECRET` (SecretStr) — the App client secret; routed
        through `SecretStr`, `.get_secret_value()` only at the `oauth.v2.access`
        call site (`notify/slack_oauth.py`), never at log/audit construction.
      - `OUTRIDER_SLACK_REDIRECT_URI` (str) — the OAuth callback URL; must match the
        redirect URL registered in the Slack App config exactly.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_SLACK_",
        extra="forbid",
        frozen=True,
    )

    client_id: str
    client_secret: SecretStr
    redirect_uri: str

    @field_validator("client_id", mode="after")
    @classmethod
    def _reject_empty_client_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("OUTRIDER_SLACK_CLIENT_ID is empty or whitespace-only.")
        return v

    @field_validator("client_secret", mode="after")
    @classmethod
    def _reject_empty_or_placeholder_secret(cls, v: SecretStr) -> SecretStr:
        """A placeholder/empty client secret would fail `oauth.v2.access` with an
        opaque `invalid_client_id`/`bad_client_secret` at callback time; fail-loud at
        startup instead. Validation only — legitimate secrets are returned as-is."""
        stripped = v.get_secret_value().strip()
        if not stripped:
            raise ValueError("OUTRIDER_SLACK_CLIENT_SECRET is empty or whitespace-only.")
        if stripped.lower() in _PLACEHOLDER_SECRETS:
            raise ValueError(
                f"OUTRIDER_SLACK_CLIENT_SECRET is a known placeholder ({stripped!r}); "
                "set the real Slack App client secret."
            )
        return v

    @field_validator("redirect_uri", mode="after")
    @classmethod
    def _validate_redirect_uri(cls, v: str) -> str:
        """The redirect URI is both sent to Slack and registered in the App config;
        a malformed value silently breaks the install flow. Require a well-formed
        http(s) URL with a host (operator-trusted, so a light shape gate suffices)."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("OUTRIDER_SLACK_REDIRECT_URI is empty or whitespace-only.")
        parsed = urlparse(stripped)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"OUTRIDER_SLACK_REDIRECT_URI must be an http(s) URL with a host; got {v!r}."
            )
        return stripped
