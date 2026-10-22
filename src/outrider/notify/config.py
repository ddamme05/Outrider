"""Slack credentials for local transport tests and installation OAuth.

SlackSettings reads a dev bot token and channel. Production posting uses encrypted
per-installation configuration loaded by resolver.py, populated by the OAuth flow
(specs/2026-06-15-slack-dashboard-in-slack.md).

SlackOAuthSettings reads OUTRIDER_SLACK_CLIENT_ID, CLIENT_SECRET, and REDIRECT_URI
under the OUTRIDER_SLACK_ prefix. SecretStr credentials are unwrapped at SDK call
sites, never in log or audit construction.
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
    """Immutable env credentials for local notifier use."""

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
    """Immutable App credentials for exchanging codes for per-install bot tokens.

    The callback encrypts tokens before persistence (DECISIONS.md#051). Required
    settings are validated when OAuth is enabled; prefix-matching variables with no
    field are ignored, so local SlackSettings variables can coexist.
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
