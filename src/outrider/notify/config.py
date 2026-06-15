"""`SlackSettings` — DEV/LOCAL env bootstrap for the Slack notifier.

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

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["SlackSettings"]

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
