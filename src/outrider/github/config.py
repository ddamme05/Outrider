# GitHub App settings, env-backed.
"""Env-mode GitHub App identity and webhook secret.

Read OUTRIDER_GITHUB_APP_ID, OUTRIDER_GITHUB_APP_PRIVATE_KEY (raw PEM), and
OUTRIDER_GITHUB_WEBHOOK_SECRET. Frozen settings validate the selected env source
at boot. SecretStr masks accidental string representations; consumers unwrap
credentials at authentication and signature-verification boundaries.

Signature verification enforces webhook-signature-constant-time-compare; GitHub
payload handling enforces webhook-strings-are-data-not-format-strings.
"""

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["GitHubAppSettings"]

# Known placeholder values shipped in `.env.example` (and the usual suspects). A verbatim
# `.env.example` copy would otherwise authenticate UNSIGNED webhooks against a PUBLIC string —
# reject these obvious non-secrets at startup. Exact-match (case-insensitive, stripped) so a
# real secret is never rejected by accident. Mirrored in `api/dashboard/config.py`; keep in sync.
_PLACEHOLDER_SECRETS: frozenset[str] = frozenset(
    {
        "replace-me",
        "replace-me-with-a-long-random-secret",
        "change-me",
        "changeme",
        "secret",
        "password",
        "your-secret-here",
    }
)


class GitHubAppSettings(BaseSettings):
    """Validate immutable env-mode credentials; reject unknown explicit fields."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_GITHUB_",
        extra="forbid",
        frozen=True,
    )

    app_id: int
    app_private_key: SecretStr
    webhook_secret: SecretStr

    @field_validator("app_private_key", "webhook_secret", mode="after")
    @classmethod
    def _reject_empty_or_whitespace(cls, v: SecretStr) -> SecretStr:
        """Reject blank or placeholder credentials before consumers use them.

        Check a trimmed copy, but return the original SecretStr so validation does not
        rewrite private-key or signing-secret bytes.
        """
        raw = v.get_secret_value()
        stripped = raw.strip()
        if not stripped:
            msg = (
                "OUTRIDER_GITHUB_* secret is empty or whitespace-only. "
                "Set a non-empty value: empty webhook_secret admits "
                "unsigned webhooks (hmac.compare_digest of two empty "
                "byte-strings is True); empty app_private_key fails "
                "at JWT-mint time with an opaque cryptography error."
            )
            raise ValueError(msg)
        if stripped.lower() in _PLACEHOLDER_SECRETS:
            msg = (
                f"An OUTRIDER_GITHUB_* secret is set to a known placeholder ({stripped!r}). A "
                "verbatim .env.example copy would authenticate unsigned webhooks against a "
                'public value. Set a real secret: python -c "import secrets; '
                'print(secrets.token_urlsafe(32))".'
            )
            raise ValueError(msg)
        # Validation only — return v unchanged. Stripping the secret
        # would mutate credential material; legitimate secrets that
        # happen to include leading/trailing whitespace (rare but
        # possible for some key formats) would silently fail at
        # consumer sites if rewritten here.
        return v
