# GitHub App settings, env-backed.
"""`GitHubAppSettings` — env-backed GitHub App identity + webhook secret.

Backs the input-boundary invariants `webhook-signature-constant-time-compare`
and `webhook-strings-are-data-not-format-strings` by routing secret material
through `pydantic.SecretStr` rather than plain `str`. `.get_secret_value()`
is called only at the wrapper call site (`auth.make_installation_client_factory`'s
inner closure for the private key; `api/webhooks/signature.verify_signature`
for the webhook secret) — never at log or audit construction.

Co-located in `github/` rather than `api/webhooks/config.py` because:
  1. `auth.make_installation_client_factory(settings)` returns a callable
     that reads `settings.app_id` + `settings.app_private_key.get_secret_value()`
     per call; if those settings lived under `api/webhooks/`, the github/
     wrapper would import from the api/ layer — an inverted dependency.
  2. `webhook_secret` is logically App-scoped (one secret per App); it
     belongs with the App identity it pairs with.

The spec text suggests `api/webhooks/config.py` but explicitly allows
"extension of existing settings module"; this is the closest
existing-module-shaped placement that avoids the dependency inversion.

Env vars (prefix `OUTRIDER_GITHUB_`):
  - `OUTRIDER_GITHUB_APP_ID` (int) — the App's numeric id from GitHub.
  - `OUTRIDER_GITHUB_APP_PRIVATE_KEY` (SecretStr) — raw PEM contents.
    Repo precedent at `llm/anthropic_provider.py:180` + `api/lifespan.py:160`:
    secrets land in env as raw values (no file-secret pattern yet).
  - `OUTRIDER_GITHUB_WEBHOOK_SECRET` (SecretStr) — per-App signing secret
    for HMAC verification.
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
    """GitHub App identity + webhook secret, env-backed.

    `frozen=True` matches the `ModelConfig` and `RetentionSettings`
    precedents at `llm/config.py:48` and `audit/config.py:35`:
    construction-time-only configuration; per-tier runtime overrides go
    via re-construction with explicit kwargs in tests, NOT mutation.

    `extra="forbid"` so a typo'd env var fails loudly at app start rather
    than silently selecting a default.
    """

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
        """Empty / whitespace-only `app_private_key` or `webhook_secret`
        would silently admit broken state at the consumer sites:

          - Empty `webhook_secret` → `hmac.compare_digest(b"", b"")` is
            True, so unsigned webhooks (empty signature header) would
            authenticate as valid. Critical input-boundary defense.
          - Empty `app_private_key` → JWT signing in `github/auth.py`
            would fail at install-token mint time with an opaque
            cryptography error; fail-loud at startup is clearer.

        Mirrors the `DashboardSettings.admin_api_key` validator.
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
