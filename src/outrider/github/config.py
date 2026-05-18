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

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["GitHubAppSettings"]


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
