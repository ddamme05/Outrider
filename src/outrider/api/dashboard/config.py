"""DashboardSettings — env-backed bearer-token credential for the
HITL endpoints.

Separate Pydantic-settings class from `GitHubAppSettings` because the
dashboard's V1.5 surface will grow (per-reviewer auth, OAuth, etc.)
and conflating those settings into the GitHub-App namespace would
collide with the `OUTRIDER_GITHUB_*` env prefix. `frozen=True` +
`extra="forbid"` mirror the existing config-class precedent.

`SecretStr` wraps the API key so accidental string logging shows
`SecretStr('**********')` rather than the cleartext value. Callers
unwrap via `.get_secret_value()` at the HMAC-compare site (one place:
`api/dashboard/auth.py`).

V1: a single global admin key used by an internal dashboard. V1.5+:
expand to a per-reviewer-id table; the Protocol seam is the existing
`require_admin_api_key` FastAPI dependency.
"""

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    """Dashboard bearer-token credential.

    Reads `OUTRIDER_ADMIN_API_KEY` from the environment at app
    startup. Construction is fail-loud (missing env var raises
    ValidationError at FastAPI lifespan startup, NOT at the first
    request) so misconfiguration surfaces immediately.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_",
        extra="forbid",
        frozen=True,
    )

    admin_api_key: SecretStr

    @field_validator("admin_api_key", mode="after")
    @classmethod
    def _reject_empty_or_whitespace(cls, v: SecretStr) -> SecretStr:
        """Empty / whitespace-only `OUTRIDER_ADMIN_API_KEY` would silently
        admit empty bearer tokens at the `hmac.compare_digest` site
        because both `compare_digest(b"", b"")` is True. Reject at
        startup so misconfiguration surfaces at lifespan-init, not at
        the first authenticated request when a `Bearer ` header (no
        value) would compare equal to the empty configured secret.
        """
        raw = v.get_secret_value()
        stripped = raw.strip()
        if not stripped:
            msg = (
                "OUTRIDER_ADMIN_API_KEY is empty or whitespace-only. "
                "An empty admin key would admit empty `Bearer` headers "
                "at the auth site (hmac.compare_digest of two empty "
                "byte-strings is True). Set a non-empty secret."
            )
            raise ValueError(msg)
        return SecretStr(stripped) if stripped != raw else v
