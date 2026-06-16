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

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known placeholder values shipped in `.env.example` (and the usual suspects). A verbatim
# `.env.example` copy would otherwise authenticate against a PUBLIC string — reject these
# obvious non-secrets at startup so the operator must set a real value. Exact-match
# (case-insensitive, stripped) so a real secret is never rejected by accident. Mirrored in
# `github/config.py`; keep the two sets in sync.
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

    # Optional read-only token for the agent-view endpoint (feature 3 / S2,
    # `GET /reviews/{id}/agent-view`). SEPARATE scope from the admin key: agents
    # must NEVER hold the admin key (it can `POST /decide`). `None` = the agent
    # surface is disabled — `require_agent_api_key` returns a uniform 401 — so a
    # deployment that doesn't expose `/agent-view` simply omits the env var.
    agent_api_key: SecretStr | None = None

    # Public base URL of the Outrider dashboard (e.g. "https://outrider.example.com"),
    # read from `OUTRIDER_DASHBOARD_BASE_URL`. Optional: when set, the publish node
    # embeds per-finding + aggregate deep-links in the review body (DECISIONS.md#050);
    # when None (unset) the body renders the no-link fallback. Plain `str` (not
    # SecretStr — it is a public URL, not a credential); the publish-side
    # `_is_markdown_link_safe_url` validates it at render time and degrades to no-link
    # on a malformed value, so a misconfigured URL never produces a broken link.
    dashboard_base_url: str | None = None

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
        if stripped.lower() in _PLACEHOLDER_SECRETS:
            msg = (
                f"OUTRIDER_ADMIN_API_KEY is set to a known placeholder ({stripped!r}). A "
                "verbatim .env.example copy would admit any client sending that public value "
                'as a Bearer token. Set a real secret: python -c "import secrets; '
                'print(secrets.token_urlsafe(32))".'
            )
            raise ValueError(msg)
        # Validation only — return v unchanged. Stripping the secret
        # would mutate credential material; the auth comparison at
        # `api/dashboard/auth.py` uses `compare_digest` on the raw
        # bytes, so any rewrite here would silently desync from the
        # operator's configured value.
        return v

    @field_validator("agent_api_key", mode="after")
    @classmethod
    def _reject_empty_or_placeholder_agent_key(cls, v: SecretStr | None) -> SecretStr | None:
        """Same guard as the admin key, but ONLY when set. `None` is valid (the
        agent surface is simply disabled); a set-but-empty/whitespace/placeholder
        key is a misconfiguration (an empty key would admit empty `Bearer`
        headers at the `hmac.compare_digest` site) and fails loud at startup."""
        if v is None:
            return None
        stripped = v.get_secret_value().strip()
        if not stripped:
            raise ValueError(
                "OUTRIDER_AGENT_API_KEY is set but empty or whitespace-only. An empty agent "
                "key would admit empty `Bearer` headers at the auth site. Unset it to disable "
                "the agent surface, or set a non-empty secret."
            )
        if stripped.lower() in _PLACEHOLDER_SECRETS:
            raise ValueError(
                f"OUTRIDER_AGENT_API_KEY is a known placeholder ({stripped!r}); set a real "
                'secret or unset it: python -c "import secrets; print(secrets.token_urlsafe(32))".'
            )
        return v

    @model_validator(mode="after")
    def _reject_admin_agent_key_collision(self) -> "DashboardSettings":
        """When set, the agent key MUST differ from the admin key. Sharing one secret
        collapses the read-only/admin scope separation `require_agent_api_key` promises:
        the single secret satisfies BOTH gates, so an agent token could `POST /decide`.
        Fail loud at startup rather than silently authenticating both surfaces with one
        secret. Plain equality is fine — both operands are trusted operator config,
        compared once at startup (no per-request timing surface to defend)."""
        if (
            self.agent_api_key is not None
            and self.agent_api_key.get_secret_value() == self.admin_api_key.get_secret_value()
        ):
            msg = (
                "OUTRIDER_AGENT_API_KEY must differ from OUTRIDER_ADMIN_API_KEY. Sharing one "
                "secret defeats the read-only/admin scope separation: the agent token would also "
                "authorize POST /reviews/{id}/decide. Set a distinct secret for the agent surface, "
                "or unset it to disable /agent-view."
            )
            raise ValueError(msg)
        return self
