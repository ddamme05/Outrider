# See DECISIONS.md#070 — setup/onboarding config surface (database-mode).
"""Config for App-Manifest onboarding (`DECISIONS.md#070`).

Two new settings, both **required in `database` mode** (ignored in `env` mode):

- `OUTRIDER_PUBLIC_BASE_URL` — the canonical externally-reachable base (e.g. `https://ci.acme.com`).
  Every manifest URL (setup, redirect, webhook, callback) is built from THIS, never the request
  `Host` header (an attacker-controlled header must not steer where GitHub sends the code/creds).
- `OUTRIDER_SETUP_STATE_SECRET` — the HMAC secret for the signed `state`; validated in
  `state_token.validate_setup_state_secret` (present + non-placeholder + ≥32 chars + distinct from
  sibling secret roots).

`validate_setup_config()` is the eager boot check the composition root calls in `database` mode so a
missing/malformed value fails loud at startup, not at the first `POST /setup`.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from outrider.api.setup.state_token import validate_setup_state_secret

__all__ = ["SetupSettings", "validate_setup_config"]


class SetupSettings(BaseSettings):
    """Onboarding config, env-backed. `frozen` + `extra="forbid"` per the `GitHubAppSettings`
    precedent (construction-time-only config; a typo'd env var fails loud at app start). The narrow
    `OUTRIDER_PUBLIC_` prefix maps only `OUTRIDER_PUBLIC_BASE_URL → base_url` (no broad-prefix
    collision); the state secret is read directly by `state_token` (fresh per call, for test
    monkeypatch + restart-free rotation)."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_PUBLIC_",
        extra="forbid",
        frozen=True,
    )

    base_url: str

    @field_validator("base_url", mode="after")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        """A well-formed http(s) base URL with a host and no path/query/fragment — every manifest
        URL is derived from it. Returns the canonical form (trailing slash stripped). Production is
        HTTPS (DECISIONS.md#070 bootstrap security); http is permitted for local/tunnel testing, the
        same shape gate `notify` uses for its redirect URI."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("OUTRIDER_PUBLIC_BASE_URL is empty or whitespace-only.")
        parsed = urlparse(stripped)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"OUTRIDER_PUBLIC_BASE_URL must be an http(s) URL with a host; got {v!r}."
            )
        if parsed.path not in ("", "/") or parsed.query or parsed.params or parsed.fragment:
            raise ValueError(
                "OUTRIDER_PUBLIC_BASE_URL must be a bare origin (scheme://host[:port]) with no "
                f"path, query, or fragment; got {v!r}."
            )
        return stripped.rstrip("/")


def validate_setup_config() -> SetupSettings:
    """Eager `database`-mode boot validation: the state secret (via `state_token`) AND the public
    base URL. Raises (`SetupStateError` / pydantic `ValidationError`) on any failure; returns the
    validated `SetupSettings` when both are good. The composition root calls this only when the
    credential source is `database`."""
    validate_setup_state_secret()
    return SetupSettings()
