"""Unit tests for `github.credentials` — the credential-source mode + provider (#070).

Covers mode resolution (default env / database / invalid / case-insensitive), the env-mode
snapshot + always-configured, and the fail-loud selected-source validation in
`build_credential_provider`. The `database` provider's DB read + decrypt + fail-closed is an
integration test (real Postgres) — see tests/integration/test_credential_provider_database.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from cryptography.fernet import Fernet

from outrider.github.config import GitHubAppSettings
from outrider.github.credential_crypto import CREDENTIAL_ENC_KEY_ENV
from outrider.github.credentials import (
    CREDENTIAL_SOURCE_ENV,
    DatabaseCredentialProvider,
    EnvCredentialProvider,
    build_credential_provider,
    resolve_credential_source,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def _settings() -> GitHubAppSettings:
    return GitHubAppSettings(
        app_id=123456,
        app_private_key="-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----",
        webhook_secret="a-real-webhook-secret-value",  # noqa: S106  (test fixture value)
    )


# ── mode resolution ──────────────────────────────────────────────────────────


def test_resolve_source_defaults_to_env() -> None:
    assert resolve_credential_source(env={}) == "env"


def test_resolve_source_database() -> None:
    assert resolve_credential_source(env={CREDENTIAL_SOURCE_ENV: "database"}) == "database"


def test_resolve_source_case_insensitive() -> None:
    assert resolve_credential_source(env={CREDENTIAL_SOURCE_ENV: "  DataBase "}) == "database"


@pytest.mark.parametrize("bad", ["envdb", "postgres", "1", "true", "en v"])
def test_resolve_source_invalid_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid"):
        resolve_credential_source(env={CREDENTIAL_SOURCE_ENV: bad})


# ── env provider ─────────────────────────────────────────────────────────────


async def test_env_provider_snapshot_matches_settings() -> None:
    settings = _settings()
    provider = EnvCredentialProvider(settings)
    assert await provider.is_configured() is True
    snap = await provider.current()
    assert snap.app_id == 123456
    assert snap.app_private_key.get_secret_value() == settings.app_private_key.get_secret_value()
    assert snap.webhook_secret.get_secret_value() == settings.webhook_secret.get_secret_value()
    assert snap.slug is None and snap.client_id is None  # env mode has no metadata


# ── build_credential_provider — validates ONLY the selected source ───────────


def test_build_env_mode_constructs_env_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_GITHUB_APP_ID", "123456")
    monkeypatch.setenv(
        "OUTRIDER_GITHUB_APP_PRIVATE_KEY", "-----BEGIN KEY-----\nx\n-----END KEY-----"
    )
    monkeypatch.setenv("OUTRIDER_GITHUB_WEBHOOK_SECRET", "a-real-webhook-secret")
    monkeypatch.delenv(CREDENTIAL_SOURCE_ENV, raising=False)  # default env
    provider = build_credential_provider(
        session_factory=cast("async_sessionmaker[AsyncSession]", None)
    )
    assert isinstance(provider, EnvCredentialProvider)


def test_build_database_mode_validates_enc_key_not_triad(monkeypatch: pytest.MonkeyPatch) -> None:
    """database mode: the App triad is NOT required (creds come from the DB), but the enc key IS."""
    monkeypatch.setenv(CREDENTIAL_SOURCE_ENV, "database")
    monkeypatch.delenv("OUTRIDER_GITHUB_APP_ID", raising=False)  # triad absent — must NOT matter
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, Fernet.generate_key().decode("ascii"))
    provider = build_credential_provider(
        session_factory=cast("async_sessionmaker[AsyncSession]", None)
    )
    assert isinstance(provider, DatabaseCredentialProvider)


def test_build_database_mode_missing_enc_key_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CREDENTIAL_SOURCE_ENV, "database")
    monkeypatch.delenv(CREDENTIAL_ENC_KEY_ENV, raising=False)
    with pytest.raises(Exception, match="unset or empty"):
        build_credential_provider(session_factory=cast("async_sessionmaker[AsyncSession]", None))
