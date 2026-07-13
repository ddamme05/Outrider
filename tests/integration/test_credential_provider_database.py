"""Integration tests for `DatabaseCredentialProvider` against a real Postgres (#070).

Covers what the unit tests can't (they cover env mode + mode resolution): the DB read + decrypt
round-trip, zero-row fail-closed, wrong-key/tamper fail-closed, the multiple-active-row integrity
guard, and `is_configured()` reading the authoritative `setup_state.status` — the four gaps the
security review flagged on the foundation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

from outrider.db.models.github_app_credentials import GitHubAppCredential
from outrider.github.credential_crypto import (
    CREDENTIAL_ENC_KEY_ENV,
    CredentialCryptoError,
    encrypt_credential,
)
from outrider.github.credentials import (
    DatabaseCredentialProvider,
    GitHubCredentialIntegrityError,
    GitHubUnconfiguredError,
)

_PEM = "-----BEGIN RSA PRIVATE KEY-----\nMANIFEST\n-----END RSA PRIVATE KEY-----"
_WEBHOOK = "onboarded-webhook-secret-value"


@pytest.fixture
def _enc_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, key)
    return key


@pytest_asyncio.fixture
async def session_factory(migrated_db: str) -> AsyncGenerator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(migrated_db)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _insert_active(
    sf: async_sessionmaker[AsyncSession], *, app_id: int = 999, version: int = 1, pem: str = _PEM
) -> None:
    async with sf() as session, session.begin():
        session.add(
            GitHubAppCredential(
                version=version,
                app_id=app_id,
                slug="octo-outrider",
                client_id="Iv1.deadbeef",
                pem_ciphertext=encrypt_credential(SecretStr(pem)),
                webhook_secret_ciphertext=encrypt_credential(SecretStr(_WEBHOOK)),
                is_active=True,
            )
        )


async def _set_status(sf: async_sessionmaker[AsyncSession], status: str) -> None:
    async with sf() as session, session.begin():
        await session.execute(
            text("UPDATE setup_state SET status = :s WHERE id = 1"), {"s": status}
        )


async def test_zero_rows_unconfigured(
    session_factory: async_sessionmaker[AsyncSession], _enc_key: str
) -> None:
    provider = DatabaseCredentialProvider(session_factory)
    assert await provider.is_configured() is False  # seeded status is UNCONFIGURED
    with pytest.raises(GitHubUnconfiguredError):
        await provider.current()


async def test_round_trip_decrypts(
    session_factory: async_sessionmaker[AsyncSession], _enc_key: str
) -> None:
    await _insert_active(session_factory, app_id=4242)
    await _set_status(session_factory, "CONFIGURED")
    provider = DatabaseCredentialProvider(session_factory)
    assert await provider.is_configured() is True  # status-authoritative
    snap = await provider.current()
    assert snap.app_id == 4242
    assert snap.app_private_key.get_secret_value() == _PEM
    assert snap.webhook_secret.get_secret_value() == _WEBHOOK
    assert snap.slug == "octo-outrider"


async def test_is_configured_is_status_not_row(
    session_factory: async_sessionmaker[AsyncSession], _enc_key: str
) -> None:
    """An active row present but status NOT yet CONFIGURED → is_configured() is False AND current()
    FAILS CLOSED (the state machine is authoritative; a row inserted-but-not-activated is not usable
    — credential exposure is gated on CONFIGURED, not on row presence)."""
    await _insert_active(session_factory)
    # status left at the seeded UNCONFIGURED
    provider = DatabaseCredentialProvider(session_factory)
    assert await provider.is_configured() is False
    with pytest.raises(GitHubUnconfiguredError):
        await provider.current()


async def test_missing_singleton_fails_loud(
    session_factory: async_sessionmaker[AsyncSession], _enc_key: str
) -> None:
    """The migration always seeds setup_state (id=1); if it's absent that is integrity corruption,
    NOT 'unconfigured' — both methods raise (fail loud) rather than silently reopen onboarding over
    an intended configuration."""
    async with session_factory() as session, session.begin():
        await session.execute(text("DELETE FROM setup_state WHERE id = 1"))
    provider = DatabaseCredentialProvider(session_factory)
    with pytest.raises(GitHubCredentialIntegrityError):
        await provider.is_configured()
    with pytest.raises(GitHubCredentialIntegrityError):
        await provider.current()


async def test_wrong_key_fails_closed(
    session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, Fernet.generate_key().decode("ascii"))
    await _insert_active(session_factory)  # encrypted under key A
    await _set_status(session_factory, "CONFIGURED")
    monkeypatch.setenv(
        CREDENTIAL_ENC_KEY_ENV, Fernet.generate_key().decode("ascii")
    )  # rotate to key B
    provider = DatabaseCredentialProvider(session_factory)
    # configured-but-broken: is_configured stays True (status), operations fail closed at current().
    assert await provider.is_configured() is True
    with pytest.raises(CredentialCryptoError):
        await provider.current()


@pytest.mark.parametrize("status", ["AWAITING_CALLBACK", "CONVERTING", "ORPHANED"])
async def test_non_configured_status_with_active_row_fails_closed(
    session_factory: async_sessionmaker[AsyncSession], _enc_key: str, status: str
) -> None:
    """The gate is an ALLOWLIST (`status == CONFIGURED`), not a denylist (`!= UNCONFIGURED`): a
    present, decryptable active row never exposes credentials while status is ANY non-CONFIGURED
    value — the transient onboarding states (AWAITING_CALLBACK, CONVERTING) and the terminal error
    state (ORPHANED). Without this, a `!= "UNCONFIGURED"` inversion of the gate leaks the App key +
    webhook secret over a mid-onboarding or orphaned instance and the rest of the suite stays green
    (every other current() test uses CONFIGURED or UNCONFIGURED)."""
    await _insert_active(session_factory)
    await _set_status(session_factory, status)
    provider = DatabaseCredentialProvider(session_factory)
    assert await provider.is_configured() is False
    with pytest.raises(GitHubUnconfiguredError):
        await provider.current()


async def test_missing_singleton_with_active_row_fails_closed(
    session_factory: async_sessionmaker[AsyncSession], _enc_key: str
) -> None:
    """MISSING singleton WITH a decryptable active row present: the integrity check runs BEFORE the
    row read + decrypt, so current() raises the integrity error and NEVER exposes the credentials.
    The 0-row variant (test_missing_singleton_fails_loud) can't prove this ordering — it raises
    regardless of order; this cell locks singleton-check-before-decrypt against a future reorder."""
    await _insert_active(session_factory)
    async with session_factory() as session, session.begin():
        await session.execute(text("DELETE FROM setup_state WHERE id = 1"))
    provider = DatabaseCredentialProvider(session_factory)
    with pytest.raises(GitHubCredentialIntegrityError):
        await provider.current()
    with pytest.raises(GitHubCredentialIntegrityError):
        await provider.is_configured()


async def test_multiple_active_fails_closed(
    session_factory: async_sessionmaker[AsyncSession], _enc_key: str
) -> None:
    """Two active rows (invariant violated) → refuse to choose, raise the integrity error (fail
    closed at a root credential boundary), never silently pick one. The unique index normally makes
    this impossible, so we DROP it first to simulate the violated state (bad migration / tampering)
    and prove the code-level guard is the second line of defence."""
    ct_pem = encrypt_credential(SecretStr(_PEM))
    ct_wh = encrypt_credential(SecretStr(_WEBHOOK))
    async with session_factory() as session, session.begin():
        await session.execute(text("DROP INDEX uq_github_app_credentials_one_active"))
        for app_id in (111, 222):
            await session.execute(
                text(
                    "INSERT INTO github_app_credentials "
                    "(version, app_id, slug, pem_ciphertext, webhook_secret_ciphertext, is_active) "
                    "VALUES (:v, :a, :s, :p, :w, true)"
                ),
                {"v": app_id, "a": app_id, "s": f"app-{app_id}", "p": ct_pem, "w": ct_wh},
            )
    # CONFIGURED so current() passes the status gate and reaches the multiple-active integrity guard
    # (the realistic scenario: an onboarded instance whose credential rows were then tampered).
    await _set_status(session_factory, "CONFIGURED")
    provider = DatabaseCredentialProvider(session_factory)
    with pytest.raises(GitHubCredentialIntegrityError):
        await provider.current()
