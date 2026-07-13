# See DECISIONS.md#070 — the single GitHub App credential source (mode + provider).
"""`GitHubCredentialProvider` — the single source of GitHub App credentials (`DECISIONS.md#070`).

Credential source is an explicit mode, `OUTRIDER_GITHUB_CREDENTIAL_SOURCE ∈ {env, database}`
(default `env`), with **no dynamic fallback**: only the selected source is validated, and a
missing/unreadable `database` record never falls back to `env` (that could silently switch App
identity + webhook trust).

Every consumer that today closes over `GitHubAppSettings` — `auth.make_installation_client_factory`,
`api/webhooks/signature.verify_signature`, `authz.make_installation_authorizer`, the reconciliation
janitor — instead obtains one immutable `GitHubAppCredentials` snapshot from the provider, per
operation. `env` mode wraps `GitHubAppSettings` (behavior unchanged); `database` mode reads the one
active `github_app_credentials` row and decrypts the PEM + webhook secret via `credential_crypto`,
raising `GitHubUnconfiguredError` (fail-closed) while unconfigured — so a booted-but-unonboarded
instance denies webhooks/dispatch/authorization until activation, with no restart on activation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from sqlalchemy import select

from outrider.db.models.github_app_credentials import GitHubAppCredential
from outrider.github.config import GitHubAppSettings
from outrider.github.credential_crypto import decrypt_credential, validate_credential_enc_key

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import SecretStr
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

__all__ = [
    "CREDENTIAL_SOURCE_ENV",
    "DatabaseCredentialProvider",
    "EnvCredentialProvider",
    "GitHubAppCredentials",
    "GitHubCredentialProvider",
    "GitHubUnconfiguredError",
    "build_credential_provider",
    "resolve_credential_source",
]

CREDENTIAL_SOURCE_ENV = "OUTRIDER_GITHUB_CREDENTIAL_SOURCE"


class GitHubUnconfiguredError(RuntimeError):
    """No active GitHub App credentials — `database` mode with no onboarded record (the setup-only
    bootstrap state). Consumers translate this into a fail-closed refusal (webhook 503, dispatch
    denied, authorization denied), never a fall-through to `env`."""


@dataclass(frozen=True)
class GitHubAppCredentials:
    """An immutable App-credential snapshot handed to a single consumer operation. Field names
    match `GitHubAppSettings` (`app_id`, `app_private_key`, `webhook_secret`) so the existing
    client factory / authorizer / signature verifier consume it unchanged; `slug`/`client_id` are
    non-secret metadata present only in `database` mode."""

    app_id: int
    app_private_key: SecretStr
    webhook_secret: SecretStr
    slug: str | None = None
    client_id: str | None = None


@runtime_checkable
class GitHubCredentialProvider(Protocol):
    """The single credential source. `current()` returns one immutable snapshot or raises
    `GitHubUnconfiguredError`; `is_configured()` is the cheap check for `/setup/status` + gating."""

    async def current(self) -> GitHubAppCredentials: ...

    async def is_configured(self) -> bool: ...


@dataclass(frozen=True)
class EnvCredentialProvider:
    """`env` mode — wraps the validated `GitHubAppSettings` triad; always configured."""

    settings: GitHubAppSettings

    async def current(self) -> GitHubAppCredentials:
        s = self.settings
        return GitHubAppCredentials(
            app_id=s.app_id,
            app_private_key=s.app_private_key,
            webhook_secret=s.webhook_secret,
        )

    async def is_configured(self) -> bool:
        return True


@dataclass(frozen=True)
class DatabaseCredentialProvider:
    """`database` mode — the one active `github_app_credentials` row, decrypted per operation.

    Read per-operation (not cached) so activation takes effect with no restart; the row is small
    and Fernet decryption is microsecond-cheap. Raises `GitHubUnconfiguredError` when no row is
    active (the setup-only state) — never falls back to `env`.
    """

    session_factory: async_sessionmaker[AsyncSession]

    async def _active(self, session: AsyncSession) -> GitHubAppCredential | None:
        # `.first()` after ordering (not `scalar_one_or_none`): the unique partial index
        # guarantees ≤1 active row, but if that invariant is ever violated we take the latest and
        # keep serving rather than making every credential read an unhandled MultipleResultsFound.
        result = await session.execute(
            select(GitHubAppCredential)
            .where(GitHubAppCredential.is_active)
            .order_by(GitHubAppCredential.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def current(self) -> GitHubAppCredentials:
        # Build the snapshot INSIDE the session context — decrypt while the row is still attached,
        # so no detached-instance access can arise from a future expire-on-commit / deferred column.
        async with self.session_factory() as session:
            row = await self._active(session)
            if row is None:
                raise GitHubUnconfiguredError(
                    "no active GitHub App credentials — the instance has not been onboarded "
                    "(POST /setup). database credential mode never falls back to env."
                )
            return GitHubAppCredentials(
                app_id=row.app_id,
                app_private_key=decrypt_credential(row.pem_ciphertext),
                webhook_secret=decrypt_credential(row.webhook_secret_ciphertext),
                slug=row.slug,
                client_id=row.client_id,
            )

    async def is_configured(self) -> bool:
        async with self.session_factory() as session:
            return await self._active(session) is not None


def resolve_credential_source(env: Mapping[str, str] | None = None) -> Literal["env", "database"]:
    """Resolve `OUTRIDER_GITHUB_CREDENTIAL_SOURCE` (default `env`); raise on any other value —
    an invalid mode is a configuration error, never a silent default (`DECISIONS.md#070`)."""
    env = os.environ if env is None else env
    raw = env.get(CREDENTIAL_SOURCE_ENV, "env").strip().lower()
    if raw not in ("env", "database"):
        raise ValueError(
            f"{CREDENTIAL_SOURCE_ENV}={raw!r} is invalid: expected 'env' or 'database'."
        )
    return raw  # type: ignore[return-value]


def build_credential_provider(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    env: Mapping[str, str] | None = None,
) -> GitHubCredentialProvider:
    """Construct the provider for the configured mode, validating ONLY the selected source.

    `env` → construct + validate `GitHubAppSettings` (the triad must be present). `database` →
    validate the credential encryption key is present + well-formed (fail-loud at boot), and return
    the DB provider; the App triad is NOT required (credentials come from the onboarded record).
    """
    source = resolve_credential_source(env)
    if source == "env":
        return EnvCredentialProvider(GitHubAppSettings())
    validate_credential_enc_key()
    return DatabaseCredentialProvider(session_factory)
