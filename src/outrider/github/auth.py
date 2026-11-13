"""Create authenticated GitHub clients from the selected App credentials.

Each operation takes a fresh credentials snapshot so rotation cannot mix App
identities within a request. Installation tokens and App authentication use
their respective clients.

See DECISIONS.md#065, #066, #070 and docs/trust-boundaries.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import httpx
from githubkit import AppAuthStrategy, AppInstallationAuthStrategy, GitHub

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from outrider.github.credentials import GitHubAppCredentials, GitHubCredentialProvider

__all__ = [
    "AppGitHubClient",
    "InstallationGitHubClient",
    "make_app_client",
    "make_installation_client_factory",
]


# githubkit defaults to no timeout; an upstream stall could pin a review after
# the webhook ACK. Bound each HTTP phase here, as in llm/anthropic_provider.py.
# Operator configuration is tracked separately under FUP-034.
_GITHUB_CLIENT_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=5.0, read=30.0, write=30.0, pool=10.0
)


# Own the SDK strategy type here and re-export the client alias from
# github/__init__.py so downstream consumers need no githubkit import.
InstallationGitHubClient = GitHub[AppInstallationAuthStrategy]

# Type alias for a GitHub client authenticated AS THE APP (App-JWT), not as an
# installation. Used ONLY by the #065 live-authorization check (App-level endpoints:
# `GET /app/installations/{id}`, `POST .../access_tokens`). `auth.py` is the sole file
# importing `githubkit.AppAuthStrategy`, mirroring the `AppInstallationAuthStrategy`
# confinement above; `github/authz.py` consumes this alias without touching githubkit.
AppGitHubClient = GitHub[AppAuthStrategy]


def make_installation_client_factory(
    provider: GitHubCredentialProvider,
) -> Callable[[int], Awaitable[InstallationGitHubClient]]:
    """Bind a provider and return an async factory for fresh installation clients.

    Resolve credentials per call so database activation takes effect without restart.
    An unconfigured provider raises; it never falls back to another source
    (DECISIONS.md#070). Consumers must await the returned factory.
    """

    async def make_installation_client(
        installation_id: int,
    ) -> InstallationGitHubClient:
        """Return a fresh `GitHub` client authenticated as a specific installation.

        Args:
            installation_id: The numeric installation id from the webhook payload, validated
                upstream by the webhook handler against the `installations` +
                `installation_repositories` tables.
        """
        creds = await provider.current()
        return GitHub(
            AppInstallationAuthStrategy(
                creds.app_id,
                creds.app_private_key.get_secret_value(),
                installation_id,
            ),
            timeout=_GITHUB_CLIENT_TIMEOUT,
        )

    return make_installation_client


def make_app_client(credentials: GitHubAppCredentials) -> AppGitHubClient:
    """Construct an App-JWT client for live authorization (DECISIONS.md#065, #070).

    The authorizer resolves credentials per check and uses this fresh client under
    async with for its GET/POST pair. githubkit's HTTP client is task-local, so a
    lifespan-entered client would not provide the same reuse in a review task.
    The caller owns context entry and exit; timeouts match installation clients.
    """
    return GitHub(
        AppAuthStrategy(
            credentials.app_id,
            credentials.app_private_key.get_secret_value(),
        ),
        timeout=_GITHUB_CLIENT_TIMEOUT,
    )
