# Vendor wrapper for githubkit's GitHub App installation authentication.
# See DECISIONS.md#065-authorization-is-a-live-github-check-the-local-install-db-is-a-cache
"""Construct GitHub clients from the selected credential provider.

The lifespan binds a provider once; the installation factory resolves its current
credentials and creates a fresh client per operation. This keeps short-lived
installation tokens isolated and lets database-mode onboarding activate without
a restart (DECISIONS.md#070). githubkit handles token minting and refresh.

auth.py owns the SDK authentication strategies (vendor-sdks-only-in-wrappers).
App-JWT clients are separate from installation clients and are context-managed
per live authorization check (DECISIONS.md#065).
Each deployment has its own App identity (DECISIONS.md#066).
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


# Explicit per-operation timeouts on the githubkit GitHub client.
# Default is `timeout=None` (no timeout — requests hang indefinitely on
# upstream stalls). Intake runs in a background task post-webhook-ACK;
# a stalled API call would otherwise pin the whole review pipeline.
# Shape mirrors `llm/anthropic_provider.py`'s explicit-httpx-Timeout
# pattern so the two SDK-wrapping surfaces have consistent operational
# behavior. Values: connect timeout is short (TCP handshake or auth
# resolution), read is the dominant cost (paginated file fetches), write
# matches read, pool covers connection-pool acquisition contention.
# A future operator-tunable shape can move these to `GitHubAppSettings`;
# tracked at FUP-034 alongside other input-boundary configuration.
_GITHUB_CLIENT_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=5.0, read=30.0, write=30.0, pool=10.0
)


# Type alias for a GitHub client authenticated as a specific installation.
# Defined here (not in `github/__init__.py`) so that `auth.py` remains
# the only file in the codebase importing `githubkit.AppInstallationAuthStrategy`
# per the intake-and-webhook spec. `github/__init__.py` re-exports the
# alias for ergonomic import from `outrider.github`; downstream
# consumers (`agent/`, etc.) import via `outrider.github` and never
# touch `githubkit` directly.
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
