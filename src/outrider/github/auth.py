# Vendor wrapper for githubkit's GitHub App installation authentication.
# See DECISIONS.md#065-authorization-is-a-live-github-check-the-local-install-db-is-a-cache
"""Thin wrapper over `githubkit.AppInstallationAuthStrategy`.

Only file in the codebase that imports `githubkit.AppInstallationAuthStrategy`
per `vendor-sdks-only-in-wrappers`. `api/lifespan.py` calls
`make_installation_client_factory(provider)` once at startup and binds
the returned per-installation callable as `github_factory`, which
`build_graph(...)` injects into intake. Intake `await`s
`github_factory(state.pr_context.installation_id)` at the moment a fresh
client is needed.

Why a fresh client per call (the inner factory):
  - GitHub installation tokens are short-lived (1 hour) and per-installation;
    caching a single client across installations is a cross-tenant leak.
  - `githubkit` handles JWT minting + installation-token refresh internally;
    we don't manually mint, we just construct the strategy.
  - The lexical-capture variant (e.g., `lambda _iid: pre_built_client`) is
    the canonical violation — type-checks pass, one-installation tests pass,
    production silently uses one installation's token for cross-tenant PRs.
    Test `test_github_factory_distinct_clients.py` exercises this.

Why provider-bound at lifespan (outer factory), credentials resolved lazily (`DECISIONS.md#070`):
  - The `GitHubCredentialProvider` is constructed once at boot; the inner factory `await`s
    `provider.current()` per call so activation of a `database`-mode instance takes effect with no
    restart, and the PEM is read at the last moment. In `env` mode the provider wraps the
    boot-validated `GitHubAppSettings` (behavior unchanged); in `database` mode it fails closed
    (raises `GitHubUnconfiguredError`) while not `CONFIGURED`, surfaced by the setup-only route
    gating before any review runs.
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
    """Build a per-installation `GitHub` client factory over the credential `provider`
    (`DECISIONS.md#070`).

    The returned callable is **async**: it fetches a fresh `GitHubAppCredentials` snapshot from
    the provider PER CALL (`await provider.current()`), then reads `app_id` + `app_private_key`
    from it — so the PEM is in plain memory for as short a window as possible, AND the credentials
    come from the live source. In `env` mode the provider wraps `GitHubAppSettings` (behavior
    unchanged); in `database` mode it reads the onboarded row and raises `GitHubUnconfiguredError`
    while not `CONFIGURED` (fail-closed) — so activation takes effect with no restart (the factory
    was built once at boot, but resolves credentials lazily).

    Returns:
        `async (installation_id: int) -> InstallationGitHubClient` — a fresh authenticated client
        per call. Token minting + refresh is handled internally by githubkit. The async return
        type makes every consumer's missed `await github_factory(...)` a static-type error, not a
        silent bug.
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
    """Build a GitHub client authenticated AS THE APP (App-JWT) for the #065
    live-authorization check.

    Takes a `GitHubAppCredentials` snapshot (`DECISIONS.md#070`) — the #065 authorizer resolves it
    from the credential provider per authorization. Reads `app_id` + `app_private_key` (same field
    names the env-mode `GitHubAppSettings` carried).

    Unlike `make_installation_client_factory` (a fresh per-installation client so
    short-lived installation tokens stay tenant-isolated), the App-JWT client carries
    NO per-installation token — it is the single app identity (one app, one private key
    per deployment, per `#066`); githubkit mints/refreshes the App-JWT internally. It
    authorizes the App-level endpoints the live check calls (`GET /app/installations/{id}`,
    `POST /app/installations/{id}/access_tokens`). The #065 authorizer constructs a fresh
    client PER AUTHORIZATION — NOT once at startup — and `async with`-scopes it; see
    LIFECYCLE below for why a shared long-lived client is not usable here.

    Reads `credentials.app_private_key.get_secret_value()` once per construction; because the
    client is fresh per authorization, that is a brief per-review PEM window — the same
    pattern as the per-installation client factory, not a long-lived exposure. Same explicit
    `_GITHUB_CLIENT_TIMEOUT` as the installation client so both SDK surfaces behave
    consistently under upstream stalls.

    LIFECYCLE — the returned `GitHub` is an async context manager (no `aclose`). Callers
    MUST use it under `async with` so its underlying httpx client is created once, reused
    across the calls in the block, and closed on exit — githubkit's reusing-client guidance
    (0.15.3) warns the un-entered path creates a NEW client per request and that repeatedly
    doing so "may lead to memory leaks". The #065 authorizer (`github/authz.py`) constructs a
    fresh client here per authorization and `async with`-scopes it for the GET + POST pair
    (one client, closed together). A githubkit context manager cannot be entered twice, so a
    fresh client per authorization — not a shared long-lived one — is the correct shape; the
    PEM read here is the same brief-window pattern as the per-installation client.
    """
    return GitHub(
        AppAuthStrategy(
            credentials.app_id,
            credentials.app_private_key.get_secret_value(),
        ),
        timeout=_GITHUB_CLIENT_TIMEOUT,
    )
