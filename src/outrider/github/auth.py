# Vendor wrapper for githubkit's GitHub App installation authentication.
# See DECISIONS.md#065-authorization-is-a-live-github-check-the-local-install-db-is-a-cache
"""Thin wrapper over `githubkit.AppInstallationAuthStrategy`.

Only file in the codebase that imports `githubkit.AppInstallationAuthStrategy`
per `vendor-sdks-only-in-wrappers`. `api/lifespan.py` calls
`make_installation_client_factory(settings)` once at startup and binds
the returned per-installation callable as `github_factory`, which
`build_graph(...)` injects into intake. Intake calls
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

Why settings-bound at lifespan (the outer factory):
  - Lifespan startup constructs `GitHubAppSettings()` once where a
    missing/typo'd env var fails loud with the project's friendly
    RuntimeError shape. A nested `GitHubAppSettings()` call on every
    `make_installation_client(...)` would defeat that gate — env-var
    disappearance on a running pod would surface as `ValidationError`
    deep inside intake (a graph node), not at boot.
"""

from collections.abc import Callable
from typing import Final

import httpx
from githubkit import AppAuthStrategy, AppInstallationAuthStrategy, GitHub

from outrider.github.config import GitHubAppSettings

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
    settings: GitHubAppSettings,
) -> Callable[[int], InstallationGitHubClient]:
    """Build a per-installation `GitHub` client factory closed over the
    given `GitHubAppSettings`.

    Reads `settings.app_id` once at factory-build time; reads
    `settings.app_private_key.get_secret_value()` inside the returned
    closure at every call site so the PEM is in plain memory for as
    short a window as possible.

    Args:
        settings: The lifespan-validated `GitHubAppSettings` instance.
            Caller is responsible for constructing this once at startup
            (where missing env vars fail loud against the project's
            friendly RuntimeError shape).

    Returns:
        A callable `(installation_id: int) -> InstallationGitHubClient`
        that returns a fresh authenticated client per call. Token
        minting + refresh is handled internally by githubkit on first
        API call. The return type uses the `InstallationGitHubClient`
        alias (= `GitHub[AppInstallationAuthStrategy]`) for one-name-
        one-concept symmetry with consumer-side annotations in
        `agent/nodes/intake.py` and the build_graph signature.
    """

    def make_installation_client(
        installation_id: int,
    ) -> InstallationGitHubClient:
        """Return a fresh `GitHub` client authenticated as a specific
        installation.

        Args:
            installation_id: The numeric installation id from the webhook
                payload, validated upstream by the webhook handler
                against the `installations` + `installation_repositories`
                tables.
        """
        return GitHub(
            AppInstallationAuthStrategy(
                settings.app_id,
                settings.app_private_key.get_secret_value(),
                installation_id,
            ),
            timeout=_GITHUB_CLIENT_TIMEOUT,
        )

    return make_installation_client


def make_app_client(settings: GitHubAppSettings) -> AppGitHubClient:
    """Build a GitHub client authenticated AS THE APP (App-JWT) for the #065
    live-authorization check.

    Unlike `make_installation_client_factory` (a fresh per-installation client so
    short-lived installation tokens stay tenant-isolated), the App-JWT client carries
    NO per-installation token — it is the single app identity (one app, one private key
    per deployment, per `#066`). So it is constructed ONCE at startup and reused across
    installations; githubkit mints/refreshes the App-JWT internally. It authorizes the
    App-level endpoints the live check calls (`GET /app/installations/{id}`,
    `POST /app/installations/{id}/access_tokens`).

    Reads `settings.app_private_key.get_secret_value()` once here (the client is
    long-lived, so unlike the per-call installation factory there is no per-call PEM
    window to minimize). Same explicit `_GITHUB_CLIENT_TIMEOUT` as the installation
    client so both SDK surfaces behave consistently under upstream stalls.
    """
    return GitHub(
        AppAuthStrategy(
            settings.app_id,
            settings.app_private_key.get_secret_value(),
        ),
        timeout=_GITHUB_CLIENT_TIMEOUT,
    )
