# GitHub vendor SDK wrapper folder.
"""GitHub-related wrappers around `githubkit`.

Re-exports the type alias `InstallationGitHubClient` for consumers outside
the wrapper folder that need to type-annotate parameters but must not
import `githubkit` directly per `vendor-sdks-only-in-wrappers`.


This folder is the ONLY place in the codebase allowed to `import githubkit`
per `docs/conventions.md` "Imports" and `CLAUDE.md` rule 3 (the
`vendor-sdks-only-in-wrappers` invariant). Consumers in `api/`, `agent/`,
and elsewhere reach githubkit only through the thin helpers exported
here:

  - `outrider.github.auth.make_installation_client_factory(provider)` —
    binds a `GitHubCredentialProvider` at lifespan startup (`#070`); returns
    an ASYNC `Callable[[int], Awaitable[GitHub]]` that resolves credentials
    per call and mints per-installation clients wrapping
    `githubkit.AppInstallationAuthStrategy`.
  - `outrider.github.webhooks.verify_webhook_signature` — wraps
    `githubkit.webhooks.verify`.
  - `outrider.github.fetch` (later spec) — per-file content fetch helpers.

The credential source + provider live in `outrider.github.credentials`; the
`env`-mode settings triad lives in `outrider.github.config.GitHubAppSettings`.
"""

# Re-export the type alias from auth.py. The alias's definition
# requires importing `githubkit.AppInstallationAuthStrategy`; per the
# spec ("github/auth.py is the only file importing
# `githubkit.AppInstallationAuthStrategy`"), the alias is defined in
# auth.py and re-exported here for ergonomic import from
# `outrider.github`.
from outrider.github.auth import (
    AppGitHubClient,
    InstallationGitHubClient,
    make_app_client,
)
from outrider.github.authz import (
    InstallationAuthorizer,
    LiveAuthOutcome,
    LiveAuthResult,
    check_installation_authorization,
    make_installation_authorizer,
)
from outrider.github.publisher import (
    GitHubKitPublisher,
    GitHubPublisher,
    GitHubPublishError,
    GitHubReviewValidationError,
    GitHubSecondaryRateLimitError,
)

__all__ = [
    "AppGitHubClient",
    "GitHubKitPublisher",
    "GitHubPublishError",
    "GitHubPublisher",
    "GitHubReviewValidationError",
    "GitHubSecondaryRateLimitError",
    "InstallationAuthorizer",
    "InstallationGitHubClient",
    "LiveAuthOutcome",
    "LiveAuthResult",
    "check_installation_authorization",
    "make_app_client",
    "make_installation_authorizer",
]
