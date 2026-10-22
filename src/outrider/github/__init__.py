# GitHub vendor SDK wrapper folder.
"""GitHub authentication, authorization, fetch, webhook, and publishing wrappers.

Vendor SDK imports stay inside this folder (vendor-sdks-only-in-wrappers;
docs/conventions.md "Imports"). Consumers use the exported client types and
protocols. Credentials come from github/credentials.py's env or database provider
(DECISIONS.md#070); auth.py owns the SDK authentication strategies.
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
