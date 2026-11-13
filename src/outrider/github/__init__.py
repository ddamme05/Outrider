"""Expose the GitHub API wrappers used to authenticate, fetch and publish.

GitHub SDK calls belong in this package. Callers select the credentials source
explicitly; production does not fall back between stored and environment
credentials (DECISIONS.md#070).
"""

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
