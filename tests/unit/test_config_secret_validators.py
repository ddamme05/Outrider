"""Secret-validator unit tests for DashboardSettings + GitHubAppSettings.

Both validators reject empty / whitespace-only secrets AND known `.env.example`
placeholders ('replace-me', ...) at construction, so a verbatim `.env.example`
copy fails loud at startup rather than authenticating against a public value
(an empty or placeholder secret would otherwise admit empty/forged bearer tokens
and unsigned webhooks via `hmac.compare_digest`). Whole-repo review MEDIUM.
"""

import pytest
from pydantic import SecretStr, ValidationError

from outrider.api.dashboard.config import DashboardSettings
from outrider.github.config import GitHubAppSettings

# A real-shaped PEM (not a placeholder word) so the app_private_key validator passes.
_REAL_PEM = "-----BEGIN PRIVATE KEY-----\nMIIabc123\n-----END PRIVATE KEY-----\n"
_REAL_SECRET = "kJ8xQ2-not-a-placeholder-randomish-value"  # noqa: S105 (test fixture, not a real secret)

# Empty/whitespace (the pre-existing reject) + every known placeholder (the new reject),
# with case + surrounding-whitespace variants to prove the normalization.
_BAD_SECRETS = [
    "",
    "   ",
    "replace-me",
    "REPLACE-ME",
    " replace-me ",
    "replace-me-with-a-long-random-secret",
    "change-me",
    "changeme",
    "secret",
    "password",
]


def _github(webhook_secret: SecretStr | None = None) -> GitHubAppSettings:
    return GitHubAppSettings(
        app_id=12345,
        app_private_key=SecretStr(_REAL_PEM),
        webhook_secret=webhook_secret if webhook_secret is not None else SecretStr(_REAL_SECRET),
    )


@pytest.mark.parametrize("bad", _BAD_SECRETS)
def test_dashboard_admin_key_rejects_empty_and_placeholders(bad: str) -> None:
    with pytest.raises(ValidationError):
        DashboardSettings(admin_api_key=SecretStr(bad))


def test_dashboard_admin_key_accepts_a_real_secret() -> None:
    settings = DashboardSettings(admin_api_key=SecretStr(_REAL_SECRET))
    assert settings.admin_api_key.get_secret_value() == _REAL_SECRET


@pytest.mark.parametrize("bad", _BAD_SECRETS)
def test_github_webhook_secret_rejects_empty_and_placeholders(bad: str) -> None:
    with pytest.raises(ValidationError):
        _github(webhook_secret=SecretStr(bad))


def test_github_accepts_real_secrets() -> None:
    settings = _github()
    assert settings.webhook_secret.get_secret_value() == _REAL_SECRET
    assert settings.app_private_key.get_secret_value() == _REAL_PEM
