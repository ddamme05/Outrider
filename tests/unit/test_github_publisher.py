# Tests for github/publisher.py — focus on the 422 taxonomy split.
"""Pin `GitHubKitPublisher`'s 422 discrimination logic.

Per spec §VI line 406 (Codex 2026-05-22 review): GitHub returns
HTTP 422 for two distinct failure classes — per-comment validation
failure AND secondary-rate-limit (abuse-detection throttle). The
status code alone is ambiguous; the publisher MUST inspect the
response body to discriminate.

Prior to this round, the publisher conflated both into
`GitHubReviewValidationError`, which would mis-record
`failure_class` on `PublishAttemptEvent` for rate-limited cases
(critical for retry-diagnosis dashboards).

This file pins:
- Validation-failure body → `GitHubReviewValidationError`
- Secondary-rate-limit body (case-insensitive marker) →
  `GitHubSecondaryRateLimitError`
- Non-422 status → generic `GitHubPublishError`
- Both wrapper exceptions carry `.status_code` AND `.body_text`
"""

from __future__ import annotations

from typing import Any

import pytest

from outrider.github.publisher import (
    GitHubKitPublisher,
    GitHubPublishError,
    GitHubReviewValidationError,
    GitHubSecondaryRateLimitError,
)


class _FakeResponse:
    """Minimal stub mimicking httpx-style response shape."""

    def __init__(self, *, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeRequestFailed(Exception):  # noqa: N818  (test fake mirrors githubkit's name shape)
    """Stand-in for githubkit's RequestFailed exception shape."""

    def __init__(self, *, status_code: int, text: str) -> None:
        super().__init__(f"HTTP {status_code}: {text[:50]}")
        self.response = _FakeResponse(status_code=status_code, text=text)


class _FakeGitHub:
    """Stub `gh` client whose `arequest` raises a chosen exception."""

    def __init__(self, *, raises: Exception) -> None:
        self._raises = raises

    async def arequest(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        raise self._raises


@pytest.fixture
def publisher() -> GitHubKitPublisher:
    return GitHubKitPublisher()


# ---------------------------------------------------------------------------
# 422 — validation failure (the common case, per Q6 sandbox)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_422_validation_body_raises_review_validation_error(
    publisher: GitHubKitPublisher,
) -> None:
    """422 with the Q6-observed validation body → GitHubReviewValidationError."""
    body = (
        '{"message":"Unprocessable Entity",'
        '"errors":["Position could not be resolved and Path could not be resolved"]}'
    )
    gh = _FakeGitHub(raises=_FakeRequestFailed(status_code=422, text=body))

    with pytest.raises(GitHubReviewValidationError) as exc_info:
        await publisher.create_review(
            gh=gh,  # type: ignore[arg-type]
            owner="o",
            repo="r",
            pull_number=1,
            head_sha="0" * 40,
            review_status="COMMENT",
            body_marker="<!-- test -->",
            comments=(),
        )

    assert exc_info.value.status_code == 422
    assert "Unprocessable Entity" in exc_info.value.body_text


# ---------------------------------------------------------------------------
# 422 — secondary-rate-limit (the ambiguity Codex flagged)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_422_secondary_rate_limit_body_raises_rate_limit_error(
    publisher: GitHubKitPublisher,
) -> None:
    """422 carrying GitHub's documented 'secondary rate limit' phrase →
    `GitHubSecondaryRateLimitError`, NOT `GitHubReviewValidationError`.

    Pins the spec §VI line 406 discriminator: the publisher branches
    on body content, not status code alone.
    """
    body = (
        '{"message":"You have exceeded a secondary rate limit. '
        'Please wait a few minutes before you try again.",'
        '"documentation_url":"https://docs.github.com/rest/overview/rate-limits-for-the-rest-api"}'
    )
    gh = _FakeGitHub(raises=_FakeRequestFailed(status_code=422, text=body))

    with pytest.raises(GitHubSecondaryRateLimitError) as exc_info:
        await publisher.create_review(
            gh=gh,  # type: ignore[arg-type]
            owner="o",
            repo="r",
            pull_number=1,
            head_sha="0" * 40,
            review_status="COMMENT",
            body_marker="<!-- test -->",
            comments=(),
        )

    assert exc_info.value.status_code == 422
    assert "secondary rate limit" in exc_info.value.body_text


@pytest.mark.asyncio
async def test_422_secondary_rate_limit_marker_match_is_case_insensitive(
    publisher: GitHubKitPublisher,
) -> None:
    """The marker check uses `.lower()` substring; the rate-limit phrase
    may appear in mixed case across GitHub deployments."""
    body = '{"message":"SECONDARY RATE LIMIT exceeded"}'
    gh = _FakeGitHub(raises=_FakeRequestFailed(status_code=422, text=body))

    with pytest.raises(GitHubSecondaryRateLimitError):
        await publisher.create_review(
            gh=gh,  # type: ignore[arg-type]
            owner="o",
            repo="r",
            pull_number=1,
            head_sha="0" * 40,
            review_status="COMMENT",
            body_marker="<!-- test -->",
            comments=(),
        )


# ---------------------------------------------------------------------------
# Non-422 → generic GitHubPublishError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_403_permission_denied_raises_generic_publish_error(
    publisher: GitHubKitPublisher,
) -> None:
    """403 (App lacks pull_requests:write) → generic `GitHubPublishError`."""
    body = '{"message":"Resource not accessible by integration"}'
    gh = _FakeGitHub(raises=_FakeRequestFailed(status_code=403, text=body))

    with pytest.raises(GitHubPublishError) as exc_info:
        await publisher.create_review(
            gh=gh,  # type: ignore[arg-type]
            owner="o",
            repo="r",
            pull_number=1,
            head_sha="0" * 40,
            review_status="COMMENT",
            body_marker="<!-- test -->",
            comments=(),
        )

    # Generic error class, NOT GitHubReviewValidationError or rate-limit.
    assert not isinstance(exc_info.value, GitHubReviewValidationError)
    assert not isinstance(exc_info.value, GitHubSecondaryRateLimitError)


@pytest.mark.asyncio
async def test_500_upstream_failure_raises_generic_publish_error(
    publisher: GitHubKitPublisher,
) -> None:
    """5xx from GitHub → generic `GitHubPublishError`."""
    gh = _FakeGitHub(raises=_FakeRequestFailed(status_code=503, text="Service Unavailable"))

    with pytest.raises(GitHubPublishError) as exc_info:
        await publisher.create_review(
            gh=gh,  # type: ignore[arg-type]
            owner="o",
            repo="r",
            pull_number=1,
            head_sha="0" * 40,
            review_status="COMMENT",
            body_marker="<!-- test -->",
            comments=(),
        )

    assert not isinstance(exc_info.value, GitHubReviewValidationError)
    assert not isinstance(exc_info.value, GitHubSecondaryRateLimitError)


# ---------------------------------------------------------------------------
# Class hierarchy pin
# ---------------------------------------------------------------------------


def test_both_422_exception_classes_inherit_from_github_publish_error() -> None:
    """The publish node's `except Exception` is the catch-all, but
    middleware that wants to handle "any publish failure" generically
    relies on the `GitHubPublishError` base. Pin the inheritance."""
    assert issubclass(GitHubReviewValidationError, GitHubPublishError)
    assert issubclass(GitHubSecondaryRateLimitError, GitHubPublishError)


def test_422_exception_classes_carry_status_and_body_attributes() -> None:
    """Both wrapper exceptions MUST expose `.status_code` and `.body_text`
    directly on the exception object — `_extract_status_code` in
    `agent/nodes/publish.py` reads `exc.status_code` to record the
    correct value on `PublishAttemptEvent` (Codex 2026-05-22 fix)."""
    val_err = GitHubReviewValidationError("test", status_code=422, body_text="body")
    rate_err = GitHubSecondaryRateLimitError("test", status_code=422, body_text="body")
    assert val_err.status_code == 422
    assert val_err.body_text == "body"
    assert rate_err.status_code == 422
    assert rate_err.body_text == "body"
