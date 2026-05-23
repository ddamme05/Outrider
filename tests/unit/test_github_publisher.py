# Tests for github/publisher.py — focus on the 422 taxonomy split.
"""Pin `GitHubKitPublisher`'s 422 discrimination logic.

Per `docs/spec.md` §VI: GitHub returns HTTP 422 for two distinct
failure classes — per-comment validation failure AND secondary-rate-
limit (abuse-detection throttle). The status code alone is ambiguous;
the publisher MUST inspect the response body to discriminate.

Without the split, both would land as `GitHubReviewValidationError`,
which would mis-record `failure_class` on `PublishAttemptEvent` for
rate-limited cases
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

    Pins the publish-node spec §VI discriminator: the publisher
    branches on body content, not status code alone.
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
# 422 — wording drift defense (multi-phrase marker)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrasing",
    [
        "secondary rate limit",  # canonical
        "secondary-rate limit",  # hyphenated
        "abuse detection mechanism",  # legacy
    ],
)
@pytest.mark.asyncio
async def test_422_each_documented_rate_limit_phrasing_matches(
    phrasing: str,
    publisher: GitHubKitPublisher,
) -> None:
    """Multiple GitHub phrasings can trigger the throttle. The publisher
    matches ALL documented phrasings; single-phrase wording drift
    cannot silently mis-classify the throttle as validation.

    Each phrasing → `GitHubSecondaryRateLimitError`.
    """
    body = f'{{"message":"You triggered {phrasing}. Try again later."}}'
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
# 422 — attacker-echo defense (envelope-shape check)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_422_validation_body_with_echoed_rate_limit_phrase_still_validation(
    publisher: GitHubKitPublisher,
) -> None:
    """A 422 validation body that happens to ECHO the rate-limit phrase
    (via an attacker-controlled file path or comment body surfaced in
    `errors[]`) MUST still be classified as
    `GitHubReviewValidationError`, NOT `GitHubSecondaryRateLimitError`.

    The envelope check (`'"errors":' in body`) is the discriminator:
    validation 422s always carry `errors[]`; rate-limit 422s do not.
    The phrase substring alone is insufficient defense.
    """
    # Validation 422 body whose `errors[]` array happens to contain
    # the rate-limit phrase (via an attacker-controlled file path).
    body = (
        '{"message":"Unprocessable Entity",'
        '"errors":["Path could not be resolved: src/secondary rate limit.py"]}'
    )
    gh = _FakeGitHub(raises=_FakeRequestFailed(status_code=422, text=body))

    # MUST raise GitHubReviewValidationError (the validation case),
    # NOT GitHubSecondaryRateLimitError (the rate-limit case).
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
    # Confirm the echoed phrase is in the body but classification is
    # still validation — the envelope check (presence of `errors`)
    # wins over the phrase substring match.
    assert "secondary rate limit" in exc_info.value.body_text.lower()
    assert '"errors"' in exc_info.value.body_text


@pytest.mark.asyncio
async def test_422_rate_limit_body_without_errors_envelope_classified_correctly(
    publisher: GitHubKitPublisher,
) -> None:
    """The standard rate-limit body has NO `errors[]` array; the
    envelope check correctly admits it as a rate-limit. Pins the
    distinction with the prior attacker-echo test.
    """
    body = (
        '{"message":"You have exceeded a secondary rate limit. '
        'Please wait a few minutes before you try again."}'
    )
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
    correct value on `PublishAttemptEvent`."""
    val_err = GitHubReviewValidationError("test", status_code=422, body_text="body")
    rate_err = GitHubSecondaryRateLimitError("test", status_code=422, body_text="body")
    assert val_err.status_code == 422
    assert val_err.body_text == "body"
    assert rate_err.status_code == 422
    assert rate_err.body_text == "body"


# ---------------------------------------------------------------------------
# PublishEventSink Protocol method-set structural pin
# ---------------------------------------------------------------------------


def test_publish_event_sink_protocol_method_set_pinned() -> None:
    """`PublishEventSink` has exactly 5 methods (4 emit + 1 query).

    Pins the Protocol surface so a future addition (6th method)
    surfaces as a test failure rather than silently breaking test
    stubs that don't implement it. Without this pin, a Protocol
    member added in V1.5 wouldn't crash existing test fixtures —
    Python's structural typing only flags MISSING declared members
    at isinstance() time, NOT extra members on the implementation
    side. So a new Protocol method = silent stub-incompleteness
    across all 5+ test fixtures.
    """
    from outrider.audit.sinks import PublishEventSink

    expected = {
        "emit_publish_routing",
        "emit_publish_eligibility",
        "emit_publish_attempt",
        "emit_publish_result",
        "query_prior_publish_event",
    }
    # `dir()` includes inherited dunder methods; filter for non-dunder.
    actual = {name for name in dir(PublishEventSink) if not name.startswith("_")}
    # Remove typing.Protocol's machinery (`_is_protocol`, etc are already
    # filtered by the underscore check above).
    assert actual == expected, (
        f"PublishEventSink method set drift: missing={expected - actual}, "
        f"extra={actual - expected}. If adding a method, update this pin AND "
        f"verify all 5 test-stub sites (`tests/unit/test_publish_node_end_to_end.py`, "
        f"`tests/unit/test_publish_routing.py`, `tests/unit/test_publish_idempotency.py`, "
        f"`tests/unit/test_agent_graph_builder.py`, `tests/unit/test_graph_skip_routing.py`, "
        f"`tests/integration/test_review_state_langgraph_merge.py`, "
        f"`tests/integration/test_analyze_graph_wiring.py`) carry the new method."
    )


# ---------------------------------------------------------------------------
# Body marker UUID shape pin
# ---------------------------------------------------------------------------


def test_body_marker_shape_pinned_to_uuid_format() -> None:
    """The publish node's body marker MUST be `<!-- outrider-review-id:{uuid} -->`
    with the UUID in canonical 8-4-4-4-12 lowercase hex form.

    Pins the load-bearing shape for crash-after-success recovery
    (`find_existing_review_on_head_sha` does literal `startswith`
    matching on the marker). A silent type drift on `review_id` (e.g.,
    UUID → int sequence ID) would change the marker shape and
    invisibly break the matcher. The publish node's explicit
    `str(state.review_id)` + isinstance(UUID) guard at `publish.py`
    surfaces the type-drift case as a TypeError; this test pins the
    marker SHAPE so a wording change in the template ALSO trips here.
    """
    import re
    from uuid import uuid4

    from outrider.agent.nodes.publish import _BODY_MARKER_TEMPLATE

    review_id = uuid4()
    marker = _BODY_MARKER_TEMPLATE.format(review_id=str(review_id))
    expected_pattern = (
        r"^<!-- outrider-review-id:"
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12} -->$"
    )
    assert re.match(expected_pattern, marker), (
        f"Body marker shape drift: produced {marker!r}. The marker is "
        f"load-bearing for crash-after-success recovery — "
        f"`find_existing_review_on_head_sha` does literal startswith "
        f"matching, so any template/shape change MUST update the matcher "
        f"in lockstep."
    )
