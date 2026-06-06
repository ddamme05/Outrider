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

import json
from typing import Any

import pytest

from outrider.github.publisher import (
    _REVIEWS_LIST_MAX_PAGES,
    _REVIEWS_LIST_PER_PAGE,
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
    """`PublishEventSink` has exactly 6 methods (4 emit + 1 query + 1 lock acquire).

    `acquire_publish_lock` (the lock-acquire method) landed per
    `DECISIONS.md#027` for the V1 per-review publish-side advisory lock.

    Pins the Protocol surface so a future addition (7th method)
    surfaces as a test failure rather than silently breaking test
    stubs that don't implement it. Without this pin, a Protocol
    member added in V1.5 wouldn't crash existing test fixtures —
    Python's structural typing only flags MISSING declared members
    at isinstance() time, NOT extra members on the implementation
    side. So a new Protocol method = silent stub-incompleteness
    across all 6+ test fixtures.
    """
    from outrider.audit.sinks import PublishEventSink

    expected = {
        "emit_publish_routing",
        "emit_publish_eligibility",
        "emit_publish_attempt",
        "emit_publish_result",
        "query_prior_publish_event",
        "acquire_publish_lock",
    }
    # `__dict__` is the Protocol's own namespace (excludes inherited
    # `typing.Protocol` machinery); filter for non-dunder.
    actual = {name for name in PublishEventSink.__dict__ if not name.startswith("_")}
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


# ---------------------------------------------------------------------------
# find_existing_review_on_head_sha — matcher-hardening (anti-forgery) tests
# ---------------------------------------------------------------------------


class _FakeGitHubReview:
    """Builder for a GitHub review dict matching the REST response shape."""

    def __init__(
        self,
        *,
        review_id: int,
        body: str | None = None,
        commit_id: str | None = None,
        user_type: str | None = None,
    ) -> None:
        self._dict: dict[str, Any] = {"id": review_id}
        if body is not None:
            self._dict["body"] = body
        if commit_id is not None:
            self._dict["commit_id"] = commit_id
        if user_type is not None:
            self._dict["user"] = {"type": user_type}

    def to_dict(self) -> dict[str, Any]:
        return self._dict


class _FakeGitHubPaginated:
    """Stub `gh` client returning a fixed sequence of paginated pages.

    `arequest` reads the `page` query param and returns that page's reviews
    JSON-serialized via `response.text` — matching the real method's wire
    contract (`json.loads(response.text)`).
    """

    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self.pages = pages

    async def arequest(self, *args: Any, **kwargs: Any) -> _FakeResponse:  # noqa: ARG002
        page_num = kwargs.get("params", {}).get("page", 1)
        if page_num < 1 or page_num > len(self.pages):
            raise IndexError(f"page {page_num} out of range")
        return _FakeResponse(status_code=200, text=json.dumps(self.pages[page_num - 1]))


_FORGE_MARKER = "<!-- outrider-review-id:12345678-1234-1234-1234-123456789012 -->"
_FORGE_SHA = "abc" * 13 + "deadbeef"


@pytest.mark.asyncio
async def test_find_existing_review_match_found_on_first_page(
    publisher: GitHubKitPublisher,
) -> None:
    """Marker-at-start + Bot author + matching commit_id → returns the id."""
    match = _FakeGitHubReview(
        review_id=999,
        body=_FORGE_MARKER + "\n\nbody text",
        commit_id=_FORGE_SHA,
        user_type="Bot",
    )
    other = _FakeGitHubReview(
        review_id=998, body="unrelated", commit_id=_FORGE_SHA, user_type="Bot"
    )
    gh = _FakeGitHubPaginated([[other.to_dict(), match.to_dict()]])  # type: ignore[arg-type]

    result = await publisher.find_existing_review_on_head_sha(
        gh=gh,
        owner="o",
        repo="r",
        pull_number=42,
        head_sha=_FORGE_SHA,
        body_marker=_FORGE_MARKER,
    )

    assert result == 999


@pytest.mark.asyncio
async def test_find_existing_review_human_pasted_marker_rejected(
    publisher: GitHubKitPublisher,
) -> None:
    """Forgery defense: a human (`user.type == 'User'`) who pastes the marker
    on the same head_sha does NOT match — only Bot-authored reviews count."""
    forged = _FakeGitHubReview(
        review_id=998,
        body=_FORGE_MARKER + "\n\nhuman pasted the marker",
        commit_id=_FORGE_SHA,
        user_type="User",
    )
    gh = _FakeGitHubPaginated([[forged.to_dict()]])  # type: ignore[arg-type]

    result = await publisher.find_existing_review_on_head_sha(
        gh=gh,
        owner="o",
        repo="r",
        pull_number=42,
        head_sha=_FORGE_SHA,
        body_marker=_FORGE_MARKER,
    )

    assert result is None


@pytest.mark.asyncio
async def test_find_existing_review_different_commit_id_rejected(
    publisher: GitHubKitPublisher,
) -> None:
    """Marker + Bot but a different commit_id → no match (stale-sha guard)."""
    stale = _FakeGitHubReview(
        review_id=997,
        body=_FORGE_MARKER + "\n\nfrom an earlier push",
        commit_id="def" * 13 + "feedface",
        user_type="Bot",
    )
    gh = _FakeGitHubPaginated([[stale.to_dict()]])  # type: ignore[arg-type]

    result = await publisher.find_existing_review_on_head_sha(
        gh=gh,
        owner="o",
        repo="r",
        pull_number=42,
        head_sha=_FORGE_SHA,
        body_marker=_FORGE_MARKER,
    )

    assert result is None


@pytest.mark.asyncio
async def test_find_existing_review_marker_mid_body_rejected(
    publisher: GitHubKitPublisher,
) -> None:
    """Forgery defense: marker embedded mid-body (not line-anchored at the
    start) fails the `startswith` check even with Bot + matching commit."""
    mid = _FakeGitHubReview(
        review_id=996,
        body="prose first\n" + _FORGE_MARKER + "\nmore",
        commit_id=_FORGE_SHA,
        user_type="Bot",
    )
    gh = _FakeGitHubPaginated([[mid.to_dict()]])  # type: ignore[arg-type]

    result = await publisher.find_existing_review_on_head_sha(
        gh=gh,
        owner="o",
        repo="r",
        pull_number=42,
        head_sha=_FORGE_SHA,
        body_marker=_FORGE_MARKER,
    )

    assert result is None


@pytest.mark.asyncio
async def test_find_existing_review_empty_page_terminates(
    publisher: GitHubKitPublisher,
) -> None:
    """An empty first page (shorter than per_page) terminates the walk → None."""
    gh = _FakeGitHubPaginated([[]])  # type: ignore[arg-type]

    result = await publisher.find_existing_review_on_head_sha(
        gh=gh,
        owner="o",
        repo="r",
        pull_number=42,
        head_sha=_FORGE_SHA,
        body_marker=_FORGE_MARKER,
    )

    assert result is None


@pytest.mark.asyncio
async def test_find_existing_review_match_on_second_page(
    publisher: GitHubKitPublisher,
) -> None:
    """Pagination: a match on page 2 is found ONLY if page 1 is a full page.

    The method terminates when `len(reviews) < _REVIEWS_LIST_PER_PAGE`, so
    page 1 must carry exactly `_REVIEWS_LIST_PER_PAGE` non-matching reviews
    for the walk to continue to page 2.
    """
    page1 = [
        _FakeGitHubReview(
            review_id=900 + i, body=f"review {i}", commit_id=_FORGE_SHA, user_type="Bot"
        ).to_dict()
        for i in range(_REVIEWS_LIST_PER_PAGE)
    ]
    page2 = [
        _FakeGitHubReview(
            review_id=1000,
            body=_FORGE_MARKER + "\n\non page two",
            commit_id=_FORGE_SHA,
            user_type="Bot",
        ).to_dict()
    ]
    gh = _FakeGitHubPaginated([page1, page2])  # type: ignore[arg-type]

    result = await publisher.find_existing_review_on_head_sha(
        gh=gh,
        owner="o",
        repo="r",
        pull_number=42,
        head_sha=_FORGE_SHA,
        body_marker=_FORGE_MARKER,
    )

    assert result == 1000


@pytest.mark.asyncio
async def test_find_existing_review_page_cap_raises(
    publisher: GitHubKitPublisher,
) -> None:
    """Walking past `_REVIEWS_LIST_MAX_PAGES` full pages without a match raises
    `GitHubPublishError` rather than silently returning None."""
    full_pages = [
        [
            _FakeGitHubReview(
                review_id=p * _REVIEWS_LIST_PER_PAGE + i,
                body=f"p{p} r{i}",
                commit_id=_FORGE_SHA,
                user_type="Bot",
            ).to_dict()
            for i in range(_REVIEWS_LIST_PER_PAGE)
        ]
        for p in range(_REVIEWS_LIST_MAX_PAGES + 1)
    ]
    gh = _FakeGitHubPaginated(full_pages)  # type: ignore[arg-type]

    with pytest.raises(GitHubPublishError) as exc_info:
        await publisher.find_existing_review_on_head_sha(
            gh=gh,
            owner="o",
            repo="r",
            pull_number=42,
            head_sha=_FORGE_SHA,
            body_marker=_FORGE_MARKER,
        )

    assert "_REVIEWS_LIST_MAX_PAGES" in str(exc_info.value)


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
