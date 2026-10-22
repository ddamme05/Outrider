# GitHub PR-review publisher per specs/2026-05-21-publish-node.md §6 + §3h.
"""Post a GitHub review or find a prior successful post during crash recovery.

The publish node supplies sanitized comments and a marker-first review body.
create_review makes one POST and normalizes the response into GitHubReviewCreated
(vendor-payloads-normalized-at-boundary). HTTP 422 becomes a typed validation or
secondary-rate-limit failure; there is no per-comment retry.

find_existing_review_on_head_sha walks bounded review pages and matches the
body prefix, head commit, and Bot author. This recovers a POST that succeeded
before its PublishEvent was recorded. Every request here pins the REST version
required by specs/2026-05-21-publish-node.md §6 and docs/spec.md §4.1.7.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from outrider.schemas import GitHubReviewCreated

if TYPE_CHECKING:
    from outrider.github.auth import InstallationGitHubClient
    from outrider.schemas import InlineComment

__all__ = [
    "GitHubKitPublisher",
    "GitHubPublishError",
    "GitHubPublisher",
    "GitHubReviewValidationError",
    "GitHubSecondaryRateLimitError",
]


# Pin the API version on every request — see DECISIONS.md notes on
# version-stability and the spec's reliance on the 2026-03-10 contract
# (Q6 atomicity + 4d line+side semantics verified under this version).
# A future version bump is a deliberate decision, not a silent
# githubkit-side change.
_API_VERSION_HEADER: Final[dict[str, str]] = {"X-GitHub-Api-Version": "2026-03-10"}

# Default pagination size for `GET .../pulls/{n}/reviews` when walking
# pages in `find_existing_review_on_head_sha`. 100 is the GitHub
# documented maximum per the REST docs (apps/pull-requests reviews.md
# under apiVersion 2026-03-10) — larger page sizes are silently capped
# at 100 by GitHub. Walking 100-per-page minimizes round-trips for PRs
# with many reviews.
_REVIEWS_LIST_PER_PAGE: Final[int] = 100

# Safety cap on review-list pages walked when searching for a body
# marker. 100 pages × 100 per page = 10,000 reviews is a defensive
# upper bound. A PR with that many reviews is pathological; the
# wrapper raises rather than walking indefinitely so a misbehaving
# input can't burn API quota.
_REVIEWS_LIST_MAX_PAGES: Final[int] = 100


class GitHubPublishError(Exception):
    """Base for all publish-side GitHub API failures.

    Subclasses pin specific HTTP-status-to-meaning translations so the
    publish node can record a discriminating `failure_class` string
    on `PublishAttemptEvent` without re-parsing exception messages.
    The exception class name itself IS the discriminator
    (`type(exc).__name__` rides on `PublishAttemptEvent.failure_class`).
    """


class GitHubReviewValidationError(GitHubPublishError):
    """HTTP 422 from `POST .../pulls/{n}/reviews` — atomic VALIDATION rejection.

    GitHub atomically rejects multi-comment reviews where any comment
    has an invalid position / path / commit_id. Zero reviews are
    created and zero comments are posted on 422 — the publish node
    does NOT retry per-comment.

    Distinct from `GitHubSecondaryRateLimitError`: GitHub's 422 wording
    ("Validation failed, or the endpoint has been spammed") is ambiguous
    between per-comment validation failure and a secondary-rate-limit;
    the publisher discriminates by inspecting the response body, NOT
    status code alone.

    Carries the raw 422 response body as `.body_text` for diagnostic
    logging (NOT for parsing decision logic).
    """

    def __init__(self, message: str, *, status_code: int, body_text: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body_text = body_text


class GitHubSecondaryRateLimitError(GitHubPublishError):
    """HTTP 422 classified as a secondary-rate-limit failure.

    Classification requires a recognized phrase and no errors envelope; see
    _looks_like_secondary_rate_limit and docs/spec.md §VI. The publish audit records
    the exception type as failure_class. V1 does not automatically retry.
    """

    def __init__(self, message: str, *, status_code: int, body_text: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body_text = body_text


# Recognized throttle phrases. A phrase match is required; the separate errors
# check can reject a match but cannot identify a throttle by itself. New upstream
# wording needs an explicit marker update (docs/spec.md §VI).
_SECONDARY_RATE_LIMIT_MARKERS: Final[tuple[str, ...]] = (
    "secondary rate limit",  # canonical phrasing per GitHub's docs
    "secondary-rate limit",  # hyphenated variant
    "abuse detection mechanism",  # legacy GitHub phrasing (still observed)
    "rate-limit",  # broadest fallback; intersect with envelope shape below
)


def _looks_like_secondary_rate_limit(body_text: str) -> bool:
    """Require a rate-limit phrase and reject validation-shaped errors envelopes.

    HTTP 422 alone is ambiguous (docs/spec.md §VI). The errors check prevents a
    validation body echoing attacker-controlled rate-limit wording from being
    classified as a throttle.
    """
    if not body_text:
        return False
    lower = body_text.lower()
    has_phrase = any(marker in lower for marker in _SECONDARY_RATE_LIMIT_MARKERS)
    has_errors_envelope = '"errors"' in body_text
    # Rate-limit if phrase matches AND envelope does NOT look validation-shaped.
    return has_phrase and not has_errors_envelope


@runtime_checkable
class GitHubPublisher(Protocol):
    """Publish inline review comments to GitHub.

    Consumed by the publish node (`agent/nodes/publish.py`). Real
    implementation: `GitHubKitPublisher`. Test implementation:
    hand-rolled stub per the established test-double pattern (see
    `tests/unit/test_intake_node.py` for the analogous stub shape on
    fetch.py).
    """

    async def create_review(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        review_status: str,
        body_marker: str,
        body: str | None = None,
        comments: tuple[InlineComment, ...],
    ) -> GitHubReviewCreated:
        """Post one review pinned to head_sha with sanitized comments and review_status.

        The publish node supplies the status and coordinates (nodes-receive-deps-via-closure;
        docs/spec.md §V). body must begin with body_marker for crash recovery; None uses
        the marker alone. Body composition, sanitization, and size caps belong to the node
        (DECISIONS.md#050).

        Return GitHubReviewCreated with the review id and posted count. A misplaced marker
        raises ValueError. HTTP 422 raises GitHubReviewValidationError or
        GitHubSecondaryRateLimitError according to the body (docs/spec.md §VI); other
        request or success-payload failures raise GitHubPublishError.
        """
        ...

    async def find_existing_review_on_head_sha(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        body_marker: str,
    ) -> int | None:
        """Return a prior Bot review id matching body-marker prefix and head_sha.

        Return None after exhausting the list without a match; request, malformed-page,
        and page-cap failures raise. Recovery checks this before repeating a POST whose
        success may have preceded the local PublishEvent write.
        """
        ...


class GitHubKitPublisher:
    """Implement GitHubPublisher using the per-call client, with no shared state."""

    async def create_review(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        review_status: str,
        body_marker: str,
        body: str | None = None,
        comments: tuple[InlineComment, ...],
    ) -> GitHubReviewCreated:
        """Implementation per Protocol docstring above."""
        # The review body is either the node-composed body (marker-first,
        # sanitized, size-capped — DECISIONS.md#050) or the bare marker when no
        # body was passed (inline-only path). Guard the marker-first invariant
        # that crash-recovery's `startswith(body_marker)` depends on.
        if body is not None and not body.startswith(body_marker):
            raise ValueError(
                "create_review: composed body must start with body_marker "
                "(offset 0) for crash-recovery; got a body that does not"
            )
        review_body = body if body is not None else body_marker
        # Construct the request payload. `event` is the review_status
        # (APPROVE / REQUEST_CHANGES / COMMENT); `body` carries the marker
        # (+ optional "Related concerns" / dashboard-only note); `comments[]`
        # are sanitized.
        inline_comment_payloads: list[dict[str, Any]] = []
        for inline_comment in comments:
            inline_comment_payloads.append(
                {
                    "path": inline_comment.path,
                    "line": inline_comment.line,
                    "side": inline_comment.side,
                    "body": inline_comment.body,
                }
            )
        request_payload: dict[str, Any] = {
            "commit_id": head_sha,
            "event": review_status,
            "body": review_body,
            "comments": inline_comment_payloads,
        }
        try:
            response = await gh.arequest(
                "POST",
                f"/repos/{owner}/{repo}/pulls/{pull_number}/reviews",
                json=request_payload,
                headers=_API_VERSION_HEADER,
            )
        except Exception as exc:
            # githubkit raises on non-2xx. Inspect for 422 specifically
            # (atomic-rejection contract) vs everything else (auth,
            # 5xx, etc.). The response object lives on the exception
            # as `exc.response` for githubkit's RequestFailed; if the
            # exception shape doesn't match (network error pre-
            # response), wrap as a generic GitHubPublishError.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            text = getattr(getattr(exc, "response", None), "text", str(exc))
            if status == 422:
                # 422 is ambiguous per `docs/spec.md` §VI: validation
                # failure OR secondary-rate-limit (abuse throttle).
                # `_looks_like_secondary_rate_limit` combines a
                # multi-phrase substring match with a JSON-envelope
                # shape check (validation 422s carry `errors[]`;
                # rate-limit 422s do not) — defends against both
                # GitHub wording drift AND attacker echo-injection
                # via paths/content surfaced in a validation response.
                if _looks_like_secondary_rate_limit(text or ""):
                    raise GitHubSecondaryRateLimitError(
                        f"GitHub secondary-rate-limit on create-review (422): {text[:200]!r}",
                        status_code=422,
                        body_text=text,
                    ) from exc
                raise GitHubReviewValidationError(
                    f"GitHub rejected the review (422): {text[:200]!r}",
                    status_code=422,
                    body_text=text,
                ) from exc
            raise GitHubPublishError(
                f"GitHub create-review failed (status={status}): {text[:200]!r}"
            ) from exc

        # Parse response. Under apiVersion 2026-03-10 the success
        # status is 200 (not 201) and the body carries `id` for the
        # review_id. `comments_posted` is len(comments) under atomic
        # semantics — if the call returned 2xx, ALL comments posted.
        # Wrap parse + shape failures in the wrapper's typed exception
        # so a malformed 2xx surfaces as `GitHubPublishError` (the
        # wrapper's contract) rather than escaping as a raw
        # `json.JSONDecodeError` / `KeyError` / `TypeError` /
        # `ValueError` that the publish node's audit chain would
        # classify under the wrong `failure_class`.
        try:
            parsed = json.loads(response.text)
            github_review_id = int(parsed["id"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise GitHubPublishError(
                "GitHub create-review returned an unexpected success payload: "
                f"{response.text[:200]!r}"
            ) from exc
        return GitHubReviewCreated(
            github_review_id=github_review_id,
            comments_posted=len(comments),
        )

    async def find_existing_review_on_head_sha(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        body_marker: str,
    ) -> int | None:
        """Search bounded pages for a marker-first Bot review on the requested head.

        All three checks are required: a copied marker in human review prose must not
        suppress publication. The Bot check does not identify a specific App; the
        review-specific marker supplies the additional identity. Exhaustion returns
        None; reaching the page cap raises rather than claiming no match.
        """
        for page in range(1, _REVIEWS_LIST_MAX_PAGES + 1):
            try:
                response = await gh.arequest(
                    "GET",
                    f"/repos/{owner}/{repo}/pulls/{pull_number}/reviews",
                    params={"per_page": _REVIEWS_LIST_PER_PAGE, "page": page},
                    headers=_API_VERSION_HEADER,
                )
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                text = getattr(getattr(exc, "response", None), "text", str(exc))
                raise GitHubPublishError(
                    f"GitHub list-reviews failed (status={status}, page={page}): {text[:200]!r}"
                ) from exc

            try:
                reviews = json.loads(response.text)
            except json.JSONDecodeError as exc:
                raise GitHubPublishError(
                    "GitHub list-reviews returned non-JSON payload on "
                    f"page={page}: {response.text[:200]!r}"
                ) from exc
            if not isinstance(reviews, list):
                # Defensive — REST doc says list, but if GitHub returns
                # an envelope object on some future version, we don't
                # silently mis-iterate.
                raise GitHubPublishError(
                    f"GitHub list-reviews returned non-list "
                    f"({type(reviews).__name__}); expected list per "
                    f"REST doc apiVersion 2026-03-10."
                )
            for review in reviews:
                if not isinstance(review, dict):
                    continue
                # Explicit str-type narrowing on every vendor-supplied
                # field consumed below. GitHub's REST contract documents
                # `body` and `commit_id` as strings, but a future API
                # change (or a malformed-yet-2xx response) could supply
                # a list/dict/null shape; without the type-narrowing
                # `.startswith(...)` / equality would raise
                # `AttributeError` / no-op outside the typed
                # `GitHubPublishError` hierarchy. Treat any non-string
                # as "no match" rather than raising — matches the
                # broader "best-effort search" intent of this method.
                raw_body = review.get("body")
                review_body = raw_body if isinstance(raw_body, str) else ""
                raw_commit = review.get("commit_id")
                review_commit = raw_commit if isinstance(raw_commit, str) else ""
                # Require the marker prefix, Bot author, and requested head together.
                review_user = review.get("user")
                user_type = review_user.get("type") if isinstance(review_user, dict) else None
                if (
                    review_body.startswith(body_marker)
                    and review_commit == head_sha
                    and user_type == "Bot"
                ):
                    review_id = review.get("id")
                    if isinstance(review_id, int) and review_id >= 1:
                        return review_id
            # Last page is shorter than per_page (or empty) → exit.
            if len(reviews) < _REVIEWS_LIST_PER_PAGE:
                return None

        # Hit the page cap without finding a match. Pathological — a
        # PR with >10,000 reviews shouldn't exist in practice. Raise
        # rather than return None so the caller's audit trail records
        # the abnormal case explicitly.
        raise GitHubPublishError(
            f"GitHub list-reviews exceeded _REVIEWS_LIST_MAX_PAGES="
            f"{_REVIEWS_LIST_MAX_PAGES} without finding body_marker. "
            f"PR {owner}/{repo}#{pull_number} has more reviews than "
            f"the defensive cap; investigate before retrying."
        )
