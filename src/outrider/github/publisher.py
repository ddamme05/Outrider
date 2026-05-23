# GitHub PR-review publisher per specs/2026-05-21-publish-node.md §6 + §3h.
"""GitHubPublisher Protocol + GitHubKitPublisher concrete implementation.

The publish node consumes `GitHubPublisher` to post inline review
comments to GitHub. This module is the only place in the codebase that
calls `gh.arequest("POST", "/repos/{owner}/{repo}/pulls/{n}/reviews",
json=body)` — per the established githubkit escape hatch (cookbook
recommends `arequest` over guessing generated method names that may
not exist).

Two surfaces:

  - `create_review(...)` — POST the review with inline comments. The
    `body_marker` (e.g., `<!-- outrider-review-id:{review_id} -->`)
    is embedded in the review body for crash-after-success recovery.
    Returns `GitHubReviewCreated` on success (HTTP 200; per MCP doc
    verification 2026-05-22, this endpoint returns 200, not 201).

  - `find_existing_review_on_head_sha(...)` — GET reviews on the PR
    + filter by body containing `body_marker` to detect a prior
    matching review on retry. Defends against the crash-after-GitHub-
    success race (prior process succeeded at the POST but died before
    persisting `PublishEvent`).

Per Q6 sandbox (2026-05-22): GitHub atomically rejects multi-comment
reviews where ANY comment has an invalid position / path / commit_id.
HTTP 422 + zero reviews created + zero comments posted. The publisher
does NOT attempt per-comment retry on 422; the atomicity is the
contract V1 relies on (Q2a confirmed). On 422, raise a typed exception
and let the publish node emit `PublishAttemptEvent(outcome=failed,
failure_class="GitHubReviewValidationError")`.

Per `vendor-payloads-normalized-at-boundary`: githubkit's `arequest`
response.text is JSON; we parse and extract `id` + `comments` count
inside this wrapper, then return a frozen `GitHubReviewCreated`. The
publish node never sees the raw response shape.

All requests carry `X-GitHub-Api-Version: 2026-03-10` per spec §4.1.7
+ MCP doc verification. The wrapper is the only place this header is
set; downstream consumers don't see it.
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
    """HTTP 422 from `POST .../pulls/{n}/reviews` — atomic rejection.

    Per Q6 sandbox (2026-05-22): GitHub atomically rejects multi-comment
    reviews where any comment has an invalid position / path /
    commit_id. Zero reviews are created and zero comments are posted
    on 422 — the publish node does NOT retry per-comment.

    Carries the raw 422 response body as `.body_text` for diagnostic
    logging (NOT for parsing decision logic; per Q6 the body shape is
    docs-silent and behavior may vary across GitHub deployments).
    """

    def __init__(self, message: str, *, status_code: int, body_text: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body_text = body_text


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
        comments: tuple[InlineComment, ...],
    ) -> GitHubReviewCreated:
        """POST a new review with inline comments.

        Args:
            gh: Installation-authenticated githubkit client (minted by
                `github_factory(installation_id)` per
                `nodes-receive-deps-via-closure`).
            owner, repo: Repository identifiers (validated upstream
                in the webhook handler via `PRContext`).
            pull_number: PR number from `PRContext`.
            head_sha: HEAD commit SHA to pin the review to. Required —
                without it, GitHub may post against a stale ref if
                the PR has been force-pushed.
            review_status: One of `"APPROVE"`, `"REQUEST_CHANGES"`,
                `"COMMENT"`. V1 publish derives this from the highest-
                severity finding that actually posted (per docs/spec.md
                §V "GitHub review status"). Must be one of these three
                strings exactly — bounded by `PublishEvent.review_status`'s
                Literal upstream.
            body_marker: HTML comment to embed in the review body for
                crash-after-success recovery (e.g.,
                `<!-- outrider-review-id:{review_id} -->`).
            comments: Sanitized inline comments (path / line /
                side=RIGHT / body). Atomic — any invalid comment
                triggers 422 + zero reviews created.

        Returns:
            `GitHubReviewCreated(github_review_id, comments_posted)`.

        Raises:
            `GitHubReviewValidationError`: HTTP 422 (validation
                failure, atomic).
            `GitHubPublishError`: any other HTTP error (403 permission,
                404 PR not found, 5xx upstream).
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
        """GET reviews on the PR; return `github_review_id` of the
        first review whose body contains `body_marker` AND whose
        `commit_id` matches `head_sha`.

        Crash-after-GitHub-success defense: if the prior process
        successfully POSTed the review but died before persisting
        `PublishEvent`, the retry must NOT re-POST. This query is
        how the retry detects the prior success.

        Returns `None` if no matching review is found. Distinguishes:
          - Maintainer-deleted prior bot review → `None` (no match;
            no false positive).
          - Different-installation Outrider review on same head_sha
            → `None` (different review_id → different body_marker).
          - Human review on same head_sha → `None` (no body_marker).
        """
        ...


class GitHubKitPublisher:
    """Concrete `GitHubPublisher` wrapping githubkit's `arequest` surface.

    No state — all coordination happens via the per-call `gh` client
    argument. The publish node constructs ONE `GitHubKitPublisher`
    instance per process at `build_graph` time; multiple reviews share
    it.

    Why a class instead of module-level functions? The Protocol shape
    is the contract; a class lets test fixtures swap in alternate
    implementations without touching call sites. Same pattern as the
    `LLMProvider` Protocol in `llm/base.py`.
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
        comments: tuple[InlineComment, ...],
    ) -> GitHubReviewCreated:
        """Implementation per Protocol docstring above."""
        # Construct the request body. `event` is the review_status
        # (APPROVE / REQUEST_CHANGES / COMMENT); `body` carries the
        # marker for crash-recovery; `comments[]` are sanitized.
        body: dict[str, Any] = {
            "commit_id": head_sha,
            "event": review_status,
            "body": body_marker,
            "comments": [
                {
                    "path": c.path,
                    "line": c.line,
                    "side": c.side,
                    "body": c.body,
                }
                for c in comments
            ],
        }
        try:
            response = await gh.arequest(
                "POST",
                f"/repos/{owner}/{repo}/pulls/{pull_number}/reviews",
                json=body,
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
                raise GitHubReviewValidationError(
                    f"GitHub rejected the review (422): {text[:200]!r}",
                    status_code=422,
                    body_text=text,
                ) from exc
            raise GitHubPublishError(
                f"GitHub create-review failed (status={status}): {text[:200]!r}"
            ) from exc

        # Parse response. Per MCP doc verification (2026-05-22), the
        # success status is 200 (not 201) and the body carries `id`
        # for the review_id. `comments_posted` is len(comments) under
        # atomic semantics — if the call returned 2xx, ALL comments
        # posted (Q6 atomicity).
        parsed = json.loads(response.text)
        github_review_id = parsed["id"]
        return GitHubReviewCreated(
            github_review_id=int(github_review_id),
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
        """Implementation per Protocol docstring above.

        Walks paginated review list (GET .../pulls/{n}/reviews) until
        either: (a) a matching review is found, (b) pagination
        exhausts, or (c) `_REVIEWS_LIST_MAX_PAGES` is hit (raises).

        **Matcher hardening (Wave-3 audit convergent fix):** the body
        marker alone is forgeable — a PR author with `pull_requests:
        write` could post a human review carrying a copy of the marker
        and trick a retry into `idempotently_skipped_external_record`,
        silently suppressing the legitimate findings. Defenses applied:

          1. The marker must appear at the START of the review body
             (line-anchored, not substring) so a marker embedded inside
             prose elsewhere in a review can't match.
          2. The review's `user.type` must be `"Bot"` (Outrider runs
             as a GitHub App; App-posted reviews always carry `Bot`).
          3. The `commit_id` must equal `head_sha` (preserved from
             prior shape).

        A human review on the same head_sha that pastes the marker
        text fails check (2). A Bot-posted review on the same head_sha
        that ISN'T Outrider (e.g., a different App with a borrowed
        marker) fails check (1) iff it puts the marker mid-body, AND
        the marker carries `state.review_id` (high-entropy UUID) so
        cross-App marker collision requires the UUID to leak — defense
        in depth.
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

            reviews = json.loads(response.text)
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
                review_body = review.get("body") or ""
                review_commit = review.get("commit_id") or ""
                # Hardened matcher: line-anchored marker + Bot author +
                # commit match. See method docstring "Matcher hardening".
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
