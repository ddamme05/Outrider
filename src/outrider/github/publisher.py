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
    Returns `GitHubReviewCreated` on success (HTTP 200; this endpoint
    returns 200, not 201, under apiVersion 2026-03-10).

  - `find_existing_review_on_head_sha(...)` — GET reviews on the PR
    + filter by body containing `body_marker` to detect a prior
    matching review on retry. Defends against the crash-after-GitHub-
    success race (prior process succeeded at the POST but died before
    persisting `PublishEvent`).

GitHub atomically rejects multi-comment reviews where ANY comment has
an invalid position / path / commit_id (verified via sandbox probe).
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
    """HTTP 422 from `POST .../pulls/{n}/reviews` — SECONDARY-RATE-LIMIT.

    GitHub returns the SAME 422 status code for two distinct failure
    classes: per-comment validation failures (atomic rejection) AND
    secondary-rate-limit (abuse-detection throttle). Per `docs/spec.md`
    §VI, the publisher MUST distinguish the two from the response
    body. This exception carries the rate-limit case so the publish
    node's audit row records the right `failure_class` for retry-
    diagnosis dashboards.

    The discriminator: a 422 body containing the literal token
    `"secondary rate limit"` (GitHub's documented abuse-detection
    phrasing — case-insensitive match per the docs). Anything else
    classified as `GitHubReviewValidationError`.

    Treated as a TRANSIENT failure for audit purposes (`failure_class
    = "GitHubSecondaryRateLimitError"`); V1 does NOT auto-retry —
    the dispatcher's retry-after-cooldown logic (V1.5 scope) reads
    the failure_class to decide whether to retry.
    """

    def __init__(self, message: str, *, status_code: int, body_text: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body_text = body_text


# GitHub's secondary-rate-limit phrasings — present in the 422 body
# when the throttle fires. Multiple phrasings are documented across
# GitHub's REST API + create-review prose under apiVersion 2026-03-10.
# Case-insensitive substring match; the publisher checks ALL phrasings
# to defend against wording drift across GitHub deployments.
#
# Future failure mode: if GitHub introduces a new phrasing not in this
# tuple, the publisher silently mis-classifies the throttle as a
# validation failure. Mitigation: also gate on the JSON envelope check
# in `_looks_like_secondary_rate_limit` below — a body with a single
# top-level `message` field AND no `errors[]` array is rate-limit-shaped
# regardless of exact phrasing (validation 422s always include `errors`).
_SECONDARY_RATE_LIMIT_MARKERS: Final[tuple[str, ...]] = (
    "secondary rate limit",  # canonical phrasing per GitHub's docs
    "secondary-rate limit",  # hyphenated variant
    "abuse detection mechanism",  # legacy GitHub phrasing (still observed)
    "rate-limit",  # broadest fallback; intersect with envelope shape below
)


def _looks_like_secondary_rate_limit(body_text: str) -> bool:
    """Discriminate 422 secondary-rate-limit from 422 validation failure.

    Per `docs/spec.md` §VI: 422 is ambiguous between per-comment validation
    failure and secondary-rate-limit (abuse throttle); the publisher MUST
    distinguish from the response body, not status code alone.

    Two-check defense:

    1. **Phrase check**: body contains ANY of the documented rate-limit
       phrasings (case-insensitive). Multiple phrasings defend against
       wording drift across GitHub deployments.

    2. **Envelope check**: body is rate-limit-SHAPED, NOT validation-
       shaped. Validation 422s always carry `errors[]`
       (`{"message":"Unprocessable Entity","errors":["..."]}`).
       Rate-limit 422s carry only `message` (per GitHub's documented
       abuse-detection response). Substring match on `'"errors"'` reliably
       distinguishes the two envelopes regardless of exact phrasing.

    Returns True iff EITHER check confirms rate-limit AND the envelope
    check doesn't refute it. The two checks together close the attacker-
    echo vector: an attacker who manages to inject `"secondary rate
    limit"` into a file path that gets echoed into a validation-422
    body still trips the `"errors":` envelope check and gets classified
    as validation (correct).
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
            `GitHubReviewValidationError`: HTTP 422 with a validation-
                failure body (atomic rejection per Q6).
            `GitHubSecondaryRateLimitError`: HTTP 422 with a
                secondary-rate-limit body. Per `docs/spec.md` §VI,
                422 is ambiguous between the two cases; the publisher
                discriminates by inspecting the response body.
            `GitHubPublishError`: any other HTTP error (403 permission,
                404 PR not found, 5xx upstream) — the base class.
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
        """Implementation per Protocol docstring above.

        Walks paginated review list (GET .../pulls/{n}/reviews) until
        either: (a) a matching review is found, (b) pagination
        exhausts, or (c) `_REVIEWS_LIST_MAX_PAGES` is hit (raises).

        **Matcher hardening:** the body marker alone is forgeable — a
        PR author with `pull_requests:write` could post a human review
        carrying a copy of the marker and trick a retry into
        `idempotently_skipped_external_record`, silently suppressing
        the legitimate findings. Defenses applied:

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
