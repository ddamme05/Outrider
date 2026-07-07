# Cross-boundary publish carriers per docs/spec.md §4.1.7 + specs/2026-05-21-publish-node.md.
"""InlineComment + PublishResult + GitHubReviewCreated — publish-side carriers.

All three are frozen + extra="forbid" per the output-boundary trust rule
(`docs/trust-boundaries.md` §6): the model proposes, deterministic systems
dispose. The publisher constructs these from `ReviewFinding` + the
`coordinates.tree_sitter_to_github` location — no model field controls
publish routing or comment body content; sanitizer + coordinates own those.

`InlineComment.from_finding(...)` is the canonical production construction
path; the trust-boundary checklist (boundary #6) mandates this factory be
the only call site inside `src/outrider/`. Direct Pydantic construction
remains permitted for test fixtures that need to bypass the sanitizer
path, but an import-graph unit test forbids it inside `src/outrider/`.

`PublishResult` is the publish node's terminal state field — captured on
`ReviewState.publish_result` and rolled up into the dashboard's review-level
summary. Its four constructors (`success`, `empty`, `skipped`,
`skipped_external`) line up with the four non-`failed` `PublishAttemptOutcome`
variants; the publisher raises on `failed` outcomes rather than returning
a degraded result, matching the analyze convention of "failures propagate
and the start phase event is left dangling as the audit signal."
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Self
from uuid import (
    UUID,  # noqa: TC003  (Pydantic v2 needs UUID resolvable at runtime, not TYPE_CHECKING)
)

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from outrider.schemas.review_finding import ReviewFinding


class InlineComment(BaseModel):
    """One inline review comment, ready for the GitHub create-review API.

    Constructed via `InlineComment.from_finding(*, finding, path, line,
    side, body)` in production. Direct construction is permitted by the
    schema for test fixtures but forbidden in `src/outrider/` by an
    import-graph unit test (per the spec's structural-routing assertion
    at §4.1.7 sub-rule 5).

    Fields mirror the per-comment shape the publisher posts to
    `POST /repos/{owner}/{repo}/pulls/{n}/reviews` `comments[]` items
    (verified via 4d sandbox 2026-05-22 + githubkit cookbook): `path`
    + `line` + `side` + `body`. `side` flows through from
    `GitHubCommentLocation.side` per `coordinates-module-is-sole-translator`
    — V1 always sees `"RIGHT"` because `tree_sitter_to_github` only
    accepts head_content. The `position` parameter is NOT used; V1 uses
    source-line coordinates exclusively.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # GitHub-side coordinates. Path is repo-relative POSIX form; line is
    # 1-indexed source-line on the head version of the diff. Both come
    # from `coordinates.tree_sitter_to_github(...)` — never from model
    # output, never from finding.publish_destination (which is an
    # advisory hint coordinates may overwrite).
    path: Annotated[str, Field(max_length=1024)]
    line: int = Field(ge=1)
    # `side` ("LEFT" | "RIGHT") flows from `GitHubCommentLocation.side` and
    # is not defaulted at the schema layer: per
    # `coordinates-module-is-sole-translator`, side is a translation
    # decision that belongs in `coordinates/`, not in any consumer's
    # field default. V1's `tree_sitter_to_github` always returns "RIGHT";
    # this field carries that decision through unchanged.
    side: Annotated[str, Field(pattern=r"^(LEFT|RIGHT)$")]

    # Pre-sanitized body. Capped at `GITHUB_COMMENT_BODY_MAX` UTF-8 bytes
    # by the sanitizer (Outrider policy cap per DECISIONS.md #023 + 4a
    # sandbox); the schema doesn't re-enforce the byte cap because
    # `policy/output_sanitizer.py` is the single canonical authority on
    # that boundary. Schema enforces a coarser char-count cap as a
    # defense-in-depth floor — any body that exceeds this is a sanitizer
    # bug, not a routing decision.
    body: Annotated[str, Field(min_length=1, max_length=131072)]

    # Backed by the originating ReviewFinding so the publisher can join
    # back to the FindingEvent / PublishRoutingEvent identity tuple on
    # the audit side. Not serialized to GitHub; the comment body itself
    # is the user-visible surface.
    finding_id: UUID

    @classmethod
    def from_finding(
        cls,
        *,
        finding: ReviewFinding,
        path: str,
        line: int,
        side: str,
        body: str,
    ) -> Self:
        """Canonical production constructor.

        The caller (publisher) supplies the coordinates from
        `tree_sitter_to_github(...)` (`path`, `line`, `side`) and the
        sanitized `body` from `policy/output_sanitizer.py`. `side` is
        passed through unchanged from `GitHubCommentLocation` — the
        factory does NOT independently decide head-vs-base, per
        `coordinates-module-is-sole-translator` + spec §V publisher-
        input-contract sub-rule. V1 always sees "RIGHT" because
        `tree_sitter_to_github` accepts only `head_content`.

        This factory exists so the trust-boundary structural-routing
        assertion can be enforced via an import-graph test rather
        than reviewer discipline alone.
        """
        return cls(
            path=path,
            line=line,
            side=side,
            body=body,
            finding_id=finding.finding_id,
        )


class GitHubReviewCreated(BaseModel):
    """Publisher's success-path response — what GitHub returned for the POST.

    Carries the minimum identifiers the publish node needs to emit the
    canonical `PublishEvent` audit row (review-level summary) and to
    surface in `PublishResult`. Distinct from the verbose githubkit
    response object: the wrapper extracts the two load-bearing fields
    and discards the rest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # GitHub-assigned review ID. Used as the dedup key on
    # `PublishEvent` (consumer-side dedup is `(review_id, github_review_id)`).
    github_review_id: int = Field(ge=1)

    # Count of inline comments GitHub accepted as posted. V1 atomicity
    # (per Q6 sandbox 2026-05-22): if any comment is invalid, GitHub
    # rejects the entire review with 422 and creates zero rows. So when
    # the publisher returns a `GitHubReviewCreated`, all comments
    # posted; this value equals `len(comments)` from the request.
    comments_posted: int = Field(ge=0)


class PublishResult(BaseModel):
    """Publish node's terminal state field; rolled up into ReviewState.

    Five outcome shapes correspond 1:1 to `PublishAttemptOutcome` minus
    `FAILED` (failed attempts raise rather than producing a result):

    - `success` — review posted; `github_review_id` populated.
    - `empty` — zero eligible/surfaced findings across all three tiers
      (inline + review-body + dashboard-only); no GitHub call (DECISIONS.md#050).
    - `skipped` — prior PublishEvent for this review_id; no GitHub call.
    - `skipped_external` — body-marker query found existing review on
      head_sha (crash-after-success recovery path).
    - `not_published_auth_revoked` — GitHub rejected a publish call (the external-record
      GET or the review POST) with 401/403/404; authorization revoked at GitHub's gate
      (DECISIONS.md#065). No review created; review retained + marked `completed`. Distinct
      from `FAILED` (authorized-away, not an error, so it produces a result rather than raising).

    The publish node returns `{"publish_result": result}` from its body;
    LangGraph merges into `ReviewState.publish_result` via the default
    overwrite reducer (single-writer field, no dedup needed).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The canonical outcome string — mirrors PublishAttemptOutcome.value
    # so dashboard consumers can switch on a single string. NOT typed
    # `PublishAttemptOutcome` directly to avoid the audit→schemas import
    # circular and to keep `schemas/` free of audit-layer dependencies.
    outcome: Annotated[
        str,
        Field(
            pattern=(
                r"^(success|empty|idempotently_skipped|"
                r"idempotently_skipped_external_record|not_published_auth_revoked)$"
            )
        ),
    ]

    # Populated on `success` and `skipped_external`; None on `empty` and `skipped`.
    github_review_id: int | None = Field(default=None, ge=1)

    # Publish accounting splits into three channels (see DECISIONS.md#050):
    # "posted" (inline + review-body) is distinct from "surfaced" (dashboard-only).
    # `comments_posted` is the INLINE count — comments materialized as inline review
    # comments (passed to GitHub). Distinct from `comments_attempted` on
    # `PublishAttemptEvent` (the publisher's outgoing payload; same number for
    # `success`). Zero on `empty`/`skipped` paths.
    comments_posted: int = Field(ge=0, default=0)
    # Eligible REVIEW_BODY findings materialized into the "Related concerns" body
    # section (eligibility-gated like inline). Defaulted (replay-tolerant for rows
    # predating DECISIONS.md#050; the real count is threaded by the routing loop).
    review_body_findings_posted: int = Field(ge=0, default=0)
    # DASHBOARD_ONLY findings surfaced in the aggregate body note (count + link,
    # never per-finding) — "surfaced", not "posted". Defaulted (replay-tolerant).
    dashboard_only_findings_surfaced: int = Field(ge=0, default=0)

    @classmethod
    def success(
        cls,
        *,
        github_review_id: int,
        comments_posted: int,
        review_body_findings_posted: int = 0,
        dashboard_only_findings_surfaced: int = 0,
    ) -> Self:
        """Publisher posted the review; github_review_id is the new row.

        All three accounting channels (DECISIONS.md#050): `comments_posted`
        (inline), `review_body_findings_posted` (Related concerns), and
        `dashboard_only_findings_surfaced` (aggregate note).
        """
        return cls(
            outcome="success",
            github_review_id=github_review_id,
            comments_posted=comments_posted,
            review_body_findings_posted=review_body_findings_posted,
            dashboard_only_findings_surfaced=dashboard_only_findings_surfaced,
        )

    @classmethod
    def empty(cls) -> Self:
        """No eligible/surfaced findings across all three tiers; no GitHub call.

        Per DECISIONS.md#050 `no_op_empty` fires only when inline, review-body,
        AND dashboard-only are all zero. Audit emits no_op_empty.
        """
        return cls(outcome="empty", github_review_id=None, comments_posted=0)

    @classmethod
    def skipped(
        cls,
        *,
        comments_posted: int = 0,
        review_body_findings_posted: int = 0,
        dashboard_only_findings_surfaced: int = 0,
    ) -> Self:
        """Prior PublishEvent for this review_id; no GitHub call.

        Distinct from `skipped_external` because the local audit log
        had the prior row — this is intra-review-id retry idempotency,
        not crash-after-success recovery. The three counts mirror the PRIOR
        PublishEvent's channels (DECISIONS.md#050) so the dashboard / Slack FYI
        reports what the original publish posted, not zero.
        """
        return cls(
            outcome="idempotently_skipped",
            github_review_id=None,
            comments_posted=comments_posted,
            review_body_findings_posted=review_body_findings_posted,
            dashboard_only_findings_surfaced=dashboard_only_findings_surfaced,
        )

    @classmethod
    def skipped_external(
        cls,
        *,
        existing_review_id: int,
        comments_posted: int = 0,
        review_body_findings_posted: int = 0,
        dashboard_only_findings_surfaced: int = 0,
    ) -> Self:
        """find_existing_review_on_head_sha matched a body marker.

        The prior process succeeded at the GitHub call but died before
        persisting PublishEvent. The current process discovers the
        prior review via the embedded `<!-- outrider-review-id:{review_id} -->`
        marker and treats it as the canonical outcome. No prior PublishEvent
        exists, so the three counts (DECISIONS.md#050) reflect the CURRENT
        routing pass (what this run would have posted).
        """
        return cls(
            outcome="idempotently_skipped_external_record",
            github_review_id=existing_review_id,
            comments_posted=comments_posted,
            review_body_findings_posted=review_body_findings_posted,
            dashboard_only_findings_surfaced=dashboard_only_findings_surfaced,
        )

    @classmethod
    def not_published_auth_revoked(cls) -> Self:
        """GitHub rejected a publish call (the external-record GET or the review POST) with
        401/403/404 — authorization was revoked at GitHub's gate (DECISIONS.md#065). No review
        was created, so `github_review_id=None` and all three post/surface counts are 0 (like
        `empty`): nothing landed. The attempted counts live on the paired `PublishAttemptEvent`
        (`comments_attempted`); the review is retained + marked `completed`, and the
        dashboard surfaces "reviewed but not posted (access revoked)" by reading this
        outcome value (no new review status — spec §Output boundary). No `reason` field
        anywhere: the outcome string is self-describing and #023 forbids widening the
        publish-attempt hash-recipe input shape.
        """
        return cls(outcome="not_published_auth_revoked", github_review_id=None, comments_posted=0)
