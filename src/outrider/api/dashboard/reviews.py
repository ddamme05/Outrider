# See specs/2026-05-31-dashboard-v1.md.
"""Read review queues, findings, and replay data under `/api/reviews`.

Routes use `require_admin_api_key` and do not mutate review or audit data.
Dashboard aggregate, finding, and raw-event queries match both review ID and
the review's own `is_eval` value, even on mixed eval/production pages. This
read-side defense filters divergent rows an emit path may have admitted
(FUP-130). Replay reconstruction instead verifies eval coherence and rejects drift.

Metrics come from the audit stream (DECISIONS.md#037, FUP-127). LLM totals use
`aggregate_review_llm_metrics`, shared with synthesis (FUP-093). File counts and
duration come from the latest `SynthesizeCompletedEvent`; absent completion
means pending (`None`), not zero. Severity tallies use permanent finding
events so they survive content retention (DECISIONS.md#014).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict
from uuid import UUID  # noqa: TC003  (runtime: Pydantic/route field type)

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import AwareDatetime, BaseModel, ConfigDict
from sqlalchemy import func, select

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.audit.aggregates import aggregate_review_llm_metrics
from outrider.audit.events import (  # noqa: TC001 (runtime: Pydantic response-model field type)
    AuditEvent as AuditEventUnion,
)
from outrider.audit.events import HITLDecisionEvent, ReplayVerdictEvent
from outrider.audit.replay import (
    AuditReplayer,
    ReconstructedPhase,
    ReplayEquivalenceError,
    ReplayReviewNotFoundError,
    reconstruct_event_from_row,
)
from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.findings import Finding
from outrider.db.models.installations import InstallationRepository, active_repo_membership
from outrider.db.models.purge_audit import PurgeAudit
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession

    from outrider.audit.replay import ReconstructedReview

# Retention's global TTL sweep uses installation ID 0. Installation-specific
# purge deletes the review too; a surviving review's redacted content normally
# comes from the global sweep. Sweep lookups accept both IDs.
_GLOBAL_SWEEP_INSTALLATION_ID = 0

# The `reviews.status` PG ENUM values (`db/models/_base.py::review_status_enum`).
# A `Literal` so FastAPI returns 422 on an unknown `?status=` rather than
# silently matching nothing.
ReviewStatusFilter = Literal[
    "running",
    "awaiting_approval",
    "awaiting_approval_expired",
    "completed",
    "failed",
    "skipped",
]


class ReviewMetricsView(BaseModel):
    """Audit-stream-computed metrics for one review (see module docstring).

    File/wall-clock fields are `None` when the review has no
    `SynthesizeCompletedEvent` yet — render pending, never zero.
    """

    model_config = ConfigDict(extra="forbid")

    llm_calls_made: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    # Completeness (openai-native-host arc): `total_cost_usd` is the KNOWN SUBTOTAL —
    # a lower bound when `cost_complete` is False (unpriceable completed calls carry a
    # null cost with a typed reason). The SPA renders "at least $X" / "incomplete"
    # from these; defaults keep pre-field clients and fixtures byte-compatible.
    unpriced_call_count: int = 0
    cost_complete: bool = True
    files_examined: int | None
    files_traced_beyond_diff: int | None
    wall_clock_seconds: float | None


class SeverityCounts(BaseModel):
    """Per-severity counts of a review's REPORT-EQUIVALENT findings — the
    synthesize-deduplicated set (`COUNT(DISTINCT content_hash)` per tier),
    NOT raw admitted `findings` rows. Closed key set (the five
    `FindingSeverity` tiers). On `ReviewListItem` this is `None` until a
    `SynthesizeCompletedEvent` exists, because before synthesize there is no
    deduplicated report set to count. Severity is policy-set baseline; a HITL
    override is a review-detail concern, not the list tally.
    """

    model_config = ConfigDict(extra="forbid")

    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    info: int = 0


class StatusCounts(BaseModel):
    """Per-status review counts over the list's BASE filters (`include_eval`
    + `repo_id`), independent of the active `status` filter — so the queue's
    filter chips stay stable while a status is selected. "All N" = the sum.
    """

    model_config = ConfigDict(extra="forbid")

    running: int = 0
    awaiting_approval: int = 0
    awaiting_approval_expired: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0


class ReviewListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    installation_id: int
    repo_id: int
    # Joined from `installation_repositories` (active membership); `None` when
    # no membership row exists for `(installation_id, repo_id)` — the client
    # falls back to `repo {repo_id}`.
    repo_full_name: str | None
    pr_number: int
    # Persisted at review creation from the webhook payload; `None` for rows
    # created before the `pr_title` column landed (no backfill).
    pr_title: str | None
    head_sha: str
    status: str
    is_eval: bool
    created_at: AwareDatetime
    updated_at: AwareDatetime
    completed_at: AwareDatetime | None
    metrics: ReviewMetricsView
    # Report-equivalent per-severity tally; `None` until the review reaches
    # synthesize (see `SeverityCounts`).
    severity_counts: SeverityCounts | None


class ReviewListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[ReviewListItem]
    total: int
    limit: int
    offset: int
    # Per-status counts over the base filters (eval + repo), independent of
    # the active status filter — backs the queue's filter chips.
    status_counts: StatusCounts


class ReviewDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    installation_id: int
    repo_id: int
    # Joined from active installation_repositories membership; null fallback to
    # `repo {repo_id}` client-side (same contract as ReviewListItem).
    repo_full_name: str | None
    pr_number: int
    # Persisted at creation from the webhook payload; null for pre-migration rows.
    pr_title: str | None
    head_sha: str
    status: str
    is_eval: bool
    created_at: AwareDatetime
    updated_at: AwareDatetime
    completed_at: AwareDatetime | None
    expires_at: AwareDatetime | None
    metrics: ReviewMetricsView
    # The per-review policy-version snapshot (DECISIONS.md#028), read from the
    # audit stream — `reviews` has no policy_version column; it rides on the
    # review's events (FindingEvent / SynthesizeCompletedEvent /
    # PublishEligibilityEvent, all the same snapshot). `None` for a review too
    # early to have emitted any policy-version-bearing event.
    policy_version: str | None
    # The authoritative HITL gated set, read from `reviews.hitl_request`
    # (FUP-134). `None` when no HITL request snapshot exists (the gate hasn't
    # fired); `[]` when a snapshot exists but nothing requires approval;
    # otherwise the exact finding ids a `/decide` payload must cover (== the
    # set the decide endpoint enforces). The dashboard uses this instead of
    # inferring the gate from finding severity. The UI still gates the controls
    # on `status` ∈ awaiting_approval[_expired]; this field only defines the set.
    findings_requiring_approval: list[str] | None


class HITLDecisionView(BaseModel):
    """One finding's HITL decision, projected from the canonical audit stream.

    Per DECISIONS.md#034 the per-review `HITLDecisionEvent` is the single
    canonical record of a reviewer's override; the `findings`-table override
    columns (`original_severity` / `override_reason` / `overrider_id`) are
    read-model projections, NULL in V1 (no post-HITL findings writer). This
    view reads the stream by `finding_id`, never the table — the same
    stream-canonical sourcing `publish_destination` already uses for
    `PublishRoutingEvent`.

    Present-or-absent as a unit: a finding the reviewer never decided on (not
    crit/high, so never HITL-gated; or HITL not yet reached) has
    `hitl_decision=None` on its `FindingView`, not a half-populated object.
    `original_severity` / `override_severity` are non-null ONLY when
    `outcome == "severity_override"` (the schema-enforced override contract,
    `schemas/hitl.py::PerFindingDecision.enforce_override_fields`); the other
    three outcomes (`approve` / `reject` / `suppress`) carry neither.
    `reviewer_id` is the event-level reviewer (a string, `"admin"` in V1 per
    DECISIONS.md#011) surfaced per-finding for the dashboard's convenience.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: str
    reviewer_id: str
    reason: str
    original_severity: str | None
    override_severity: str | None


class FindingView(BaseModel):
    """One finding, assembled from the permanent audit record + content.

    Metadata (type/severity/file/line/dimension/tier/proof) comes from the
    `FindingEvent` audit row (permanent). Content (`title`/`description`/
    `evidence`/`suggested_fix`) comes from the `findings` table — present
    within the retention window, `None` with `content_redacted=True` once the
    row is purged but the `FindingEvent` survives (DECISIONS.md#014 point 3:
    "render a dangling finding_id as content redacted per retention policy").

    Lifecycle fields joined from the audit stream (per DECISIONS.md#023's
    routing≠eligibility split): `publish_destination` from `PublishRoutingEvent`
    (where coordinates classified it); `eligibility` / `eligibility_reason`
    from `PublishEligibilityEvent` (whether it actually materialized). A
    high/critical finding pre-HITL shows `inline_comment` + `withheld` +
    `hitl_required_node_absent` — routed but not posted. All three are `None`
    until publish runs.

    HITL override-provenance (`hitl_decision`) is joined from the same audit
    stream per DECISIONS.md#034 — the per-review `HITLDecisionEvent` indexed by
    `finding_id`. `None` when the reviewer rendered no decision on this finding
    (not crit/high, or HITL not yet reached). It survives content redaction the
    same way the publish lifecycle does: the provenance lives in the append-only
    stream, not the retention-purged `findings` row. See `HITLDecisionView`.
    """

    model_config = ConfigDict(extra="forbid")

    finding_id: UUID
    finding_type: str
    dimension: str
    severity: str
    evidence_tier: str
    file_path: str
    line_start: int
    line_end: int
    # Content — None on a retention-redacted stub (findings row purged).
    content_redacted: bool
    title: str | None
    description: str | None
    evidence: str | None
    suggested_fix: str | None
    query_match_id: str | None
    trace_path: list[str] | None
    # Publish lifecycle (routing ≠ eligibility, DECISIONS.md#023).
    publish_destination: str | None
    eligibility: str | None
    eligibility_reason: str | None
    # HITL override-provenance, projected from the canonical HITLDecisionEvent
    # stream (DECISIONS.md#034). None when the reviewer never decided this
    # finding.
    hitl_decision: HITLDecisionView | None
    # Retention: for a `content_redacted` stub, the findings-retention-SWEEP
    # date (latest `target_table='findings'` `purge_audit` row for the global
    # TTL-sweep sentinel or this review's installation); `None` otherwise. The
    # sweep timestamp, NOT a proven per-finding delete time — `purge_audit` is
    # per-table-per-sweep, so exact per-finding provenance is out of reach
    # (FUP-129). Frontend renders "content redacted in the findings retention
    # sweep on <date>".
    redaction_sweep_at: AwareDatetime | None


class FindingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: UUID
    findings: list[FindingView]


class TimelineFindingContentView(BaseModel):
    """Expandable finding content for the replay timeline (ROADMAP §6, PR 2).

    Joined to its `FindingEvent` row in the timeline by `finding_id`. Carries the
    CONTENT (`title`/`description`/`evidence`/`suggested_fix`) the event shadow lacks
    — all `None` with `content_redacted=True` once the `findings` row is purged but the
    event survives (DECISIONS.md#014 point 3). The finding's metadata + proof artifacts
    (severity/tier/location/`query_match_id`/`trace_path`) live on the joined `FindingEvent`
    (in `events`/`phases`) and survive redaction — NOT duplicated here. `hitl_decision` is
    the stream-canonical override provenance (DECISIONS.md#034) projected from the per-review
    `HITLDecisionEvent`, never the V1-null `findings` projection columns.
    """

    model_config = ConfigDict(extra="forbid")

    finding_id: UUID
    content_redacted: bool
    title: str | None
    description: str | None
    evidence: str | None
    suggested_fix: str | None
    hitl_decision: HITLDecisionView | None
    # Findings-retention-SWEEP date for a redacted stub (latest `target_table='findings'`
    # `purge_audit` row), `None` otherwise. Per-table-per-sweep, NOT a per-finding delete
    # time (FUP-129). Frontend renders "content redacted in the findings retention sweep on
    # <date>".
    redaction_sweep_at: AwareDatetime | None


class TimelineLLMExchangeView(BaseModel):
    """Expandable LLM-exchange content for the replay timeline (ROADMAP §6, PR 2).

    Joined to its `LLMCallEvent` row in the timeline by `event_id`. Carries the
    `prompt`/`completion` text from `llm_call_content` — both `None` with
    `content_redacted=True` once the content row is purged but the event survives
    (DECISIONS.md#016 point 5). The call's metadata (model/tokens/cost/hashes) lives on the
    joined `LLMCallEvent` and survives redaction — NOT duplicated here.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    content_redacted: bool
    prompt: str | None
    completion: str | None
    # LLM-content-retention-SWEEP date for a redacted stub (latest
    # `target_table='llm_call_content'` `purge_audit` row), `None` otherwise. Same
    # per-table-per-sweep caveat (FUP-129).
    redaction_sweep_at: AwareDatetime | None


class ReplayTimelineResponse(BaseModel):
    """The grouped, replay-VERIFIED timeline read-model (ROADMAP feature 6).

    A thin serialization of `reconstruct()`'s canonical `ReconstructedReview` — the same
    reconstruction `assert_equivalent` consumes — NOT a re-interpretation of the audit
    tables. The `events`/`phases` are metadata-only (`reconstruct` reads the content tables
    server-side to verify + classify `mode`); `findings`/`llm_exchanges` carry the expandable
    CONTENT (PR 2), serialized from the same verified snapshot and gated on the equivalent
    verdict. Content from the `findings`/`llm_call_content` tables only — never from an
    `audit_events` row (DECISIONS.md#014/#016 metadata-only-audit contract).

    FUP-125 gate: `reconstruct().phases` is trustworthy only after equivalence verification,
    so `phases` is populated IFF `replay_equivalent` is true; otherwise it is `None` and the
    consumer falls back to the flat `events`. `findings`/`llm_exchanges` ride the same gate
    (empty on a non-equivalent verdict). The failure contract:
    reconstruct-raised `ReplayEquivalenceError` → verdict only (`mode`/`status`/`phases` null,
    `events`/`findings`/`llm_exchanges` empty); reconstruct-raised `ValidationError` → 500;
    assert-raised → verdict false + `mode` present + `phases`/content suppressed.

    `events` / `inter_phase_events` EXCLUDE the projected `ReplayVerdictEvent` (post-completion
    replay metadata surfaced via the verdict, not a review-work operation — the same exclusion
    `ReplayVerdictEvent.event_count` applies to the judged stream).
    `inter_phase_events` is the positional set-difference (ordered
    events not in any `phase.events`) — the transitions `_group_phases` drops from the grouped
    view, NOT an enumeration of `_PHASE_UNBOUNDED_EVENTS`.
    """

    model_config = ConfigDict(extra="forbid")

    review_id: UUID
    replay_equivalent: bool
    mode: str | None
    reason: str | None
    status: str | None
    events: tuple[AuditEventUnion, ...]
    phases: tuple[ReconstructedPhase, ...] | None
    inter_phase_events: tuple[AuditEventUnion, ...]
    findings: tuple[TimelineFindingContentView, ...]
    llm_exchanges: tuple[TimelineLLMExchangeView, ...]


class ReviewEventsResponse(BaseModel):
    """A review's full audit-event stream — the typed `AuditEvent` union per
    DECISIONS-stable schema, ordered by `sequence_number` (FUP-133).

    `events` exposes the metadata-only audit record as-is (no content joins, no
    redaction — `audit_events` is metadata-only by `DECISIONS.md#014`). Each event
    is reconstructed through `reconstruct_event_from_row` — the shared replay path —
    so historical rows tolerate post-#025 field additions and every row's mirrored
    base columns are verified against its payload. `total == len(events)` (a single
    review's stream is bounded; no pagination — see the spec non-goals).
    """

    model_config = ConfigDict(extra="forbid")

    review_id: UUID
    events: list[AuditEventUnion]
    total: int


class _FindingLifecycleFields(TypedDict):
    """Audit-derived fields shared by full and retention-redacted findings."""

    publish_destination: str | None
    eligibility: str | None
    eligibility_reason: str | None
    hitl_decision: HITLDecisionView | None


router = APIRouter(
    prefix="/api/reviews",
    tags=["dashboard"],
    dependencies=[Depends(require_admin_api_key)],
)


async def _aggregate_metrics(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> ReviewMetricsView:
    """Combine LLM aggregates with the latest synthesis metrics for one review.

    The caller supplies `reviews.is_eval` for the read-side isolation rule
    in the module docstring.
    """
    # Share LLM aggregation with synthesis; deduplication belongs to that helper.
    agg = await aggregate_review_llm_metrics(session, review_id=review_id, is_eval=review_is_eval)

    # Synthesis completion emits do not enforce review/eval coherence; filter here.
    # Recovery can emit another completion row. Take the latest by sequence;
    # no completion leaves file counts and duration pending.
    synth_stmt = (
        select(AuditEvent.payload)
        .where(
            AuditEvent.review_id == review_id,
            AuditEvent.is_eval == review_is_eval,
            AuditEvent.event_type == "synthesize_completed",
        )
        .order_by(AuditEvent.sequence_number.desc())
        .limit(1)
    )
    synth_payload = (await session.execute(synth_stmt)).scalars().first()
    if synth_payload is None:
        files_examined = files_traced_beyond_diff = None
        wall_clock_seconds = None
    else:
        files_examined = synth_payload["files_examined"]
        files_traced_beyond_diff = synth_payload["files_traced_beyond_diff"]
        wall_clock_seconds = synth_payload["wall_clock_seconds"]

    return ReviewMetricsView(
        llm_calls_made=agg.llm_calls_made,
        total_input_tokens=agg.total_input_tokens,
        total_output_tokens=agg.total_output_tokens,
        total_cost_usd=agg.total_cost_usd,
        unpriced_call_count=agg.unpriced_call_count,
        cost_complete=agg.cost_complete,
        files_examined=files_examined,
        files_traced_beyond_diff=files_traced_beyond_diff,
        wall_clock_seconds=(None if wall_clock_seconds is None else float(wall_clock_seconds)),
    )


async def _aggregate_severity_counts(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> SeverityCounts | None:
    """Count distinct finding hashes per baseline severity after synthesis.

    Use permanent FindingEvents so content purge cannot erase the tally
    (DECISIONS.md#014). Severity is policy-set; HITL overrides belong to
    review detail. Return None until a synthesis completion exists.
    """
    reached_synthesize = (
        await session.execute(
            select(func.count())
            .select_from(AuditEvent)
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.is_eval == review_is_eval,
                AuditEvent.event_type == "synthesize_completed",
            )
        )
    ).scalar_one()
    if not reached_synthesize:
        return None

    severity_key = AuditEvent.payload["severity"].astext
    rows = (
        await session.execute(
            select(
                severity_key,
                func.count(func.distinct(AuditEvent.payload["finding_content_hash"].astext)),
            )
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.is_eval == review_is_eval,
                AuditEvent.event_type == "finding",
            )
            .group_by(severity_key)
        )
    ).all()
    valid = SeverityCounts.model_fields
    return SeverityCounts(**{severity: n for severity, n in rows if severity in valid})


@router.get("", response_model=ReviewListResponse)
async def list_reviews(
    request: Request,
    status_filter: Annotated[ReviewStatusFilter | None, Query(alias="status")] = None,
    repo_id: Annotated[int | None, Query()] = None,
    include_eval: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ReviewListResponse:
    """The review queue. Excludes `is_eval=True` rows unless
    `include_eval=true` (eval-isolation default per `docs/testing.md`).
    """
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        # Base filters (eval + repo) scope BOTH the list and the status-count
        # chips; the active `status` filter narrows ONLY the list + total, so
        # the chips stay stable across status selection.
        base_conditions: list[ColumnElement[bool]] = []
        if not include_eval:
            base_conditions.append(Review.is_eval.is_(False))
        if repo_id is not None:
            base_conditions.append(Review.repo_id == repo_id)
        list_conditions = list(base_conditions)
        if status_filter is not None:
            list_conditions.append(Review.status == status_filter)

        total = (
            await session.execute(select(func.count()).select_from(Review).where(*list_conditions))
        ).scalar_one()

        # Status-count chips: GROUP BY status over the BASE filters only
        # (independent of the active status filter). "All N" = sum.
        status_rows = (
            await session.execute(
                select(Review.status, func.count()).where(*base_conditions).group_by(Review.status)
            )
        ).all()
        valid_statuses = StatusCounts.model_fields
        status_counts = StatusCounts(
            **{status: n for status, n in status_rows if status in valid_statuses}
        )

        # Repo name via LEFT JOIN to active installation_repositories membership
        # (`(installation_id, repo_id)` is unique, so at most one row; removed
        # rows yield NULL → client falls back to `repo {repo_id}`).
        rows = (
            await session.execute(
                select(Review, InstallationRepository.repo_full_name)
                .outerjoin(
                    InstallationRepository,
                    active_repo_membership(Review.installation_id, Review.repo_id),
                )
                .where(*list_conditions)
                .order_by(Review.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        ).all()

        # Aggregate each row sequentially; the page limit bounds these queries.
        items: list[ReviewListItem] = []
        for review, repo_full_name in rows:
            items.append(
                ReviewListItem(
                    id=review.id,
                    installation_id=review.installation_id,
                    repo_id=review.repo_id,
                    repo_full_name=repo_full_name,
                    pr_number=review.pr_number,
                    pr_title=review.pr_title,
                    head_sha=review.head_sha,
                    status=review.status,
                    is_eval=review.is_eval,
                    created_at=review.created_at,
                    updated_at=review.updated_at,
                    completed_at=review.completed_at,
                    metrics=await _aggregate_metrics(session, review.id, review.is_eval),
                    severity_counts=await _aggregate_severity_counts(
                        session, review.id, review.is_eval
                    ),
                )
            )

    return ReviewListResponse(
        reviews=items,
        total=total,
        limit=limit,
        offset=offset,
        status_counts=status_counts,
    )


@router.get("/{review_id}", response_model=ReviewDetail)
async def get_review(request: Request, review_id: UUID) -> ReviewDetail:
    """One review's detail + audit-stream-computed metrics. 404 if absent.

    A direct fetch by id is not `is_eval`-filtered — the list endpoint is the
    eval-isolation surface; holding the id is sufficient to view it.
    """
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        # LEFT JOIN active installation_repositories membership for the repo
        # name (null → client falls back to `repo {repo_id}`); same contract as
        # the list. `(installation_id, repo_id)` is unique, so at most one row.
        detail_row = (
            await session.execute(
                select(Review, InstallationRepository.repo_full_name)
                .outerjoin(
                    InstallationRepository,
                    active_repo_membership(Review.installation_id, Review.repo_id),
                )
                .where(Review.id == review_id)
            )
        ).first()
        if detail_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="review not found")
        review, repo_full_name = detail_row
        metrics = await _aggregate_metrics(session, review.id, review.is_eval)
        # The earliest event carrying the review's policy snapshot supplies its
        # version (DECISIONS.md#028); None before analysis. Policy-version events
        # can use unguarded emit paths, so filter by the review's eval flag.
        policy_version = (
            await session.execute(
                select(AuditEvent.payload["policy_version"].astext)
                .where(
                    AuditEvent.review_id == review.id,
                    AuditEvent.is_eval == review.is_eval,
                    AuditEvent.payload["policy_version"].astext.isnot(None),
                )
                .order_by(AuditEvent.sequence_number.asc())
                .limit(1)
            )
        ).scalar_one_or_none()
        # Authoritative gated set from the persisted HITL request snapshot
        # (FUP-134). None when no snapshot; the stored ids are JSON strings.
        hitl_request = review.hitl_request
        findings_requiring_approval = (
            [str(fid) for fid in (hitl_request.get("findings_requiring_approval") or [])]
            if hitl_request is not None
            else None
        )
        return ReviewDetail(
            id=review.id,
            installation_id=review.installation_id,
            repo_id=review.repo_id,
            repo_full_name=repo_full_name,
            pr_number=review.pr_number,
            pr_title=review.pr_title,
            head_sha=review.head_sha,
            status=review.status,
            is_eval=review.is_eval,
            created_at=review.created_at,
            updated_at=review.updated_at,
            completed_at=review.completed_at,
            expires_at=review.expires_at,
            metrics=metrics,
            policy_version=policy_version,
            findings_requiring_approval=findings_requiring_approval,
        )


async def _latest_publish_routing(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> dict[str, str | None]:
    """Map findings to their latest routing destination, in sequence order."""
    rows = (
        await session.execute(
            select(
                AuditEvent.payload["finding_id"].astext,
                AuditEvent.payload["destination"].astext,
            )
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.is_eval == review_is_eval,
                AuditEvent.event_type == "publish_routing",
            )
            .order_by(AuditEvent.sequence_number.asc())
        )
    ).all()
    return {fid: destination for fid, destination in rows if fid is not None}


async def _latest_publish_eligibility(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> dict[str, tuple[str | None, str | None]]:
    """Map findings to their latest (eligibility, reason) decision.

    Routing does not imply eligibility (DECISIONS.md#023).
    """
    rows = (
        await session.execute(
            select(
                AuditEvent.payload["finding_id"].astext,
                AuditEvent.payload["eligibility"].astext,
                AuditEvent.payload["reason"].astext,
            )
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.is_eval == review_is_eval,
                AuditEvent.event_type == "publish_eligibility",
            )
            .order_by(AuditEvent.sequence_number.asc())
        )
    ).all()
    return {fid: (eligibility, reason) for fid, eligibility, reason in rows if fid is not None}


async def _hitl_decisions(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> dict[str, HITLDecisionView]:
    """Project per-finding decisions from the review's HITLDecisionEvent.

    The audit stream owns override provenance (DECISIONS.md#034). A database
    natural key allows at most one decision event; no event returns an empty
    dict. Copy the event's reviewer ID onto each decision view.
    """
    payload = (
        await session.execute(
            select(AuditEvent.payload)
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.is_eval == review_is_eval,
                AuditEvent.event_type == "hitl_decision",
            )
            .order_by(AuditEvent.sequence_number.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if payload is None:
        return {}
    reviewer_id = payload["reviewer_id"]
    decisions: dict[str, HITLDecisionView] = {}
    for decision in payload["decisions"]:
        fid = decision["finding_id"]
        decisions[fid] = HITLDecisionView(
            outcome=decision["outcome"],
            reviewer_id=reviewer_id,
            reason=decision["reason"],
            original_severity=decision.get("original_severity"),
            override_severity=decision.get("override_severity"),
        )
    return decisions


async def _latest_sweep_at(
    session: AsyncSession,
    target_table: str,
    installation_ids: tuple[int, ...],
) -> datetime | None:
    """Latest `purge_audit` sweep timestamp for `target_table`, scoped to the given
    installation ids (the review's installation + the global TTL-sweep sentinel `0`).

    The per-table-per-sweep date — NOT a proven per-item delete time (FUP-129); not
    eval-scoped (`PurgeAudit` has no `is_eval` column). `None` if no sweep is recorded.
    """
    return (
        await session.execute(
            select(PurgeAudit.timestamp)
            .where(
                PurgeAudit.installation_id.in_(installation_ids),
                PurgeAudit.target_table == target_table,
            )
            .order_by(PurgeAudit.timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


def _hitl_decisions_from_events(
    events: tuple[AuditEventUnion, ...],
) -> dict[str, HITLDecisionView]:
    """`{finding_id: HITLDecisionView}` projected from the `HITLDecisionEvent` in the VERIFIED
    reconstruction (DECISIONS.md#034) — the same stream-canonical source as `_hitl_decisions`, but
    read from `reconstruct()`'s single snapshot rather than a fresh query, so the content panel can
    never surface a decision the verified `events`/`phases` don't contain (single-snapshot
    consistency). At most one `HITLDecisionEvent` per review (DB-guaranteed).
    """
    out: dict[str, HITLDecisionView] = {}
    for event in events:
        if not isinstance(event, HITLDecisionEvent):
            continue
        for decision in event.decisions:
            out[str(decision.finding_id)] = HITLDecisionView(
                outcome=decision.outcome.value,
                reviewer_id=event.reviewer_id,
                reason=decision.reason,
                original_severity=(
                    decision.original_severity.value
                    if decision.original_severity is not None
                    else None
                ),
                override_severity=(
                    decision.override_severity.value
                    if decision.override_severity is not None
                    else None
                ),
            )
    return out


async def _timeline_content(
    session: AsyncSession,
    reconstructed: ReconstructedReview,
) -> tuple[tuple[TimelineFindingContentView, ...], tuple[TimelineLLMExchangeView, ...]]:
    """Build timeline content from the replay's single verified snapshot.

    Missing finding content or LLM prompts means retention redaction (#014/#016).
    Approval provenance comes from this snapshot's decision events (#034), never
    a fresh query. Only sweep dates need separate reads, once per redacted content type.
    """
    decisions_by_finding = _hitl_decisions_from_events(reconstructed.events)
    findings_redacted = any(finding.content is None for finding in reconstructed.findings)
    llm_redacted = any(exchange.prompt is None for exchange in reconstructed.llm_exchanges)
    # Review row survives → scope the sweep lookup to its installation + the global TTL
    # sentinel; purged (METADATA_ONLY) → only the global TTL sweep is reachable.
    installation_ids = (
        (reconstructed.review.installation_id, _GLOBAL_SWEEP_INSTALLATION_ID)
        if reconstructed.review is not None
        else (_GLOBAL_SWEEP_INSTALLATION_ID,)
    )
    findings_sweep_at = (
        await _latest_sweep_at(session, "findings", installation_ids) if findings_redacted else None
    )
    llm_sweep_at = (
        await _latest_sweep_at(session, "llm_call_content", installation_ids)
        if llm_redacted
        else None
    )

    finding_views = tuple(
        TimelineFindingContentView(
            finding_id=finding.event.finding_id,
            content_redacted=finding.content is None,
            title=finding.content.title if finding.content is not None else None,
            description=finding.content.description if finding.content is not None else None,
            evidence=finding.content.evidence if finding.content is not None else None,
            suggested_fix=finding.content.suggested_fix if finding.content is not None else None,
            hitl_decision=decisions_by_finding.get(str(finding.event.finding_id)),
            redaction_sweep_at=findings_sweep_at if finding.content is None else None,
        )
        for finding in reconstructed.findings
    )
    exchange_views = tuple(
        TimelineLLMExchangeView(
            event_id=exchange.event.event_id,
            content_redacted=exchange.prompt is None,
            prompt=exchange.prompt,
            completion=exchange.completion,
            redaction_sweep_at=llm_sweep_at if exchange.prompt is None else None,
        )
        for exchange in reconstructed.llm_exchanges
    )
    return finding_views, exchange_views


async def _assemble_finding_views(
    session: AsyncSession,
    *,
    review_id: UUID,
    installation_id: int,
    review_is_eval: bool,
) -> list[FindingView]:
    """Assemble a review's findings from the permanent audit record + content rows.

    Shared by `GET /{id}/findings` and the agent-view endpoint (`api/dashboard/
    agent_view.py`). Surviving `findings` rows render as full findings; a
    `FindingEvent` whose `findings` row was purged under retention renders as a
    `content_redacted=True` stub from `FindingEvent` metadata (DECISIONS.md#014
    point 3 — the audit trail outlives the content). Publish lifecycle
    (`publish_destination` / `eligibility` / `eligibility_reason`, DECISIONS.md#023)
    and HITL override-provenance (`hitl_decision`, DECISIONS.md#034) are joined from
    the audit stream, never the (V1-null) `findings` columns, so they survive
    content redaction. Apply the module's per-review eval isolation rule to each read.
    """
    destinations_by_finding = await _latest_publish_routing(session, review_id, review_is_eval)
    eligibility_by_finding = await _latest_publish_eligibility(session, review_id, review_is_eval)
    decisions_by_finding = await _hitl_decisions(session, review_id, review_is_eval)

    def lifecycle_for_finding(finding_id: str) -> _FindingLifecycleFields:
        eligibility, reason = eligibility_by_finding.get(finding_id, (None, None))
        return {
            "publish_destination": destinations_by_finding.get(finding_id),
            "eligibility": eligibility,
            "eligibility_reason": reason,
            "hitl_decision": decisions_by_finding.get(finding_id),
        }

    # Read surviving content under the review's eval flag, even with write-side checks.
    content_rows = (
        (
            await session.execute(
                select(Finding).where(
                    Finding.review_id == review_id,
                    Finding.is_eval == review_is_eval,
                )
            )
        )
        .scalars()
        .all()
    )
    content_finding_ids = {str(finding.finding_id) for finding in content_rows}
    views: list[FindingView] = []
    for content_finding in content_rows:
        views.append(
            FindingView(
                finding_id=content_finding.finding_id,
                finding_type=content_finding.finding_type,
                dimension=content_finding.dimension,
                severity=content_finding.severity,
                evidence_tier=content_finding.evidence_tier,
                file_path=content_finding.file_path,
                line_start=content_finding.line_start,
                line_end=content_finding.line_end,
                content_redacted=False,
                title=content_finding.title,
                description=content_finding.description,
                evidence=content_finding.evidence,
                suggested_fix=content_finding.suggested_fix,
                query_match_id=content_finding.query_match_id,
                trace_path=content_finding.trace_path,
                redaction_sweep_at=None,
                **lifecycle_for_finding(str(content_finding.finding_id)),
            )
        )

    # Dangling FindingEvents (content row purged, event survives) ->
    # retention-redacted stubs. Dedup by finding_id, keep latest.
    event_payloads = (
        (
            await session.execute(
                select(AuditEvent.payload)
                .where(
                    AuditEvent.review_id == review_id,
                    AuditEvent.is_eval == review_is_eval,
                    AuditEvent.event_type == "finding",
                )
                .order_by(AuditEvent.sequence_number.asc())
            )
        )
        .scalars()
        .all()
    )
    redacted_by_finding: dict[str, dict[str, Any]] = {
        payload["finding_id"]: payload
        for payload in event_payloads
        if payload["finding_id"] not in content_finding_ids
    }
    # All redacted stubs share the latest recorded sweep date. It describes the
    # table's sweep, not a proven per-finding deletion time (FUP-129).
    redaction_sweep_at = None
    if redacted_by_finding:
        redaction_sweep_at = await _latest_sweep_at(
            session, "findings", (installation_id, _GLOBAL_SWEEP_INSTALLATION_ID)
        )
    views.extend(
        FindingView(
            finding_id=UUID(finding_id),
            finding_type=metadata["finding_type"],
            dimension=metadata["dimension"],
            severity=metadata["severity"],
            evidence_tier=metadata["evidence_tier"],
            file_path=metadata["file_path"],
            line_start=metadata["line_start"],
            line_end=metadata["line_end"],
            content_redacted=True,
            title=None,
            description=None,
            evidence=None,
            suggested_fix=None,
            query_match_id=metadata.get("query_match_id"),
            trace_path=metadata.get("trace_path"),
            redaction_sweep_at=redaction_sweep_at,
            **lifecycle_for_finding(finding_id),
        )
        for finding_id, metadata in redacted_by_finding.items()
    )
    views.sort(key=lambda view: (view.file_path, view.line_start, str(view.finding_id)))
    return views


@router.get("/{review_id}/findings", response_model=FindingsResponse)
async def list_findings(request: Request, review_id: UUID) -> FindingsResponse:
    """A review's findings, assembled from the permanent audit record. 404 if the
    review doesn't exist. The assembly — surviving content rows vs retention-
    redacted stubs, publish lifecycle + HITL provenance from the audit stream, all
    `is_eval`-scoped — lives in `_assemble_finding_views` (DECISIONS.md#014/#023/#034).
    """
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        review_row = (
            await session.execute(
                select(Review.installation_id, Review.is_eval).where(Review.id == review_id)
            )
        ).one_or_none()
        if review_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="review not found")
        installation_id, review_is_eval = review_row
        views = await _assemble_finding_views(
            session,
            review_id=review_id,
            installation_id=installation_id,
            review_is_eval=review_is_eval,
        )
        return FindingsResponse(review_id=review_id, findings=views)


@router.get("/{review_id}/replay-timeline", response_model=ReplayTimelineResponse)
async def get_replay_timeline(request: Request, review_id: UUID) -> ReplayTimelineResponse:
    """Grouped, replay-verified timeline read-model (ROADMAP feature 6).

    Single-snapshot compose: `reconstruct` (one REPEATABLE READ snapshot)
    then `assert_equivalent` over THAT SAME object — never `assert_replay_equivalent` (which
    reconstructs again, risking phases-from-snapshot-A / verdict-from-snapshot-B). `reconstruct`
    inherits historical-field tolerance + row-consistency + `is_eval` coherence by construction.
    Phases (`reconstruct().phases`) AND the expandable `findings`/`llm_exchanges` content are
    exposed ONLY on an equivalent verdict (FUP-125 — phases are trustworthy only once
    `_verify_phase_wellformed` has proven the non-nesting precondition that makes `_group_phases`
    lossless; content rides the same gate). Content (PR 2) comes from `reconstruct()`'s already-
    hydrated `findings`/`llm_exchanges` (the content tables, not audit rows) via
    `_timeline_content`. See `ReplayTimelineResponse` for the failure contract + verdict-exclusion.
    """
    replayer = AuditReplayer(session_factory=request.app.state.session_factory)
    try:
        reconstructed = await replayer.reconstruct(review_id)
    except ReplayReviewNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="review not found"
        ) from exc
    except ReplayEquivalenceError as exc:
        # Reconstruction itself non-equivalent (row-vs-payload / is_eval drift) — verdict only;
        # no trustworthy stream/phases exist. NOT a 500 (the verdict is the product). A corrupt
        # payload that won't deserialize surfaces as ValidationError → 500 (genuine corruption).
        return ReplayTimelineResponse(
            review_id=review_id,
            replay_equivalent=False,
            mode=None,
            reason=str(exc),
            status=None,
            events=(),
            phases=None,
            inter_phase_events=(),
            findings=(),
            llm_exchanges=(),
        )

    equivalent = True
    reason: str | None = None
    try:
        await replayer.assert_equivalent(reconstructed)
    except ReplayEquivalenceError as exc:
        equivalent = False
        reason = str(exc)

    # The flat ordered stream, EXCLUDING the projected ReplayVerdictEvent (post-completion replay
    # metadata, surfaced via the verdict — not a review-work row; the same judged-stream
    # exclusion ReplayVerdictEvent.event_count applies).
    events = tuple(e for e in reconstructed.events if not isinstance(e, ReplayVerdictEvent))
    review_status = reconstructed.review.status if reconstructed.review is not None else None

    phases: tuple[ReconstructedPhase, ...] | None
    if equivalent:
        phases = reconstructed.phases
        # Positional set-difference: ordered events the phase structure does NOT account for — the
        # inter-phase transitions `_group_phases` drops from the grouped view. A phase accounts for
        # its `events` (between the markers) AND its `start`/`end` markers themselves, so exclude
        # both; the verdict is already filtered out of `events`.
        accounted_ids = {e.event_id for phase in phases for e in phase.events}
        for phase in phases:
            if phase.start is not None:
                accounted_ids.add(phase.start.event_id)
            if phase.end is not None:
                accounted_ids.add(phase.end.event_id)
        inter_phase = tuple(e for e in events if e.event_id not in accounted_ids)
    else:
        # FUP-125: the grouping is unverified — suppress phases; the consumer renders the flat
        # `events` + a "not replay-equivalent — grouping unavailable" banner.
        phases = None
        inter_phase = ()

    # Content expansion (PR 2): serialize the finding + LLM content `reconstruct()` already
    # hydrated under its verified snapshot, gated on the equivalent verdict (parallel to the
    # `phases` gate). The FindingEvent/LLMCallEvent metadata + proof artifacts ride the
    # `events`/`phases` rows; these views carry only the content the event shadow lacks + the
    # redaction signal. Suppressed (empty) on a non-equivalent verdict.
    finding_views: tuple[TimelineFindingContentView, ...] = ()
    llm_views: tuple[TimelineLLMExchangeView, ...] = ()
    if equivalent:
        async with request.app.state.session_factory() as session:
            finding_views, llm_views = await _timeline_content(session, reconstructed)

    return ReplayTimelineResponse(
        review_id=review_id,
        replay_equivalent=equivalent,
        mode=reconstructed.mode.value,
        reason=reason,
        status=review_status,
        events=events,
        phases=phases,
        inter_phase_events=inter_phase,
        findings=finding_views,
        llm_exchanges=llm_views,
    )


@router.get("/{review_id}/events", response_model=ReviewEventsResponse)
async def get_review_events(request: Request, review_id: UUID) -> ReviewEventsResponse:
    """The review's full audit-event stream, ordered by `sequence_number` (FUP-133).

    Read-only over `audit_events`, which is metadata-only by `DECISIONS.md#014`
    (hashes/ids/costs — never raw prompt or finding content), so there are no
    content-table joins and no redaction. Each row is rebuilt through the shared
    `reconstruct_event_from_row` (the replay read-path), so historical rows tolerate
    post-#025 field additions (DECISIONS.md#032) AND every row's mirrored base
    columns are verified against its payload. 404 when the review doesn't exist.
    Like the detail endpoint, a by-id fetch is NOT gated on the caller's eval
    preference — holding the id is sufficient to view it — but the stream is
    scoped to the review's OWN `is_eval` (FUP-130 read-side defense): a divergent
    eval event can't surface on a production review's explorer (the firehose was
    the broadest unguarded leak). A single review's stream is bounded by its graph
    run, so it is returned whole (no pagination).
    """
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        review_is_eval = (
            await session.execute(select(Review.is_eval).where(Review.id == review_id))
        ).scalar_one_or_none()
        if review_is_eval is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="review not found")
        rows = (
            await session.execute(
                select(
                    AuditEvent.event_id,
                    AuditEvent.review_id,
                    AuditEvent.event_type,
                    AuditEvent.timestamp,
                    AuditEvent.is_eval,
                    AuditEvent.phase_key,
                    AuditEvent.payload,
                    AuditEvent.sequence_number,
                )
                .where(
                    AuditEvent.review_id == review_id,
                    AuditEvent.is_eval == review_is_eval,
                )
                .order_by(AuditEvent.sequence_number.asc())
            )
        ).all()

    # Reconstruct off the materialized rows (sync, no DB). A row whose mirrored
    # base columns disagree with its payload raises ReplayEquivalenceError — that
    # is genuine corruption, surfaced loudly as a structured 500, never a silent
    # mismatched event.
    try:
        events: list[AuditEventUnion] = []
        for event_row in rows:
            events.append(
                reconstruct_event_from_row(
                    payload=event_row.payload,
                    sequence_number=event_row.sequence_number,
                    event_id=event_row.event_id,
                    review_id=event_row.review_id,
                    event_type=event_row.event_type,
                    timestamp=event_row.timestamp,
                    is_eval=event_row.is_eval,
                    phase_key=event_row.phase_key,
                )
            )
    except ReplayEquivalenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "audit_row_inconsistent", "note": str(exc)},
        ) from exc

    return ReviewEventsResponse(review_id=review_id, events=events, total=len(events))
