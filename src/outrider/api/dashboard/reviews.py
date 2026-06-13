"""Dashboard read-API — the reviews queue + detail endpoints.

Per `specs/2026-05-31-dashboard-v1.md` (increment 1). Read-only over the
existing tables and the audit stream; this module NEVER issues an
`UPDATE`/`DELETE` (audit-append-only boundary, `docs/trust-boundaries.md` §7).
The only dashboard write path stays the existing `POST /reviews/{id}/decide`
HITL endpoint (`api/dashboard/hitl.py`) — not touched here.

Mounted at prefix `/api/reviews` (per `docs/architecture.md`'s `/api/*`
dashboard namespace; the legacy HITL write stays at `/reviews/{id}/decide`).
Every route is gated by the existing bearer-auth dependency
`require_admin_api_key` (reused, not re-implemented — `hmac.compare_digest`).

**Metric contract (the load-bearing part).** Review metrics are computed
read-through from the audit stream. There is no `reviews.*` aggregate-column
copy — the seeded-zero columns were dropped per DECISIONS.md#037 (FUP-127);
the audit stream is the source of truth. Per metric:

  - `llm_calls_made` / `total_input_tokens` / `total_output_tokens` /
    `total_cost_usd` are summed from `LLMCallEvent` rows
    (`event_type='llm_call'`) on `review_id`. These are the only metrics
    summed from raw rows, and the only ones available for a review that has
    not yet reached synthesize. `SynthesizeCompletedEvent`'s LLM-aggregate
    fields are populated from this same SUM going forward (FUP-093) but are
    `None` for historical rows — so the dashboard sums `LLMCallEvent` directly
    (the single source) rather than reading them.
  - `files_examined` / `files_traced_beyond_diff` / `wall_clock_seconds`
    are read from the per-review `SynthesizeCompletedEvent`
    (`event_type='synthesize_completed'`) payload — the persisted
    `ReviewMetrics` mirror. A review with no such event (synthesize never
    emitted: still `running`, or `failed` per `intake.py`) has these as
    `None` — the UI renders pending, NOT zero.

Severity filtering is intentionally NOT here (increment 2): `reviews` has no
severity field (severity is per-finding, policy-set), so it needs the
findings join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal
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
from outrider.db.models.installations import InstallationRepository
from outrider.db.models.purge_audit import PurgeAudit
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession

    from outrider.audit.replay import ReconstructedReview

# The sentinel `purge_audit.installation_id` the time-based retention sweep
# writes (`sweep/purge_expired.py::_GLOBAL_SWEEP_INSTALLATION_ID` = 0; the sweep
# isn't scoped to one install). The reachable "review survives, findings purged"
# case IS the TTL sweep, so a redacted finding's sweep row carries this sentinel,
# NOT the review's installation_id (the installation-purge path uses the real id
# but also deletes the review, so this endpoint 404s). The lookup matches both.
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
    pr_number: int
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


router = APIRouter(
    prefix="/api/reviews",
    tags=["dashboard"],
    dependencies=[Depends(require_admin_api_key)],
)


async def _aggregate_metrics(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> ReviewMetricsView:
    """Compute one review's metrics read-through from the audit stream.

    Every event read is scoped by BOTH `review_id` AND `is_eval == review_is_eval`
    (FUP-130 read-side defense). A review's `is_eval` is a single value
    (`ReviewState.is_eval`) every emit-site copies onto its events, so the stream
    is is_eval-homogeneous under producer discipline — but the persister only
    enforces that at the two content-bearing sites it resolves the reviews row
    for (`persist()` / `emit_finding()`), and `SynthesizeCompletedEvent` reaches
    the row through a non-resolving emit path. So the `is_eval` predicate here is
    what actually prevents a divergent eval `synthesize_completed` (or `llm_call`)
    event from surfacing on a production review's metrics, mirroring replay's
    read-time `_verify_is_eval_consistent`. Caller passes `reviews.is_eval`.
    """
    # LLM aggregates — read-through SUM over llm_call rows, single-sourced through
    # `aggregate_review_llm_metrics`: the synthesize node populates the audit row from
    # this SAME helper (FUP-093), so the badge and the persisted audit row share one
    # aggregation path (no divergence). Scoped by is_eval (FUP-130). The V1-naive-SUM /
    # V2-dedup contract lives in that helper.
    agg = await aggregate_review_llm_metrics(session, review_id=review_id, is_eval=review_is_eval)

    # File / wall-clock — read from the persisted SynthesizeCompletedEvent
    # (NOT recomputed from raw FileExaminationEvent/TraceDecisionEvent rows).
    # Absent => synthesize never emitted => None (pending, not zero).
    # Duplicate completion rows CAN exist: SynthesizeCompletedEvent is
    # event_id-PK (no V1 natural-key dedup), so a crash-recovery re-emit mints
    # a fresh UUID and lands a second row (per its docstring). Order by
    # `sequence_number` (monotonic on insert) and take the latest — the
    # resumed/successful completion wins, never an arbitrary stale row.
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
        files_examined=files_examined,
        files_traced_beyond_diff=files_traced_beyond_diff,
        wall_clock_seconds=(None if wall_clock_seconds is None else float(wall_clock_seconds)),
    )


async def _aggregate_severity_counts(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> SeverityCounts | None:
    """Report-equivalent per-severity tally for one review, or `None` when the
    review has not reached synthesize (no `SynthesizeCompletedEvent`) — before
    synthesize there is no deduplicated report set to count, so a raw
    `findings` aggregate would not be report-equivalent.

    When populated: `COUNT(DISTINCT content_hash)` per severity over the
    review's `findings` rows. Synthesize deduplicates in-memory by
    `content_hash` (and `findings` carries `content_hash`), so distinct-hash
    count reproduces the final reported set; severity is constant within a
    `content_hash` because it is policy-set from `finding_type`. `is_eval` is
    matched to the review's OWN value (FUP-130 per-row defense, mirroring
    `_aggregate_metrics`), never a global predicate — so a mixed
    `include_eval=true` page never leaks an eval review's findings into a
    production tally. Severity is the policy-set baseline (`severity-set-by-
    policy`); a HITL override is a review-detail concern, not the list tally.
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

    rows = (
        await session.execute(
            select(Finding.severity, func.count(func.distinct(Finding.content_hash)))
            .where(Finding.review_id == review_id, Finding.is_eval == review_is_eval)
            .group_by(Finding.severity)
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
                    (InstallationRepository.installation_id == Review.installation_id)
                    & (InstallationRepository.repo_id == Review.repo_id)
                    & (InstallationRepository.removed_at.is_(None)),
                )
                .where(*list_conditions)
                .order_by(Review.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        ).all()

        # Per-review metric + severity aggregation. N+1 over a page bounded by
        # `limit` is an accepted V1 simplification (read-through-at-query-time
        # per the spec); batch later if a page's latency warrants it.
        items = [
            ReviewListItem(
                id=r.id,
                installation_id=r.installation_id,
                repo_id=r.repo_id,
                repo_full_name=repo_full_name,
                pr_number=r.pr_number,
                pr_title=r.pr_title,
                head_sha=r.head_sha,
                status=r.status,
                is_eval=r.is_eval,
                created_at=r.created_at,
                updated_at=r.updated_at,
                completed_at=r.completed_at,
                metrics=await _aggregate_metrics(session, r.id, r.is_eval),
                severity_counts=await _aggregate_severity_counts(session, r.id, r.is_eval),
            )
            for (r, repo_full_name) in rows
        ]

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
        review = (
            await session.execute(select(Review).where(Review.id == review_id))
        ).scalar_one_or_none()
        if review is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="review not found")
        metrics = await _aggregate_metrics(session, review.id, review.is_eval)
        # The per-review policy-version snapshot lives on the review's audit
        # events, not the `reviews` row. Take the earliest event carrying one
        # (they share the snapshot per DECISIONS.md#028); None pre-analyze.
        # Scoped by `is_eval` (FUP-130 read-side defense) — several
        # policy_version-bearing event types reach the row through unguarded
        # emit paths, so without this a divergent eval event could set a
        # production review's displayed policy_version.
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
            pr_number=review.pr_number,
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
    """`{finding_id: destination}` from `PublishRoutingEvent`, latest per
    finding (sequence_number ASC -> last write wins; re-route lands a new row).
    Scoped by `is_eval` (FUP-130 read-side defense): publish events reach the
    row through an unguarded emit path, so the predicate is what isolates a
    production review's lifecycle from a divergent eval event.
    """
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
    """`{finding_id: (eligibility, reason)}` from `PublishEligibilityEvent`,
    latest per finding. Separate from routing per DECISIONS.md#023: a routed
    finding can still be withheld (e.g. `withheld`/`hitl_required_node_absent`).
    Scoped by `is_eval` (FUP-130 read-side defense, same rationale as routing).
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
    """`{finding_id: HITLDecisionView}` projected from the review's single
    `HITLDecisionEvent`.

    Canonical source of override provenance per DECISIONS.md#034: the audit
    stream, never the (V1-null) `findings` override columns. At most one
    `HITLDecisionEvent` exists per review — DB-guaranteed by
    `uq_audit_events_hitl_decision_natural_key` — so unlike the publish helpers
    there is no per-finding last-write-wins; `desc()` + `LIMIT 1` is defensive
    determinism, not a real multiplicity. Empty dict if the review never gated
    (no crit/high finding) or hasn't reached HITL yet. The per-finding
    `reviewer_id` is the event-level reviewer (a string, `"admin"` in V1)
    denormalized onto each decision for the view. Scoped by `is_eval` (FUP-130
    read-side defense, same rationale as the publish helpers).
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
    """Serialize the reconstructed finding + LLM content for the timeline (ROADMAP §6, PR 2).

    Content is already hydrated on `reconstructed` (the verified single snapshot); `content
    is None` / `prompt is None` ⇒ retention-redacted (DECISIONS.md#014/#016). Override provenance
    is the stream-canonical `HITLDecisionEvent` (DECISIONS.md#034), projected from `reconstruct()`'s
    VERIFIED events (NOT a fresh query — single-snapshot consistency; NOT the V1-null `findings`
    projection columns). The only direct read is the forensic `purge_audit` sweep date, queried once
    per content type when a stub exists.
    """
    hitl_by_fid = _hitl_decisions_from_events(reconstructed.events)
    findings_redacted = any(f.content is None for f in reconstructed.findings)
    llm_redacted = any(x.prompt is None for x in reconstructed.llm_exchanges)
    # Review row survives → scope the sweep lookup to its installation + the global TTL
    # sentinel; purged (METADATA_ONLY) → only the global TTL sweep is reachable.
    installation_ids = (
        (reconstructed.review.installation_id, _GLOBAL_SWEEP_INSTALLATION_ID)
        if reconstructed.review is not None
        else (_GLOBAL_SWEEP_INSTALLATION_ID,)
    )
    findings_sweep = (
        await _latest_sweep_at(session, "findings", installation_ids) if findings_redacted else None
    )
    llm_sweep = (
        await _latest_sweep_at(session, "llm_call_content", installation_ids)
        if llm_redacted
        else None
    )

    finding_views = tuple(
        TimelineFindingContentView(
            finding_id=f.event.finding_id,
            content_redacted=f.content is None,
            title=f.content.title if f.content is not None else None,
            description=f.content.description if f.content is not None else None,
            evidence=f.content.evidence if f.content is not None else None,
            suggested_fix=f.content.suggested_fix if f.content is not None else None,
            hitl_decision=hitl_by_fid.get(str(f.event.finding_id)),
            redaction_sweep_at=findings_sweep if f.content is None else None,
        )
        for f in reconstructed.findings
    )
    llm_views = tuple(
        TimelineLLMExchangeView(
            event_id=x.event.event_id,
            content_redacted=x.prompt is None,
            prompt=x.prompt,
            completion=x.completion,
            redaction_sweep_at=llm_sweep if x.prompt is None else None,
        )
        for x in reconstructed.llm_exchanges
    )
    return finding_views, llm_views


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
    content redaction. EVERY read is `is_eval`-scoped (FUP-130 read-side defense).
    """
    dest_by_fid = await _latest_publish_routing(session, review_id, review_is_eval)
    elig_by_fid = await _latest_publish_eligibility(session, review_id, review_is_eval)
    hitl_by_fid = await _hitl_decisions(session, review_id, review_is_eval)

    def _lifecycle(fid: str) -> dict[str, Any]:
        eligibility, reason = elig_by_fid.get(fid, (None, None))
        return {
            "publish_destination": dest_by_fid.get(fid),
            "eligibility": eligibility,
            "eligibility_reason": reason,
            "hitl_decision": hitl_by_fid.get(fid),
        }

    # Surviving content rows -> full findings. Scoped by `is_eval` (FUP-130):
    # leaking a finding's CONTENT across eval/production isolation is the worst
    # case, so the read filters even though `emit_finding` enforces it write-side.
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
    content_fids = {str(f.finding_id) for f in content_rows}
    views = [
        FindingView(
            finding_id=f.finding_id,
            finding_type=f.finding_type,
            dimension=f.dimension,
            severity=f.severity,
            evidence_tier=f.evidence_tier,
            file_path=f.file_path,
            line_start=f.line_start,
            line_end=f.line_end,
            content_redacted=False,
            title=f.title,
            description=f.description,
            evidence=f.evidence,
            suggested_fix=f.suggested_fix,
            query_match_id=f.query_match_id,
            trace_path=f.trace_path,
            redaction_sweep_at=None,
            **_lifecycle(str(f.finding_id)),
        )
        for f in content_rows
    ]

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
    redacted_by_fid: dict[str, dict[str, Any]] = {
        payload["finding_id"]: payload
        for payload in event_payloads
        if payload["finding_id"] not in content_fids
    }
    # Findings-retention-SWEEP date — the latest `target_table='findings'`
    # purge_audit row matching EITHER the global TTL-sweep sentinel (the
    # reachable "review survives" case) OR this review's installation. Best
    # purge_audit can offer (per-table-per-sweep, not per-finding; FUP-129);
    # shared by every redacted stub. None if no findings-purge sweep is recorded.
    redaction_sweep_at = None
    if redacted_by_fid:
        redaction_sweep_at = await _latest_sweep_at(
            session, "findings", (installation_id, _GLOBAL_SWEEP_INSTALLATION_ID)
        )
    views.extend(
        FindingView(
            finding_id=UUID(fid),
            finding_type=meta["finding_type"],
            dimension=meta["dimension"],
            severity=meta["severity"],
            evidence_tier=meta["evidence_tier"],
            file_path=meta["file_path"],
            line_start=meta["line_start"],
            line_end=meta["line_end"],
            content_redacted=True,
            title=None,
            description=None,
            evidence=None,
            suggested_fix=None,
            query_match_id=meta.get("query_match_id"),
            trace_path=meta.get("trace_path"),
            redaction_sweep_at=redaction_sweep_at,
            **_lifecycle(fid),
        )
        for fid, meta in redacted_by_fid.items()
    )
    views.sort(key=lambda v: (v.file_path, v.line_start, str(v.finding_id)))
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
        events = [
            reconstruct_event_from_row(
                payload=row.payload,
                sequence_number=row.sequence_number,
                event_id=row.event_id,
                review_id=row.review_id,
                event_type=row.event_type,
                timestamp=row.timestamp,
                is_eval=row.is_eval,
                phase_key=row.phase_key,
            )
            for row in rows
        ]
    except ReplayEquivalenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "audit_row_inconsistent", "note": str(exc)},
        ) from exc

    return ReviewEventsResponse(review_id=review_id, events=events, total=len(events))
