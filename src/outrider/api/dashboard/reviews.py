"""Read review lists, details, findings and replay history for the dashboard.

Admin authentication protects these read-only routes. Findings are assembled
in findings.py; replay uses AuditReplayer. Metrics come from recorded events,
so retained history can still explain a review after content expires.

Joined content and event rows must match both the review ID and its is_eval
flag. Missing completion metrics are unknown, not zero.
See specs/2026-05-31-dashboard-v1.md,
DECISIONS.md#014, #037 and FUP-130, FUP-127, FUP-093.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID  # noqa: TC003  (runtime: Pydantic/route field type)

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import AwareDatetime, BaseModel, ConfigDict
from sqlalchemy import func, select

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.api.dashboard.findings import (
    GLOBAL_SWEEP_INSTALLATION_ID,
    assemble_finding_views,
    latest_sweep_at,
)
from outrider.api.dashboard.findings import FindingsResponse as FindingsResponse
from outrider.api.dashboard.findings import FindingView as FindingView
from outrider.api.dashboard.findings import HITLDecisionView as HITLDecisionView
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
from outrider.db.models.installations import InstallationRepository, active_repo_membership
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession

    from outrider.audit.replay import ReconstructedReview

# Preserve the former route-module import path for finding assembly.
_assemble_finding_views = assemble_finding_views

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
) -> tuple[ReviewMetricsView, bool]:
    """Return audit metrics and whether a matching synthesis completion exists.

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
    synth_row = (await session.execute(synth_stmt)).first()
    has_synthesis_completion = synth_row is not None
    synth_payload = synth_row[0] if synth_row is not None else None
    if synth_payload is None:
        files_examined = files_traced_beyond_diff = None
        wall_clock_seconds = None
    else:
        files_examined = synth_payload["files_examined"]
        files_traced_beyond_diff = synth_payload["files_traced_beyond_diff"]
        wall_clock_seconds = synth_payload["wall_clock_seconds"]

    metrics = ReviewMetricsView(
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
    return metrics, has_synthesis_completion


async def _aggregate_severity_counts(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> SeverityCounts:
    """Count distinct finding hashes per baseline severity after synthesis.

    Use permanent FindingEvents so content purge cannot erase the tally
    (DECISIONS.md#014). Severity is policy-set; HITL overrides belong to
    review detail. The caller must establish that a matching synthesis exists.
    """
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
            metrics, has_synthesis_completion = await _aggregate_metrics(
                session, review.id, review.is_eval
            )
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
                    metrics=metrics,
                    severity_counts=(
                        await _aggregate_severity_counts(session, review.id, review.is_eval)
                        if has_synthesis_completion
                        else None
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
        metrics, _ = await _aggregate_metrics(session, review.id, review.is_eval)
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


def _hitl_decisions_from_events(
    events: tuple[AuditEventUnion, ...],
) -> dict[str, HITLDecisionView]:
    """`{finding_id: HITLDecisionView}` projected from the `HITLDecisionEvent` in the VERIFIED
    reconstruction (DECISIONS.md#034) — the same stream-canonical source as
    `findings._hitl_decisions`, but read from `reconstruct()`'s single snapshot rather than a fresh
    query, so the content panel can
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
        (reconstructed.review.installation_id, GLOBAL_SWEEP_INSTALLATION_ID)
        if reconstructed.review is not None
        else (GLOBAL_SWEEP_INSTALLATION_ID,)
    )
    findings_sweep_at = (
        await latest_sweep_at(session, "findings", installation_ids) if findings_redacted else None
    )
    llm_sweep_at = (
        await latest_sweep_at(session, "llm_call_content", installation_ids)
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
        views = await assemble_finding_views(
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
