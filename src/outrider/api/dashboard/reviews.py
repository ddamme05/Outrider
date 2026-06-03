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
read-through from the audit stream, NOT from the `reviews.*` aggregate
columns (which are seeded to zero and never rolled up — FUP-127 / FUP-093).
Per metric:

  - `llm_calls_made` / `total_input_tokens` / `total_output_tokens` /
    `total_cost_usd` are summed from `LLMCallEvent` rows
    (`event_type='llm_call'`) on `review_id`. These are the only metrics
    summed from raw rows, and the only ones available for a review that has
    not yet reached synthesize. `SynthesizeCompletedEvent`'s LLM-aggregate
    fields are `None` in V1 (FUP-093) — never read them.
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
from sqlalchemy import Integer, Numeric, cast, func, select

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.audit.events import (  # noqa: TC001 (runtime: Pydantic response-model field type)
    AuditEvent as AuditEventUnion,
)
from outrider.audit.replay import (
    AuditReplayer,
    ReplayEquivalenceError,
    ReplayReviewNotFoundError,
    reconstruct_event_from_row,
)
from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.findings import Finding
from outrider.db.models.purge_audit import PurgeAudit
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession

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


class ReviewListItem(BaseModel):
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
    metrics: ReviewMetricsView


class ReviewListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[ReviewListItem]
    total: int
    limit: int
    offset: int


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


class ReplayVerdict(BaseModel):
    """Replay-equivalence verdict for one review — a thin wrapper over
    `audit/replay.py::AuditReplayer`.

    Deliberately does NOT expose `reconstruct()`'s `phases` grouping (FUP-125:
    trustworthy only after `assert_replay_equivalent`) — only the mode, the
    counts, and the pass/fail verdict. `mode`/`event_count`/`finding_count`/
    `orphan_finding_count` are `None` only when `reconstruct` itself raised
    (corrupt row/payload/is_eval drift). `reason` carries the failing-check
    message when not equivalent — metadata only (ids / hashes / counts /
    enum values), never finding content (per the replay verifier's design).
    """

    model_config = ConfigDict(extra="forbid")

    review_id: UUID
    replay_equivalent: bool
    mode: str | None
    event_count: int | None
    finding_count: int | None
    orphan_finding_count: int | None
    reason: str | None


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


async def _aggregate_metrics(session: AsyncSession, review_id: UUID) -> ReviewMetricsView:
    """Compute one review's metrics read-through from the audit stream.

    Filtering by `review_id` alone is the correct `is_eval` scope under V1
    wiring: a review's `is_eval` is a single value (`ReviewState.is_eval`) that
    every emit-site copies onto its events, so a review's stream is
    is_eval-homogeneous. This is PRODUCER DISCIPLINE, not a persister-enforced
    guarantee — the persister copies `event.is_eval` verbatim without
    cross-checking it against `reviews.is_eval` (unlike `installation_id`, which
    it does cross-check). See FUP-130 for adding that guard.
    """
    # LLM aggregates — COUNT/SUM over llm_call payloads (never the None
    # SynthesizeCompletedEvent LLM fields, per FUP-093). The SUM assumes one
    # llm_call row per logical call — true in V1 (the non-durable BackgroundTasks
    # dispatcher never replays a node body, so no crash-recovery re-emit lands a
    # duplicate row; HITL-resume re-enters at hitl, after the LLM-calling nodes).
    # Durable retry (V2 Celery + Redis) WOULD land duplicate rows with fresh
    # event_ids → the SUM would double-count; dedup then needs the V2 `llm_call_event_id`
    # binding (DECISIONS.md#029) — folded into FUP-093.
    llm_stmt = select(
        func.count().label("calls"),
        func.coalesce(func.sum(cast(AuditEvent.payload["input_tokens"].astext, Integer)), 0).label(
            "input_tokens"
        ),
        func.coalesce(func.sum(cast(AuditEvent.payload["output_tokens"].astext, Integer)), 0).label(
            "output_tokens"
        ),
        func.coalesce(func.sum(cast(AuditEvent.payload["cost_usd"].astext, Numeric)), 0).label(
            "cost_usd"
        ),
    ).where(
        AuditEvent.review_id == review_id,
        AuditEvent.event_type == "llm_call",
    )
    llm_row = (await session.execute(llm_stmt)).one()

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
        llm_calls_made=llm_row.calls,
        total_input_tokens=llm_row.input_tokens,
        total_output_tokens=llm_row.output_tokens,
        total_cost_usd=float(llm_row.cost_usd),
        files_examined=files_examined,
        files_traced_beyond_diff=files_traced_beyond_diff,
        wall_clock_seconds=(None if wall_clock_seconds is None else float(wall_clock_seconds)),
    )


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
        conditions: list[ColumnElement[bool]] = []
        if not include_eval:
            conditions.append(Review.is_eval.is_(False))
        if status_filter is not None:
            conditions.append(Review.status == status_filter)
        if repo_id is not None:
            conditions.append(Review.repo_id == repo_id)

        total = (
            await session.execute(select(func.count()).select_from(Review).where(*conditions))
        ).scalar_one()

        rows = (
            (
                await session.execute(
                    select(Review)
                    .where(*conditions)
                    .order_by(Review.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            )
            .scalars()
            .all()
        )

        # Per-review metric aggregation. N+1 over a page bounded by `limit`
        # is an accepted V1 simplification (read-through-at-query-time per
        # the spec); batch later if a page's latency warrants it.
        items = [
            ReviewListItem(
                id=r.id,
                installation_id=r.installation_id,
                repo_id=r.repo_id,
                pr_number=r.pr_number,
                head_sha=r.head_sha,
                status=r.status,
                is_eval=r.is_eval,
                created_at=r.created_at,
                updated_at=r.updated_at,
                completed_at=r.completed_at,
                metrics=await _aggregate_metrics(session, r.id),
            )
            for r in rows
        ]

    return ReviewListResponse(reviews=items, total=total, limit=limit, offset=offset)


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
        metrics = await _aggregate_metrics(session, review.id)
        # The per-review policy-version snapshot lives on the review's audit
        # events, not the `reviews` row. Take the earliest event carrying one
        # (they share the snapshot per DECISIONS.md#028); None pre-analyze.
        policy_version = (
            await session.execute(
                select(AuditEvent.payload["policy_version"].astext)
                .where(
                    AuditEvent.review_id == review.id,
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


async def _latest_publish_routing(session: AsyncSession, review_id: UUID) -> dict[str, str | None]:
    """`{finding_id: destination}` from `PublishRoutingEvent`, latest per
    finding (sequence_number ASC -> last write wins; re-route lands a new row).
    """
    rows = (
        await session.execute(
            select(
                AuditEvent.payload["finding_id"].astext,
                AuditEvent.payload["destination"].astext,
            )
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.event_type == "publish_routing",
            )
            .order_by(AuditEvent.sequence_number.asc())
        )
    ).all()
    return {fid: destination for fid, destination in rows if fid is not None}


async def _latest_publish_eligibility(
    session: AsyncSession, review_id: UUID
) -> dict[str, tuple[str | None, str | None]]:
    """`{finding_id: (eligibility, reason)}` from `PublishEligibilityEvent`,
    latest per finding. Separate from routing per DECISIONS.md#023: a routed
    finding can still be withheld (e.g. `withheld`/`hitl_required_node_absent`).
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
                AuditEvent.event_type == "publish_eligibility",
            )
            .order_by(AuditEvent.sequence_number.asc())
        )
    ).all()
    return {fid: (eligibility, reason) for fid, eligibility, reason in rows if fid is not None}


async def _hitl_decisions(session: AsyncSession, review_id: UUID) -> dict[str, HITLDecisionView]:
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
    denormalized onto each decision for the view.
    """
    payload = (
        await session.execute(
            select(AuditEvent.payload)
            .where(
                AuditEvent.review_id == review_id,
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


@router.get("/{review_id}/findings", response_model=FindingsResponse)
async def list_findings(request: Request, review_id: UUID) -> FindingsResponse:
    """A review's findings, assembled from the permanent audit record.

    404 if the review doesn't exist. Surviving `findings`-table rows render as
    full findings; a `FindingEvent` whose `findings` row was purged under
    retention renders as a `content_redacted=True` stub from `FindingEvent`
    metadata (DECISIONS.md#014 point 3 — the audit trail outlives the content).
    Each finding's publish lifecycle is joined from the audit stream:
    `publish_destination` (`PublishRoutingEvent`) and `eligibility` /
    `eligibility_reason` (`PublishEligibilityEvent`) — separate per
    DECISIONS.md#023, so a routed finding can still show `withheld`. None of
    these are read from the (V1-null) `findings.publish_destination` column.
    HITL override-provenance (`hitl_decision`) is joined the same way from the
    review's single `HITLDecisionEvent` (DECISIONS.md#034) — never the
    (V1-null) `findings` override columns. Both lifecycle and override
    provenance survive content redaction (stream-sourced), so a
    `content_redacted` stub still carries them.
    """
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        installation_id = (
            await session.execute(select(Review.installation_id).where(Review.id == review_id))
        ).scalar_one_or_none()
        if installation_id is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="review not found")

        dest_by_fid = await _latest_publish_routing(session, review_id)
        elig_by_fid = await _latest_publish_eligibility(session, review_id)
        hitl_by_fid = await _hitl_decisions(session, review_id)

        def _lifecycle(fid: str) -> dict[str, Any]:
            eligibility, reason = elig_by_fid.get(fid, (None, None))
            return {
                "publish_destination": dest_by_fid.get(fid),
                "eligibility": eligibility,
                "eligibility_reason": reason,
                "hitl_decision": hitl_by_fid.get(fid),
            }

        # Surviving content rows -> full findings.
        content_rows = (
            (await session.execute(select(Finding).where(Finding.review_id == review_id)))
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
        # reachable "review survives" case) OR this review's installation (the
        # rarer installation-purge case). Best purge_audit can offer
        # (per-table-per-sweep, not per-finding; FUP-129); shared by every
        # redacted stub here. None if no findings-purge sweep is recorded.
        redaction_sweep_at = None
        if redacted_by_fid:
            redaction_sweep_at = (
                await session.execute(
                    select(PurgeAudit.timestamp)
                    .where(
                        PurgeAudit.installation_id.in_(
                            (installation_id, _GLOBAL_SWEEP_INSTALLATION_ID)
                        ),
                        PurgeAudit.target_table == "findings",
                    )
                    .order_by(PurgeAudit.timestamp.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
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
        return FindingsResponse(review_id=review_id, findings=views)


@router.get("/{review_id}/replay", response_model=ReplayVerdict)
async def get_replay_verdict(request: Request, review_id: UUID) -> ReplayVerdict:
    """Replay-equivalence verdict — wraps `audit/replay.py::AuditReplayer`.

    404 if the review has no audit-event rows (`ReplayReviewNotFoundError`).
    Single-snapshot: `reconstruct` (one REPEATABLE READ snapshot) gives mode +
    counts, then `assert_equivalent` verifies THAT SAME reconstruction — so the
    verdict can't mix counts from one snapshot with pass/fail from another (the
    bug if we re-ran `assert_replay_equivalent`, which reconstructs again). Both
    read-only (no mutation). A `ReplayEquivalenceError` from either step yields
    `replay_equivalent=False` carrying the failing check's message, NOT a 500 —
    the verdict IS the product. (A corrupt payload that won't even deserialize
    surfaces as the underlying `ValidationError` → 500, the genuine-corruption
    case.) `phases` is intentionally not exposed (FUP-125).
    """
    replayer = AuditReplayer(session_factory=request.app.state.session_factory)
    try:
        reconstructed = await replayer.reconstruct(review_id)
    except ReplayReviewNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="review not found"
        ) from exc
    except ReplayEquivalenceError as exc:
        # Reconstruction itself is non-equivalent (row-vs-payload / is_eval
        # drift) — report the verdict; mode/counts are unavailable.
        return ReplayVerdict(
            review_id=review_id,
            replay_equivalent=False,
            mode=None,
            event_count=None,
            finding_count=None,
            orphan_finding_count=None,
            reason=str(exc),
        )

    equivalent = True
    reason: str | None = None
    try:
        # Verify the SAME reconstruction (single snapshot), not a re-reconstruct.
        await replayer.assert_equivalent(reconstructed)
    except ReplayEquivalenceError as exc:
        equivalent = False
        reason = str(exc)

    return ReplayVerdict(
        review_id=review_id,
        replay_equivalent=equivalent,
        mode=reconstructed.mode.value,
        event_count=len(reconstructed.events),
        finding_count=len(reconstructed.findings),
        orphan_finding_count=len(reconstructed.orphan_finding_ids),
        reason=reason,
    )


@router.get("/{review_id}/events", response_model=ReviewEventsResponse)
async def get_review_events(request: Request, review_id: UUID) -> ReviewEventsResponse:
    """The review's full audit-event stream, ordered by `sequence_number` (FUP-133).

    Read-only over `audit_events`, which is metadata-only by `DECISIONS.md#014`
    (hashes/ids/costs — never raw prompt or finding content), so there are no
    content-table joins and no redaction. Each row is rebuilt through the shared
    `reconstruct_event_from_row` (the replay read-path), so historical rows tolerate
    post-#025 field additions (DECISIONS.md#032) AND every row's mirrored base
    columns are verified against its payload. 404 when the review has no audit rows
    (parity with detail/replay). Like the detail endpoint, a by-id fetch is NOT
    `is_eval`-filtered — holding the id is sufficient to view it. A single review's
    stream is bounded by its graph run, so it is returned whole (no pagination).
    """
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
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
                .where(AuditEvent.review_id == review_id)
                .order_by(AuditEvent.sequence_number.asc())
            )
        ).all()
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="review not found")

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
