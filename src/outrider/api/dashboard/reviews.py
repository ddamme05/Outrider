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

from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID  # noqa: TC003  (runtime: Pydantic/route field type)

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import AwareDatetime, BaseModel, ConfigDict
from sqlalchemy import Integer, Numeric, cast, func, select

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession

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


router = APIRouter(
    prefix="/api/reviews",
    tags=["dashboard"],
    dependencies=[Depends(require_admin_api_key)],
)


async def _aggregate_metrics(session: AsyncSession, review_id: UUID) -> ReviewMetricsView:
    """Compute one review's metrics read-through from the audit stream.

    Filtering by `review_id` alone is the correct `is_eval` scope: every
    audit event a review emits carries that review's `is_eval` value (set
    once per review, propagated to all its events), so there is no
    cross-`is_eval` contamination within a single review's stream.
    """
    # LLM aggregates — summed from llm_call payloads (never the None
    # SynthesizeCompletedEvent fields, per FUP-093).
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
    synth_stmt = (
        select(AuditEvent.payload)
        .where(
            AuditEvent.review_id == review_id,
            AuditEvent.event_type == "synthesize_completed",
        )
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
        )
