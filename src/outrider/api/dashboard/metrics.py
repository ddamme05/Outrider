"""Dashboard analytics: cross-review metrics aggregation endpoint (`GET /api/metrics`).

Honest read-only aggregation over the audit stream + reviews table for the Signal
Overview analytics (`DECISIONS.md#039`): per-bucket reviews / cost / findings / failed
series, the findings distribution by severity AND evidence-tier, and period-over-period
totals. Every number traces to an `audit_events` / `reviews` row; sparse windows render
honest ZEROS, never interpolation. Eval rows are excluded by default (`is_eval`), like the
rest of the dashboard. No writes, no new event type — `audit-events-append-only` is
preserved by construction.

Time bucketing forces **UTC** (`date_trunc(field, col, 'UTC')`) regardless of the DB session
timezone, and the granularity follows the window (hourly for `24h`, daily for `7d`/`30d`).

"Findings" means DEDUPED logical findings — distinct `(review_id, finding_content_hash)`,
counted at the FIRST emission, NOT raw `FindingEvent` rows (raw is `n_findings_emitted`,
semantically inflated). The severity/tier distributions read the REPRESENTATIVE
(earliest-emission, `min(sequence_number)`) row per logical finding — a deterministic proxy
for synthesize's kept finding (exact for the common cross-round re-emit case; bounded edge
per the spec's Proof-boundary note, since analyze dedups on `(content_hash, proposal_hash)`).
min-then-filter: the representative is chosen over ALL emissions up to the window END (NOT
lower-bounded by the window start), then filtered by ITS first-emission timestamp — so a
pre-window first emission isn't re-dated by an in-window re-emit. Under the production
invariant (earlier emission = lower sequence_number = earlier timestamp) that representative's
timestamp IS the true minimum; out-of-order/backdated inserts are the spec's bounded edge.

Replay-% is NOT here: it has no stored verdict (computed on demand per review), so it ships
via a sibling replay-verdict-projection feature, not this read-only endpoint.
"""

# See DECISIONS.md#039.
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from sqlalchemy import Numeric, and_, case, cast, func, select

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.audit.events import FindingEvent, LLMCallEvent
from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.reviews import Review
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import FindingSeverity

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement, Subquery
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import InstrumentedAttribute

_WindowParam = Literal["24h", "7d", "30d"]
_WINDOWS: dict[str, timedelta] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}
# Hourly buckets for 24h (a usable sparkline); daily for the longer windows.
_GRANULARITY: dict[str, str] = {"24h": "hour", "7d": "day", "30d": "day"}

# Source the audit discriminators from the event models (matches `audit/aggregates.py`)
# so a discriminator rename can't silently zero this endpoint's cost/findings.
_LLM_CALL: str = LLMCallEvent.model_fields["event_type"].default
_FINDING: str = FindingEvent.model_fields["event_type"].default
_FAILED = "failed"  # a `review_status_enum` value (db/models/_base.py); DB-enforced, stable.


class MetricBucket(BaseModel):
    """One UTC time bucket (hour or day) of honest counts; zero-filled across the window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket: AwareDatetime
    reviews: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    findings: int = Field(ge=0)
    failed: int = Field(ge=0)


class PeriodTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviews: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    findings: int = Field(ge=0)
    failed: int = Field(ge=0)


class MetricDeltas(BaseModel):
    """Current vs the immediately-prior equal-length window (frontend computes the %)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current: PeriodTotals
    previous: PeriodTotals


class DashboardMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    window: _WindowParam
    granularity: str
    generated_at: AwareDatetime
    buckets: tuple[MetricBucket, ...]
    severity_distribution: dict[str, int]
    evidence_tier_distribution: dict[str, int]
    deltas: MetricDeltas


router = APIRouter(
    prefix="/api/metrics",
    tags=["dashboard"],
    dependencies=[Depends(require_admin_api_key)],
)


def _not_eval(column: InstrumentedAttribute[bool], include_eval: bool) -> list[ColumnElement[bool]]:
    """Mirror `list_reviews`: exclude `is_eval=True` rows unless `include_eval`."""
    return [] if include_eval else [column.is_(False)]


def _truncate(ts: datetime, granularity: str) -> datetime:
    """UTC-truncate a timestamp to the bucket boundary — matches the SQL `date_trunc(..,'UTC')`."""
    ts = ts.astimezone(UTC)
    if granularity == "hour":
        return ts.replace(minute=0, second=0, microsecond=0)
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def _bucket_starts(start: datetime, end: datetime, granularity: str) -> list[datetime]:
    """The zero-fill scaffold: every UTC bucket boundary touched by [start, end]."""
    step = timedelta(hours=1) if granularity == "hour" else timedelta(days=1)
    out: list[datetime] = []
    cur = _truncate(start, granularity)
    while cur <= end:
        out.append(cur)
        cur += step
    return out


def _finding_representatives(start: datetime, end: datetime, *, include_eval: bool) -> Subquery:
    """One REPRESENTATIVE row per logical finding whose FIRST emission is in [start, end).

    Carries `first_ts`, `sev`, `tier`. The inner `min(sequence_number)` runs over ALL emissions
    up to `end` (NOT lower-bounded by `start`) so a pre-window first emission is not re-dated by
    an in-window re-emit; the outer keeps only representatives whose first emission is in-window.
    The join on `(review_id, min sequence_number)` selects the earliest-emitted row — its tier
    is the representative tier (see the module docstring for the seq-vs-timestamp bounded edge).
    """
    hash_col = AuditEvent.payload["finding_content_hash"].astext
    rep_seq = (
        select(
            AuditEvent.review_id.label("rid"),
            hash_col.label("h"),
            func.min(AuditEvent.sequence_number).label("seq"),
        )
        .where(
            AuditEvent.event_type == _FINDING,
            AuditEvent.timestamp < end,
            *_not_eval(AuditEvent.is_eval, include_eval),
        )
        .group_by(AuditEvent.review_id, hash_col)
        .subquery()
    )
    return (
        select(
            AuditEvent.timestamp.label("first_ts"),
            AuditEvent.payload["severity"].astext.label("sev"),
            AuditEvent.payload["evidence_tier"].astext.label("tier"),
        )
        .join(
            rep_seq,
            and_(
                AuditEvent.review_id == rep_seq.c.rid,
                AuditEvent.sequence_number == rep_seq.c.seq,
            ),
        )
        .where(AuditEvent.timestamp >= start)
        .subquery()
    )


async def _representatives(
    session: AsyncSession, start: datetime, end: datetime, *, include_eval: bool
) -> list[tuple[datetime, str, str]]:
    """Materialize the deduped logical findings ONCE: `(first_ts, severity, tier)` per finding.

    One query feeds the findings/bucket series + both distributions + the count, instead of
    re-executing the representative scan per aggregation.
    """
    rep = _finding_representatives(start, end, include_eval=include_eval)
    rows = (await session.execute(select(rep.c.first_ts, rep.c.sev, rep.c.tier))).all()
    return [(r.first_ts, r.sev, r.tier) for r in rows]


async def _reviews_failed_by_bucket(
    session: AsyncSession, start: datetime, end: datetime, granularity: str, *, include_eval: bool
) -> dict[datetime, tuple[int, int]]:
    bucket = func.date_trunc(granularity, Review.created_at, "UTC").label("bucket")
    stmt = (
        select(
            bucket,
            func.count().label("reviews"),
            func.coalesce(func.sum(case((Review.status == _FAILED, 1), else_=0)), 0).label(
                "failed"
            ),
        )
        .where(
            Review.created_at >= start,
            Review.created_at < end,
            *_not_eval(Review.is_eval, include_eval),
        )
        .group_by(bucket)
    )
    return {
        r.bucket.astimezone(UTC): (int(r.reviews), int(r.failed))
        for r in (await session.execute(stmt)).all()
    }


async def _cost_by_bucket(
    session: AsyncSession, start: datetime, end: datetime, granularity: str, *, include_eval: bool
) -> dict[datetime, float]:
    bucket = func.date_trunc(granularity, AuditEvent.timestamp, "UTC").label("bucket")
    stmt = (
        select(
            bucket,
            func.coalesce(func.sum(cast(AuditEvent.payload["cost_usd"].astext, Numeric)), 0).label(
                "cost"
            ),
        )
        .where(
            AuditEvent.event_type == _LLM_CALL,
            AuditEvent.timestamp >= start,
            AuditEvent.timestamp < end,
            *_not_eval(AuditEvent.is_eval, include_eval),
        )
        .group_by(bucket)
    )
    return {r.bucket.astimezone(UTC): float(r.cost) for r in (await session.execute(stmt)).all()}


async def _review_cost_totals(
    session: AsyncSession, start: datetime, end: datetime, *, include_eval: bool
) -> tuple[int, int, float]:
    """(reviews, failed, cost) over [start, end) — used for the PREVIOUS window only.

    The current window's totals are derived in Python from the already-fetched buckets + reps,
    so they cannot drift from the rendered series.
    """
    rstmt = select(
        func.count().label("reviews"),
        func.coalesce(func.sum(case((Review.status == _FAILED, 1), else_=0)), 0).label("failed"),
    ).where(
        Review.created_at >= start,
        Review.created_at < end,
        *_not_eval(Review.is_eval, include_eval),
    )
    rrow = (await session.execute(rstmt)).one()
    cstmt = select(
        func.coalesce(func.sum(cast(AuditEvent.payload["cost_usd"].astext, Numeric)), 0)
    ).where(
        AuditEvent.event_type == _LLM_CALL,
        AuditEvent.timestamp >= start,
        AuditEvent.timestamp < end,
        *_not_eval(AuditEvent.is_eval, include_eval),
    )
    cost = float((await session.execute(cstmt)).scalar_one())
    return int(rrow.reviews), int(rrow.failed), cost


@router.get("", response_model=DashboardMetricsResponse)
async def get_metrics(
    request: Request,
    window: Annotated[_WindowParam, Query()] = "7d",
    include_eval: Annotated[bool, Query()] = False,
) -> DashboardMetricsResponse:
    """Honest Signal Overview analytics — see module docstring + `DECISIONS.md#039`."""
    delta = _WINDOWS[window]
    granularity = _GRANULARITY[window]
    now = datetime.now(UTC)
    start = now - delta
    prev_start = start - delta

    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        rf = await _reviews_failed_by_bucket(
            session, start, now, granularity, include_eval=include_eval
        )
        cost = await _cost_by_bucket(session, start, now, granularity, include_eval=include_eval)
        reps = await _representatives(session, start, now, include_eval=include_eval)
        prev_reviews, prev_failed, prev_cost = await _review_cost_totals(
            session, prev_start, start, include_eval=include_eval
        )
        prev_reps = await _representatives(session, prev_start, start, include_eval=include_eval)

    findings_by_bucket: Counter[datetime] = Counter(_truncate(ts, granularity) for ts, _, _ in reps)
    sev_dist: Counter[str] = Counter(sev for _, sev, _ in reps)
    tier_dist: Counter[str] = Counter(tier for _, _, tier in reps)

    buckets = tuple(
        MetricBucket(
            bucket=b,
            reviews=rf.get(b, (0, 0))[0],
            failed=rf.get(b, (0, 0))[1],
            cost_usd=cost.get(b, 0.0),
            findings=findings_by_bucket.get(b, 0),
        )
        for b in _bucket_starts(start, now, granularity)
    )
    # `current` derived from the rendered series (cannot drift from the buckets); only
    # `previous` needs its own queries.
    current = PeriodTotals(
        reviews=sum(b.reviews for b in buckets),
        failed=sum(b.failed for b in buckets),
        cost_usd=sum(b.cost_usd for b in buckets),
        findings=len(reps),
    )
    previous = PeriodTotals(
        reviews=prev_reviews, failed=prev_failed, cost_usd=prev_cost, findings=len(prev_reps)
    )
    return DashboardMetricsResponse(
        window=window,
        granularity=granularity,
        generated_at=now,
        buckets=buckets,
        severity_distribution={s.value: sev_dist.get(s.value, 0) for s in FindingSeverity},
        evidence_tier_distribution={t.value: tier_dist.get(t.value, 0) for t in EvidenceTier},
        deltas=MetricDeltas(current=current, previous=previous),
    )
