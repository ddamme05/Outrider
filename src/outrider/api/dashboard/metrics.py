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

Replay-% ships as a SIBLING route in this module (`GET /api/metrics/replay`,
`get_replay_metrics`): it reads the PERSISTED `replay_verdict` events appended by the
background projector (`sweep/replay_verdict.py`, `DECISIONS.md#039`) — equivalent / total
bucketed by `reviews.completed_at`, with ONLY reviews carrying a persisted verdict in the
denominator (a still-running or projector-pending review is excluded, never assumed
equivalent, so projector lag does not distort the chart). The main `/api/metrics` response
above does NOT carry Replay-%: the verdict is a separate append surfaced by its own route.
"""

# See DECISIONS.md#039.
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from sqlalchemy import Boolean, Numeric, and_, case, cast, func, select

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.audit.events import FindingEvent, LLMCallEvent, ReplayVerdictEvent
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
_VERDICT: str = ReplayVerdictEvent.model_fields["event_type"].default
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


class ReplayBucket(BaseModel):
    """One UTC bucket of replay-verdict counts. Raw `equivalent` / `total` (not a precomputed
    rate): callers derive the %, and `total == 0` has no defined rate (no divide-by-zero)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket: AwareDatetime
    equivalent: int = Field(ge=0)
    total: int = Field(ge=0)


class ReplayPeriodTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    equivalent: int = Field(ge=0)
    total: int = Field(ge=0)


class ReplayDeltas(BaseModel):
    """Current vs the immediately-prior equal-length window (frontend computes the % change)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current: ReplayPeriodTotals
    previous: ReplayPeriodTotals


class ReplayMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    window: _WindowParam
    granularity: str
    generated_at: AwareDatetime
    buckets: tuple[ReplayBucket, ...]
    deltas: ReplayDeltas


router = APIRouter(
    prefix="/api/metrics",
    tags=["dashboard"],
    dependencies=[Depends(require_admin_api_key)],
)


def _not_eval(column: InstrumentedAttribute[bool], include_eval: bool) -> list[ColumnElement[bool]]:
    """Mirror `list_reviews`: exclude `is_eval=True` rows unless `include_eval`."""
    return [] if include_eval else [column.is_(False)]


def _truncate(ts: datetime, granularity: str) -> datetime:
    """UTC-truncate a timestamp to the bucket boundary — matches the SQL `date_trunc(..,'UTC')`.

    Fail-loud on an unknown granularity: this Python truncation must agree with the SQL
    `date_trunc` that buckets reviews/cost. A new granularity (e.g. `week`) that SQL understands
    but this helper silently defaulted to day would desync the findings series — so raise rather
    than guess. The full single-truncation-authority refactor (all-SQL bucketing) is a follow-up.
    """
    ts = ts.astimezone(UTC)
    if granularity == "hour":
        return ts.replace(minute=0, second=0, microsecond=0)
    if granularity == "day":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"unsupported granularity for truncation: {granularity!r}")


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
    # is_eval scoped via the JOINED review, matching the FUP-130 read-side EQUALITY rule
    # (`reviews.py`: `AuditEvent.is_eval == review_is_eval`) + the replay consistency check
    # (`_verify_is_eval_consistent`): an event must AGREE with its review's is_eval, so a drift
    # row is rejected in BOTH directions (eval review w/ prod-labeled event AND prod review w/
    # eval-labeled event), and production scope uses the review's authoritative is_eval. Identical
    # under producer homogeneity; the predicate is the read-side defense the rest of the dashboard
    # carries (a one-sided `Review.is_eval` filter alone would leak the prod-review->eval case).
    rep_seq = (
        select(
            AuditEvent.review_id.label("rid"),
            hash_col.label("h"),
            func.min(AuditEvent.sequence_number).label("seq"),
        )
        .join(Review, Review.id == AuditEvent.review_id)
        .where(
            AuditEvent.event_type == _FINDING,
            AuditEvent.timestamp < end,
            AuditEvent.is_eval == Review.is_eval,  # reject is_eval drift (either direction)
            *_not_eval(Review.is_eval, include_eval),
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
    # is_eval via the joined REVIEW with the FUP-130 equality predicate — see
    # `_finding_representatives`. The reviews/failed series is reviews-table-only (no event join).
    stmt = (
        select(
            bucket,
            func.coalesce(func.sum(cast(AuditEvent.payload["cost_usd"].astext, Numeric)), 0).label(
                "cost"
            ),
        )
        .join(Review, Review.id == AuditEvent.review_id)
        .where(
            AuditEvent.event_type == _LLM_CALL,
            AuditEvent.timestamp >= start,
            AuditEvent.timestamp < end,
            AuditEvent.is_eval == Review.is_eval,  # reject is_eval drift (either direction)
            *_not_eval(Review.is_eval, include_eval),
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
    cstmt = (
        select(func.coalesce(func.sum(cast(AuditEvent.payload["cost_usd"].astext, Numeric)), 0))
        .join(Review, Review.id == AuditEvent.review_id)
        .where(
            AuditEvent.event_type == _LLM_CALL,
            AuditEvent.timestamp >= start,
            AuditEvent.timestamp < end,
            AuditEvent.is_eval == Review.is_eval,  # FUP-130 equality: reject is_eval drift
            *_not_eval(Review.is_eval, include_eval),
        )
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


def _replay_equivalent_predicate() -> ColumnElement[bool]:
    """`replay_verdict` payload's `replay_equivalent` JSON bool as a SQL boolean predicate.

    JSONB `->>` (`.astext`) renders the JSON boolean as the text `'true'`/`'false'`; casting
    that text to `Boolean` yields a real SQL boolean — the `cast(payload[...].astext, <type>)`
    idiom the rest of this module uses (`cost_usd` → `Numeric`). One verdict per review (partial
    unique index), so counting verdict rows IS counting reviewed-and-verdicted reviews.
    """
    return cast(AuditEvent.payload["replay_equivalent"].astext, Boolean)


async def _replay_by_bucket(
    session: AsyncSession, start: datetime, end: datetime, granularity: str, *, include_eval: bool
) -> dict[datetime, tuple[int, int]]:
    """`{bucket: (equivalent, total)}` over persisted verdicts, bucketed by `reviews.completed_at`.

    Bucketing on the REVIEW's completion (not the verdict event's timestamp) keeps the chart's
    time axis aligned to when work completed, so projector lag never shifts a point. is_eval via
    the joined review with the FUP-130 equality predicate (see `_cost_by_bucket`).
    """
    bucket = func.date_trunc(granularity, Review.completed_at, "UTC").label("bucket")
    stmt = (
        select(
            bucket,
            func.count().label("total"),
            func.coalesce(func.sum(case((_replay_equivalent_predicate(), 1), else_=0)), 0).label(
                "equivalent"
            ),
        )
        .join(Review, Review.id == AuditEvent.review_id)
        .where(
            AuditEvent.event_type == _VERDICT,
            Review.completed_at >= start,
            Review.completed_at < end,
            AuditEvent.is_eval == Review.is_eval,  # reject is_eval drift (either direction)
            *_not_eval(Review.is_eval, include_eval),
        )
        .group_by(bucket)
    )
    return {
        r.bucket.astimezone(UTC): (int(r.equivalent), int(r.total))
        for r in (await session.execute(stmt)).all()
    }


async def _replay_totals(
    session: AsyncSession, start: datetime, end: datetime, *, include_eval: bool
) -> tuple[int, int]:
    """`(equivalent, total)` over [start, end) — used for the PREVIOUS window only.

    The current window's totals are summed in Python from the rendered buckets, so they cannot
    drift from the series (mirrors `_review_cost_totals`).
    """
    stmt = (
        select(
            func.count().label("total"),
            func.coalesce(func.sum(case((_replay_equivalent_predicate(), 1), else_=0)), 0).label(
                "equivalent"
            ),
        )
        .join(Review, Review.id == AuditEvent.review_id)
        .where(
            AuditEvent.event_type == _VERDICT,
            Review.completed_at >= start,
            Review.completed_at < end,
            AuditEvent.is_eval == Review.is_eval,  # FUP-130 equality: reject is_eval drift
            *_not_eval(Review.is_eval, include_eval),
        )
    )
    row = (await session.execute(stmt)).one()
    return int(row.equivalent), int(row.total)


@router.get("/replay", response_model=ReplayMetricsResponse)
async def get_replay_metrics(
    request: Request,
    window: Annotated[_WindowParam, Query()] = "7d",
    include_eval: Annotated[bool, Query()] = False,
) -> ReplayMetricsResponse:
    """Replay-equivalence rate over time, from persisted verdicts — see module docstring + `#039`.

    `equivalent / total` per bucket (frontend derives the %); only reviews with a persisted
    `replay_verdict` are counted, so a projector-pending review is excluded, never assumed
    equivalent.
    """
    delta = _WINDOWS[window]
    granularity = _GRANULARITY[window]
    now = datetime.now(UTC)
    start = now - delta
    prev_start = start - delta

    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        by_bucket = await _replay_by_bucket(
            session, start, now, granularity, include_eval=include_eval
        )
        prev_equivalent, prev_total = await _replay_totals(
            session, prev_start, start, include_eval=include_eval
        )

    buckets = tuple(
        ReplayBucket(
            bucket=b,
            equivalent=by_bucket.get(b, (0, 0))[0],
            total=by_bucket.get(b, (0, 0))[1],
        )
        for b in _bucket_starts(start, now, granularity)
    )
    current = ReplayPeriodTotals(
        equivalent=sum(b.equivalent for b in buckets),
        total=sum(b.total for b in buckets),
    )
    previous = ReplayPeriodTotals(equivalent=prev_equivalent, total=prev_total)
    return ReplayMetricsResponse(
        window=window,
        granularity=granularity,
        generated_at=now,
        buckets=buckets,
        deltas=ReplayDeltas(current=current, previous=previous),
    )
