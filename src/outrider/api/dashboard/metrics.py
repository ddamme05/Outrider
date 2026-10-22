"""Read dashboard analytics from reviews and the permanent audit stream.

UTC buckets contain review counts, known model costs, logical finding counts, and
failed reviews (DECISIONS.md#039). Eval rows are excluded by default; joined event
rows must agree with their review's is_eval flag (FUP-130). No writes are made
(audit-events-append-only). Unpriced calls keep cost completeness visible.

Count each (review_id, finding_content_hash) once at its earliest sequence number
before the window end, then filter by that representative's timestamp. A later
re-emission cannot move an older finding into a new window. This represents the
synthesized result for ordinary cross-round repeats; proposal-hash differences
and backdated timestamps remain the spec's bounded edge.

The separate replay route counts only persisted replay verdicts, bucketed by
review completion. Projector-pending reviews are excluded rather than assumed
equivalent. Demo windows anchor to the latest scoped data and expose that anchor.
"""

# See DECISIONS.md#039.
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from sqlalchemy import Boolean, Numeric, and_, case, cast, func, select, type_coerce

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
    # openai-native-host arc: False when this bucket contains unpriceable completed
    # calls (null cost, typed reason) — its cost_usd is then a LOWER BOUND and the
    # chart/tooltip must mark or gap the bucket, never render it as exact.
    cost_complete: bool = True
    findings: int = Field(ge=0)
    failed: int = Field(ge=0)


class PeriodTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviews: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    # Period-level completeness: a delta between two incomplete lower bounds is not
    # a valid lower bound, so the frontend SUPPRESSES the cost delta when either
    # period has cost_complete=False (openai-native-host arc).
    unpriced_calls: int = Field(default=0, ge=0)
    cost_complete: bool = True
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
    # Demo-mode snapshot anchoring (#039-honest): `window_end` is the instant the
    # buckets/deltas computed against — wall clock in production, the seeded data's
    # latest instant on a demo box. `anchored=True` marks the demo case so the
    # frontend labels the window "ending <date>" instead of implying live recency.
    window_end: AwareDatetime
    anchored: bool
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
    # Same demo-snapshot anchoring contract as `DashboardMetricsResponse`.
    window_end: AwareDatetime
    anchored: bool
    buckets: tuple[ReplayBucket, ...]
    deltas: ReplayDeltas


async def _resolve_window_end(
    request: Request, session: AsyncSession, wall_now: datetime, *, include_eval: bool
) -> tuple[datetime, bool]:
    """Use wall_now in production or anchor a demo window to its latest scoped data.

    The response exposes window_end and anchored so the UI cannot imply live recency
    (DECISIONS.md#039). Respect include_eval and event/review equality (FUP-130).
    An empty database falls back to wall_now without an anchor.
    """
    if not bool(getattr(request.app.state, "demo_mode", False)):
        return wall_now, False
    review_clock = (
        select(func.greatest(func.max(Review.created_at), func.max(Review.completed_at)))
        .where(*_not_eval(Review.is_eval, include_eval))
        .scalar_subquery()
    )
    event_clock = (
        select(func.max(AuditEvent.timestamp))
        .select_from(AuditEvent)
        .join(Review, AuditEvent.review_id == Review.id)
        .where(
            AuditEvent.is_eval == Review.is_eval,  # reject is_eval drift (either direction)
            *_not_eval(Review.is_eval, include_eval),
        )
        .scalar_subquery()
    )
    anchor = (
        await session.execute(select(func.greatest(review_clock, event_clock)))
    ).scalar_one_or_none()
    if anchor is None:
        return wall_now, False
    # Every window predicate is half-open ([start, end) — `timestamp < end`), so an
    # end sitting exactly ON the newest row's timestamp would exclude that row.
    # Nudge past it by one second; nothing is newer than the anchor by construction.
    return anchor + timedelta(seconds=1), True


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
    bucket_starts: list[datetime] = []
    bucket_start = _truncate(start, granularity)
    while bucket_start <= end:
        bucket_starts.append(bucket_start)
        bucket_start += step
    return bucket_starts


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
    representative_query = _finding_representatives(start, end, include_eval=include_eval)
    finding_rows = (
        await session.execute(
            select(
                representative_query.c.first_ts,
                representative_query.c.sev,
                representative_query.c.tier,
            )
        )
    ).all()
    return [
        (finding_row.first_ts, finding_row.sev, finding_row.tier) for finding_row in finding_rows
    ]


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
) -> dict[datetime, tuple[float, int]]:
    """Per bucket: (known cost subtotal, unpriced call count). SUM skips the null
    costs of unpriceable calls; the count keeps the gap visible per bucket."""
    bucket = func.date_trunc(granularity, AuditEvent.timestamp, "UTC").label("bucket")
    # is_eval via the joined REVIEW with the FUP-130 equality predicate — see
    # `_finding_representatives`. The reviews/failed series is reviews-table-only (no event join).
    stmt = (
        select(
            bucket,
            func.coalesce(func.sum(cast(AuditEvent.payload["cost_usd"].astext, Numeric)), 0).label(
                "cost"
            ),
            func.coalesce(
                func.sum(case((AuditEvent.payload["cost_usd"].astext.is_(None), 1), else_=0)), 0
            ).label("unpriced"),
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
    return {
        r.bucket.astimezone(UTC): (float(r.cost), int(r.unpriced))
        for r in (await session.execute(stmt)).all()
    }


async def _review_cost_totals(
    session: AsyncSession, start: datetime, end: datetime, *, include_eval: bool
) -> tuple[int, int, float, int]:
    """(reviews, failed, cost, unpriced_calls) over [start, end) — PREVIOUS window only.

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
        select(
            func.coalesce(func.sum(cast(AuditEvent.payload["cost_usd"].astext, Numeric)), 0),
            func.coalesce(
                func.sum(case((AuditEvent.payload["cost_usd"].astext.is_(None), 1), else_=0)), 0
            ),
        )
        .join(Review, Review.id == AuditEvent.review_id)
        .where(
            AuditEvent.event_type == _LLM_CALL,
            AuditEvent.timestamp >= start,
            AuditEvent.timestamp < end,
            AuditEvent.is_eval == Review.is_eval,  # FUP-130 equality: reject is_eval drift
            *_not_eval(Review.is_eval, include_eval),
        )
    )
    crow = (await session.execute(cstmt)).one()
    return int(rrow.reviews), int(rrow.failed), float(crow[0]), int(crow[1])


@router.get("", response_model=DashboardMetricsResponse)
async def get_metrics(
    request: Request,
    window: Annotated[_WindowParam, Query()] = "7d",
    include_eval: Annotated[bool, Query()] = False,
) -> DashboardMetricsResponse:
    """Honest Signal Overview analytics — see module docstring + `DECISIONS.md#039`."""
    window_duration = _WINDOWS[window]
    granularity = _GRANULARITY[window]
    generated_at = datetime.now(UTC)

    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        end, anchored = await _resolve_window_end(
            request, session, generated_at, include_eval=include_eval
        )
        start = end - window_duration
        previous_start = start - window_duration
        review_counts_by_bucket = await _reviews_failed_by_bucket(
            session, start, end, granularity, include_eval=include_eval
        )
        costs_by_bucket = await _cost_by_bucket(
            session, start, end, granularity, include_eval=include_eval
        )
        finding_representatives = await _representatives(
            session, start, end, include_eval=include_eval
        )
        (
            previous_reviews,
            previous_failed,
            previous_cost,
            previous_unpriced,
        ) = await _review_cost_totals(session, previous_start, start, include_eval=include_eval)
        previous_findings = await _representatives(
            session, previous_start, start, include_eval=include_eval
        )

    findings_by_bucket: Counter[datetime] = Counter(
        _truncate(first_emission_at, granularity)
        for first_emission_at, _, _ in finding_representatives
    )
    severity_distribution: Counter[str] = Counter(
        severity for _, severity, _ in finding_representatives
    )
    evidence_tier_distribution: Counter[str] = Counter(
        tier for _, _, tier in finding_representatives
    )

    buckets = tuple(
        MetricBucket(
            bucket=bucket_start,
            reviews=review_counts_by_bucket.get(bucket_start, (0, 0))[0],
            failed=review_counts_by_bucket.get(bucket_start, (0, 0))[1],
            cost_usd=costs_by_bucket.get(bucket_start, (0.0, 0))[0],
            cost_complete=costs_by_bucket.get(bucket_start, (0.0, 0))[1] == 0,
            findings=findings_by_bucket.get(bucket_start, 0),
        )
        for bucket_start in _bucket_starts(start, end, granularity)
    )
    # `current` derived from the rendered series (cannot drift from the buckets); only
    # `previous` needs its own queries.
    current_unpriced = sum(unpriced for _, unpriced in costs_by_bucket.values())
    current = PeriodTotals(
        reviews=sum(bucket.reviews for bucket in buckets),
        failed=sum(bucket.failed for bucket in buckets),
        cost_usd=sum(bucket.cost_usd for bucket in buckets),
        unpriced_calls=current_unpriced,
        cost_complete=current_unpriced == 0,
        findings=len(finding_representatives),
    )
    previous = PeriodTotals(
        reviews=previous_reviews,
        failed=previous_failed,
        cost_usd=previous_cost,
        unpriced_calls=previous_unpriced,
        cost_complete=previous_unpriced == 0,
        findings=len(previous_findings),
    )
    return DashboardMetricsResponse(
        window=window,
        granularity=granularity,
        generated_at=generated_at,
        window_end=end,
        anchored=anchored,
        buckets=buckets,
        severity_distribution={
            s.value: severity_distribution.get(s.value, 0) for s in FindingSeverity
        },
        evidence_tier_distribution={
            t.value: evidence_tier_distribution.get(t.value, 0) for t in EvidenceTier
        },
        deltas=MetricDeltas(current=current, previous=previous),
    )


def _replay_equivalent_predicate() -> ColumnElement[bool]:
    """`replay_verdict` payload's `replay_equivalent` JSON bool as a SQL boolean predicate.

    Compares the JSONB `->>` text (`.astext`) to `'true'` rather than `cast(... AS boolean)`:
    Postgres `CAST(text AS boolean)` RAISES on any non-recognized literal, so one malformed
    payload value would abort the whole aggregate and 500 the endpoint for the entire window.
    The text comparison degrades a malformed value to `false` (counts inequivalent) instead of
    erroring. `type_coerce` types the boolean expression for the caller without emitting a SQL
    CAST (no runtime parse). One verdict per review (partial unique index), so counting verdict
    rows IS counting reviewed-and-verdicted reviews.
    """
    return type_coerce(AuditEvent.payload["replay_equivalent"].astext == "true", Boolean)


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

    **`include_eval` is production-scoped (inert today).** The background projector verdicts
    PRODUCTION reviews only (`sweep/replay_verdict.py`, docs/testing.md), so no `replay_verdict`
    row is ever `is_eval=True` and `include_eval=true` returns the same production rate — the
    param is retained for `/api/metrics` symmetry + forward-compatibility (if eval projection is
    ever added) and still threads the FUP-130 is_eval-drift equality defense either way. The
    Overview Replay card therefore does NOT pass the global eval toggle (it is always production).
    """
    window_duration = _WINDOWS[window]
    granularity = _GRANULARITY[window]
    generated_at = datetime.now(UTC)

    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        end, anchored = await _resolve_window_end(
            request, session, generated_at, include_eval=include_eval
        )
        start = end - window_duration
        previous_start = start - window_duration
        verdict_counts_by_bucket = await _replay_by_bucket(
            session, start, end, granularity, include_eval=include_eval
        )
        previous_equivalent, previous_total = await _replay_totals(
            session, previous_start, start, include_eval=include_eval
        )

    buckets = tuple(
        ReplayBucket(
            bucket=bucket_start,
            equivalent=verdict_counts_by_bucket.get(bucket_start, (0, 0))[0],
            total=verdict_counts_by_bucket.get(bucket_start, (0, 0))[1],
        )
        for bucket_start in _bucket_starts(start, end, granularity)
    )
    current = ReplayPeriodTotals(
        equivalent=sum(bucket.equivalent for bucket in buckets),
        total=sum(bucket.total for bucket in buckets),
    )
    previous = ReplayPeriodTotals(equivalent=previous_equivalent, total=previous_total)
    return ReplayMetricsResponse(
        window=window,
        granularity=granularity,
        generated_at=generated_at,
        window_end=end,
        anchored=anchored,
        buckets=buckets,
        deltas=ReplayDeltas(current=current, previous=previous),
    )
