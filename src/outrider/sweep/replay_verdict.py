# See DECISIONS.md#039 — the sibling verdict-projection of the read-only metrics endpoint.
"""Background replay-verdict projector.

Computes each completed review's replay-equivalence verdict ONCE, off the hot
path, and appends it as a `ReplayVerdictEvent` so the dashboard Replay-% can
aggregate a persisted verdict instead of reconstructing per 2s poll
(`DECISIONS.md#039`, the cost the read-only `/api/metrics` endpoint ruled out).

Sweep-family placement: it reads `reviews WHERE status='completed'` on a cadence
and appends an audit row — the sweep job profile — but unlike hitl-expiry it flips
no status, so it needs no `SWEEP_LOCK_ID` advisory lock: its only side effect is
the natural-key-idempotent verdict INSERT (the anomaly two-caller-class precedent —
a lock is load-bearing only for a non-idempotent status flip).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError
from sqlalchemy import func, select

from outrider.audit.events import ReplayVerdictEvent
from outrider.audit.replay import AuditReplayer, ReplayEquivalenceError, ReplayError
from outrider.db.models.audit_events import AuditEvent as AuditEventRow
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.audit.persister import AuditPersister

logger = logging.getLogger(__name__)

_VERDICT_EVENT_TYPE: str = ReplayVerdictEvent.model_fields["event_type"].default


async def project_replay_verdicts(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    audit_persister: AuditPersister,
) -> dict[str, int]:
    """Project a replay verdict for each completed PRODUCTION review lacking one.

    Candidates: `status='completed'`, `is_eval=False` (the sweep contract,
    docs/testing.md — eval reviews are not projected, so eval Replay-% is
    production-scoped), and no `replay_verdict` event yet (an anti-join
    optimization; correctness holds without it since `emit_replay_verdict` is
    idempotent). Per review: target = `max(sequence_number)` over the NON-verdict
    rows (so a re-projection excludes any prior verdict), reconstruct over that
    prefix + assert_equivalent, build the verdict, emit. Per-row try/except so one
    bad review never aborts the tick. Returns `{"projected": N, "failed": M}`.
    """
    replayer = AuditReplayer(session_factory=session_factory)
    projected = 0
    failed = 0
    for review_id, is_eval in await _candidate_reviews(session_factory):
        try:
            verdict = await _compute_verdict(replayer, session_factory, review_id, is_eval=is_eval)
            # Count only a FRESH insert: under concurrent ticks both can select the
            # same candidate, but only one INSERT wins (the rest no-op) — so the
            # telemetry counts real projections, not no-op re-emits.
            if await audit_persister.emit_replay_verdict(verdict):
                projected += 1
        except Exception:
            logger.exception(
                "replay_verdict_projection_failed", extra={"review_id": str(review_id)}
            )
            failed += 1
    return {"projected": projected, "failed": failed}


async def _candidate_reviews(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[tuple[UUID, bool]]:
    verdict_exists = (
        select(AuditEventRow.review_id)
        .where(
            AuditEventRow.review_id == Review.id,
            AuditEventRow.event_type == _VERDICT_EVENT_TYPE,
        )
        .exists()
    )
    stmt = select(Review.id, Review.is_eval).where(
        Review.status == "completed",
        Review.is_eval.is_(False),
        ~verdict_exists,
    )
    async with session_factory() as session:
        rows = (await session.execute(stmt)).all()
    return [(row.id, row.is_eval) for row in rows]


async def _compute_verdict(
    replayer: AuditReplayer,
    session_factory: async_sessionmaker[AsyncSession],
    review_id: UUID,
    *,
    is_eval: bool,
) -> ReplayVerdictEvent:
    """Reconstruct over the judged prefix + verify; build the verdict.

    Reconstruction failure (corrupt row / is_eval drift / not found) → inequivalent
    with an ABSENT envelope (couldn't reconstruct). An `assert_equivalent` failure →
    inequivalent with the FULL envelope (reconstruction succeeded). Success →
    equivalent with the full envelope.
    """
    target = await _max_non_verdict_sequence(session_factory, review_id)
    try:
        review = await replayer.reconstruct(review_id, max_sequence_number=target)
    except (ReplayError, ValidationError) as exc:
        return ReplayVerdictEvent(
            review_id=review_id,
            replay_equivalent=False,
            reason=_reconstruct_failure_reason(exc),
            target_max_sequence_number=target,
            is_eval=is_eval,
        )
    try:
        await replayer.assert_equivalent(review)
        replay_equivalent = True
        reason: str | None = None
    except ReplayEquivalenceError as exc:
        replay_equivalent = False
        # Fallback guards the `reason`-required-iff-inequivalent contract against a
        # hypothetical empty exception message (all current raise sites carry one).
        reason = str(exc).strip()[:500] or "replay inequivalent (no detail available)"
    return ReplayVerdictEvent(
        review_id=review_id,
        replay_equivalent=replay_equivalent,
        mode=review.mode.value,
        event_count=len(review.events),
        finding_count=len(review.findings),
        orphan_finding_count=len(review.orphan_finding_ids),
        reason=reason,
        target_max_sequence_number=target,
        is_eval=is_eval,
    )


def _reconstruct_failure_reason(exc: ReplayError | ValidationError) -> str:
    """A metadata-only reason for a reconstruct failure.

    A `pydantic.ValidationError` (a corrupt audit row) is SANITIZED to its error
    LOCATIONS + CODES — never `str(exc)`, whose message can echo the offending
    `input` value, i.e. raw payload content, into this metadata-only verdict event
    (`DECISIONS.md#014`/`#016`: audit events carry no content). `ReplayError` messages
    are already metadata (field names, ids, sequence numbers).
    """
    if isinstance(exc, ValidationError):
        # ONLY the error `type` slugs (fixed programmatic identifiers like
        # "missing"/"int_parsing"/"extra_forbidden") — NOT `loc` (which for an
        # `extra_forbidden` error carries the unexpected payload KEY), nor `input`/`msg`
        # (which echo the offending VALUE). Fully content-free per DECISIONS#014/#016.
        codes = "; ".join(
            err["type"]
            for err in exc.errors(include_url=False, include_input=False, include_context=False)
        )
        return f"reconstruct failed: ValidationError ({exc.error_count()}): {codes}"[:500]
    return f"reconstruct failed: {type(exc).__name__}: {exc}"[:500]


async def _max_non_verdict_sequence(
    session_factory: async_sessionmaker[AsyncSession], review_id: UUID
) -> int:
    """Judged-prefix high-water mark: `max(sequence_number)` over the review's
    NON-verdict rows. A completed review always has graph events, so this is never
    None; raise loudly if it somehow is (a completed review with no stream is a data
    anomaly, not a verdict to fabricate)."""
    async with session_factory() as session:
        result = await session.scalar(
            select(func.max(AuditEventRow.sequence_number)).where(
                AuditEventRow.review_id == review_id,
                AuditEventRow.event_type != _VERDICT_EVENT_TYPE,
            )
        )
    if result is None:
        raise ReplayError(
            f"completed review {review_id} has no non-verdict audit events; "
            f"cannot project a replay verdict"
        )
    return int(result)


__all__ = ["project_replay_verdicts"]
