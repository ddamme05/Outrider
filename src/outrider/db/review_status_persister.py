"""Durable implementation of `ReviewStatusSink` + `ReviewStatusReader`.

Single class implementing both Protocols from `db/sinks.py`. Per-method
fresh `AsyncSession` from the injected `async_sessionmaker`. Write methods
run ONE atomic UPDATE statement covering status + JSONB column +
(where applicable) `expires_at`; read method runs two SELECTs in one
session.

Mirrors the `AuditPersister.__init__` precedent for dependency-injection
discipline: `session_factory` is required, `None` raises at construction
(not at first call). Tests that wrap `async_sessionmaker` for
instrumentation remain compatible — runtime flexibility wins over
isinstance gating.
"""

from collections.abc import Mapping
from typing import Any, Final
from uuid import UUID

from pydantic import AwareDatetime
from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.reviews import Review
from outrider.db.sinks import ReviewDecidePreflight
from outrider.policy import FindingSeverity
from outrider.schemas.hitl import HITLDecision, HITLRequest

# Defensive cap on `_load_gated_severities` IN-list length. Mirrors the
# `HITLRequest.findings_requiring_approval` producer-side bound; reaching
# this cap means the producer dropped its own check, which is a fail-loud
# condition not a silent truncation.
_MAX_GATED_FINDING_LOOKUP: Final[int] = 256


class ReviewStatusPersisterConfigError(ValueError):
    """Raised when `ReviewStatusPersister` is constructed without
    `session_factory`. Fail-loud at construction time per the
    `AuditPersister.__init__` precedent.
    """

    def __init__(self) -> None:
        super().__init__(
            "ReviewStatusPersister requires session_factory: "
            "pass an async_sessionmaker[AsyncSession] from "
            "db/session.py at build_graph(...) time."
        )


class ReviewStatusPersister:
    """Durable persister implementing both `ReviewStatusSink` and
    `ReviewStatusReader`.

    Three sink methods (`mark_awaiting_approval`, `mark_running`,
    `mark_awaiting_approval_expired`) — each opens ONE fresh
    `AsyncSession`, runs ONE atomic UPDATE, commits. All return
    successfully on rowcount=0 (idempotency contract).

    One reader method (`fetch_for_decide`) — runs two SELECTs in one
    session: the row state + (conditionally) the gated-finding severity
    map sourced from `FindingEvent.severity` rows in `audit_events`.
    """

    def __init__(self, *, session_factory: async_sessionmaker[AsyncSession]) -> None:
        if session_factory is None:
            raise ReviewStatusPersisterConfigError()
        self._session_factory = session_factory

    async def mark_awaiting_approval(
        self,
        *,
        review_id: UUID,
        expires_at: AwareDatetime,
        hitl_request_payload: dict[str, Any],
    ) -> None:
        """Single-transaction status flip + expires_at + JSONB write.

        Predicate filters on `status='running' AND hitl_request IS NULL`
        — the `hitl_request IS NULL` discriminator is load-bearing for
        first-write-only semantics (see `ReviewStatusSink` Protocol
        docstring).
        """
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(Review)
                .where(
                    and_(
                        Review.id == review_id,
                        Review.status == "running",
                        Review.hitl_request.is_(None),
                    )
                )
                .values(
                    status="awaiting_approval",
                    expires_at=expires_at,
                    hitl_request=hitl_request_payload,
                )
            )

    async def mark_running(
        self,
        *,
        review_id: UUID,
        hitl_decision_payload: dict[str, Any],
    ) -> None:
        """Single-transaction status flip + JSONB write on resume.

        Predicate filters on `status IN ('awaiting_approval',
        'awaiting_approval_expired') AND hitl_decision IS NULL`. The
        `hitl_decision IS NULL` discriminator makes the method
        first-write-only, mirroring `mark_awaiting_approval`'s
        `hitl_request IS NULL` defense: a concurrent second background
        task whose audit-layer emit lost the natural-key race (or whose
        body re-runs after a window (g) crash) sees `hitl_decision`
        already populated and no-ops (rowcount=0). Without this
        discriminator, a divergent-content second decision whose
        `AuditPersisterHITLDecisionNaturalKeyConflict` was caught by
        the endpoint's failure wrapper could STILL overwrite the JSONB
        cache with content the audit row never ratified.

        Source states admitted: `awaiting_approval` (canonical
        transition) + `awaiting_approval_expired` (remediation path per
        spec §4.1.6). `running` is intentionally NOT in the predicate;
        once status moves past awaiting_approval, the row is
        terminal-from-the-HITL-node's-perspective.

        `expires_at` is intentionally left in place for forensic
        visibility — the sweep filter `status='awaiting_approval' AND
        expires_at < NOW()` rules out the row once status moves past
        `awaiting_approval`, so the value's persistence is correctness-
        free and forensically useful.
        """
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(Review)
                .where(
                    and_(
                        Review.id == review_id,
                        or_(
                            Review.status == "awaiting_approval",
                            Review.status == "awaiting_approval_expired",
                        ),
                        Review.hitl_decision.is_(None),
                    )
                )
                .values(
                    status="running",
                    hitl_decision=hitl_decision_payload,
                )
            )

    async def mark_awaiting_approval_expired(self, *, review_id: UUID) -> None:
        """Single-transaction status flip for the sweep job's expiry
        transition.

        Predicate admits `awaiting_approval` (canonical transition) and
        `awaiting_approval_expired` (idempotent re-fire when a sweep
        tick re-processes an already-expired row).
        """
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(Review)
                .where(
                    and_(
                        Review.id == review_id,
                        or_(
                            Review.status == "awaiting_approval",
                            Review.status == "awaiting_approval_expired",
                        ),
                    )
                )
                .values(status="awaiting_approval_expired")
            )

    async def fetch_for_decide(self, *, review_id: UUID) -> ReviewDecidePreflight | None:
        """Two-SELECT preflight read: row state + (conditional)
        gated-finding severity map.

        Returns `None` if the review row does not exist. Empty
        `gated_finding_severities` Mapping when `hitl_request` is NULL
        (no gate, no severity lookup needed).
        """
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(
                        Review.status,
                        Review.hitl_request,
                        Review.hitl_decision,
                    ).where(Review.id == review_id)
                )
            ).one_or_none()
            if row is None:
                return None
            status_str = str(row.status)
            hitl_request = (
                HITLRequest.model_validate(row.hitl_request)
                if row.hitl_request is not None
                else None
            )
            hitl_decision = (
                HITLDecision.model_validate(row.hitl_decision)
                if row.hitl_decision is not None
                else None
            )
            severities: Mapping[UUID, FindingSeverity]
            if hitl_request is None:
                severities = {}
            else:
                severities = await self._load_gated_severities(
                    session=session,
                    review_id=review_id,
                    gated_ids=hitl_request.findings_requiring_approval,
                )
            return ReviewDecidePreflight(
                status=status_str,
                hitl_request=hitl_request,
                hitl_decision=hitl_decision,
                gated_finding_severities=severities,
            )

    async def _load_gated_severities(
        self,
        *,
        session: AsyncSession,
        review_id: UUID,
        gated_ids: tuple[UUID, ...],
    ) -> Mapping[UUID, FindingSeverity]:
        """Sibling SELECT populating finding_id -> severity for the
        gated set.

        Reads `FindingEvent.severity` from `audit_events.payload`
        JSONB, filtered to the gated finding ids via JSONB-key match.
        Returns an empty dict when `gated_ids` is empty so the caller
        avoids constructing an unbounded IN clause. Enforces the
        spec-documented cap of <=256 finding ids defensively at the
        persister boundary — the schema-level cap is at producer side
        (HITLRequest construction), so this guard catches a future
        producer drop without unbounded IN-list growth.
        """
        if not gated_ids:
            return {}
        if len(gated_ids) > _MAX_GATED_FINDING_LOOKUP:
            raise ValueError(
                f"_load_gated_severities: gated_ids length "
                f"{len(gated_ids)} exceeds the {_MAX_GATED_FINDING_LOOKUP} "
                f"cap. The producer-side `HITLRequest.findings_requiring_approval` "
                f"contract bounds the set at 256; reaching this raise means a "
                f"schema-level cap regressed."
            )
        gated_str = [str(fid) for fid in gated_ids]
        stmt = select(
            AuditEvent.payload["finding_id"].astext.label("finding_id"),
            AuditEvent.payload["severity"].astext.label("severity"),
        ).where(
            and_(
                AuditEvent.review_id == review_id,
                AuditEvent.event_type == "finding",
                AuditEvent.payload["finding_id"].astext.in_(gated_str),
            )
        )
        result = await session.execute(stmt)
        severities: dict[UUID, FindingSeverity] = {}
        for row in result.all():
            severities[UUID(row.finding_id)] = FindingSeverity(row.severity)
        # Completeness check: every gated_id MUST have a matching
        # FindingEvent row. A missing id means the
        # `reviews.hitl_request.findings_requiring_approval` JSONB cache
        # diverged from the canonical `audit_events` stream (replay-
        # equivalence guarantee broken) — fail loud here so the
        # downstream `_build_domain_decisions` doesn't KeyError mid-
        # request and produce an opaque 500. The endpoint catches this
        # at the preflight boundary and surfaces a 500 with a clear
        # state-corruption diagnostic, not a stack trace.
        missing = sorted(str(fid) for fid in gated_ids if fid not in severities)
        if missing:
            msg = (
                f"_load_gated_severities: {len(missing)} of {len(gated_ids)} "
                f"gated finding_ids have no matching FindingEvent row for "
                f"review_id={review_id}. Missing: {missing}. State corruption: "
                f"reviews.hitl_request.findings_requiring_approval diverged "
                f"from audit_events; replay-equivalence broken."
            )
            raise ValueError(msg)
        return severities


__all__ = [
    "ReviewStatusPersister",
    "ReviewStatusPersisterConfigError",
]
