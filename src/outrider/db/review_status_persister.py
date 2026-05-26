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
from typing import Any
from uuid import UUID

from pydantic import AwareDatetime
from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.reviews import Review
from outrider.db.sinks import ReviewDecidePreflight
from outrider.policy import FindingSeverity
from outrider.schemas.hitl import HITLDecision, HITLRequest


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

        Predicate admits `awaiting_approval`, `awaiting_approval_expired`,
        and `running` source states (last for idempotent re-fire
        no-op). `expires_at` is intentionally left in place for
        forensic visibility.
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
                            Review.status == "running",
                        ),
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
        avoids constructing an unbounded IN clause.
        """
        if not gated_ids:
            return {}
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
        return severities


__all__ = [
    "ReviewStatusPersister",
    "ReviewStatusPersisterConfigError",
]
