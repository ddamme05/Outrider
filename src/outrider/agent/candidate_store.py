"""Load candidate metadata and expiring model content within one review's scope.

Sessions are opened per operation. Missing retained content stays unavailable;
this reader never reconstructs it from checkpoints or fetches repository files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from outrider.audit.events import (
    CandidateAssessmentEvent,
    CandidateCheckStartedEvent,
    LLMCallEvent,
    NativeCandidateEvent,
)
from outrider.audit.replay import reconstruct_event_from_row
from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.llm_call_content import LLMCallContent
from outrider.db.models.reviews import Review
from outrider.llm.base import LLMResponse
from outrider.policy.canonical import compute_response_hash
from outrider.schemas.native_candidate import candidate_check_contract_for_version

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.audit.events import AuditEvent as AuditEventValue
    from outrider.schemas.native_candidate import (
        CandidateAssessmentRef,
        CandidateCheckStarted,
        NativeCandidateRef,
    )


def _event_from_row(row: AuditEvent) -> AuditEventValue:
    return reconstruct_event_from_row(
        payload=row.payload,
        sequence_number=row.sequence_number,
        event_id=row.event_id,
        review_id=row.review_id,
        event_type=row.event_type,
        timestamp=row.timestamp,
        is_eval=row.is_eval,
        phase_key=row.phase_key,
    )


class CandidateIntegrityError(ValueError):
    """Stored candidate references disagree with their review or source exchange."""


@dataclass(frozen=True)
class RetainedCandidateExchange:
    event: LLMCallEvent
    response: LLMResponse | None = field(repr=False)


@dataclass(frozen=True)
class CandidateAttemptReceipt:
    check: CandidateCheckStarted | None = None
    assessment: CandidateAssessmentRef | None = None
    exchange: RetainedCandidateExchange | None = field(default=None, repr=False)


class CandidateStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def _check_scope(
        self, session: AsyncSession, *, review_id: UUID, installation_id: int, is_eval: bool
    ) -> bool:
        row = (
            await session.execute(
                select(Review.installation_id, Review.is_eval).where(Review.id == review_id)
            )
        ).one_or_none()
        if row is None:
            return False
        if row.installation_id != installation_id or row.is_eval != is_eval:
            raise CandidateIntegrityError("candidate review scope differs from its owner")
        return True

    async def _load_exchange(
        self, session: AsyncSession, *, row: AuditEvent, installation_id: int
    ) -> RetainedCandidateExchange:
        event = _event_from_row(row)
        if not isinstance(event, LLMCallEvent):
            raise CandidateIntegrityError("candidate exchange reference is not a model call")
        content = await session.get(LLMCallContent, event.event_id)
        if content is None:
            return RetainedCandidateExchange(event, None)
        if content.installation_id != installation_id or content.is_eval != event.is_eval:
            raise CandidateIntegrityError("candidate content scope differs from its audit event")
        if content.retention_expires_at <= datetime.now(UTC):
            return RetainedCandidateExchange(event, None)
        response = LLMResponse(
            text=content.completion,
            model=event.model,
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            cache_read_tokens=event.cached_tokens,
            cache_write_tokens=event.cache_write_tokens or 0,
            finish_reason=event.finish_reason or "unknown",
            latency_ms=event.latency_ms,
            profile_id=event.profile_id,
            reasoning_enabled=event.reasoning_enabled,
            profile_contract_digest=event.profile_contract_digest,
            billed_prompt_tokens=event.billed_prompt_tokens,
            service_tier_actual=event.service_tier,
            exchange_event_id=event.event_id,
            exchange_recorded_at=event.timestamp,
        )
        return RetainedCandidateExchange(event, response)

    async def load_source(
        self, ref: NativeCandidateRef, *, installation_id: int, is_eval: bool
    ) -> RetainedCandidateExchange | None:
        async with self._session_factory() as session:
            if not await self._check_scope(
                session, review_id=ref.review_id, installation_id=installation_id, is_eval=is_eval
            ):
                return None
            row = await session.get(AuditEvent, ref.source_exchange_event_id)
            if row is None or row.review_id != ref.review_id or row.is_eval != is_eval:
                raise CandidateIntegrityError(
                    "candidate source event is absent or outside review scope"
                )
            exchange = await self._load_exchange(session, row=row, installation_id=installation_id)
            if exchange.event.node_id != "analyze":
                raise CandidateIntegrityError("candidate source is not an analyzer exchange")
            if exchange.response is not None and (
                compute_response_hash(exchange.response.text) != ref.response_hash
            ):
                raise CandidateIntegrityError("candidate source response hash differs")
            return exchange

    async def load_records(
        self, *, review_id: UUID, installation_id: int, is_eval: bool
    ) -> tuple[
        tuple[NativeCandidateRef, ...],
        tuple[CandidateCheckStarted, ...],
        tuple[CandidateAssessmentRef, ...],
    ]:
        async with self._session_factory() as session:
            if not await self._check_scope(
                session, review_id=review_id, installation_id=installation_id, is_eval=is_eval
            ):
                return (), (), ()
            rows = (
                await session.scalars(
                    select(AuditEvent)
                    .where(
                        AuditEvent.review_id == review_id,
                        AuditEvent.is_eval == is_eval,
                        AuditEvent.event_type.in_(
                            ("native_candidate", "candidate_check_started", "candidate_assessment")
                        ),
                    )
                    .order_by(AuditEvent.sequence_number)
                )
            ).all()
        candidates, checks, assessments = [], [], []
        for row in rows:
            event = _event_from_row(row)
            if isinstance(event, NativeCandidateEvent):
                if event.candidate.review_id != review_id:
                    raise CandidateIntegrityError("candidate metadata differs from its review")
                candidates.append(event.candidate)
            elif isinstance(event, CandidateCheckStartedEvent):
                checks.append(event.check)
            elif isinstance(event, CandidateAssessmentEvent):
                assessments.append(event.assessment)
            else:
                raise CandidateIntegrityError("unexpected candidate metadata event")
        by_candidate = {ref.candidate_id: ref for ref in candidates}
        by_attempt = {check.attempt_id: check for check in checks}
        if len(by_candidate) != len(candidates) or len(by_attempt) != len(checks):
            raise CandidateIntegrityError("candidate metadata has repeated identities")
        for started_check in checks:
            ref = by_candidate.get(started_check.candidate_id)
            if (
                ref is None
                or ref.source_exchange_event_id != started_check.source_exchange_event_id
            ):
                raise CandidateIntegrityError("candidate check differs from its source inventory")
        if len({item.attempt_id for item in assessments}) != len(assessments):
            raise CandidateIntegrityError("candidate attempt has repeated assessments")
        for item in assessments:
            assessment_check = by_attempt.get(item.attempt_id)
            if assessment_check is None or item.candidate_id != assessment_check.candidate_id:
                raise CandidateIntegrityError("candidate assessment has no matching started check")
            if item.outcome == "completed":
                contract = assessment_check.response_contract
                expects_impact = contract in {
                    "head-change-and-impact-v1",
                    "head-change-impact-and-repair-v1",
                }
                if (item.impact_assessment is not None) != expects_impact:
                    raise CandidateIntegrityError(
                        "candidate impact assessment differs from its recorded response contract"
                    )
                expects_repair = contract == "head-change-impact-and-repair-v1"
                if (item.repair_assessment is not None) != expects_repair:
                    raise CandidateIntegrityError(
                        "candidate repair assessment differs from its recorded response contract"
                    )
        return tuple(candidates), tuple(checks), tuple(assessments)

    async def load_attempt(
        self,
        *,
        review_id: UUID,
        installation_id: int,
        candidate_id: str,
        attempt_id: UUID,
        is_eval: bool,
    ) -> CandidateAttemptReceipt:
        candidates, checks, assessments = await self.load_records(
            review_id=review_id, installation_id=installation_id, is_eval=is_eval
        )
        if not any(ref.candidate_id == candidate_id for ref in candidates):
            raise CandidateIntegrityError("candidate attempt has no source inventory record")
        starts = [check for check in checks if check.attempt_id == attempt_id]
        results = [item for item in assessments if item.attempt_id == attempt_id]
        if (
            len(starts) > 1
            or len(results) > 1
            or any(item.candidate_id != candidate_id for item in starts)
            or any(item.candidate_id != candidate_id for item in results)
        ):
            raise CandidateIntegrityError("candidate attempt metadata is ambiguous")
        async with self._session_factory() as session:
            if not await self._check_scope(
                session, review_id=review_id, installation_id=installation_id, is_eval=is_eval
            ):
                return CandidateAttemptReceipt()
            rows = (
                await session.scalars(
                    select(AuditEvent).where(
                        AuditEvent.review_id == review_id,
                        AuditEvent.is_eval == is_eval,
                        AuditEvent.event_type == "llm_call",
                        AuditEvent.payload["candidate_attempt_id"].astext == str(attempt_id),
                    )
                )
            ).all()
            if len(rows) > 1 or ((rows or results) and not starts):
                raise CandidateIntegrityError(
                    "candidate attempt exchange is ambiguous or unstarted"
                )
            exchange = (
                None
                if not rows
                else await self._load_exchange(
                    session, row=rows[0], installation_id=installation_id
                )
            )
        check = starts[0] if starts else None
        if (
            exchange is not None
            and check is not None
            and (
                exchange.event.node_id != "trace"
                or exchange.event.model != check.model
                or exchange.event.prompt_hash != check.prompt_hash
                or check.response_contract
                != candidate_check_contract_for_version(exchange.event.prompt_template_version)
            )
        ):
            raise CandidateIntegrityError(
                "candidate attempt exchange differs from its prepared check"
            )
        if (
            results
            and results[0].outcome == "completed"
            and (
                exchange is None
                or results[0].assessment_exchange_event_id != exchange.event.event_id
            )
        ):
            raise CandidateIntegrityError(
                "candidate assessment differs from its completed exchange"
            )
        return CandidateAttemptReceipt(check, results[0] if results else None, exchange)
