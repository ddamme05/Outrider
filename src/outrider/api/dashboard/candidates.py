"""Show original model warnings, draft selection and provisional assessments.

Only metadata survives content expiry. This read projection never qualifies a
warning, changes its category or grants permission to publish it.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID  # noqa: TC003 — route and response field types

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import from_json
from sqlalchemy import select

from outrider.agent.candidate_checks import CandidateText, candidate_text
from outrider.agent.candidate_store import CandidateStore
from outrider.audit.events import AnalyzeCompletedEvent, FindingEvent, SynthesizeCompletedEvent
from outrider.audit.replay import reconstruct_event_from_row
from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.reviews import Review
from outrider.llm.parsing import strip_outer_json_fence
from outrider.schemas.native_candidate import (  # noqa: TC001 — Pydantic response field types
    CandidateAssessmentRef,
    NativeCandidateRef,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

DraftDisposition = Literal[
    "included_in_draft",
    "selected_in_analysis",
    "duplicate",
    "held_by_limit",
    "needs_classification",
    "unqualified",
    "malformed",
    "not_recorded",
]


class NativeCandidateView(NativeCandidateRef):
    """Show the source side explicitly without changing durable reference bytes."""

    source_side: Literal["head", "base"] = Field(default="head")


class CandidateView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: NativeCandidateView
    draft_disposition: DraftDisposition
    original: CandidateText | None
    raw_preview: str | None
    preview_truncated: bool
    content_redacted: bool
    assessment: CandidateAssessmentRef | None
    assessment_reason: str | None
    assessment_content_redacted: bool


class CandidatesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: UUID
    total: int
    limit: int
    offset: int
    candidates: tuple[CandidateView, ...]


def draft_disposition(
    candidate: NativeCandidateRef,
    *,
    selected_ids: tuple[UUID, ...] | None,
    report_complete: bool,
    finding_hashes: dict[UUID, str],
) -> DraftDisposition:
    if candidate.initial_disposition == "malformed":
        return "malformed"
    if candidate.initial_disposition == "rejected":
        return (
            "needs_classification"
            if candidate.rejection_reason == "finding_type_not_in_enum"
            else "unqualified"
        )
    if selected_ids is None:
        return "not_recorded"
    if candidate.finding_id in selected_ids:
        return "included_in_draft" if report_complete else "selected_in_analysis"
    if any(finding_hashes.get(finding_id) == candidate.content_hash for finding_id in selected_ids):
        return "duplicate"
    return "held_by_limit"


async def _selection_records(
    session: AsyncSession, review: Review
) -> tuple[dict[int, tuple[UUID, ...] | None], tuple[UUID, ...] | None, dict[UUID, str]]:
    rows = (
        await session.scalars(
            select(AuditEvent)
            .where(
                AuditEvent.review_id == review.id,
                AuditEvent.is_eval == review.is_eval,
                AuditEvent.event_type.in_(("finding", "analyze_completed", "synthesize_completed")),
            )
            .order_by(AuditEvent.sequence_number)
        )
    ).all()
    rounds: dict[int, tuple[UUID, ...] | None] = {}
    report_ids: tuple[UUID, ...] | None = None
    finding_hashes: dict[UUID, str] = {}
    for row in rows:
        event = reconstruct_event_from_row(
            payload=row.payload,
            sequence_number=row.sequence_number,
            event_id=row.event_id,
            review_id=row.review_id,
            event_type=row.event_type,
            timestamp=row.timestamp,
            is_eval=row.is_eval,
            phase_key=row.phase_key,
        )
        if isinstance(event, AnalyzeCompletedEvent):
            rounds[event.pass_index] = event.selected_finding_ids
        elif isinstance(event, SynthesizeCompletedEvent):
            report_ids = event.selected_finding_ids
        elif isinstance(event, FindingEvent):
            finding_hashes[event.finding_id] = event.finding_content_hash
    return rounds, report_ids, finding_hashes


def _raw_preview(text: str, candidate: NativeCandidateRef) -> tuple[str | None, bool]:
    try:
        envelope = from_json(strip_outer_json_fence(text), allow_partial=False, allow_inf_nan=False)
        rendered = json.dumps(envelope["findings"][candidate.proposal_index], ensure_ascii=False)
    except (ValueError, TypeError, KeyError, IndexError, RecursionError):
        return None, False
    return rendered[:4096], len(rendered) > 4096


def _assessment_reason(text: str, assessment: CandidateAssessmentRef) -> str | None:
    try:
        value = from_json(strip_outer_json_fence(text), allow_partial=False, allow_inf_nan=False)
        reason = value.get("reason") if isinstance(value, dict) else None
    except (ValueError, RecursionError):
        return None
    if (
        not isinstance(reason, str)
        or len(reason) > 4000
        or len(reason) != assessment.reason_length
        or hashlib.sha256(reason.encode()).hexdigest() != assessment.reason_hash
    ):
        return None
    return reason


router = APIRouter()


@router.get("/{review_id}/candidates", response_model=CandidatesResponse)
async def list_candidates(
    review_id: UUID,
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CandidatesResponse:
    """Read one page of original occurrences, including unqualified warnings."""
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        review = await session.get(Review, review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")
        rounds, report_ids, finding_hashes = await _selection_records(session, review)
    store = CandidateStore(session_factory)
    candidates, _checks, assessments = await store.load_records(
        review_id=review.id, installation_id=review.installation_id, is_eval=review.is_eval
    )
    latest = {assessment.candidate_id: assessment for assessment in assessments}
    sources = {}
    views = []
    for candidate in candidates[offset : offset + limit]:
        source_key = (candidate.source_exchange_event_id, candidate.response_hash)
        if source_key not in sources:
            sources[source_key] = await store.load_source(
                candidate, installation_id=review.installation_id, is_eval=review.is_eval
            )
        source = sources[source_key]
        source_text = (
            source.response.text if source is not None and source.response is not None else None
        )
        original = candidate_text(source_text, candidate) if source_text is not None else None
        preview, truncated = (
            _raw_preview(source_text, candidate) if source_text is not None else (None, False)
        )
        assessment = latest.get(candidate.candidate_id)
        reason = None
        if assessment is not None:
            receipt = await store.load_attempt(
                review_id=review.id,
                installation_id=review.installation_id,
                candidate_id=candidate.candidate_id,
                attempt_id=assessment.attempt_id,
                is_eval=review.is_eval,
            )
            if receipt.exchange is not None and receipt.exchange.response is not None:
                reason = _assessment_reason(receipt.exchange.response.text, assessment)
        views.append(
            CandidateView(
                candidate=NativeCandidateView.model_validate(candidate.model_dump()),
                draft_disposition=draft_disposition(
                    candidate,
                    selected_ids=report_ids
                    if report_ids is not None
                    else rounds.get(candidate.pass_index),
                    report_complete=report_ids is not None,
                    finding_hashes=finding_hashes,
                ),
                original=original,
                raw_preview=preview,
                preview_truncated=truncated,
                content_redacted=source_text is None,
                assessment=assessment,
                assessment_reason=reason,
                assessment_content_redacted=(
                    assessment is not None and assessment.outcome == "completed" and reason is None
                ),
            )
        )
    return CandidatesResponse(
        review_id=review.id,
        total=len(candidates),
        limit=limit,
        offset=offset,
        candidates=tuple(views),
    )
