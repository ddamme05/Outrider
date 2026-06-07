# Agent-view endpoint per ROADMAP.md section 3 / S2 + FUP-154.
"""GET /reviews/{review_id}/agent-view — read-only structured review for AI agents.

The unforgeable channel (feature 3 / S2): an AI agent reads trust-critical fields
(severity, evidence tier, HITL decision) from this authenticated JSON endpoint
instead of grepping the forgeable comment prose (the S1/S1.5 markers; FUP-154).

Mounted on its OWN router gated by `require_agent_api_key` — a SEPARATE read-only
token from the admin key, so an agent never holds the key that can `POST /decide`.
Read-only: a pure projection of existing state (the `reviews` row, FindingEvent /
Finding content, HITLDecisionEvent, PublishEvent, InstallationRepository); it emits
no audit event.

Every audit read is `is_eval`-scoped (FUP-130). Unstored fields are OMITTED from the
V1 contract, not faked: `github_comment_url` (Outrider records only the review-level
`github_review_id`) and `suggested_patch` (ROADMAP feature 2). `schema_version` lets
agents branch on future contract revisions.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID  # noqa: TC003  (runtime: route field type)

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import AwareDatetime, BaseModel, ConfigDict
from sqlalchemy import select

from outrider.api.dashboard.auth import require_agent_api_key
from outrider.api.dashboard.reviews import _assemble_finding_views
from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.installations import InstallationRepository
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from outrider.api.dashboard.reviews import FindingView


class AgentReviewerDecision(BaseModel):
    """One finding's HITL decision, for an agent. Projected from the canonical
    HITLDecisionEvent (DECISIONS.md#034); `reviewer_id` is `"admin"` in V1
    (DECISIONS.md#011). Present only when the finding was decided (gated + reached)."""

    model_config = ConfigDict(extra="forbid")

    outcome: str
    reviewer_id: str
    reason: str
    decided_at: AwareDatetime


class AgentFindingView(BaseModel):
    """One finding in the agent shape. `severity` / `evidence_tier` are the
    already-decided values (never model output). `title` / `description` are `None`
    on a retention-redacted stub. `github_comment_url` + `suggested_patch` are
    omitted in V1 (not stored / feature-2)."""

    model_config = ConfigDict(extra="forbid")

    finding_id: UUID
    finding_type: str
    severity: str
    file_path: str
    line_start: int
    line_end: int
    evidence_tier: str
    title: str | None
    description: str | None
    hitl_gated: bool
    reviewer_decision: AgentReviewerDecision | None


class AgentPublishEvent(BaseModel):
    """The review's GitHub publish outcome. `None` when publish never ran
    (zero-eligible / no-op)."""

    model_config = ConfigDict(extra="forbid")

    github_review_id: int
    review_status: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"]
    comments_posted: int


class AgentReviewView(BaseModel):
    """The structured agent-view of a review (ROADMAP §3 / S2). `schema_version` is
    the versioned contract handle so agents can branch on future shape changes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    review_id: UUID
    pr_url: str | None
    status: str
    policy_version: str | None
    findings: list[AgentFindingView]
    publish_event: AgentPublishEvent | None


router = APIRouter(
    prefix="/api/reviews",
    tags=["agent"],
    dependencies=[Depends(require_agent_api_key)],
)


async def _pr_url(
    session: AsyncSession, *, installation_id: int, repo_id: int, pr_number: int
) -> str | None:
    """`https://github.com/{repo_full_name}/pull/{pr_number}` from the active
    InstallationRepository membership. `None` when the membership row is absent or
    removed (a repo rename/removal) — never 500 the whole view for a missing join."""
    repo_full_name = (
        await session.execute(
            select(InstallationRepository.repo_full_name).where(
                InstallationRepository.installation_id == installation_id,
                InstallationRepository.repo_id == repo_id,
                InstallationRepository.removed_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if repo_full_name is None:
        return None
    return f"https://github.com/{repo_full_name}/pull/{pr_number}"


async def _latest_publish_event(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> AgentPublishEvent | None:
    """The review's latest PublishEvent (github_review_id / review_status /
    comments_posted), `is_eval`-scoped (FUP-130). `None` when publish never ran."""
    payload = (
        await session.execute(
            select(AuditEvent.payload)
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.is_eval == review_is_eval,
                AuditEvent.event_type == "publish",
            )
            .order_by(AuditEvent.sequence_number.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if payload is None:
        return None
    return AgentPublishEvent(
        github_review_id=payload["github_review_id"],
        review_status=payload["review_status"],
        comments_posted=payload["comments_posted"],
    )


async def _hitl_decided_at(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> AwareDatetime | None:
    """The review's single HITLDecisionEvent `decided_at` (event-level — shared by
    every per-finding decision), `is_eval`-scoped. `None` when no decision exists."""
    payload = (
        await session.execute(
            select(AuditEvent.payload)
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.is_eval == review_is_eval,
                AuditEvent.event_type == "hitl_decision",
            )
            .order_by(AuditEvent.sequence_number.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if payload is None:
        return None
    # Stored as a tz-aware ISO string in the JSONB payload (HITLDecisionEvent.
    # decided_at is AwareDatetime). Parse to a real datetime so the typed return
    # holds; AgentReviewerDecision.decided_at (AwareDatetime) accepts it.
    raw = payload["decided_at"]
    return datetime.fromisoformat(raw) if raw is not None else None


def _to_agent_finding(
    fv: FindingView, *, gated_fids: set[str], decided_at: AwareDatetime | None
) -> AgentFindingView:
    """Map a `FindingView` (the shared assembly) to the agent shape. `hitl_gated` =
    the finding is in the persisted gated set; `reviewer_decision` is present only
    when the finding carries a HITL decision (gated + decided), with the event-level
    `decided_at`."""
    reviewer_decision = None
    if fv.hitl_decision is not None and decided_at is not None:
        reviewer_decision = AgentReviewerDecision(
            outcome=fv.hitl_decision.outcome,
            reviewer_id=fv.hitl_decision.reviewer_id,
            reason=fv.hitl_decision.reason,
            decided_at=decided_at,
        )
    return AgentFindingView(
        finding_id=fv.finding_id,
        finding_type=fv.finding_type,
        severity=fv.severity,
        file_path=fv.file_path,
        line_start=fv.line_start,
        line_end=fv.line_end,
        evidence_tier=fv.evidence_tier,
        title=fv.title,
        description=fv.description,
        hitl_gated=str(fv.finding_id) in gated_fids,
        reviewer_decision=reviewer_decision,
    )


@router.get("/{review_id}/agent-view", response_model=AgentReviewView)
async def get_agent_view(request: Request, review_id: UUID) -> AgentReviewView:
    """Structured, read-only review for an AI agent. 404 if the review is absent
    (holding the id suffices, like `get_review`). Reuses `_assemble_finding_views`
    then maps each finding to the agent shape. Every audit read is `is_eval`-scoped."""
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        review = (
            await session.execute(select(Review).where(Review.id == review_id))
        ).scalar_one_or_none()
        if review is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="review not found")

        # Policy-version snapshot (earliest policy-bearing event), `is_eval`-scoped —
        # mirror of get_review (DECISIONS.md#028 / FUP-130).
        policy_version = (
            await session.execute(
                select(AuditEvent.payload["policy_version"].astext)
                .where(
                    AuditEvent.review_id == review.id,
                    AuditEvent.is_eval == review.is_eval,
                    AuditEvent.payload["policy_version"].astext.isnot(None),
                )
                .order_by(AuditEvent.sequence_number.asc())
                .limit(1)
            )
        ).scalar_one_or_none()

        # Authoritative gated set from the persisted HITL request snapshot (FUP-134);
        # stored ids are JSON strings.
        hitl_request = review.hitl_request
        gated_fids = (
            {str(fid) for fid in (hitl_request.get("findings_requiring_approval") or [])}
            if hitl_request is not None
            else set()
        )

        finding_views = await _assemble_finding_views(
            session,
            review_id=review.id,
            installation_id=review.installation_id,
            review_is_eval=review.is_eval,
        )
        decided_at = await _hitl_decided_at(session, review.id, review.is_eval)
        publish_event = await _latest_publish_event(session, review.id, review.is_eval)
        pr_url = await _pr_url(
            session,
            installation_id=review.installation_id,
            repo_id=review.repo_id,
            pr_number=review.pr_number,
        )

        agent_findings = [
            _to_agent_finding(fv, gated_fids=gated_fids, decided_at=decided_at)
            for fv in finding_views
        ]
        return AgentReviewView(
            review_id=review.id,
            pr_url=pr_url,
            status=review.status,
            policy_version=policy_version,
            findings=agent_findings,
            publish_event=publish_event,
        )
