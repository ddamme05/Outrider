# Agent-view endpoint per ROADMAP.md section 3 / S2 + FUP-154.
"""Authenticated, read-only JSON for agents following one review.

The router uses a separate agent token; it cannot authorize dashboard decisions.
It projects stored review, finding, decision and publication data without writing
new audit events. Audit queries stay scoped to the review's `is_eval` value.

V1 omits per-comment URLs and suggested fixes. Publication records a GitHub review
ID, not individual comment URLs. Analyze persists findings before synthesis adds
patches to graph state, so this endpoint does not project those generated patches.
`schema_version` identifies the response contract (ROADMAP §3 / S2, FUP-154).
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
from outrider.db.models.installations import InstallationRepository, active_repo_membership
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from outrider.api.dashboard.reviews import FindingView


class AgentReviewerDecision(BaseModel):
    """A finding's recorded human decision and severity-override provenance.

    Projected from HITLDecisionEvent (DECISIONS.md#034); V1 reviewer_id is "admin"
    (DECISIONS.md#011).
    Original and override severities are present only for severity overrides.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: str
    reviewer_id: str
    reason: str
    decided_at: AwareDatetime
    original_severity: str | None
    override_severity: str | None


class AgentFindingView(BaseModel):
    """An admitted finding with effective severity and its recorded decision.

    Severity comes from policy or a human override; original_severity preserves
    the policy value. Evidence tier comes from the validated finding record.
    Title and description may come from query metadata or a model, and can be
    absent after retention. V1 omits per-comment URLs and suggested fixes.
    """

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
    """A recorded GitHub review and its posted comment count."""

    model_config = ConfigDict(extra="forbid")

    github_review_id: int
    review_status: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"]
    comments_posted: int


class AgentReviewView(BaseModel):
    """Stored review state for agents; schema_version identifies the contract."""

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
    """Build the PR URL from active repository membership, or return None.

    Missing membership is a normal state, not malformed audit data.
    """
    repo_full_name = (
        await session.execute(
            select(InstallationRepository.repo_full_name).where(
                active_repo_membership(installation_id, repo_id)
            )
        )
    ).scalar_one_or_none()
    if repo_full_name is None:
        return None
    return f"https://github.com/{repo_full_name}/pull/{pr_number}"


async def _latest_publish_event(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> AgentPublishEvent | None:
    """Read the latest recorded GitHub publication in this review's eval scope.

    No event returns None, even if the graph completed without posting.
    """
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
    # Missing keys indicate corrupt validated audit data. Do not hide them as
    # absent publication, unlike the normal missing-membership case in _pr_url.
    return AgentPublishEvent(
        github_review_id=payload["github_review_id"],
        review_status=payload["review_status"],
        comments_posted=payload["comments_posted"],
    )


async def _hitl_decided_at(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> AwareDatetime | None:
    """Read the latest decision timestamp, shared by its per-finding decisions."""
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
    # Validated audit timestamps are stored as aware ISO strings in JSONB.
    raw = payload["decided_at"]
    return datetime.fromisoformat(raw) if raw is not None else None


def _to_agent_finding(
    fv: FindingView, *, gated_fids: set[str], decided_at: AwareDatetime | None
) -> AgentFindingView:
    """Project gate history and any timestamped decision onto an admitted finding.

    A severity override becomes the effective value; the decision retains the
    original policy severity. Gate membership comes from the saved HITL request.
    """
    decision = fv.hitl_decision
    effective_severity = fv.severity
    reviewer_decision = None
    if decision is not None and decided_at is not None:
        if decision.outcome == "severity_override" and decision.override_severity is not None:
            effective_severity = decision.override_severity
        reviewer_decision = AgentReviewerDecision(
            outcome=decision.outcome,
            reviewer_id=decision.reviewer_id,
            reason=decision.reason,
            decided_at=decided_at,
            original_severity=decision.original_severity,
            override_severity=decision.override_severity,
        )
    return AgentFindingView(
        finding_id=fv.finding_id,
        finding_type=fv.finding_type,
        severity=effective_severity,
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
    """Read an authenticated review projection; return 404 for an unknown ID.

    Shared finding assembly supplies content/proof and retention behavior.
    Audit reads use the stored review's production/eval scope.
    """
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
