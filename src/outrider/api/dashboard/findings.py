"""Load finding details for the dashboard and agent APIs.

`assemble_finding_views` combines stored content with posting decisions and
human decisions. When content has been removed, return the recorded finding
type, severity, location and decisions. Finding and review-event queries keep
production and evaluation records separate.

Replay reads decisions from its saved, verified events in `reviews.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict
from uuid import UUID  # noqa: TC003  (runtime: Pydantic field type)

from pydantic import AwareDatetime, BaseModel, ConfigDict
from sqlalchemy import select

from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.findings import Finding
from outrider.db.models.purge_audit import PurgeAudit

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

# Retention's global TTL sweep uses installation ID 0. Installation-specific
# purge deletes the review too; a surviving review's redacted content normally
# comes from the global sweep. Sweep lookups accept both IDs.
GLOBAL_SWEEP_INSTALLATION_ID = 0


class HITLDecisionView(BaseModel):
    """One finding's HITL decision, projected from the canonical audit stream.

    Per DECISIONS.md#034 the per-review `HITLDecisionEvent` is the single
    canonical record of a reviewer's override; the `findings`-table override
    columns (`original_severity` / `override_reason` / `overrider_id`) are
    read-model projections, NULL in V1 (no post-HITL findings writer). This
    view reads the stream by `finding_id`, never the table — the same
    stream-canonical sourcing `publish_destination` already uses for
    `PublishRoutingEvent`.

    Present-or-absent as a unit: a finding the reviewer never decided on (not
    crit/high, so never HITL-gated; or HITL not yet reached) has
    `hitl_decision=None` on its `FindingView`, not a half-populated object.
    `original_severity` / `override_severity` are non-null ONLY when
    `outcome == "severity_override"` (the schema-enforced override contract,
    `schemas/hitl.py::PerFindingDecision.enforce_override_fields`); the other
    three outcomes (`approve` / `reject` / `suppress`) carry neither.
    `reviewer_id` is the event-level reviewer (a string, `"admin"` in V1 per
    DECISIONS.md#011) surfaced per-finding for the dashboard's convenience.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: str
    reviewer_id: str
    reason: str
    original_severity: str | None
    override_severity: str | None


class FindingView(BaseModel):
    """One finding, assembled from the permanent audit record + content.

    Metadata (type/severity/file/line/dimension/tier/proof) comes from the
    `FindingEvent` audit row (permanent). Content (`title`/`description`/
    `evidence`/`suggested_fix`) comes from the `findings` table — present
    within the retention window, `None` with `content_redacted=True` once the
    row is purged but the `FindingEvent` survives (DECISIONS.md#014 point 3:
    "render a dangling finding_id as content redacted per retention policy").

    Lifecycle fields joined from the audit stream (per DECISIONS.md#023's
    routing≠eligibility split): `publish_destination` from `PublishRoutingEvent`
    (where coordinates classified it); `eligibility` / `eligibility_reason`
    from `PublishEligibilityEvent` (whether it actually materialized). A
    high/critical finding pre-HITL shows `inline_comment` + `withheld` +
    `hitl_required_node_absent` — routed but not posted. All three are `None`
    until publish runs.

    HITL override-provenance (`hitl_decision`) is joined from the same audit
    stream per DECISIONS.md#034 — the per-review `HITLDecisionEvent` indexed by
    `finding_id`. `None` when the reviewer rendered no decision on this finding
    (not crit/high, or HITL not yet reached). It survives content redaction the
    same way the publish lifecycle does: the provenance lives in the append-only
    stream, not the retention-purged `findings` row. See `HITLDecisionView`.
    """

    model_config = ConfigDict(extra="forbid")

    finding_id: UUID
    finding_type: str
    dimension: str
    severity: str
    evidence_tier: str
    file_path: str
    line_start: int
    line_end: int
    # Content — None on a retention-redacted stub (findings row purged).
    content_redacted: bool
    title: str | None
    description: str | None
    evidence: str | None
    suggested_fix: str | None
    query_match_id: str | None
    trace_path: list[str] | None
    # Publish lifecycle (routing ≠ eligibility, DECISIONS.md#023).
    publish_destination: str | None
    eligibility: str | None
    eligibility_reason: str | None
    # HITL override-provenance, projected from the canonical HITLDecisionEvent
    # stream (DECISIONS.md#034). None when the reviewer never decided this
    # finding.
    hitl_decision: HITLDecisionView | None
    # Retention: for a `content_redacted` stub, the findings-retention-SWEEP
    # date (latest `target_table='findings'` `purge_audit` row for the global
    # TTL-sweep sentinel or this review's installation); `None` otherwise. The
    # sweep timestamp, NOT a proven per-finding delete time — `purge_audit` is
    # per-table-per-sweep, so exact per-finding provenance is out of reach
    # (FUP-129). Frontend renders "content redacted in the findings retention
    # sweep on <date>".
    redaction_sweep_at: AwareDatetime | None


class FindingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: UUID
    findings: list[FindingView]


class _FindingLifecycleFields(TypedDict):
    """Audit-derived fields shared by full and retention-redacted findings."""

    publish_destination: str | None
    eligibility: str | None
    eligibility_reason: str | None
    hitl_decision: HITLDecisionView | None


async def assemble_finding_views(
    session: AsyncSession,
    *,
    review_id: UUID,
    installation_id: int,
    review_is_eval: bool,
) -> list[FindingView]:
    """Assemble a review's findings from the permanent audit record + content rows.

    Shared by `GET /{id}/findings` and the agent-view endpoint (`api/dashboard/
    agent_view.py`). Surviving `findings` rows render as full findings; a
    `FindingEvent` whose `findings` row was purged under retention renders as a
    `content_redacted=True` stub from `FindingEvent` metadata (DECISIONS.md#014
    point 3 — the audit trail outlives the content). Publish lifecycle
    (`publish_destination` / `eligibility` / `eligibility_reason`, DECISIONS.md#023)
    and HITL override-provenance (`hitl_decision`, DECISIONS.md#034) are joined from
    the audit stream, never the (V1-null) `findings` columns, so they survive
    content redaction. Apply the module's per-review eval isolation rule to each read.
    """
    destinations_by_finding = await _latest_publish_routing(session, review_id, review_is_eval)
    eligibility_by_finding = await _latest_publish_eligibility(session, review_id, review_is_eval)
    decisions_by_finding = await _hitl_decisions(session, review_id, review_is_eval)

    def lifecycle_for_finding(finding_id: str) -> _FindingLifecycleFields:
        eligibility, reason = eligibility_by_finding.get(finding_id, (None, None))
        return {
            "publish_destination": destinations_by_finding.get(finding_id),
            "eligibility": eligibility,
            "eligibility_reason": reason,
            "hitl_decision": decisions_by_finding.get(finding_id),
        }

    # Read surviving content under the review's eval flag, even with write-side checks.
    content_rows = (
        (
            await session.execute(
                select(Finding).where(
                    Finding.review_id == review_id,
                    Finding.is_eval == review_is_eval,
                )
            )
        )
        .scalars()
        .all()
    )
    content_finding_ids = {str(finding.finding_id) for finding in content_rows}
    views: list[FindingView] = []
    for content_finding in content_rows:
        views.append(
            FindingView(
                finding_id=content_finding.finding_id,
                finding_type=content_finding.finding_type,
                dimension=content_finding.dimension,
                severity=content_finding.severity,
                evidence_tier=content_finding.evidence_tier,
                file_path=content_finding.file_path,
                line_start=content_finding.line_start,
                line_end=content_finding.line_end,
                content_redacted=False,
                title=content_finding.title,
                description=content_finding.description,
                evidence=content_finding.evidence,
                suggested_fix=content_finding.suggested_fix,
                query_match_id=content_finding.query_match_id,
                trace_path=content_finding.trace_path,
                redaction_sweep_at=None,
                **lifecycle_for_finding(str(content_finding.finding_id)),
            )
        )

    # Dangling FindingEvents (content row purged, event survives) ->
    # retention-redacted stubs. Dedup by finding_id, keep latest.
    event_payloads = (
        (
            await session.execute(
                select(AuditEvent.payload)
                .where(
                    AuditEvent.review_id == review_id,
                    AuditEvent.is_eval == review_is_eval,
                    AuditEvent.event_type == "finding",
                )
                .order_by(AuditEvent.sequence_number.asc())
            )
        )
        .scalars()
        .all()
    )
    redacted_by_finding: dict[str, dict[str, Any]] = {
        payload["finding_id"]: payload
        for payload in event_payloads
        if payload["finding_id"] not in content_finding_ids
    }
    # All redacted stubs share the latest recorded sweep date. It describes the
    # table's sweep, not a proven per-finding deletion time (FUP-129).
    redaction_sweep_at = None
    if redacted_by_finding:
        redaction_sweep_at = await latest_sweep_at(
            session, "findings", (installation_id, GLOBAL_SWEEP_INSTALLATION_ID)
        )
    views.extend(
        FindingView(
            finding_id=UUID(finding_id),
            finding_type=metadata["finding_type"],
            dimension=metadata["dimension"],
            severity=metadata["severity"],
            evidence_tier=metadata["evidence_tier"],
            file_path=metadata["file_path"],
            line_start=metadata["line_start"],
            line_end=metadata["line_end"],
            content_redacted=True,
            title=None,
            description=None,
            evidence=None,
            suggested_fix=None,
            query_match_id=metadata.get("query_match_id"),
            trace_path=metadata.get("trace_path"),
            redaction_sweep_at=redaction_sweep_at,
            **lifecycle_for_finding(finding_id),
        )
        for finding_id, metadata in redacted_by_finding.items()
    )
    views.sort(key=lambda view: (view.file_path, view.line_start, str(view.finding_id)))
    return views


async def _latest_publish_routing(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> dict[str, str | None]:
    """Map findings to their latest routing destination, in sequence order."""
    rows = (
        await session.execute(
            select(
                AuditEvent.payload["finding_id"].astext,
                AuditEvent.payload["destination"].astext,
            )
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.is_eval == review_is_eval,
                AuditEvent.event_type == "publish_routing",
            )
            .order_by(AuditEvent.sequence_number.asc())
        )
    ).all()
    return {fid: destination for fid, destination in rows if fid is not None}


async def _latest_publish_eligibility(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> dict[str, tuple[str | None, str | None]]:
    """Map findings to their latest (eligibility, reason) decision.

    Routing does not imply eligibility (DECISIONS.md#023).
    """
    rows = (
        await session.execute(
            select(
                AuditEvent.payload["finding_id"].astext,
                AuditEvent.payload["eligibility"].astext,
                AuditEvent.payload["reason"].astext,
            )
            .where(
                AuditEvent.review_id == review_id,
                AuditEvent.is_eval == review_is_eval,
                AuditEvent.event_type == "publish_eligibility",
            )
            .order_by(AuditEvent.sequence_number.asc())
        )
    ).all()
    return {fid: (eligibility, reason) for fid, eligibility, reason in rows if fid is not None}


async def _hitl_decisions(
    session: AsyncSession, review_id: UUID, review_is_eval: bool
) -> dict[str, HITLDecisionView]:
    """Project per-finding decisions from the review's HITLDecisionEvent.

    The audit stream owns override provenance (DECISIONS.md#034). A database
    natural key allows at most one decision event; no event returns an empty
    dict. Copy the event's reviewer ID onto each decision view.
    """
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
        return {}
    reviewer_id = payload["reviewer_id"]
    decisions: dict[str, HITLDecisionView] = {}
    for decision in payload["decisions"]:
        fid = decision["finding_id"]
        decisions[fid] = HITLDecisionView(
            outcome=decision["outcome"],
            reviewer_id=reviewer_id,
            reason=decision["reason"],
            original_severity=decision.get("original_severity"),
            override_severity=decision.get("override_severity"),
        )
    return decisions


async def latest_sweep_at(
    session: AsyncSession,
    target_table: str,
    installation_ids: tuple[int, ...],
) -> datetime | None:
    """Latest `purge_audit` sweep timestamp for `target_table`, scoped to the given
    installation ids (the review's installation + the global TTL-sweep sentinel `0`).

    The per-table-per-sweep date — NOT a proven per-item delete time (FUP-129); not
    eval-scoped (`PurgeAudit` has no `is_eval` column). `None` if no sweep is recorded.
    """
    return (
        await session.execute(
            select(PurgeAudit.timestamp)
            .where(
                PurgeAudit.installation_id.in_(installation_ids),
                PurgeAudit.target_table == target_table,
            )
            .order_by(PurgeAudit.timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
