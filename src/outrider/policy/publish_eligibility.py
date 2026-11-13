"""Decide whether a finding is approved for publication.

Destination routing is separate. High, critical and verification-held findings
need approval tied to the request; overrides must match that decision.
Missing or inconsistent approval data cannot grant permission to post.
The explicit severity table must cover every severity, so a new value cannot
silently bypass approval.

See DECISIONS.md#023 and specs/2026-05-26-hitl-node.md.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from outrider.audit.events import PublishEligibility, PublishEligibilityReason
from outrider.policy.severity import FindingSeverity
from outrider.schemas.hitl import PerFindingOutcome

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from outrider.policy import FindingType
    from outrider.schemas.hitl import HITLDecision, HITLRequest, PerFindingDecision
    from outrider.schemas.native_candidate import CandidateAssessmentRef, NativeCandidateRef
    from outrider.schemas.review_finding import ReviewFinding


# ---------------------------------------------------------------------------
# Eligibility mapping, consulted after override validation. It says which
# severities require HITL (CRITICAL/HIGH), independently of decision lookup.
# ---------------------------------------------------------------------------


class _V1SeverityBaseline(StrEnum):
    """V1 baseline gate outcome per severity, pre-HITL-decision-check.

    Internal — the public surface is the `is_eligible_for_v1_publish`
    function which converts this into the public `(PublishEligibility,
    PublishEligibilityReason | None)` tuple. CRITICAL/HIGH severities
    require a HITL decision to materialize. MEDIUM/LOW/INFO pass this baseline;
    a separate verification hold can still require approval.
    """

    ELIGIBLE = "eligible"
    REQUIRES_HITL_DECISION = "requires_hitl_decision"


_V1_SEVERITY_GATE: Final[MappingProxyType[FindingSeverity, _V1SeverityBaseline]] = MappingProxyType(
    {
        FindingSeverity.CRITICAL: _V1SeverityBaseline.REQUIRES_HITL_DECISION,
        FindingSeverity.HIGH: _V1SeverityBaseline.REQUIRES_HITL_DECISION,
        FindingSeverity.MEDIUM: _V1SeverityBaseline.ELIGIBLE,
        FindingSeverity.LOW: _V1SeverityBaseline.ELIGIBLE,
        FindingSeverity.INFO: _V1SeverityBaseline.ELIGIBLE,
    }
)


def _assert_mapping_total_at_import() -> None:
    """Pin the mapping's totality over `FindingSeverity` at import time.

    See module docstring for the exhaustive-coverage rationale. A new
    `FindingSeverity` member that lands without an entry in
    `_V1_SEVERITY_GATE` crashes the import rather than silently treating
    the new severity as ineligible.
    """
    missing = set(FindingSeverity) - set(_V1_SEVERITY_GATE.keys())
    extra = set(_V1_SEVERITY_GATE.keys()) - set(FindingSeverity)
    if missing or extra:
        raise RuntimeError(
            f"_V1_SEVERITY_GATE mapping must be total over FindingSeverity. "
            f"Missing: {sorted(s.value for s in missing)!r}. "
            f"Extra: {sorted(s.value for s in extra)!r}. "
            f"Add an entry to publish_eligibility.py:_V1_SEVERITY_GATE "
            f"AND verify the V1 policy intent — a new severity is a "
            f"deliberate decision, not a defaultable case."
        )


_assert_mapping_total_at_import()


def is_hitl_gated_severity(severity: FindingSeverity) -> bool:
    """True iff `severity` requires a HITL decision under the V1 gate.

    Derived from `_V1_SEVERITY_GATE` so the schema-side override
    legitimacy validators (`ReviewFinding._enforce_override_triplet_
    coherence`, `PublishEligibilityEvent._enforce_override_legitimacy`)
    share the gated-set definition with the runtime gate function
    `is_eligible_for_v1_publish`. A baseline severity that is NOT in
    the gated set could not have produced a legitimate
    `SEVERITY_OVERRIDE` decision, because HITL only fires for
    CRITICAL/HIGH; a non-None `original_severity ∉ gated set` is
    therefore a producer bug or replay-injected forge.
    """
    return _V1_SEVERITY_GATE.get(severity) is _V1SeverityBaseline.REQUIRES_HITL_DECISION


# ---------------------------------------------------------------------------
# Public gate function.
# ---------------------------------------------------------------------------


def _find_decision_for(
    *, finding_id: object, hitl_decision: HITLDecision | None
) -> PerFindingDecision | None:
    """Return the `PerFindingDecision` matching `finding_id`, or None.

    Bounded by the decision tuple size (<=256 per `HITLDecisionPayload`
    schema cap); linear scan is fine.
    """
    if hitl_decision is None:
        return None
    for d in hitl_decision.decisions:
        if d.finding_id == finding_id:
            return d
    return None


# The outcome partition the publish gate applies: APPROVE and SEVERITY_OVERRIDE admit
# the finding, REJECT and SUPPRESS withhold it. Defined once and used by both
# `is_eligible_for_v1_publish` and `count_gated_approvals`, so the reviewer's reported
# verdict cannot drift from the rule that decides what actually publishes.
# `test_admitting_outcomes_matches_eligibility_gate` drives every enum member through the
# real gate and pins the two together, so adding a member to one without the other fails.
HITL_ADMITTING_OUTCOMES: Final[frozenset[PerFindingOutcome]] = frozenset(
    {PerFindingOutcome.APPROVE, PerFindingOutcome.SEVERITY_OVERRIDE}
)


def verification_hold_finding_ids(
    findings: Sequence[ReviewFinding],
    native_candidates: Sequence[NativeCandidateRef],
    assessments: Sequence[CandidateAssessmentRef],
    *,
    policy_version: str,
) -> tuple[UUID, ...]:
    """Require human review of disputed claims and removed-source findings.

    Any incomplete, disputed or stale attempt keeps the hold. Another model
    assertion cannot clear counterevidence; the existing human decision can.
    """
    by_candidate: dict[str, list[CandidateAssessmentRef]] = {}
    for assessment in assessments:
        by_candidate.setdefault(assessment.candidate_id, []).append(assessment)
    disputed: set[tuple[str, FindingType | None, FindingSeverity | None]] = set()
    for candidate in native_candidates:
        if candidate.initial_disposition != "admitted" or candidate.content_hash is None:
            continue
        for assessment in by_candidate.get(candidate.candidate_id, ()):
            if (
                candidate.policy_version != policy_version
                or assessment.policy_version != policy_version
                or assessment.outcome != "completed"
                or assessment.status != "supports"
                or (
                    assessment.suggested_finding_type is not None
                    and assessment.suggested_finding_type != candidate.finding_type
                )
            ):
                disputed.add((candidate.content_hash, candidate.finding_type, candidate.severity))
    return tuple(
        sorted(
            finding.finding_id
            for finding in findings
            if finding.source_side == "base"
            or (
                disputed
                and (finding.content_hash, finding.finding_type, finding.severity) in disputed
            )
        )
    )


def is_eligible_for_v1_publish(
    finding: ReviewFinding,
    *,
    hitl_request: HITLRequest | None,
    hitl_decision: HITLDecision | None,
    verification_hold: bool = False,
) -> tuple[PublishEligibility, PublishEligibilityReason | None]:
    """Return eligibility and a withholding reason before publish routing.

    Find the matching decision, then reject unsupported override fields first:
    the request must gate this finding, and a SEVERITY_OVERRIDE decision must
    match its baseline and applied severity. This blocks forged downgrades
    before the effective severity can pass the baseline gate.

    MEDIUM/LOW/INFO pass unless verification-held. Held findings and CRITICAL/HIGH
    require a HITL request, a matching decision, and request membership. APPROVE and
    SEVERITY_OVERRIDE admit; REJECT and SUPPRESS withhold. Missing request or
    decision gets its distinct withholding reason. ELIGIBLE permits routing;
    WITHHELD prevents GitHub materialization."""
    matching_decision = _find_decision_for(
        finding_id=finding.finding_id, hitl_decision=hitl_decision
    )
    verification_hold = verification_hold or finding.source_side == "base"

    # Validate a claimed override before effective severity can admit it.
    # Request membership blocks invented approvals; exact severity matching
    # prevents reusing a real approval for a different downgrade.
    if finding.original_severity is not None:
        has_legit_override = (
            matching_decision is not None
            and matching_decision.outcome == PerFindingOutcome.SEVERITY_OVERRIDE
            and hitl_request is not None
            and finding.finding_id in hitl_request.findings_requiring_approval
            and matching_decision.original_severity == finding.original_severity
            and matching_decision.override_severity == finding.severity
        )
        if not has_legit_override:
            return (
                PublishEligibility.WITHHELD,
                PublishEligibilityReason.UNEXPECTED_OVERRIDE_FIELDS_PRESENT,
            )

    requested_hold = (
        hitl_request is not None
        and finding.finding_id in hitl_request.verification_hold_finding_ids
    )
    if verification_hold and not requested_hold:
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.HITL_REQUIRED_NODE_ABSENT,
        )
    if (
        requested_hold
        and not is_hitl_gated_severity(finding.original_severity or finding.severity)
        and matching_decision is not None
        and matching_decision.outcome == PerFindingOutcome.SEVERITY_OVERRIDE
    ):
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.UNEXPECTED_OVERRIDE_FIELDS_PRESENT,
        )

    # Severity and recorded verification holds share the approval gate.
    baseline = _V1_SEVERITY_GATE[finding.severity]
    if baseline is _V1SeverityBaseline.ELIGIBLE and not requested_hold:
        return (PublishEligibility.ELIGIBLE, None)

    # Approval path: consult the saved HITL request and decision.
    if hitl_request is None:
        # Wiring bypass — HITL never ran. Defense-in-depth (the graph
        # routes analyze/trace -> hitl unconditionally per Group 5).
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.HITL_REQUIRED_NODE_ABSENT,
        )

    if hitl_decision is None:
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.HITL_DECISION_MISSING,
        )

    if matching_decision is None:
        # Decision landed but no entry for this finding_id (defense-in-
        # depth: the endpoint's mismatch check should have rejected
        # this before resume).
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.HITL_DECISION_MISSING,
        )

    # Recheck request membership even though /decide validates set equality.
    # A decision outside the canonical request cannot satisfy
    # hitl-gates-high-severity; treat it as missing.
    if finding.finding_id not in hitl_request.findings_requiring_approval:
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.HITL_DECISION_MISSING,
        )

    if matching_decision.outcome in HITL_ADMITTING_OUTCOMES:
        return (PublishEligibility.ELIGIBLE, None)
    if matching_decision.outcome == PerFindingOutcome.REJECT:
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.HITL_REJECTED,
        )
    if matching_decision.outcome == PerFindingOutcome.SUPPRESS:
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.HITL_SUPPRESSED,
        )
    # Exhaustive enum coverage — the four PerFindingOutcome members are
    # all branched above. A new member that lands without a branch here
    # fails loudly via the unreachable raise rather than silently
    # falling through to ELIGIBLE or WITHHELD.
    raise RuntimeError(
        f"is_eligible_for_v1_publish: unhandled PerFindingOutcome "
        f"{matching_decision.outcome!r}; add a branch above when a new "
        f"outcome lands in PerFindingOutcome."
    )


def count_gated_approvals(
    *,
    gated_finding_ids: Sequence[UUID],
    hitl_decision: HITLDecision | None,
) -> tuple[int, int]:
    """`(approved, total)` over the GATED finding set.

    The reviewer's verdict is NOT derivable from publish routing counts: those span
    auto-post (non-gated) findings too, so a review that rejected every gated finding
    can still show `posted_count > 0`, and an approved finding routed DASHBOARD_ONLY
    contributes zero. Callers that need to say "approved" or "dismissed" ask here.

    A gated finding with no matching decision counts as not-approved (fail-closed,
    matching the gate's HITL_DECISION_MISSING withholding).
    """
    approved = sum(
        1
        for fid in gated_finding_ids
        if (d := _find_decision_for(finding_id=fid, hitl_decision=hitl_decision)) is not None
        and d.outcome in HITL_ADMITTING_OUTCOMES
    )
    return approved, len(gated_finding_ids)
