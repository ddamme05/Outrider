# V1 publish eligibility gate per specs/2026-05-21-publish-node.md Q3 + the
# HITL-aware extension per specs/2026-05-26-hitl-node.md Group 6.
"""is_eligible_for_v1_publish — policy-derived publish-materialization gate.

Per DECISIONS.md #023 (publish routing and eligibility are separate
decisions, not one combined gate): this module is the **policy** half
of the V1 fabricated-override defense. The **schema** half lives at
`PublishEligibilityEvent._enforce_v1_no_overrides` in `audit/events.py`;
both must hold for the DECISIONS #023 "schema + gate" trust story.

The gate fires BEFORE materialization — the publish node calls
`is_eligible_for_v1_publish(finding, hitl_request=..., hitl_decision=...)`
per finding, and routes the finding to `publisher.create_review(...)`
only when the return is `("eligible", None)`. Withholding outcomes get
recorded in `PublishEligibilityEvent(eligibility=withheld, reason=<reason>)`
and the finding does NOT reach GitHub.

The HITL context comes through as explicit kwargs (NOT read from any
module-level state) so the gate stays a pure function over its inputs:

  - `hitl_request: HITLRequest | None` — the gate envelope the HITL
    node emitted (None means HITL didn't run for this review).
  - `hitl_decision: HITLDecision | None` — the reviewer's decision set
    after resume (None means HITL ran but no decision landed yet,
    OR HITL never ran).

Withholding reasons (post-HITL extension):

- `hitl_required_node_absent` — finding severity is `CRITICAL` or
  `HIGH` and the HITL node did not run for this review (request +
  decision are both None). Defense-in-depth: the graph wiring routes
  through HITL post-analyze/trace, so reaching this branch indicates
  a wiring bypass.

- `hitl_decision_missing` — severity ∈ {CRITICAL, HIGH}, HITL request
  landed (request is not None) but decision is None OR no matching
  `PerFindingDecision` for this `finding_id` in the submitted set
  (defense-in-depth against an endpoint mismatch check that missed
  something).

- `hitl_rejected` — severity ∈ {CRITICAL, HIGH}, reviewer's outcome
  for this finding was REJECT.

- `hitl_suppressed` — severity ∈ {CRITICAL, HIGH}, reviewer's outcome
  for this finding was SUPPRESS.

- `unexpected_override_fields_present` — finding carries a non-None
  `original_severity` despite no matching `SEVERITY_OVERRIDE` decision
  in the HITL decision set. Indicates either a producer bug or
  replay-injected state forging a pre-approved downgrade. Defended at
  this gate AND at the audit-row schema layer; the gate fires FIRST
  so no GitHub call ever happens for a forged-override finding.

The mapping is a `MappingProxyType` keyed by `FindingSeverity` — set
membership is FORBIDDEN per the spec's "implementation discipline"
clause. The mapping is total over `FindingSeverity` at import time
(`_assert_mapping_total_at_import` runs as a module-level assertion);
any new severity that lands without an entry crashes the import.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from outrider.audit.events import PublishEligibility, PublishEligibilityReason
from outrider.policy.severity import FindingSeverity
from outrider.schemas.hitl import PerFindingOutcome

if TYPE_CHECKING:
    from outrider.schemas.hitl import HITLDecision, HITLRequest, PerFindingDecision
    from outrider.schemas.review_finding import ReviewFinding


# ---------------------------------------------------------------------------
# Eligibility mapping — keyed by FindingSeverity. The HITL gate consults
# this mapping AFTER the fabricated-override check (which is HITL-aware)
# and BEFORE the HITL decision lookup, so the mapping itself only encodes
# whether a severity requires HITL gating (CRITICAL/HIGH) or passes
# through (MEDIUM/LOW/INFO).
# ---------------------------------------------------------------------------


class _V1SeverityBaseline(StrEnum):
    """V1 baseline gate outcome per severity, pre-HITL-decision-check.

    Internal — the public surface is the `is_eligible_for_v1_publish`
    function which converts this into the public `(PublishEligibility,
    PublishEligibilityReason | None)` tuple. CRITICAL/HIGH severities
    require a HITL decision to materialize; MEDIUM/LOW/INFO are always
    eligible per the gate-set partition.
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


def is_eligible_for_v1_publish(
    finding: ReviewFinding,
    *,
    hitl_request: HITLRequest | None,
    hitl_decision: HITLDecision | None,
) -> tuple[PublishEligibility, PublishEligibilityReason | None]:
    """Decide whether a finding materializes via the publisher in V1.

    Returns:
        `(PublishEligibility.ELIGIBLE, None)` — publisher materializes;
        `(PublishEligibility.WITHHELD, <reason>)` — no GitHub call.

    Inputs (kwargs-only — the gate stays a pure function over what the
    caller explicitly passes, not what's in a module-level state):

      - `finding` — the candidate; the gate reads `.severity`,
        `.original_severity`, `.finding_id`.
      - `hitl_request` — the HITL gate envelope (None means HITL didn't
        run for this review).
      - `hitl_decision` — the reviewer's decision set (None means HITL
        ran but no decision landed yet, OR HITL never ran).

    Withholding order (precedence-bearing):

      1. **Fabricated-override defense first.** If
         `finding.original_severity is not None` AND no matching
         `PerFindingDecision(outcome=SEVERITY_OVERRIDE)` exists in
         `hitl_decision`, return WITHHELD with
         `unexpected_override_fields_present`. A `CRITICAL` finding
         showing `original_severity=CRITICAL + severity=LOW` would
         otherwise pass the severity gate; the override-fields check
         FIRST blocks this.

      2. **Severity gate.** Lookup in `_V1_SEVERITY_GATE` by
         `finding.severity`. MEDIUM/LOW/INFO → ELIGIBLE (no HITL
         needed). CRITICAL/HIGH → consult HITL state:

           - `hitl_request is None`: WITHHELD with
             `hitl_required_node_absent` (graph-wiring bypass — the
             HITL node never ran).
           - `hitl_decision is None` (request landed but decision
             didn't): WITHHELD with `hitl_decision_missing`.
           - Decision exists: lookup the per-finding outcome:
             - APPROVE → ELIGIBLE
             - SEVERITY_OVERRIDE → ELIGIBLE
             - REJECT → WITHHELD with `hitl_rejected`
             - SUPPRESS → WITHHELD with `hitl_suppressed`
             - no matching `finding_id` (defense-in-depth):
               WITHHELD with `hitl_decision_missing`.
    """
    matching_decision = _find_decision_for(
        finding_id=finding.finding_id, hitl_decision=hitl_decision
    )

    # Defense (1): fabricated-override check. Fires FIRST so a forged
    # downgrade can't sneak past the severity gate — unless the
    # corresponding HITL decision actually carries SEVERITY_OVERRIDE
    # for this finding_id AND that finding_id appears in the
    # `findings_requiring_approval` set of an actual HITL request.
    # Three-condition gate (defense-in-depth):
    #   1. A SEVERITY_OVERRIDE decision exists for this finding_id.
    #   2. A `hitl_request` ran (it's not None).
    #   3. This finding_id was in the gated set of that request.
    # Without (2)+(3), a forged `finding.original_severity` paired with
    # a forged-but-Pydantic-valid `HITLDecision` (no `hitl_request`
    # backing it) would pass (1) alone — the gated-set membership check
    # closes that surface.
    if finding.original_severity is not None:
        has_legit_override = (
            matching_decision is not None
            and matching_decision.outcome == PerFindingOutcome.SEVERITY_OVERRIDE
            and hitl_request is not None
            and finding.finding_id in hitl_request.findings_requiring_approval
        )
        if not has_legit_override:
            return (
                PublishEligibility.WITHHELD,
                PublishEligibilityReason.UNEXPECTED_OVERRIDE_FIELDS_PRESENT,
            )

    # Defense (2): severity gate.
    baseline = _V1_SEVERITY_GATE[finding.severity]
    if baseline is _V1SeverityBaseline.ELIGIBLE:
        return (PublishEligibility.ELIGIBLE, None)

    # CRITICAL/HIGH path: consult HITL state.
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

    # Defense-in-depth: the decision's `finding_id` MUST be in the
    # canonical `hitl_request.findings_requiring_approval` gated set.
    # The /decide endpoint already enforces payload→request set-equality
    # at 422 (`/decide` rejects mismatched payloads); this is the
    # publish-side check for a forged-state scenario where a
    # `PerFindingDecision` was constructed for a finding_id the
    # original HITLRequest never gated. Without this check, a decision
    # not backed by the canonical request would be honored at publish
    # — bypassing `hitl-gates-high-severity` for findings the request
    # never listed. Reject as HITL_DECISION_MISSING (the canonical
    # "this finding has no decision backing it" reason).
    if finding.finding_id not in hitl_request.findings_requiring_approval:
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.HITL_DECISION_MISSING,
        )

    if matching_decision.outcome == PerFindingOutcome.APPROVE:
        return (PublishEligibility.ELIGIBLE, None)
    if matching_decision.outcome == PerFindingOutcome.SEVERITY_OVERRIDE:
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
