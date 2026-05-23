# V1 publish eligibility gate per specs/2026-05-21-publish-node.md Q3 + FUP-062.
"""is_eligible_for_v1_publish — policy-derived publish-materialization gate.

Per DECISIONS.md #023 (publish routing and eligibility are separate
decisions, not one combined gate): this module is the **policy** half
of the V1 fabricated-override defense. The **schema** half lives at
`PublishEligibilityEvent._enforce_v1_no_overrides` in `audit/events.py`;
both must hold for the DECISIONS #023 "schema + gate" trust story.

The gate fires BEFORE materialization — the publish node calls
`is_eligible_for_v1_publish(finding)` per finding, and routes the
finding to `publisher.create_review(...)` only when the return is
`("eligible", None)`. Withholding outcomes get recorded in
`PublishEligibilityEvent(eligibility=withheld, reason=<reason>)` and
the finding does NOT reach GitHub.

V1 withholding reasons:

- `hitl_required_node_absent` — finding severity is `CRITICAL` or
  `HIGH` and the HITL node isn't shipped yet. ALL `CRITICAL`/`HIGH`
  findings get this in V1; when HITL ships, the gate flips to consult
  `HITLDecisionEvent` for these severities.

- `unexpected_override_fields_present` — finding carries a non-None
  `original_severity` despite no legitimate HITL override path
  existing in V1. Indicates either a producer bug or replay-injected
  state forging a pre-approved downgrade (a `CRITICAL` finding
  showing `original_severity=CRITICAL + severity=LOW` would otherwise
  appear to have been HITL-approved). Defended at this gate AND at
  the audit-row schema layer; the gate fires FIRST so no GitHub call
  ever happens for a forged-override finding.

The mapping is a `MappingProxyType` keyed by `FindingSeverity` — set
membership is FORBIDDEN per the spec's "implementation discipline"
clause (§Severity policy of the publish-node spec, line 30). The
mapping is total over `FindingSeverity` at import time
(`_assert_mapping_total_at_import` runs as a module-level assertion);
any new severity that lands without an entry crashes the import.
This is the loud-failure pattern: a missing mapping entry is
ambiguous between "deliberate omission" and "forgotten case," so
the floor is "doesn't import" rather than "silently falls through
to a default."
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from outrider.audit.events import PublishEligibility, PublishEligibilityReason
from outrider.policy.severity import FindingSeverity

if TYPE_CHECKING:
    from outrider.schemas.review_finding import ReviewFinding


# ---------------------------------------------------------------------------
# Eligibility mapping — keyed by FindingSeverity, value is the V1 outcome
# tuple for severities WITHOUT the fabricated-override condition. The
# override defense runs before this mapping is consulted, so the mapping
# itself only encodes severity-based gating.
# ---------------------------------------------------------------------------


class _V1SeverityBaseline(StrEnum):
    """V1 baseline gate outcome per severity, pre-override-check.

    Internal — the public surface is `is_eligible_for_v1_publish(finding)`
    which converts this into the public `(PublishEligibility,
    PublishEligibilityReason | None)` tuple. The intermediate enum
    exists so `_assert_mapping_total_at_import` can pin the exhaustive
    coverage without entangling test code with `PublishEligibility`'s
    public values.
    """

    ELIGIBLE = "eligible"
    WITHHELD_HITL_ABSENT = "withheld_hitl_absent"


# The MappingProxyType wrapper makes this immutable at runtime — a test
# fixture or buggy caller can't mutate the mapping and silently change
# eligibility for the rest of the process. Same defense-in-depth shape as
# `outrider.llm.pricing.RATE_TABLE` per project convention.
#
# V1 policy:
#   CRITICAL/HIGH → withheld (HITL node absent in V1)
#   MEDIUM/LOW/INFO → eligible (no HITL gating needed; INFO is below
#                    the CRITICAL/HIGH HITL trigger)
_V1_SEVERITY_GATE: Final[MappingProxyType[FindingSeverity, _V1SeverityBaseline]] = MappingProxyType(
    {
        FindingSeverity.CRITICAL: _V1SeverityBaseline.WITHHELD_HITL_ABSENT,
        FindingSeverity.HIGH: _V1SeverityBaseline.WITHHELD_HITL_ABSENT,
        FindingSeverity.MEDIUM: _V1SeverityBaseline.ELIGIBLE,
        FindingSeverity.LOW: _V1SeverityBaseline.ELIGIBLE,
        FindingSeverity.INFO: _V1SeverityBaseline.ELIGIBLE,
    }
)


def _assert_mapping_total_at_import() -> None:
    """Pin the mapping's totality over `FindingSeverity` at import time.

    Per the publish-node spec's implementation discipline clause
    ("`is_eligible_for_v1_publish` MUST use exhaustive `match` over
    every `FindingSeverity` enum member OR a frozen `MappingProxyType`
    keyed lookup that raises `KeyError` on miss; set-
    membership rejected at review time"), we use the MappingProxyType
    + KeyError-on-miss shape; this import-time assertion is the
    "exhaustive coverage" half. If a new `FindingSeverity` member lands
    without an entry in `_V1_SEVERITY_GATE`, the module fails to import
    rather than silently treating the new severity as ineligible.
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


def is_eligible_for_v1_publish(
    finding: ReviewFinding,
) -> tuple[PublishEligibility, PublishEligibilityReason | None]:
    """Decide whether a finding materializes via the publisher in V1.

    Returns:
        `(PublishEligibility.ELIGIBLE, None)` — publisher materializes;
        `(PublishEligibility.WITHHELD, <reason>)` — no GitHub call.

    Per DECISIONS.md #023 + FUP-062, this function is a REQUIRED
    publish-node-impl precondition. The publish node MUST call it
    before passing a finding to `publisher.create_review(...)`. The
    schema-layer defense (`PublishEligibilityEvent._enforce_v1_no_overrides`)
    is a belt-and-suspenders backstop — it fires at audit-row
    construction, but by then the GitHub call already happened. The
    gate fires BEFORE materialization so a forged-override finding
    never reaches GitHub.

    Withholding order (matters for the precedence story):

      1. **Fabricated-override defense first.** If
         `finding.original_severity is not None`, return
         `WITHHELD + unexpected_override_fields_present` regardless
         of `finding.severity`. A `CRITICAL` finding showing
         `original_severity=CRITICAL + severity=LOW` would otherwise
         pass the severity gate (severity=LOW → eligible) and
         materialize. The override-fields check FIRST blocks this.

      2. **Severity gate.** Lookup in `_V1_SEVERITY_GATE` by
         `finding.severity`. `CRITICAL`/`HIGH` → withheld
         (hitl_required_node_absent); `MEDIUM`/`LOW` → eligible.

    The exhaustive-coverage discipline (`_assert_mapping_total_at_import`)
    runs at module import; this function uses direct dict access (raises
    `KeyError` on miss) rather than `.get(..., default)` so a new
    `FindingSeverity` member that somehow bypasses the import-time
    assertion still fails loudly at runtime.
    """
    # Defense (1): fabricated-override check. Fires FIRST so a forged
    # downgrade can't sneak past the severity gate.
    if finding.original_severity is not None:
        return (
            PublishEligibility.WITHHELD,
            PublishEligibilityReason.UNEXPECTED_OVERRIDE_FIELDS_PRESENT,
        )

    # Defense (2): severity gate. Direct dict access — raises KeyError
    # on miss for the same reason the import-time totality check exists.
    baseline = _V1_SEVERITY_GATE[finding.severity]
    if baseline is _V1SeverityBaseline.ELIGIBLE:
        return (PublishEligibility.ELIGIBLE, None)
    return (
        PublishEligibility.WITHHELD,
        PublishEligibilityReason.HITL_REQUIRED_NODE_ABSENT,
    )
