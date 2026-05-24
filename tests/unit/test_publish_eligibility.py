# Tests for policy/publish_eligibility.py per FUP-062 + DECISIONS.md #023.
"""Pin the V1 publish eligibility gate.

Per DECISIONS.md #023 (two-layered fabricated-override defense):
  - SCHEMA half: PublishEligibilityEvent._enforce_v1_no_overrides
  - POLICY half: is_eligible_for_v1_publish (this module)

The schema rejects the audit row; the gate must reject BEFORE the
GitHub call so the fabricated-override never materializes.
"""

from __future__ import annotations

from uuid import uuid4

from outrider.audit.events import (
    PublishEligibility,
    PublishEligibilityReason,
    compute_finding_content_hash,
)
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.publish_eligibility import (
    _V1_SEVERITY_GATE,
    is_eligible_for_v1_publish,
)
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewFinding


def _make_finding(
    *,
    severity: FindingSeverity = FindingSeverity.MEDIUM,
    original_severity: FindingSeverity | None = None,
) -> ReviewFinding:
    """Construct a minimal ReviewFinding for gate testing.

    The gate function only reads `.severity` and `.original_severity`, so
    the finding_type/severity match here is purely for construction
    (proof-boundary validator demands they agree under SEVERITY_POLICY).
    """
    # finding_type chosen to match `severity` under SEVERITY_POLICY so
    # the proof-boundary validator is satisfied at construction.
    finding_type_by_severity = {
        FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
        FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
        FindingSeverity.MEDIUM: FindingType.MISSING_INPUT_VALIDATION,
        FindingSeverity.LOW: FindingType.MISSING_ERROR_HANDLING,
        FindingSeverity.INFO: FindingType.UNUSED_IMPORT,
    }
    # When original_severity is set (HITL override case), finding_type
    # must map to original_severity (the POLICY baseline); severity is
    # the override value. When original_severity is None, finding_type
    # matches `severity` directly.
    baseline = original_severity if original_severity is not None else severity
    finding_type = finding_type_by_severity[baseline]
    file_path = "src/foo.py"
    line_start = 10
    line_end = 12
    # The HITL override triplet (original_severity + override_reason +
    # overrider_id) must be all-set-or-all-None per ReviewFinding's
    # `enforce_override_triplet` validator. To exercise the V1
    # fabricated-override defense in `is_eligible_for_v1_publish`, we
    # construct a legitimate-LOOKING override (full triplet present) and
    # verify the gate withholds it anyway — V1 has no HITL node, so a
    # populated triplet must be a replay-injected forgery.
    override_kwargs: dict[str, object] = {}
    if original_severity is not None:
        override_kwargs = {
            "original_severity": original_severity,
            "override_reason": "forged-by-test (fabricated-override-defense regression)",
            "overrider_id": uuid4(),
        }
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=finding_type,
        severity=severity,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        proposal_hash="a" * 64,  # Per DECISIONS.md#025; dummy SHA-256 hex.
        **override_kwargs,
    )


# ---------------------------------------------------------------------------
# Severity gate — CRITICAL/HIGH withheld, MEDIUM/LOW/INFO eligible
# ---------------------------------------------------------------------------


def test_critical_finding_is_withheld_hitl_absent() -> None:
    """V1: CRITICAL findings can't materialize until HITL ships."""
    finding = _make_finding(severity=FindingSeverity.CRITICAL)
    eligibility, reason = is_eligible_for_v1_publish(finding)
    assert eligibility is PublishEligibility.WITHHELD
    assert reason is PublishEligibilityReason.HITL_REQUIRED_NODE_ABSENT


def test_high_finding_is_withheld_hitl_absent() -> None:
    """V1: HIGH findings can't materialize until HITL ships."""
    finding = _make_finding(severity=FindingSeverity.HIGH)
    eligibility, reason = is_eligible_for_v1_publish(finding)
    assert eligibility is PublishEligibility.WITHHELD
    assert reason is PublishEligibilityReason.HITL_REQUIRED_NODE_ABSENT


def test_medium_finding_is_eligible() -> None:
    """MEDIUM findings materialize directly in V1."""
    finding = _make_finding(severity=FindingSeverity.MEDIUM)
    eligibility, reason = is_eligible_for_v1_publish(finding)
    assert eligibility is PublishEligibility.ELIGIBLE
    assert reason is None


def test_low_finding_is_eligible() -> None:
    """LOW findings materialize directly in V1."""
    finding = _make_finding(severity=FindingSeverity.LOW)
    eligibility, reason = is_eligible_for_v1_publish(finding)
    assert eligibility is PublishEligibility.ELIGIBLE
    assert reason is None


def test_info_finding_is_eligible() -> None:
    """INFO findings materialize directly in V1."""
    finding = _make_finding(severity=FindingSeverity.INFO)
    eligibility, reason = is_eligible_for_v1_publish(finding)
    assert eligibility is PublishEligibility.ELIGIBLE
    assert reason is None


# ---------------------------------------------------------------------------
# Fabricated-override defense — fires FIRST, regardless of severity
# ---------------------------------------------------------------------------


def test_fabricated_override_rejects_even_for_low_severity() -> None:
    """A LOW-severity finding with original_severity is rejected.

    Defends against producer-bug or replay-injected state forging a
    CRITICAL → LOW downgrade. The fact that the CURRENT severity is
    LOW (which would otherwise be eligible) does NOT matter — the
    presence of original_severity is the signal.
    """
    finding = _make_finding(
        severity=FindingSeverity.LOW,
        original_severity=FindingSeverity.CRITICAL,
    )
    eligibility, reason = is_eligible_for_v1_publish(finding)
    assert eligibility is PublishEligibility.WITHHELD
    assert reason is PublishEligibilityReason.UNEXPECTED_OVERRIDE_FIELDS_PRESENT


def test_fabricated_override_precedes_severity_gate() -> None:
    """The override check fires BEFORE the severity gate.

    A CRITICAL finding with original_severity returns
    `unexpected_override_fields_present`, NOT `hitl_required_node_absent`.
    Tests the precedence: override-defense first, then severity gate.
    """
    finding = _make_finding(
        severity=FindingSeverity.CRITICAL,
        original_severity=FindingSeverity.LOW,
    )
    eligibility, reason = is_eligible_for_v1_publish(finding)
    assert eligibility is PublishEligibility.WITHHELD
    # NOT hitl_required_node_absent — the override-defense wins.
    assert reason is PublishEligibilityReason.UNEXPECTED_OVERRIDE_FIELDS_PRESENT


# ---------------------------------------------------------------------------
# Exhaustive-mapping invariant — _V1_SEVERITY_GATE is total over FindingSeverity
# ---------------------------------------------------------------------------


def test_severity_gate_mapping_is_total_over_finding_severity() -> None:
    """Every FindingSeverity member has a gate entry.

    Per spec §Severity policy line 30: the gate MUST use exhaustive
    mapping over every FindingSeverity, set-membership rejected. The
    module-level `_assert_mapping_total_at_import` runs this at import
    time; this test pins it as a regression guard against the
    assertion being inadvertently removed.
    """
    assert set(_V1_SEVERITY_GATE.keys()) == set(FindingSeverity)


def test_severity_gate_mapping_is_immutable_proxy() -> None:
    """`_V1_SEVERITY_GATE` is wrapped in MappingProxyType.

    A test fixture or buggy caller can't mutate the mapping and
    silently change eligibility for the rest of the process. Same
    defense-in-depth shape as `outrider.llm.pricing.RATE_TABLE`.
    """
    from types import MappingProxyType

    assert isinstance(_V1_SEVERITY_GATE, MappingProxyType)
    import pytest

    with pytest.raises(TypeError):
        _V1_SEVERITY_GATE[FindingSeverity.CRITICAL] = None  # type: ignore[index]
