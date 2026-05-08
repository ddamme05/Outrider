"""RiskLevel enum has the 4 ladders from spec §7.2 (Amended 2026-05-08).

Backs the canonical-shape rule: enum membership and value casing match
docs/spec.md §7.2 (as amended) verbatim. RiskLevel was added 2026-05-08
to close the canonical-shape gap (referenced by TriageResult.overall_risk
and ReviewReport.overall_risk but never defined as a class). The
test_risk_level_matches_canonical_amendment test guards against the same
canonical-vs-implementation drift this spec is designed to close — if a
future amendment changes the vocabulary, this test fails until both the
amendment and the enum line up.

Canonical rationale for the chosen vocabulary (from spec.md §7.2 amendment
+ specs/2026-05-08-schema-foundation.md): same ladder shape as
FindingSeverity minus INFO, since "informational PR-level risk" has no
operational meaning (no auto-decline tier, no HITL gate, no display
affordance, and the LLM is already trained on the four-word ladder).
"""

from outrider.schemas import RiskLevel

EXPECTED_RISK_VALUES = {"low", "medium", "high", "critical"}


def test_risk_level_has_exact_4_values() -> None:
    """No extras, no missing — matches spec §7.2 amendment verbatim."""
    actual = {r.value for r in RiskLevel}
    assert actual == EXPECTED_RISK_VALUES, (
        f"diff: extra={actual - EXPECTED_RISK_VALUES} missing={EXPECTED_RISK_VALUES - actual}"
    )


def test_risk_level_count_is_4() -> None:
    """Per spec §7.2 amendment: exactly 4 levels (LOW/MEDIUM/HIGH/CRITICAL)."""
    assert len(list(RiskLevel)) == 4


def test_risk_level_excludes_info() -> None:
    """INFO is deliberately absent — informational PR-level risk has no operational meaning."""
    assert "info" not in {r.value for r in RiskLevel}
    assert "INFO" not in {r.name for r in RiskLevel}


def test_risk_level_matches_canonical_amendment() -> None:
    """Pin the canonical-amendment vocabulary exactly.

    If a future amendment to spec.md §7.2 changes the vocabulary, this
    test fails until both the amendment and the enum line up. Guards
    against the same canonical-vs-implementation drift this schema-
    foundation spec was designed to close.
    """
    assert RiskLevel.LOW.value == "low"
    assert RiskLevel.MEDIUM.value == "medium"
    assert RiskLevel.HIGH.value == "high"
    assert RiskLevel.CRITICAL.value == "critical"
