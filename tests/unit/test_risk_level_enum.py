"""Check PR risk levels against docs/spec.md §7.2.

The severity-like ladder excludes INFO, which has no PR approval action.
See specs/2026-05-08-schema-foundation.md."""

from outrider.schemas import RiskLevel

EXPECTED_RISK_VALUES = {"low", "medium", "high", "critical"}


def test_risk_level_has_exact_4_values() -> None:
    """No extras, no missing — matches spec §7.2 amendment verbatim."""
    actual = {r.value for r in RiskLevel}
    assert actual == EXPECTED_RISK_VALUES, (
        f"diff: extra={actual - EXPECTED_RISK_VALUES} missing={EXPECTED_RISK_VALUES - actual}"
    )


def test_risk_level_excludes_info() -> None:
    """INFO is deliberately absent — informational PR-level risk has no operational meaning."""
    assert "info" not in {r.value for r in RiskLevel}
    assert "INFO" not in {r.name for r in RiskLevel}


def test_risk_level_matches_canonical_amendment() -> None:
    """Named members serialize to the canonical docs/spec.md §7.2 vocabulary."""
    assert RiskLevel.LOW.value == "low"
    assert RiskLevel.MEDIUM.value == "medium"
    assert RiskLevel.HIGH.value == "high"
    assert RiskLevel.CRITICAL.value == "critical"
