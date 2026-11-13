"""Follow a scripted PyGoat authentication-bypass finding to the approval gate.

Check model-judgment evidence and policy severity. Critical findings must stop
before publication when the driver has no human decision."""

from outrider.policy import FindingType, lookup_severity

EXPECTED_FINDING = {
    "finding_type": FindingType.AUTH_BYPASS,
    # Severity from policy lookup per `severity-set-by-policy`.
    "severity": lookup_severity(FindingType.AUTH_BYPASS),
    # tier: per the actual finding shape (OBSERVED if a tree-sitter query matched,
    # INFERRED if trace-walked, JUDGED if model-only). Eval ground truth pins this
    # at flip time when the analyze node + queries registry are real.
}


def test_pygoat_auth_bypass_detected_with_correct_severity() -> None:
    """Agent produces AUTH_BYPASS + severity from policy."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/pygoat_auth_bypass.json")
    auth_findings = [f for f in findings if f.finding_type == FindingType.AUTH_BYPASS]
    assert len(auth_findings) >= 1
    assert auth_findings[0].severity == EXPECTED_FINDING["severity"]
