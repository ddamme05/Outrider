"""Check a scripted PyGoat model judgment against the real proof and policy gates.

The legacy fixture remains frozen; this workflow uses its JUDGED counterpart.
Independent structural coverage is checked by the query and model-refresh tests.
"""

from outrider.policy import EvidenceTier, FindingType, lookup_severity

EXPECTED_FINDING = {
    "finding_type": FindingType.SQL_INJECTION,
    "evidence_tier": EvidenceTier.JUDGED,
    # Severity derived via lookup_severity() per `severity-set-by-policy`.
    # Hard-coding the constant would drift if the policy table changes;
    # the canonical mapping is the single source of truth.
    "severity": lookup_severity(FindingType.SQL_INJECTION),
}


def test_pygoat_sql_injection_detected_with_judged_tier() -> None:
    """Admit SQL_INJECTION as JUDGED with no structural claim and policy severity."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/pygoat_sql_injection_judged.json")
    sql_findings = [f for f in findings if f.finding_type == FindingType.SQL_INJECTION]
    assert len(sql_findings) >= 1
    finding = sql_findings[0]
    assert finding.evidence_tier == EXPECTED_FINDING["evidence_tier"]
    assert finding.severity == EXPECTED_FINDING["severity"]
    assert finding.query_match_id is None
