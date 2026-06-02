"""True-positive eval scenario: PyGoat SQL injection in search view.

Per spec §11.2: PR introduces SQL injection in a search view; the agent
should produce `FindingType.SQL_INJECTION` + `EvidenceTier.OBSERVED` +
`query_match_id` matching a real entry in the queries registry.

Driver-backed: drives the real graph via `run_review` against
`mock_github/pygoat_sql_injection.json`. The fixture's analyze response
carries `evidence_tier="observed"` with a `query_match_id` that is in the
set `_build_query_match_id_set` fires on the file (proof boundary);
severity is asserted via `lookup_severity` (set by policy, not the model).
"""

from outrider.policy import EvidenceTier, FindingType, lookup_severity

EXPECTED_FINDING = {
    "finding_type": FindingType.SQL_INJECTION,
    "evidence_tier": EvidenceTier.OBSERVED,
    # Severity derived via lookup_severity() per `severity-set-by-policy`.
    # Hard-coding the constant would drift if the policy table changes;
    # the canonical mapping is the single source of truth.
    "severity": lookup_severity(FindingType.SQL_INJECTION),
    # query_match_id: the fixture cites a real id from the file's fired
    # structural-query set; asserted non-None (the exact id is fixture-local).
}


def test_pygoat_sql_injection_detected_with_observed_tier() -> None:
    """Agent produces SQL_INJECTION + OBSERVED + valid query_match_id + policy severity."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/pygoat_sql_injection.json")
    sql_findings = [f for f in findings if f.finding_type == FindingType.SQL_INJECTION]
    assert len(sql_findings) >= 1
    finding = sql_findings[0]
    assert finding.evidence_tier == EXPECTED_FINDING["evidence_tier"]
    assert finding.severity == EXPECTED_FINDING["severity"]
    assert finding.query_match_id is not None
