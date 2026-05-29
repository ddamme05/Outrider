"""True-positive eval scenario: PyGoat SQL injection in search view.

Per spec §11.2: PR introduces SQL injection in a search view; the agent
should produce `FindingType.SQL_INJECTION` + `EvidenceTier.OBSERVED` +
`query_match_id` matching a real entry in the queries registry.

V1: scaffolded; assertions wire up when the eval graph driver lands
(analyze node + queries registry shipped) per §15.3. The expected-output
fixture pins the canonical FindingType +
severity + tier; the specific `query_match_id` string is intentionally
left as "matches a real registry entry" since the queries spec is
canonical for that naming convention.
"""

import pytest

from outrider.policy import EvidenceTier, FindingType, lookup_severity

pytestmark = pytest.mark.skip(
    reason="requires eval graph driver (mock LLM provider + run_review shim + "
    "mock_github fixtures); analyze node + queries registry already shipped"
)

EXPECTED_FINDING = {
    "finding_type": FindingType.SQL_INJECTION,
    "evidence_tier": EvidenceTier.OBSERVED,
    # Severity derived via lookup_severity() per `severity-set-by-policy`.
    # Hard-coding the constant would drift if the policy table changes;
    # the canonical mapping is the single source of truth.
    "severity": lookup_severity(FindingType.SQL_INJECTION),
    # query_match_id: matches a real entry in queries/python/*.scm; specific
    # string pinned when the queries spec lands.
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
