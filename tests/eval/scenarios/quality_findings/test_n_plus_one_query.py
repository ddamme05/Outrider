"""Quality-finding eval scenario: PR introduces an N+1 query.

Per spec §11.2: PR adds a loop that issues a DB query per item; expected:
`FindingType.N_PLUS_ONE_QUERY` (canonical enum from `policy/severity.py`)
+ severity from `SEVERITY_POLICY[N_PLUS_ONE_QUERY]` + tier per the
structural evidence.

V1: scaffolded; assertions wire up when the eval graph driver lands (analyze node shipped).
"""

import pytest

from outrider.policy import FindingType, lookup_severity

pytestmark = pytest.mark.skip(
    reason="requires eval graph driver: mock LLM provider + run_review shim + "
    "mock_github fixtures (not yet shipped)"
)

EXPECTED_FINDING = {
    "finding_type": FindingType.N_PLUS_ONE_QUERY,
    # Severity from policy lookup per `severity-set-by-policy`; tracks the
    # canonical mapping rather than hard-coding a constant (which would
    # drift if the policy table changes for this finding type).
    "severity": lookup_severity(FindingType.N_PLUS_ONE_QUERY),
}


def test_n_plus_one_query_detected_with_policy_severity() -> None:
    """Agent produces N_PLUS_ONE_QUERY + severity from policy."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/n_plus_one_query.json")
    nplus = [f for f in findings if f.finding_type == FindingType.N_PLUS_ONE_QUERY]
    assert len(nplus) >= 1
    assert nplus[0].severity == EXPECTED_FINDING["severity"]
