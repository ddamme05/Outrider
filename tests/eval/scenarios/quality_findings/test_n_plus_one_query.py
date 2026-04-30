"""Quality-finding eval scenario: PR introduces an N+1 query.

Per spec §11.2: PR adds a loop that issues a DB query per item; expected:
`FindingType.N_PLUS_ONE_QUERY` (canonical enum from `policy/severity.py`)
+ severity from `SEVERITY_POLICY[N_PLUS_ONE_QUERY]` (MEDIUM) + tier per
the structural evidence.

V1: scaffolded; assertions wire up when analyze node lands.
"""

import pytest

from outrider.policy import FindingSeverity, FindingType

pytestmark = pytest.mark.skip(reason="requires analyze node")

EXPECTED_FINDING = {
    "finding_type": FindingType.N_PLUS_ONE_QUERY,
    "severity": FindingSeverity.MEDIUM,  # SEVERITY_POLICY[N_PLUS_ONE_QUERY] = MEDIUM
}


def test_n_plus_one_query_detected_with_medium_severity() -> None:
    """Agent produces N_PLUS_ONE_QUERY + MEDIUM severity from policy."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/n_plus_one_query.json")
    nplus = [f for f in findings if f.finding_type == FindingType.N_PLUS_ONE_QUERY]
    assert len(nplus) >= 1
    assert nplus[0].severity == EXPECTED_FINDING["severity"]
