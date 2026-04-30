"""Quality-finding eval scenario: PR with missing error handling on network call.

Per spec §11.2: PR adds a network call without try/except or `.raise_for_status()`;
expected: `FindingType.MISSING_ERROR_HANDLING` (canonical enum) + severity from
`SEVERITY_POLICY[MISSING_ERROR_HANDLING]`.

V1: scaffolded; assertions wire up when analyze node lands.
"""

import pytest

from outrider.policy import FindingType

pytestmark = pytest.mark.skip(reason="requires analyze node")

EXPECTED_FINDING = {
    "finding_type": FindingType.MISSING_ERROR_HANDLING,
    # severity: SEVERITY_POLICY[MISSING_ERROR_HANDLING]; pinned at flip time
}


def test_missing_error_handling_on_network_call_detected() -> None:
    """Agent flags the unguarded network call with MISSING_ERROR_HANDLING."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/missing_error_handling.json")
    matches = [f for f in findings if f.finding_type == FindingType.MISSING_ERROR_HANDLING]
    assert len(matches) >= 1
