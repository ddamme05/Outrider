"""Quality-finding eval scenario: PR with missing error handling on network call.

Per spec §11.2: PR adds a network call without try/except or `.raise_for_status()`;
expected: `FindingType.MISSING_ERROR_HANDLING` (canonical enum) + severity from
`SEVERITY_POLICY[MISSING_ERROR_HANDLING]`.

V1: scaffolded; assertions wire up when the eval graph driver lands (analyze node shipped).
"""

import pytest

from outrider.policy import FindingType, lookup_severity

pytestmark = pytest.mark.skip(
    reason="requires eval graph driver: mock LLM provider + run_review shim + "
    "mock_github fixtures (not yet shipped)"
)

EXPECTED_FINDING = {
    "finding_type": FindingType.MISSING_ERROR_HANDLING,
    # Severity from policy lookup per `severity-set-by-policy`; tracks the
    # canonical mapping rather than hard-coding a constant (which would
    # drift if the policy table changes for this finding type).
    "severity": lookup_severity(FindingType.MISSING_ERROR_HANDLING),
}


def test_missing_error_handling_on_network_call_detected() -> None:
    """Agent flags the unguarded network call with MISSING_ERROR_HANDLING."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/missing_error_handling.json")
    matches = [f for f in findings if f.finding_type == FindingType.MISSING_ERROR_HANDLING]
    assert len(matches) >= 1
