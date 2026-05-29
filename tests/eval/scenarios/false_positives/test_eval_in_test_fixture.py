"""False-positive eval scenario: `eval()` call inside a test fixture.

Per spec §11.2: a test file that uses `eval()` in a fixture (e.g., to
parse a literal in test setup) is NOT a security finding — the call
lives in test code, not production code paths. Expected: zero security
findings.

V1: scaffolded; assertions wire up when the eval graph driver lands (analyze node shipped).
"""

import pytest

from outrider.schemas import ReviewDimension

pytestmark = pytest.mark.skip(
    reason="requires eval graph driver: mock LLM provider + run_review shim + "
    "mock_github fixtures (not yet shipped)"
)

EXPECTED_SECURITY_FINDING_COUNT = 0


def test_eval_in_test_fixture_produces_no_security_findings() -> None:
    """Agent recognizes test-context use of eval() and produces zero security findings."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/eval_in_test_fixture.json")
    security_findings = [f for f in findings if f.dimension == ReviewDimension.SECURITY]
    assert len(security_findings) == EXPECTED_SECURITY_FINDING_COUNT
