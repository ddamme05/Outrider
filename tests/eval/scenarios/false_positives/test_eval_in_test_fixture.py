"""Exercise the no-findings graph path for an eval() call in test code.

The model response is scripted empty. This checks the workflow; it does not
measure a real model's false-positive rate or test finding-dimension filtering."""

from outrider.schemas import ReviewDimension

EXPECTED_SECURITY_FINDING_COUNT = 0


def test_eval_in_test_fixture_produces_no_security_findings() -> None:
    """Agent recognizes test-context use of eval() and produces zero security findings."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/eval_in_test_fixture.json")
    security_findings = [f for f in findings if f.dimension == ReviewDimension.SECURITY]
    assert len(security_findings) == EXPECTED_SECURITY_FINDING_COUNT
