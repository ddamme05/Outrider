"""Exercise the no-findings graph path for a harmless refactor.

The empty model response is scripted, so passing checks the workflow rather
than proving that a real model will recognize safe code."""

from outrider.schemas import ReviewDimension

EXPECTED_SECURITY_FINDING_COUNT = 0


def test_safe_refactor_produces_no_security_findings() -> None:
    """Agent produces zero ReviewDimension.SECURITY findings on a cosmetic refactor."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/safe_refactor.json")
    security_findings = [f for f in findings if f.dimension == ReviewDimension.SECURITY]
    assert len(security_findings) == EXPECTED_SECURITY_FINDING_COUNT
