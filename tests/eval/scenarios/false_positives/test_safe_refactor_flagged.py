"""False-positive eval scenario: safe refactor wrongly flagged as security change.

Per spec §11.2: PR refactors safe code (e.g., reorders parameters,
renames a variable inside a function) without introducing any security
issue; expected: zero security findings. Asserts the agent doesn't
hallucinate findings on cosmetic changes.

V1: scaffolded; assertions wire up when analyze node lands per §15.3.
"""

import pytest

from outrider.schemas import ReviewDimension

pytestmark = pytest.mark.skip(reason="requires analyze node")

EXPECTED_SECURITY_FINDING_COUNT = 0


def test_safe_refactor_produces_no_security_findings() -> None:
    """Agent produces zero ReviewDimension.SECURITY findings on a cosmetic refactor."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/safe_refactor.json")
    security_findings = [f for f in findings if f.dimension == ReviewDimension.SECURITY]
    assert len(security_findings) == EXPECTED_SECURITY_FINDING_COUNT
