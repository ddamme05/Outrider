"""ReviewDimension enum has the 5 axes from spec §7.3.

Backs the canonical-shape rule: enum membership and value casing match
docs/spec.md verbatim.
"""

from outrider.schemas import ReviewDimension

EXPECTED_DIMENSION_VALUES = {
    "code_quality",
    "security",
    "performance",
    "test_coverage",
    "best_practices",
}


def test_review_dimension_has_exact_5_values() -> None:
    """No extras, no missing — matches spec §7.3 verbatim."""
    actual = {d.value for d in ReviewDimension}
    assert actual == EXPECTED_DIMENSION_VALUES, (
        f"diff: extra={actual - EXPECTED_DIMENSION_VALUES} "
        f"missing={EXPECTED_DIMENSION_VALUES - actual}"
    )


def test_review_dimension_count_is_5() -> None:
    """Per spec §7.3: exactly 5 dimensions."""
    assert len(list(ReviewDimension)) == 5
