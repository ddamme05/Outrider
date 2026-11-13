"""ReviewTier enum has the 4 tiers from spec §4.1.2 + §6.10.

Membership and value casing match docs/spec.md. SKIP represents the §6.10 size
cap alongside the model-produced DEEP, STANDARD and SKIM tiers.
"""

from outrider.schemas import ReviewTier

EXPECTED_TIER_VALUES = {"deep", "standard", "skim", "skip"}


def test_review_tier_has_exact_4_values() -> None:
    """No extras, no missing — matches spec §7.2 verbatim."""
    actual = {t.value for t in ReviewTier}
    assert actual == EXPECTED_TIER_VALUES, (
        f"diff: extra={actual - EXPECTED_TIER_VALUES} missing={EXPECTED_TIER_VALUES - actual}"
    )
