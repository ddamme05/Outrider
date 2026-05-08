"""ReviewTier enum has the 4 tiers from spec §4.1.2 + §6.10.

Backs the canonical-shape rule: enum membership and value casing match
docs/spec.md verbatim. SKIP exists in the enum because §6.10 size-cap
classification maps directly to the same tier vocabulary, even though
§4.1.2's prose only enumerates DEEP / STANDARD / SKIM as the LLM-produced
tiers.
"""

from outrider.schemas import ReviewTier

EXPECTED_TIER_VALUES = {"deep", "standard", "skim", "skip"}


def test_review_tier_has_exact_4_values() -> None:
    """No extras, no missing — matches spec §7.2 verbatim."""
    actual = {t.value for t in ReviewTier}
    assert actual == EXPECTED_TIER_VALUES, (
        f"diff: extra={actual - EXPECTED_TIER_VALUES} missing={EXPECTED_TIER_VALUES - actual}"
    )


def test_review_tier_count_is_4() -> None:
    """Per spec §7.2: exactly 4 tiers (DEEP/STANDARD/SKIM/SKIP)."""
    assert len(list(ReviewTier)) == 4


def test_review_tier_values_lowercase() -> None:
    """Project convention: enum values are lowercase serialized strings."""
    for tier in ReviewTier:
        assert tier.value == tier.value.lower(), f"{tier.name} = {tier.value!r} is not lowercase"
