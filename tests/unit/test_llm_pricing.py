"""pricing.py rate table + PRICING_VERSION digest pinning (AC#19, partial).

Covers:
  - RATE_TABLE shape (every default ModelConfig model has an entry)
  - ModelPricing carries all four required Decimal rates
  - Each rate is non-negative (allows free-promo edge case per round-11 H3)
  - Anthropic ratio sanity gates (cache-write > input > cache-read; output > input)
  - compute_cost_usd four-class formula
  - PRICING_VERSION digest pinning — bumping rates without bumping
    PRICING_VERSION + EXPECTED_PRICING_DIGEST fails this test (AC#19 enforcement)
"""

import hashlib
from decimal import Decimal

import pytest

from outrider.llm.config import ModelConfig
from outrider.llm.pricing import (
    PRICING_VERSION,
    RATE_TABLE,
    ModelPricing,
    compute_cost_usd,
)

# ---------------------------------------------------------------------------
# RATE_TABLE shape.
# ---------------------------------------------------------------------------


def test_pricing_version_is_non_empty_string() -> None:
    assert isinstance(PRICING_VERSION, str)
    assert len(PRICING_VERSION) > 0


def test_every_modelconfig_default_model_has_pricing() -> None:
    """If a ModelConfig default (or env override) names a model not in
    RATE_TABLE, AnthropicProvider construction fails. That gate depends
    on this RATE_TABLE coverage."""
    cfg = ModelConfig()
    for model in (
        cfg.triage_model,
        cfg.analyze_model,
        cfg.synthesize_model,
        cfg.trace_model,
    ):
        assert model in RATE_TABLE, f"missing pricing entry for {model!r}"


@pytest.mark.parametrize("model_id,pricing", list(RATE_TABLE.items()))
def test_modelpricing_has_all_four_rates(model_id: str, pricing: ModelPricing) -> None:
    assert isinstance(pricing.in_per_token, Decimal)
    assert isinstance(pricing.cache_write_per_token, Decimal)
    assert isinstance(pricing.cache_read_per_token, Decimal)
    assert isinstance(pricing.out_per_token, Decimal)


@pytest.mark.parametrize("model_id,pricing", list(RATE_TABLE.items()))
def test_rates_non_negative(model_id: str, pricing: ModelPricing) -> None:
    """Round-11 H3 fold: `>= 0` accepts the free-promo edge case
    (some models could legitimately have zero cache rates during
    promotional windows)."""
    assert pricing.in_per_token >= 0
    assert pricing.cache_write_per_token >= 0
    assert pricing.cache_read_per_token >= 0
    assert pricing.out_per_token >= 0


@pytest.mark.parametrize("model_id,pricing", list(RATE_TABLE.items()))
def test_anthropic_ratio_sanity_gates(model_id: str, pricing: ModelPricing) -> None:
    """Anthropic's billable shape:
      - cache-write > input rate (premium, ~1.25× per public pricing)
      - cache-read < input rate (discount, ~0.10× per public pricing)
      - output > input rate (output is more expensive than input)
    Skips the check if all four rates are zero (free promo)."""
    if pricing.in_per_token == 0:
        pytest.skip(f"{model_id!r}: zero rate (free promo) — ratio gates not applicable")
    assert pricing.cache_write_per_token > pricing.in_per_token, (
        f"{model_id!r}: cache_write_per_token should be > in_per_token (premium)"
    )
    assert pricing.cache_read_per_token < pricing.in_per_token, (
        f"{model_id!r}: cache_read_per_token should be < in_per_token (discount)"
    )
    assert pricing.out_per_token > pricing.in_per_token, (
        f"{model_id!r}: out_per_token should be > in_per_token"
    )


# ---------------------------------------------------------------------------
# compute_cost_usd four-class formula.
# ---------------------------------------------------------------------------


def test_compute_cost_usd_four_class_sum() -> None:
    """All four token classes contribute to the total. Round-14 fold per
    Codex finding F2 — earlier designs missed cache-write tokens.
    """
    model = "claude-sonnet-4-6"
    rates = RATE_TABLE[model]
    expected = (
        rates.in_per_token * 100
        + rates.cache_write_per_token * 50
        + rates.cache_read_per_token * 200
        + rates.out_per_token * 300
    )
    actual = compute_cost_usd(
        model=model,
        input_tokens=100,
        cache_write_tokens=50,
        cache_read_tokens=200,
        output_tokens=300,
    )
    assert actual == expected


@pytest.mark.parametrize(
    "field",
    ["input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens"],
)
def test_each_token_class_is_a_separate_term(field: str) -> None:
    """Set each class to zero in turn; cost should drop by exactly that
    class's contribution. Confirms each is a separate term, not folded."""
    model = "claude-sonnet-4-6"
    base = compute_cost_usd(
        model=model,
        input_tokens=100,
        cache_write_tokens=100,
        cache_read_tokens=100,
        output_tokens=100,
    )
    kwargs: dict[str, int] = {
        "input_tokens": 100,
        "cache_write_tokens": 100,
        "cache_read_tokens": 100,
        "output_tokens": 100,
    }
    kwargs[field] = 0
    reduced = compute_cost_usd(model=model, **kwargs)
    rates = RATE_TABLE[model]
    rate_for_field = {
        "input_tokens": rates.in_per_token,
        "cache_write_tokens": rates.cache_write_per_token,
        "cache_read_tokens": rates.cache_read_per_token,
        "output_tokens": rates.out_per_token,
    }[field]
    expected_diff = rate_for_field * 100
    assert base - reduced == expected_diff


def test_compute_cost_usd_returns_decimal() -> None:
    """Decimal precision throughout — the cast to float happens at
    LLMCallEvent construction, NOT inside compute_cost_usd."""
    cost = compute_cost_usd("claude-haiku-4-5", 1, 1, 1, 1)
    assert isinstance(cost, Decimal)


def test_compute_cost_usd_zero_tokens_is_zero() -> None:
    cost = compute_cost_usd("claude-sonnet-4-6", 0, 0, 0, 0)
    assert cost == Decimal(0)


def test_compute_cost_usd_unknown_model_raises_keyerror() -> None:
    """compute_cost_usd doesn't translate KeyError; AnthropicProvider's
    eager pricing-coverage validation makes this unreachable in
    production, and complete() step 8 wraps it as
    LLMPricingMissingError defensively."""
    with pytest.raises(KeyError):
        compute_cost_usd("claude-fake-99-99", 1, 1, 1, 1)


# ---------------------------------------------------------------------------
# PRICING_VERSION digest pinning (round-11 H4 fold).
# ---------------------------------------------------------------------------


def _compute_rate_table_digest() -> str:
    """Hash the rate table's content. Bumping rates without bumping
    PRICING_VERSION + this expected digest fails the next test loud."""
    items = sorted(RATE_TABLE.items())
    serialized = repr(items).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


# Pinned digest for PRICING_VERSION="v1". When PRICING_VERSION bumps,
# ALSO update this constant; the test below catches mismatched updates.
EXPECTED_PRICING_DIGEST: dict[str, str] = {
    "v1": "a7c01b52255f790e",  # initial — claude-sonnet-4-7 + claude-haiku-4-5
    "v2": "761941ec53c83be1",  # round-20 — replaced claude-sonnet-4-7 with claude-sonnet-4-6
}


def test_pricing_digest_matches_pricing_version() -> None:
    """If RATE_TABLE changes without PRICING_VERSION + EXPECTED_PRICING_DIGEST
    co-update, this test fails — the deterministic floor against the
    manual-discipline gap (round-11 H4 fold)."""
    assert PRICING_VERSION in EXPECTED_PRICING_DIGEST, (
        f"PRICING_VERSION={PRICING_VERSION!r} has no pinned digest in "
        f"EXPECTED_PRICING_DIGEST. Add an entry when bumping the version."
    )
    actual = _compute_rate_table_digest()
    expected = EXPECTED_PRICING_DIGEST[PRICING_VERSION]
    assert actual == expected, (
        f"RATE_TABLE digest drifted: got {actual}, expected {expected}. "
        f"EITHER bump PRICING_VERSION + add a new EXPECTED_PRICING_DIGEST "
        f"entry (legitimate rate update), OR revert the rate-table change "
        f"(unintentional drift). Per round-11 H4: the digest catches "
        f"silent rate changes that would skew historical replay."
    )
