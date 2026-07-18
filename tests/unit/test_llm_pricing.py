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
    LONG_CONTEXT_POLICY,
    MIN_CACHEABLE_TOKENS,
    PRICING_VERSION,
    RATE_TABLE,
    SERVICE_TIER_MULTIPLIERS,
    TIER_ECHO_EXPECTED_PROFILE_IDS,
    CostUnpricedReason,
    ModelPricing,
    Priced,
    Unpriced,
    compute_cost_outcome,
    compute_cost_usd,
    min_cacheable_tokens,
    pricing_key,
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
        # ModelConfig() defaults are the anthropic host's models (#056).
        assert pricing_key("anthropic", model) in RATE_TABLE, f"missing pricing entry for {model!r}"


@pytest.mark.parametrize("model_id,pricing", list(RATE_TABLE.items()))
def test_modelpricing_has_all_four_rates(model_id: tuple[str, str], pricing: ModelPricing) -> None:
    assert isinstance(pricing.in_per_token, Decimal)
    assert isinstance(pricing.cache_write_per_token, Decimal)
    assert isinstance(pricing.cache_read_per_token, Decimal)
    assert isinstance(pricing.out_per_token, Decimal)


@pytest.mark.parametrize("model_id,pricing", list(RATE_TABLE.items()))
def test_rates_non_negative(model_id: tuple[str, str], pricing: ModelPricing) -> None:
    """Round-11 H3 fold: `>= 0` accepts the free-promo edge case
    (some models could legitimately have zero cache rates during
    promotional windows)."""
    assert pricing.in_per_token >= 0
    assert pricing.cache_write_per_token >= 0
    assert pricing.cache_read_per_token >= 0
    assert pricing.out_per_token >= 0


@pytest.mark.parametrize("model_id,pricing", list(RATE_TABLE.items()))
def test_billable_ratio_sanity_gates(model_id: tuple[str, str], pricing: ModelPricing) -> None:
    """Per-provider billable-shape sanity over every priced model:
      - cache-read < input rate (cache read is always a discount)
      - output > input rate (output is more expensive than input)
      - cache-write > input rate (premium) ONLY for providers that HAVE a
        cache-write token class. Anthropic charges cache_creation at a
        ~1.25× premium; Baseten/GLM has NO cache-write class (automatic
        prefix caching), so cache_write_per_token=0 and the premium check
        is N/A — GLMProvider always sets cache_write_tokens=0, so the term
        is inert in compute_cost_usd regardless.
    Skips entirely if the input rate is zero (free-promo edge case)."""
    if pricing.in_per_token == 0:
        pytest.skip(f"{model_id!r}: zero rate (free promo) — ratio gates not applicable")
    if pricing.cache_write_per_token > 0:
        assert pricing.cache_write_per_token > pricing.in_per_token, (
            f"{model_id!r}: a provider with a cache-write class must price it "
            f"above the input rate (premium)"
        )
    assert pricing.cache_read_per_token < pricing.in_per_token, (
        f"{model_id!r}: cache_read_per_token should be < in_per_token (discount)"
    )
    assert pricing.out_per_token > pricing.in_per_token, (
        f"{model_id!r}: out_per_token should be > in_per_token"
    )


# ---------------------------------------------------------------------------
# MIN_CACHEABLE_TOKENS floor table.
# ---------------------------------------------------------------------------


def test_min_cacheable_keys_mirror_rate_table() -> None:
    """A model priced without a declared cache floor fails loud here —
    the floor table and the rate table cover the same model set."""
    assert set(MIN_CACHEABLE_TOKENS.keys()) == set(RATE_TABLE.keys())


def test_min_cacheable_floors_match_anthropic_contract() -> None:
    """Values from the LIVE prompt-caching page ("Cache limitations",
    platform.claude.com): Sonnet 4.6 → 1024 (verified 2026-06-10),
    Sonnet 5 → 1024 (verified 2026-06-30, FUP-202), Haiku 4.5 → 4096.
    Floors are runtime API behavior — the live page governs; the pinned
    aegis-docs mirror still carries Sonnet 4.6 at a stale 2048
    (pre-snapshot-drift value)."""
    assert min_cacheable_tokens("anthropic", "claude-sonnet-4-6") == 1024
    assert min_cacheable_tokens("anthropic", "claude-sonnet-5") == 1024
    assert min_cacheable_tokens("anthropic", "claude-haiku-4-5") == 4096


def test_min_cacheable_resolves_dated_pins() -> None:
    assert min_cacheable_tokens("anthropic", "claude-haiku-4-5-20251001") == 4096


def test_min_cacheable_unknown_model_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        min_cacheable_tokens("anthropic", "claude-sonnet-9-9")


# ---------------------------------------------------------------------------
# compute_cost_usd four-class formula.
# ---------------------------------------------------------------------------


def test_compute_cost_usd_four_class_sum() -> None:
    """All four token classes contribute to the total. Round-14 fold per
    Codex finding F2 — earlier designs missed cache-write tokens.
    """
    model = "claude-sonnet-4-6"
    rates = RATE_TABLE[("anthropic", model)]
    expected = (
        rates.in_per_token * 100
        + rates.cache_write_per_token * 50
        + rates.cache_read_per_token * 200
        + rates.out_per_token * 300
    )
    actual = compute_cost_usd(
        "anthropic",
        model,
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
        "anthropic",
        model,
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
    reduced = compute_cost_usd("anthropic", model, **kwargs)
    rates = RATE_TABLE[("anthropic", model)]
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
    cost = compute_cost_usd(
        "anthropic",
        "claude-haiku-4-5",
        input_tokens=1,
        cache_write_tokens=1,
        cache_read_tokens=1,
        output_tokens=1,
    )
    assert isinstance(cost, Decimal)


def test_compute_cost_usd_zero_tokens_is_zero() -> None:
    cost = compute_cost_usd(
        "anthropic",
        "claude-sonnet-4-6",
        input_tokens=0,
        cache_write_tokens=0,
        cache_read_tokens=0,
        output_tokens=0,
    )
    assert cost == Decimal(0)


def test_compute_cost_usd_unknown_model_raises_keyerror() -> None:
    """compute_cost_usd doesn't translate KeyError; AnthropicProvider's
    eager pricing-coverage validation makes this unreachable in
    production, and complete() step 8 wraps it as
    LLMPricingMissingError defensively."""
    with pytest.raises(KeyError):
        compute_cost_usd(
            "anthropic",
            "claude-fake-99-99",
            input_tokens=1,
            cache_write_tokens=1,
            cache_read_tokens=1,
            output_tokens=1,
        )


def test_compute_cost_usd_token_args_are_keyword_only() -> None:
    """Misuse resistance: cache_write_tokens and cache_read_tokens bill
    at OPPOSITE rates (1.25× premium for writes vs 0.10× discount for
    reads). A positional swap would silently overcharge by 12.5×.
    The keyword-only barrier prevents this."""
    with pytest.raises(TypeError, match="positional argument"):
        compute_cost_usd("anthropic", "claude-haiku-4-5", 1, 1, 1, 1)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PRICING_VERSION digest pinning (round-11 H4 fold).
# ---------------------------------------------------------------------------


def _compute_rate_table_digest() -> str:
    """Hash the pricing contract's replay-bearing content.

    v1-v6 hashed RATE_TABLE alone; v7 widened the hash to the full versioned
    policy — long-context threshold/multipliers, service-tier map, and the
    unpriced-reason classification — per specs/2026-07-18-openai-native-host.md,
    so pricing BEHAVIOR outside the rate table cannot drift outside the pin.
    Historical v1-v6 entries are records of the old table-only hash; only the
    current PRICING_VERSION's digest is ever recomputed."""
    items = sorted(RATE_TABLE.items())
    long_context = sorted(LONG_CONTEXT_POLICY.items())
    tiers = sorted(SERVICE_TIER_MULTIPLIERS.items())
    reasons = [reason.value for reason in CostUnpricedReason]
    echo_expecting = sorted(TIER_ECHO_EXPECTED_PROFILE_IDS)
    serialized = repr((items, long_context, tiers, reasons, echo_expecting)).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


# Pinned digest for PRICING_VERSION="v1". When PRICING_VERSION bumps,
# ALSO update this constant; the test below catches mismatched updates.
EXPECTED_PRICING_DIGEST: dict[str, str] = {
    "v1": "a7c01b52255f790e",  # initial — claude-sonnet-4-7 + claude-haiku-4-5
    "v2": "761941ec53c83be1",  # round-20 — replaced claude-sonnet-4-7 with claude-sonnet-4-6
    "v3": "f8b6ddaad3f69eec",  # added zai-org/GLM-5.2 (Baseten) — GLM provider mode
    # v4 (#056): host-qualified re-key — keys are now (profile_id, model) tuples.
    # The digest changed because the keys changed (no rate VALUES changed).
    "v4": "da7b949a35b966ee",
    # v5 (Sonnet 5 migration): added ("anthropic", "claude-sonnet-5") at intro
    # rates; claude-sonnet-4-6 retained for historical replay + scorecard baseline.
    "v5": "e7a9702ea9b8d626",
    # v6 (#056 amendment): added ("fireworks", "accounts/fireworks/models/glm-5p2")
    # at $1.40/$0.14/$4.40 (verified live 2026-07-06). No other rate VALUES changed.
    "v6": "ff3388e38abc33f2",
    # v7 (openai-native-host spec): added the three ("openai", gpt-5.6-*) rows
    # (mirror snapshot 2026-07-18) AND widened the hash itself to cover the
    # long-context/tier policy, unpriced-reason classification, and the
    # tier-echo-expecting profile set. No prior rate VALUES changed; the digest
    # function changed shape at this (unreleased) version — re-pinned in-arc
    # when the audit fold moved tier-echo expectation into the versioned policy.
    "v7": "79d520e08eb10701",
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


# ---------------------------------------------------------------------------
# Round-27 fold (Copilot) — ambient decimal precision insulation.
# ---------------------------------------------------------------------------


def test_compute_cost_usd_insulated_from_ambient_low_precision() -> None:
    """Round-18's bare `decimal.localcontext()` copied the caller's thread
    context, so a caller setting `getcontext().prec = 5` would have
    silently truncated cost arithmetic. Round-27 (Copilot) explicitly
    resets `ctx.prec = 28` inside the local context. This test verifies
    the insulation: pre-set ambient prec=5, run a cost computation that
    would lose digits at prec=5, assert exact-Decimal result anyway."""
    import decimal

    saved_prec = decimal.getcontext().prec
    try:
        decimal.getcontext().prec = 5
        # Token counts chosen to make the four-term sum a long-precision
        # value that prec=5 would round/truncate visibly.
        cost = compute_cost_usd(
            "anthropic",
            "claude-sonnet-4-6",
            input_tokens=123_456,
            cache_write_tokens=78_901,
            cache_read_tokens=234_567,
            output_tokens=345_678,
        )
        rates = RATE_TABLE[("anthropic", "claude-sonnet-4-6")]
        expected = (
            rates.in_per_token * 123_456
            + rates.cache_write_per_token * 78_901
            + rates.cache_read_per_token * 234_567
            + rates.out_per_token * 345_678
        )
        # Recompute `expected` at prec=28 to match the function's local context.
        with decimal.localcontext() as ctx:
            ctx.prec = 28
            expected_at_28 = (
                rates.in_per_token * 123_456
                + rates.cache_write_per_token * 78_901
                + rates.cache_read_per_token * 234_567
                + rates.out_per_token * 345_678
            )
        assert cost == expected_at_28, (
            f"compute_cost_usd ran under ambient prec=5 instead of insulated "
            f"prec=28: got {cost}, expected {expected_at_28} (and "
            f"{expected} computed at the truncated ambient prec)"
        )
    finally:
        decimal.getcontext().prec = saved_prec


# ---------------------------------------------------------------------------
# Round-27 fold (Copilot) — dated model ID normalization for pricing lookup.
# ---------------------------------------------------------------------------


def test_normalize_to_pricing_key_strips_dated_suffix() -> None:
    """Dated SDK-catalog pins (e.g., `claude-haiku-4-5-20251001`) must
    resolve to their undated alias for pricing lookup. The undated alias
    is what `RATE_TABLE` keys on; the dated form is just a precise pin."""
    from outrider.llm.pricing import normalize_to_pricing_key

    assert normalize_to_pricing_key("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert normalize_to_pricing_key("claude-sonnet-4-6-20251015") == "claude-sonnet-4-6"
    assert normalize_to_pricing_key("claude-opus-4-7-20251020") == "claude-opus-4-7"


def test_normalize_to_pricing_key_idempotent_on_undated() -> None:
    """Undated input is returned unchanged."""
    from outrider.llm.pricing import normalize_to_pricing_key

    assert normalize_to_pricing_key("claude-haiku-4-5") == "claude-haiku-4-5"
    assert normalize_to_pricing_key("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_normalize_to_pricing_key_does_not_strip_wrong_digit_count() -> None:
    """Only `-YYYYMMDD` (exactly 8 trailing digits) strips. Wrong-count
    suffixes (typos like `-2025` or `-202510`) stay attached, which is
    correct: they fail the regex validator at construction anyway."""
    from outrider.llm.pricing import normalize_to_pricing_key

    assert normalize_to_pricing_key("claude-haiku-4-5-2025") == "claude-haiku-4-5-2025"
    assert normalize_to_pricing_key("claude-haiku-4-5-202510") == "claude-haiku-4-5-202510"


def test_compute_cost_usd_accepts_dated_model() -> None:
    """Step 8 cost computation must succeed for dated model IDs (which
    `LLMResponse.model` may carry from the SDK response). Without
    normalization, this would `KeyError` on RATE_TABLE lookup."""
    cost_dated = compute_cost_usd(
        "anthropic",
        "claude-sonnet-4-6-20251015",
        input_tokens=100,
        cache_write_tokens=0,
        cache_read_tokens=0,
        output_tokens=50,
    )
    cost_undated = compute_cost_usd(
        "anthropic",
        "claude-sonnet-4-6",
        input_tokens=100,
        cache_write_tokens=0,
        cache_read_tokens=0,
        output_tokens=50,
    )
    assert cost_dated == cost_undated


# ---------------------------------------------------------------------------
# #056 host-qualified key composition + the unqualified-response guard.
# ---------------------------------------------------------------------------


def test_pricing_key_composes_host_and_normalized_model() -> None:
    """`pricing_key` qualifies the model normalizer with the host: `(profile_id,
    normalize_to_pricing_key(model))`. The same slug under two hosts yields two
    distinct keys, and the model part is still date-stripped."""
    assert pricing_key("anthropic", "claude-haiku-4-5-20251001") == (
        "anthropic",
        "claude-haiku-4-5",
    )
    assert pricing_key("baseten", "zai-org/GLM-5.2") == ("baseten", "zai-org/GLM-5.2")
    # Same slug, two hosts → two keys (the whole point of #056).
    assert pricing_key("baseten", "zai-org/GLM-5.2") != pricing_key("deepinfra", "zai-org/GLM-5.2")


def test_compute_cost_usd_none_profile_id_raises() -> None:
    """An unqualified response (profile_id=None) cannot be priced — a real
    provider always stamps the host-identity triad (#056), so reaching pricing
    with None is a provider/fixture bug, surfaced as a loud ValueError rather
    than a confusing KeyError or a silently-misattributed cost."""
    with pytest.raises(ValueError, match="host-qualified profile_id"):
        compute_cost_usd(
            None,
            "claude-haiku-4-5",
            input_tokens=1,
            cache_write_tokens=1,
            cache_read_tokens=1,
            output_tokens=1,
        )


# ---------------------------------------------------------------------------
# Round-27 sweep — RATE_TABLE has no importable mutable backdoor.
# ---------------------------------------------------------------------------


def test_rate_table_raw_dict_not_importable() -> None:
    """Round-27 sweep against round-27 patterns: the round-16 fold added
    `MappingProxyType` so `RATE_TABLE['X'] = cheap_pricing` raises, but
    the underlying `_RATE_TABLE_RAW` dict was a module-level mutable name
    importable as `from outrider.llm.pricing import _RATE_TABLE_RAW` —
    same defense-in-depth class as Copilot's logging-filter finding (a
    back-door bypassing the documented protection). The literal is now
    inlined; no `_RATE_TABLE_RAW` name exists at module scope."""
    import outrider.llm.pricing as pricing_module

    assert not hasattr(pricing_module, "_RATE_TABLE_RAW"), (
        "_RATE_TABLE_RAW must not exist at module scope — the dict literal "
        "is inlined into MappingProxyType so there's no mutable back-door "
        "for `from outrider.llm.pricing import _RATE_TABLE_RAW`."
    )
    # Belt: the public surface still raises on mutation.
    with pytest.raises(TypeError):
        RATE_TABLE["claude-fake"] = ModelPricing(  # type: ignore[index]
            in_per_token=Decimal("0"),
            cache_write_per_token=Decimal("0"),
            cache_read_per_token=Decimal("0"),
            out_per_token=Decimal("0"),
        )


# ---------------------------------------------------------------------------
# v7: long-context / service-tier policy + canonical outcome
# (specs/2026-07-18-openai-native-host.md).
# ---------------------------------------------------------------------------

_SOL_KEY = ("openai", "gpt-5.6-sol")
_TOKENS = dict(
    input_tokens=100_000, cache_write_tokens=2_000, cache_read_tokens=50_000, output_tokens=4_000
)


def _flat_cost(key: tuple[str, str]) -> Decimal:
    r = RATE_TABLE[key]
    return (
        r.in_per_token * _TOKENS["input_tokens"]
        + r.cache_write_per_token * _TOKENS["cache_write_tokens"]
        + r.cache_read_per_token * _TOKENS["cache_read_tokens"]
        + r.out_per_token * _TOKENS["output_tokens"]
    )


def _long_cost(key: tuple[str, str]) -> Decimal:
    r = RATE_TABLE[key]
    lc = LONG_CONTEXT_POLICY[key]
    return (
        r.in_per_token * lc.in_mult * _TOKENS["input_tokens"]
        + r.cache_write_per_token * lc.cache_write_mult * _TOKENS["cache_write_tokens"]
        + r.cache_read_per_token * lc.cache_read_mult * _TOKENS["cache_read_tokens"]
        + r.out_per_token * lc.out_mult * _TOKENS["output_tokens"]
    )


def test_every_gpt56_model_has_long_context_policy() -> None:
    """All three 5.6 rows carry the 272K full-request repricing; no other key does."""
    gpt56_keys = {k for k in RATE_TABLE if k[0] == "openai"}
    assert gpt56_keys == set(LONG_CONTEXT_POLICY.keys())
    for policy in LONG_CONTEXT_POLICY.values():
        assert policy.threshold_tokens == 272_000
        assert (policy.in_mult, policy.cache_write_mult, policy.cache_read_mult) == (
            Decimal("2"),
            Decimal("2"),
            Decimal("2"),
        )
        assert policy.out_mult == Decimal("1.5")


def test_long_context_reprices_full_request_per_model() -> None:
    """Above the threshold, every token class of the FULL request reprices —
    pinned per model, not via a union assertion."""
    for key in LONG_CONTEXT_POLICY:
        outcome = compute_cost_outcome(
            key[0],
            key[1],
            billed_prompt_tokens=272_001,
            service_tier="default",
            expects_tier_echo=True,
            **_TOKENS,
        )
        assert outcome == Priced(_long_cost(key)), key


def test_long_context_boundary_is_strictly_greater() -> None:
    """272_000 exactly stays flat (docs: '>272K'); 272_001 reprices."""
    at = compute_cost_outcome(
        *_SOL_KEY,
        billed_prompt_tokens=272_000,
        service_tier="default",
        expects_tier_echo=True,
        **_TOKENS,
    )
    over = compute_cost_outcome(
        *_SOL_KEY,
        billed_prompt_tokens=272_001,
        service_tier="default",
        expects_tier_echo=True,
        **_TOKENS,
    )
    assert at == Priced(_flat_cost(_SOL_KEY))
    assert over == Priced(_long_cost(_SOL_KEY))


def test_flex_is_half_of_default_short_and_long() -> None:
    flex_short = compute_cost_outcome(
        *_SOL_KEY,
        billed_prompt_tokens=1_000,
        service_tier="flex",
        expects_tier_echo=True,
        **_TOKENS,
    )
    flex_long = compute_cost_outcome(
        *_SOL_KEY,
        billed_prompt_tokens=300_000,
        service_tier="flex",
        expects_tier_echo=True,
        **_TOKENS,
    )
    assert flex_short == Priced(_flat_cost(_SOL_KEY) * Decimal("0.5"))
    assert flex_long == Priced(_long_cost(_SOL_KEY) * Decimal("0.5"))


def test_priority_is_double_short_context_only() -> None:
    """Priority short = 2x the flat row; priority+long has no published rates
    and is Unpriced(PRIORITY_LONG_CONTEXT), never a guessed cost."""
    prio_short = compute_cost_outcome(
        *_SOL_KEY,
        billed_prompt_tokens=1_000,
        service_tier="priority",
        expects_tier_echo=True,
        **_TOKENS,
    )
    prio_long = compute_cost_outcome(
        *_SOL_KEY,
        billed_prompt_tokens=300_000,
        service_tier="priority",
        expects_tier_echo=True,
        **_TOKENS,
    )
    assert prio_short == Priced(_flat_cost(_SOL_KEY) * Decimal("2"))
    assert prio_long == Unpriced(CostUnpricedReason.PRIORITY_LONG_CONTEXT)


def test_unpriceable_tier_taxonomy_per_variant() -> None:
    """Each unpriceable echo maps to its own reason — pinned individually."""
    cases = [
        (None, CostUnpricedReason.ABSENT_TIER),
        ("", CostUnpricedReason.ABSENT_TIER),
        ("auto", CostUnpricedReason.AUTO_TIER),
        ("scale", CostUnpricedReason.SCALE_TIER),
        ("turbo-9000", CostUnpricedReason.NOVEL_TIER),
    ]
    for echo, reason in cases:
        outcome = compute_cost_outcome(
            *_SOL_KEY,
            billed_prompt_tokens=1_000,
            service_tier=echo,
            expects_tier_echo=True,
            **_TOKENS,
        )
        assert outcome == Unpriced(reason), (echo, reason)


def test_tierless_hosts_never_produce_unpriced() -> None:
    """With expects_tier_echo=False (Anthropic/GLM), the same absent/novel echoes
    price flat — absent_tier fires only for echo-expecting profiles."""
    for echo in (None, "", "auto", "scale", "turbo-9000"):
        outcome = compute_cost_outcome(
            "anthropic", "claude-haiku-4-5", service_tier=echo, **_TOKENS
        )
        anthropic_flat = compute_cost_usd("anthropic", "claude-haiku-4-5", **_TOKENS)
        assert outcome == Priced(anthropic_flat), echo


def test_compute_cost_usd_wrapper_refuses_unpriced() -> None:
    """The legacy Decimal wrapper cannot silently swallow an Unpriced outcome."""
    with pytest.raises(ValueError, match="compute_cost_outcome"):
        compute_cost_usd(*_SOL_KEY, service_tier="scale", expects_tier_echo=True, **_TOKENS)


def test_tier_echo_policy_coherent_with_profiles() -> None:
    """The VERSIONED echo-expectation record (this module) and the live profile
    field must agree — the persister prices from the former, the provider from
    the latter; drift would split their classifications."""
    from outrider.llm.host_profiles import HOST_PROFILES

    for host_id, profile in HOST_PROFILES.items():
        assert (profile.requested_service_tier is not None) == (
            host_id in TIER_ECHO_EXPECTED_PROFILE_IDS
        ), host_id
    assert "anthropic" not in TIER_ECHO_EXPECTED_PROFILE_IDS
