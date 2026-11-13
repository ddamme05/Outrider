"""Calculate model-call costs from versioned host and model rates.

Decimal arithmetic keeps recorded USD totals stable. Call records include
their pricing version so stored costs remain explainable after rates change.
Update PRICING_VERSION and the pricing test digest when changing the rate table.
"""

import re
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Final, NamedTuple

__all__ = [
    "COST_UNADJUDICATED_PROFILE_IDS",
    "LONG_CONTEXT_POLICY",
    "TIER_ECHO_EXPECTED_PROFILE_IDS",
    "MIN_CACHEABLE_TOKENS",
    "PRICING_VERSION",
    "RATE_TABLE",
    "SERVICE_TIER_MULTIPLIERS",
    "CostUnpricedReason",
    "LongContextPolicy",
    "ModelPricing",
    "Priced",
    "PricingOutcome",
    "Unpriced",
    "compute_cost_outcome",
    "compute_cost_usd",
    "min_cacheable_tokens",
    "normalize_to_pricing_key",
    "pricing_key",
]


# Dated exact model IDs use their undated alias's rate. Only a trailing
# eight-digit date is stripped; audit records keep the original identifier.
_DATED_SUFFIX_PATTERN: Final = re.compile(r"-\d{8}$")


def normalize_to_pricing_key(model: str) -> str:
    """Strip a trailing `-YYYYMMDD` date suffix so dated exact model-version IDs
    resolve to their undated pricing alias.

    Idempotent: undated input is returned unchanged.

    Examples:
      `claude-haiku-4-5-20251001` → `claude-haiku-4-5`
      `claude-haiku-4-5`          → `claude-haiku-4-5` (unchanged)
      `claude-haiku-4-5-2025`     → `claude-haiku-4-5-2025` (wrong digit count, unchanged)

    `LLMCallEvent.model` records the model identifier the provider chose to
    bill + audit against: `response.model` (the upstream-returned id, possibly
    substituted/dated) for `AnthropicProvider`, but `request.model` for
    `OpenAICompatibleProvider` — which keys on the request because some hosts
    (e.g. Baseten) echo an empty `response.model`. Only the pricing-table lookup
    uses the normalized key; the audit row preserves the provider's chosen model
    id so replay can reconstruct exactly what executed.
    """
    return _DATED_SUFFIX_PATTERN.sub("", model)


def pricing_key(profile_id: str, model: str) -> tuple[str, str]:
    """The host-qualified `RATE_TABLE` / `MIN_CACHEABLE_TOKENS` key (DECISIONS.md#056).

    `(profile_id, normalize_to_pricing_key(model))` — the same model slug served by
    two hosts (Baseten + DeepInfra both serve `zai-org/GLM-5.2`) bills at two rates,
    so pricing identity is `(host, model)`, not `model` alone. `profile_id` is the
    provider's host id (`AnthropicProvider` → `"anthropic"`; `OpenAICompatibleProvider`
    → `profile.host_id`), sourced from `response.profile_id` at response-derived call
    sites so a call is priced under the host that actually served it.

    `normalize_to_pricing_key` stays model-only (its scorecard model-equivalence use
    is not pricing identity); this composer adds the host qualification on top.
    """
    return (profile_id, normalize_to_pricing_key(model))


PRICING_VERSION_PATTERN: Final[str] = r"^v[1-9][0-9]*$"
"""Single-source regex shape for `PRICING_VERSION` strings.

The pricing version uses `vN` (no leading zeros, N >= 1), distinct
from the bare-semver shape `policy_version` uses. Exported so
`audit/events.py` can apply the pattern at the `pricing_version`
field on `LLMCallEvent` and `AnalyzeCompletedEvent` — without the
gate, a malformed value (e.g., "v0", "v1.0", "") could land in the
append-only audit log and break replay reconstruction's
version-keyed cost aggregation.
"""

# Version the host-qualified rates (DECISIONS.md#056) together with long-context,
# service-tier and unpriced-outcome policy (specs/2026-07-18-openai-native-host.md).
# The digest-pinning test rejects unversioned changes. Retain older model rows
# for replay and saved baselines. Current rates are verified against the official
# pricing pages; saved events retain their recorded version and cost.
PRICING_VERSION: Final[str] = "v8"
if not re.fullmatch(PRICING_VERSION_PATTERN, PRICING_VERSION):
    raise RuntimeError(
        f"PRICING_VERSION must match {PRICING_VERSION_PATTERN!r} "
        f"(vN shape, no leading zeros, N >= 1); got {PRICING_VERSION!r}"
    )


class ModelPricing(NamedTuple):
    """Per-token rates for one model. All four rates required.

    Naming mirrors Anthropic's billable token classes:
      - `in_per_token`: regular (uncached) input rate
      - `cache_write_per_token`: cache_creation_input_tokens — premium
      - `cache_read_per_token`: cache_read_input_tokens — discount
      - `out_per_token`: output rate

    `Decimal` per-token rates so the four-term sum in
    `compute_cost_usd()` is exact (no IEEE-754 drift); the float cast
    happens at `LLMCallEvent.cost_usd` construction.
    """

    in_per_token: Decimal
    cache_write_per_token: Decimal
    cache_read_per_token: Decimal
    out_per_token: Decimal


# Rates verified 2026-10-31, in USD per token (per-million rates / 1_000_000).
# Sources: platform.claude.com/docs/en/about-claude/pricing,
# baseten.co/pricing, fireworks.ai/models/fireworks/glm-5p3[-flash], and
# developers.openai.com/api/docs/pricing. Cache writes use the default lifetime
# (Anthropic: 5 minutes; OpenAI: 30 minutes). GLM hosts have no write-token class.
# Keep host-qualified keys: the same model can have different cache rates.
# Inline the backing dictionary so the immutable proxy has no mutable alias.
RATE_TABLE: Final[Mapping[tuple[str, str], ModelPricing]] = MappingProxyType(
    {
        ("anthropic", "claude-sonnet-5"): ModelPricing(
            in_per_token=Decimal("0.000002"),  # 2.00/MTok
            cache_write_per_token=Decimal("0.0000025"),  # 2.50/MTok (1.25× input)
            cache_read_per_token=Decimal("0.0000002"),  # 0.20/MTok (0.10× input)
            out_per_token=Decimal("0.00001"),  # 10.00/MTok
        ),
        ("anthropic", "claude-sonnet-4-6"): ModelPricing(
            in_per_token=Decimal("0.000003"),  # 3.00/MTok
            cache_write_per_token=Decimal("0.00000375"),  # 3.75/MTok (1.25× input)
            cache_read_per_token=Decimal("0.0000003"),  # 0.30/MTok (0.10× input)
            out_per_token=Decimal("0.000015"),  # 15.00/MTok
        ),
        ("anthropic", "claude-haiku-4-5"): ModelPricing(
            in_per_token=Decimal("0.000001"),  # 1.00/MTok
            cache_write_per_token=Decimal("0.00000125"),  # 1.25/MTok
            cache_read_per_token=Decimal("0.0000001"),  # 0.10/MTok
            out_per_token=Decimal("0.000005"),  # 5.00/MTok
        ),
        ("baseten", "zai-org/GLM-5.2"): ModelPricing(
            in_per_token=Decimal("0.0000014"),  # 1.40/MTok
            cache_write_per_token=Decimal("0"),  # no Baseten cache-write class
            cache_read_per_token=Decimal("0.00000014"),  # 0.14/MTok (cached input)
            out_per_token=Decimal("0.0000044"),  # 4.40/MTok
        ),
        ("fireworks", "accounts/fireworks/models/glm-5p2"): ModelPricing(
            in_per_token=Decimal("0.0000014"),  # 1.40/MTok
            cache_write_per_token=Decimal("0"),  # no Fireworks cache-write class
            cache_read_per_token=Decimal("0.00000014"),  # 0.14/MTok (cached input)
            out_per_token=Decimal("0.0000044"),  # 4.40/MTok
        ),
        # GPT-5.6 family on api.openai.com (PRICING_VERSION v7;
        # specs/2026-07-18-openai-native-host.md). Standard SHORT-CONTEXT rates from the
        # openai-api mirror pricing page (snapshot 2026-07-18). Cache writes are a real
        # billed class on 5.6+ (1.25× input); >272K-input requests reprice the FULL
        # request via LONG_CONTEXT_POLICY below — these flat rows are provably correct
        # only together with the provider's pre-flight input ceiling. Explicit slugs
        # only: the `gpt-5.6` alias routes to Sol server-side and is rejected by the
        # profile's model_slug_pattern so the request-side pricing key never desyncs.
        ("openai", "gpt-5.6-sol"): ModelPricing(
            in_per_token=Decimal("0.000005"),  # 5.00/MTok
            cache_write_per_token=Decimal("0.00000625"),  # 6.25/MTok (1.25× input)
            cache_read_per_token=Decimal("0.0000005"),  # 0.50/MTok (0.10× input)
            out_per_token=Decimal("0.00003"),  # 30.00/MTok
        ),
        ("openai", "gpt-5.6-terra"): ModelPricing(
            in_per_token=Decimal("0.0000025"),  # 2.50/MTok
            cache_write_per_token=Decimal("0.000003125"),  # 3.125/MTok (1.25× input)
            cache_read_per_token=Decimal("0.00000025"),  # 0.25/MTok (0.10× input)
            out_per_token=Decimal("0.000015"),  # 15.00/MTok
        ),
        ("openai", "gpt-5.6-luna"): ModelPricing(
            in_per_token=Decimal("0.000001"),  # 1.00/MTok
            cache_write_per_token=Decimal("0.00000125"),  # 1.25/MTok (1.25× input)
            cache_read_per_token=Decimal("0.0000001"),  # 0.10/MTok (0.10× input)
            out_per_token=Decimal("0.000006"),  # 6.00/MTok
        ),
        ("anthropic", "claude-sonnet-5-5"): ModelPricing(
            in_per_token=Decimal("0.000002"),
            cache_write_per_token=Decimal("0.0000025"),
            cache_read_per_token=Decimal("0.0000001"),
            out_per_token=Decimal("0.00001"),
        ),
        ("anthropic", "claude-haiku-5-5"): ModelPricing(
            in_per_token=Decimal("0.0000001"),
            cache_write_per_token=Decimal("0.000000125"),
            cache_read_per_token=Decimal("0.00000001"),
            out_per_token=Decimal("0.0000005"),
        ),
        ("baseten", "zai-org/GLM-5.3"): ModelPricing(
            in_per_token=Decimal("0.0000014"),
            cache_write_per_token=Decimal("0"),
            cache_read_per_token=Decimal("0.00000014"),
            out_per_token=Decimal("0.0000044"),
        ),
        ("baseten", "zai-org/GLM-5.3-Flash"): ModelPricing(
            in_per_token=Decimal("0.00000015"),
            cache_write_per_token=Decimal("0"),
            cache_read_per_token=Decimal("0.00000003"),
            out_per_token=Decimal("0.0000005"),
        ),
        ("fireworks", "accounts/fireworks/models/glm-5p3"): ModelPricing(
            in_per_token=Decimal("0.0000014"),
            cache_write_per_token=Decimal("0"),
            cache_read_per_token=Decimal("0.00000026"),
            out_per_token=Decimal("0.0000044"),
        ),
        ("fireworks", "accounts/fireworks/models/glm-5p3-flash"): ModelPricing(
            in_per_token=Decimal("0.00000015"),
            cache_write_per_token=Decimal("0"),
            cache_read_per_token=Decimal("0.00000003"),
            out_per_token=Decimal("0.0000005"),
        ),
        ("openai", "gpt-6.1-sol"): ModelPricing(
            in_per_token=Decimal("0.000002"),
            cache_write_per_token=Decimal("0.0000025"),
            cache_read_per_token=Decimal("0.0000001"),
            out_per_token=Decimal("0.00001"),
        ),
        ("openai", "gpt-6.1-terra"): ModelPricing(
            in_per_token=Decimal("0.000001"),
            cache_write_per_token=Decimal("0.00000125"),
            cache_read_per_token=Decimal("0.00000005"),
            out_per_token=Decimal("0.000005"),
        ),
        ("openai", "gpt-6-luna"): ModelPricing(
            in_per_token=Decimal("0.0000001"),
            cache_write_per_token=Decimal("0.000000125"),
            cache_read_per_token=Decimal("0.00000001"),
            out_per_token=Decimal("0.0000005"),
        ),
    }
)


# Minimum cacheable prompt length per model (Anthropic prompt-caching
# contract): prompts whose prefix is below the floor are "processed
# without caching, with no error returned" — both cache_creation and
# cache_read report 0. See
# DECISIONS.md#042-analyze-prompt-cache-packs-a-cross-file-invariant-prefix.
# Floors are RUNTIME API behavior, so the LIVE prompt-caching page
# governs the values (platform.claude.com "Cache limitations",
# verified 2026-06-10: Sonnet 4.6 = 1024, Haiku 4.5 = 4096). The pinned
# aegis-docs mirror (anthropic v0.100.0 snapshot) still lists Sonnet 4.6
# at 2048 — a stale snapshot value the live page lowered; the Haiku
# floor agrees in both sources. Keyed by `(profile_id, model)` per
# DECISIONS.md#056, the same shape as RATE_TABLE (dated pins resolve via
# `normalize_to_pricing_key`); `test_llm_pricing.py` asserts the key sets stay
# identical so a model priced without a declared floor fails loud. Same
# inlined-literal + MappingProxyType immutability discipline as RATE_TABLE above.
#
# Value type is `int | None`: `None` is the EXPLICIT unknown-floor sentinel
# (DECISIONS.md#056) for a host with no documented cacheable threshold —
# distinct from `0`, a DOCUMENTED no-floor (Baseten: every request caches). No
# arc-1a host needs the sentinel; it forward-provisions DeepInfra/Fireworks.
MIN_CACHEABLE_TOKENS: Final[Mapping[tuple[str, str], int | None]] = MappingProxyType(
    {
        ("anthropic", "claude-sonnet-4-6"): 1024,
        # Verified against the live prompt-caching page 2026-06-30 (FUP-202): Sonnet 5's
        # min-cacheable floor is 1024 — matching Sonnet 4.6's floor in this table. Runtime-enforced,
        # so re-verify live on a model bump. The None unknown-floor sentinel
        # (DECISIONS.md#056) is still handled by the provider diagnostic for any future
        # host with an undocumented floor, but no current Anthropic model needs it.
        ("anthropic", "claude-sonnet-5"): 1024,
        ("anthropic", "claude-haiku-4-5"): 4096,
        # Baseten documents NO minimum-cacheable-token floor for GLM 5.2
        # ("every request participates in caching automatically") — [MB-11],
        # not in the docs mirror. 0 = DOCUMENTED no-floor (not the None unknown
        # sentinel). The provider does not consult this value (it emits no
        # cache_control marker and has no silently-disabled-cache diagnostic),
        # but the key-set parity test requires an entry for every RATE_TABLE key.
        ("baseten", "zai-org/GLM-5.2"): 0,
        # Fireworks: the captured wire (spikes/fireworks/fixtures/adapted.json) shows caching
        # fire at prompt_tokens=159 (cached=158) with no apparent minimum, and Fireworks
        # documents automatic prompt caching without a published token threshold — 0 =
        # no-floor, same treatment as the sibling GLM host. The OpenAI-compat provider does
        # not consult this (Fireworks caches automatically; the provider reads cached_tokens
        # straight from usage); the entry exists for the RATE_TABLE key-set parity test.
        ("fireworks", "accounts/fireworks/models/glm-5p2"): 0,
        # OpenAI: automatic caching for prompts >= 1024 tokens (prompt-caching guide,
        # mirror snapshot 2026-07-18: "caching is enabled automatically for prompts that
        # are 1024 tokens or longer"). One documented floor for the whole 5.6 family.
        ("openai", "gpt-5.6-sol"): 1024,
        ("openai", "gpt-5.6-terra"): 1024,
        ("openai", "gpt-5.6-luna"): 1024,
        ("anthropic", "claude-sonnet-5-5"): 512,
        ("anthropic", "claude-haiku-5-5"): 512,
        # No model-specific cache floor is documented for these GLM releases.
        ("baseten", "zai-org/GLM-5.3"): None,
        ("baseten", "zai-org/GLM-5.3-Flash"): None,
        ("fireworks", "accounts/fireworks/models/glm-5p3"): None,
        ("fireworks", "accounts/fireworks/models/glm-5p3-flash"): None,
        ("openai", "gpt-6.1-sol"): 1024,
        ("openai", "gpt-6.1-terra"): 1024,
        ("openai", "gpt-6-luna"): 1024,
    }
)


def min_cacheable_tokens(profile_id: str, model: str) -> int | None:
    """Minimum cacheable prompt length (tokens) for `(profile_id, model)`.

    Resolves dated pins through `pricing_key`, same as the rate lookup. Returns
    `None` when the host is priced but documents no cacheable threshold (the
    DECISIONS.md#056 unknown-floor sentinel) — callers treat `None` as "skip the
    floor diagnostic." Raises `KeyError` when the `(profile_id, model)` pair is
    not priced at all — callers that reached pricing already passed the same
    coverage gate, so a miss means MIN_CACHEABLE_TOKENS lagged a RATE_TABLE
    addition.
    """
    return MIN_CACHEABLE_TOKENS[pricing_key(profile_id, model)]


class CostUnpricedReason(StrEnum):
    """Closed reasons a COMPLETED, billed exchange cannot be honestly priced.

    Rides `LLMCallEvent.cost_unpriced_reason`, coupled to a null `cost_usd`
    (priced ⇔ reason absent). The actual vendor tier echo is preserved
    separately as a bounded raw string — a novel echo is never collapsed
    into this enum. Absent-echo fires only for profiles that DECLARE a
    `requested_service_tier` (specs/2026-07-18-openai-native-host.md).
    """

    ABSENT_TIER = "absent_tier"
    AUTO_TIER = "auto_tier"
    SCALE_TIER = "scale_tier"
    NOVEL_TIER = "novel_tier"
    PRIORITY_LONG_CONTEXT = "priority_long_context"
    # A host whose token→billing-class mapping is NOT yet adjudicated against the
    # provider's real billing export. Pricing REFUSES to fabricate a dollar cost from a
    # merely wire-consistent normalization (GPT-5.6 SHAPER v4: writes-inside-prompt is the
    # wire envelope, not billing-confirmed — FUP-247). Cleared per host by removing it from
    # `COST_UNADJUDICATED_PROFILE_IDS` (a versioned change — bump PRICING_VERSION) once the
    # billing artifact confirms/corrects the equation.
    BILLING_PENDING = "billing_pending"


class Priced(NamedTuple):
    """Canonical priced outcome: the exact `Decimal` cost for one call."""

    cost_usd: Decimal


class Unpriced(NamedTuple):
    """Canonical unpriceable outcome: typed reason, no fabricated cost."""

    reason: CostUnpricedReason


PricingOutcome = Priced | Unpriced
"""The ONE pricing authority's result type (specs/2026-07-18-openai-native-host.md):
both providers and `audit/persister.py`'s fresh-write cross-check consume this, so
the unpriceable classification exists exactly once."""


class LongContextPolicy(NamedTuple):
    """Full-request repricing above a flat-rate input-token threshold."""

    threshold_tokens: int
    in_mult: Decimal
    cache_write_mult: Decimal
    cache_read_mult: Decimal
    out_mult: Decimal


# OpenAI reprices the whole request above 272K input tokens (2x input/cache,
# 1.5x output). Haiku 5.5 reprices every token class at 5x above 100K input
# tokens, counting uncached input plus both cache classes. Other rows stay flat.
LONG_CONTEXT_POLICY: Final[Mapping[tuple[str, str], LongContextPolicy]] = MappingProxyType(
    {
        key: LongContextPolicy(
            threshold_tokens=272_000,
            in_mult=Decimal("2"),
            cache_write_mult=Decimal("2"),
            cache_read_mult=Decimal("2"),
            out_mult=Decimal("1.5"),
        )
        for key in (
            ("openai", "gpt-5.6-sol"),
            ("openai", "gpt-5.6-terra"),
            ("openai", "gpt-5.6-luna"),
            ("openai", "gpt-6.1-sol"),
            ("openai", "gpt-6.1-terra"),
            ("openai", "gpt-6-luna"),
        )
    }
    | {
        ("anthropic", "claude-haiku-5-5"): LongContextPolicy(
            threshold_tokens=100_000,
            in_mult=Decimal("5"),
            cache_write_mult=Decimal("5"),
            cache_read_mult=Decimal("5"),
            out_mult=Decimal("5"),
        )
    }
)

# Which profile ids EXPECT a service-tier echo (an absent echo is then
# Unpriced(absent_tier) rather than priced-as-default). Part of the VERSIONED pricing
# policy — folded into the v7 digest — so historical pricing semantics stay derivable
# from this module's record alone: a future HostProfile change (or removal) cannot
# reinterpret how a v7-era event was classified. `HostProfile.requested_service_tier`
# must agree with membership here; a unit test pins the coherence.
TIER_ECHO_EXPECTED_PROFILE_IDS: Final[frozenset[str]] = frozenset({"openai"})

# Which profile ids have a NOT-yet-billing-adjudicated token→class mapping. Their cost is
# Unpriced(BILLING_PENDING) — pricing refuses to fabricate a dollar figure from a merely
# wire-consistent normalization (GPT-5.6 SHAPER v4; FUP-247). VERSIONED policy (folded into
# the v7 digest, same as the tier-echo set): a v7-era event's Unpriced classification stays
# derivable from this module. Removing a host here is a re-admission of its cost — a
# PRICING_VERSION bump — done only once the billing artifact confirms the equation.
COST_UNADJUDICATED_PROFILE_IDS: Final[frozenset[str]] = frozenset({"openai"})

# Echoed-service-tier multipliers over the corresponding default-tier rate
# (mirror pricing tables, snapshot 2026-07-18: Flex = 0.5× Standard short AND
# long; Priority = 2× Standard, SHORT CONTEXT ONLY — no published Priority
# long-context rates, so priority+long is Unpriced(PRIORITY_LONG_CONTEXT)).
SERVICE_TIER_MULTIPLIERS: Final[Mapping[str, Decimal]] = MappingProxyType(
    {
        "default": Decimal("1"),
        "flex": Decimal("0.5"),
        "priority": Decimal("2"),
    }
)


def compute_cost_outcome(
    profile_id: str | None,
    model: str,
    *,
    input_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
    billed_prompt_tokens: int | None = None,
    service_tier: str | None = None,
) -> PricingOutcome:
    """The canonical pricing authority (specs/2026-07-18-openai-native-host.md).

    Derives long-context pricing from token counts and tier requirements from
    the versioned host policy. Anthropic's disjoint input/cache counts sum to
    prompt length; OpenAI uses the raw `billed_prompt_tokens` wire count.
    Callers supply wire facts, never policy booleans. Raises `KeyError`
    on an unpriced `(profile_id, model)` (coverage bug, same as always);
    returns `Unpriced(reason)` for the documented-but-unpriceable echoes rather
    than fabricating a cost.
    """
    import decimal

    if profile_id is None:
        raise ValueError(
            f"compute_cost_outcome requires a host-qualified profile_id to price "
            f"model {model!r}; got None. A real provider stamps the host-identity "
            f"triad (DECISIONS.md#056), so an unqualified response reaching pricing "
            f"is a provider/fixture bug, not a billable call."
        )
    key = pricing_key(profile_id, model)
    rates = RATE_TABLE[key]  # KeyError on unpriced model — intentional, pre-tier.

    expects_tier_echo = profile_id in TIER_ECHO_EXPECTED_PROFILE_IDS
    tier_mult = Decimal("1")
    if expects_tier_echo:
        if not service_tier:
            return Unpriced(CostUnpricedReason.ABSENT_TIER)
        if service_tier == "auto":
            return Unpriced(CostUnpricedReason.AUTO_TIER)
        if service_tier == "scale":
            return Unpriced(CostUnpricedReason.SCALE_TIER)
        if service_tier not in SERVICE_TIER_MULTIPLIERS:
            return Unpriced(CostUnpricedReason.NOVEL_TIER)
        tier_mult = SERVICE_TIER_MULTIPLIERS[service_tier]

    long_context_candidate = LONG_CONTEXT_POLICY.get(key)
    long_policy: LongContextPolicy | None = None
    # Anthropic reports disjoint input classes; all count toward prompt length.
    # OpenAI retains its raw billed-prompt count because its normalization differs.
    prompt_tokens = (
        input_tokens + cache_write_tokens + cache_read_tokens
        if profile_id == "anthropic"
        else billed_prompt_tokens
    )
    if (
        long_context_candidate is not None
        and prompt_tokens is not None
        and prompt_tokens > long_context_candidate.threshold_tokens
    ):
        long_policy = long_context_candidate
    if long_policy is not None and expects_tier_echo and service_tier == "priority":
        return Unpriced(CostUnpricedReason.PRIORITY_LONG_CONTEXT)

    # Billing-unadjudicated host (GPT-5.6 SHAPER v4, FUP-247): the tier/long-context
    # classification above still fires (that evidence is real), but where a dollar cost
    # WOULD be computed, refuse — the writes-inside normalization is wire-consistent, not
    # billing-confirmed. The token split (read_usage) is still recorded; only the $ is
    # withheld until the billing export lands.
    if profile_id in COST_UNADJUDICATED_PROFILE_IDS:
        return Unpriced(CostUnpricedReason.BILLING_PENDING)

    with decimal.localcontext() as cost_context:
        cost_context.prec = 28  # Python's documented default; insulates against caller mutations.
        if long_policy is not None:
            cost = (
                rates.in_per_token * long_policy.in_mult * input_tokens
                + rates.cache_write_per_token * long_policy.cache_write_mult * cache_write_tokens
                + rates.cache_read_per_token * long_policy.cache_read_mult * cache_read_tokens
                + rates.out_per_token * long_policy.out_mult * output_tokens
            )
        else:
            cost = (
                rates.in_per_token * input_tokens
                + rates.cache_write_per_token * cache_write_tokens
                + rates.cache_read_per_token * cache_read_tokens
                + rates.out_per_token * output_tokens
            )
        return Priced(cost * tier_mult)


def compute_cost_usd(
    profile_id: str | None,
    model: str,
    *,
    input_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
    billed_prompt_tokens: int | None = None,
    service_tier: str | None = None,
) -> Decimal:
    """Return the canonical Decimal cost for a host-qualified model call.

    Token counts are keyword-only to prevent swapping billable classes.
    `compute_cost_outcome` owns tier and long-context policy and runs arithmetic
    at local precision 28, independent of the caller's Decimal context.

    A missing rate raises `KeyError`; an unqualified host or `Unpriced` outcome
    raises `ValueError`. Callers that handle unpriced completed calls use
    `compute_cost_outcome` directly. Rates and admission changes require a
    `PRICING_VERSION` bump (`DECISIONS.md#056`).
    """
    if profile_id is None:
        raise ValueError(
            f"compute_cost_usd requires a host-qualified profile_id to price "
            f"model {model!r}; got None. A real provider stamps the host-identity "
            f"triad (DECISIONS.md#056), so an unqualified response reaching pricing "
            f"is a provider/fixture bug, not a billable call."
        )
    # Thin wrapper over the canonical authority (v7): flat/long/tier math lives in
    # `compute_cost_outcome` exactly once. Callers that can legitimately receive an
    # `Unpriced` outcome (the tier-echoing OpenAI path, the persister guard) consume
    # `compute_cost_outcome` directly; reaching an Unpriced result THROUGH this
    # wrapper is a programming error, not a billing state.
    outcome = compute_cost_outcome(
        profile_id,
        model,
        input_tokens=input_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        output_tokens=output_tokens,
        billed_prompt_tokens=billed_prompt_tokens,
        service_tier=service_tier,
    )
    if isinstance(outcome, Unpriced):
        raise ValueError(
            f"compute_cost_usd cannot return a cost for an unpriceable outcome "
            f"({outcome.reason.value!r} on {model!r}); use compute_cost_outcome at "
            f"call sites that handle Unpriced."
        )
    return outcome.cost_usd
