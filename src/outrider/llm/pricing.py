# Host-qualified token-cost rate table, keyed by `(profile_id, model)`.
# See specs/2026-05-05-llm-provider-wrapper.md and DECISIONS.md #016 Amended 2026-05-05.
# Per DECISIONS.md#056: the same model slug served by two hosts bills at two rates,
# so identity is `(profile_id, model)` via `pricing_key`, not `model` alone.
"""Host-qualified token-cost rate table + `PRICING_VERSION` constant.

Backs `LLMCallEvent.cost_usd` provider-side computation per the canonical
contract: `AnthropicProvider.complete()` step 8 reads this table,
multiplies token counts × rates, and produces `cost_usd`. Step 9
populates `LLMCallEvent.pricing_version` from `PRICING_VERSION` so
historical-row replay reads the version directly rather than depending
on an external version-effective-range map (matches the
`severity-policy-versioned-for-replay` precedent from DECISIONS#001).

Four billable token classes per Anthropic's billing surface:
  - regular input (uncached portion)
  - cache write (`cache_creation_input_tokens` — premium over input)
  - cache read (`cache_read_input_tokens` — discount under input)
  - output

`Decimal` precision to avoid IEEE-754 drift; the cast to `float` at
`LLMCallEvent` construction is the documented precision boundary
(deferred to a future DECISIONS amendment per Pending #6 if precision
matters at scales V1 doesn't reach).

When rates change:
  1. Bump `PRICING_VERSION` (the constant string immediately below).
  2. Update `RATE_TABLE` with new `ModelPricing` entries.
  3. Update `EXPECTED_PRICING_DIGEST` in `tests/unit/test_llm_pricing.py`
     so the digest-pinning test catches the change loud.
"""

import re
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Final, NamedTuple

__all__ = [
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


# — dated model IDs (e.g., `claude-haiku-4-5-20251001`)
# are accepted by `ModelConfig`'s widened regex  but `RATE_TABLE` is
# keyed only by undated aliases (`claude-haiku-4-5`). Without normalization,
# `AnthropicProvider`'s eager pricing-coverage check would reject every dated
# pin and step-8 cost computation would `KeyError`. The dated form is the SDK
# catalog's exact-pin shape; pricing is identical to the alias.
#
# Pattern: trailing `-YYYYMMDD` (8 digits, anchored to end). Matches the
# regex exactly. Does NOT match `-1` or `-2025` (wrong digit count).
_DATED_SUFFIX_PATTERN: Final = re.compile(r"-\d{8}$")


def normalize_to_pricing_key(model: str) -> str:
    """Strip a trailing `-YYYYMMDD` date suffix so dated SDK-catalog pins
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

# Bump on every rate-table change. The digest-pinning test in
# `test_llm_pricing.py` fails if rates change without a version bump,
# preventing silent replay drift.
#
# v1 (initial): claude-sonnet-4-7 + claude-haiku-4-5
# v2 : replaced claude-sonnet-4-7 with claude-sonnet-4-6
#   (the 4-7 model didn't exist in the Anthropic SDK 0.100 catalog;
#   canonical model name correction)
# v3 : added zai-org/GLM-5.2 (Baseten) for the GLM provider mode. Existing
#   Anthropic rates unchanged; the bump records v3 on new calls so GLM-era
#   reviews replay under the GLM-aware table. Per-MTok figures confirmed against
#   baseten.co/pricing (1.40 in / 0.26 cached-in / 4.40 out); the Baseten pricing
#   page is not in the docs mirror, so re-verify live if the rate table changes.
# v4 (DECISIONS.md#056): host-qualified re-key. RATE_TABLE + MIN_CACHEABLE_TOKENS
#   are now keyed by `(profile_id, model)` (the same slug served by two hosts bills
#   at two rates), and compute_cost_usd / min_cacheable_tokens take a profile_id.
#   No rate VALUES changed — the bump records v4 so a host-qualified call replays
#   under the host-aware table rather than the v3 model-only one.
# v5 (Sonnet 5 migration): added ("anthropic", "claude-sonnet-5") at its
#   2026-06-30 INTRODUCTORY rate ($2.00 in / $10.00 out per MTok; cache read
#   $0.20 = 0.10× input, cache write $2.50 = 1.25× input). Introductory pricing
#   runs through 2026-08-31; STANDARD pricing ($3.00 in / $15.00 out) takes effect
#   2026-09-01 and needs a bump then (FUP-201, now the v8 slot). The prior
#   claude-sonnet-4-6 rates are RETAINED for historical replay + the GLM-vs-Anthropic
#   scorecard baseline. Anthropic pricing values are runtime-enforced and can drift
#   between snapshots — verify live before that bump.
# v6 (DECISIONS.md#056 amendment 2026-07-06): added
#   ("fireworks", "accounts/fireworks/models/glm-5p2") at $1.40 in / $0.14 cached-read /
#   $4.40 out (Standard path), verified live against fireworks.ai + the 6/30 blog. Fireworks'
#   cached-input DIVERGES from Baseten's $0.26 (dropped to $0.14 on 6/30) — the host-qualified
#   key resolves both. No Anthropic/Baseten rate VALUES changed; the bump records v6 so a
#   Fireworks-host call replays under this table.
# v7 (specs/2026-07-18-openai-native-host.md): added the native OpenAI host — three
#   ("openai", gpt-5.6-{sol,terra,luna}) rows at the Standard short-context rates
#   (mirror openai-api snapshot 2026-07-18: Sol $5.00/$6.25/$0.50/$30.00 per MTok
#   in/write/read/out; Terra $2.50/$3.125/$0.25/$15.00; Luna $1.00/$1.25/$0.10/$6.00;
#   writes are 1.25× input across the family) — PLUS the versioned long-context/tier
#   policy (LONG_CONTEXT_POLICY: >272K reprices the FULL request 2×/2×/2×/1.5×;
#   SERVICE_TIER_MULTIPLIERS: flex 0.5×, priority 2× short-context-only) and the
#   CostUnpricedReason classification AND the tier-echo-expecting profile set
#   (TIER_ECHO_EXPECTED_PROFILE_IDS — openai), all folded into the widened v7 digest.
#   No prior rate VALUES changed. The Sonnet 5 standard-rate bump (FUP-201) now
#   targets v8.
PRICING_VERSION: Final[str] = "v7"
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


# Anthropic per-token rates as of the V1 PRICING_VERSION bump.
# Sources :
#   - claude-sonnet-5: $2.00/MTok input, $10.00/MTok output, $0.20/MTok cache read,
#     $2.50/MTok cache write — 2026-06-30 INTRODUCTORY rate (through 2026-08-31;
#     standard $3.00/$15.00 from 2026-09-01, see PRICING_VERSION v5 note).
#   - claude-sonnet-4-6: $3.00/MTok input, $15.00/MTok output,
#     $0.30/MTok cache read, $3.75/MTok cache write (1.25× input premium).
#   - claude-haiku-4-5: $1.00/MTok input, $5.00/MTok output,
#     $0.10/MTok cache read, $1.25/MTok cache write.
# Per-token = per-MTok / 1_000_000.
#
# Wrapped in `MappingProxyType` so runtime mutation raises `TypeError`
# .
# The `Final` annotation alone is a type-checker hint; `MappingProxyType`
# enforces immutability at runtime. A test fixture that does
# `RATE_TABLE["X"] = cheap_pricing` fails loudly instead of silently
# mutating module state for the rest of the pytest session.
#
# (sweep against patterns): the dict literal is
# inlined directly into `MappingProxyType(...)` rather than bound to a
# module-level `_RATE_TABLE_RAW` name. The underscore-prefix convention
# alone doesn't prevent `from outrider.llm.pricing import _RATE_TABLE_RAW`
# followed by `_RATE_TABLE_RAW["X"] = cheap_pricing` — same defense-in-
# depth class as Copilot's logging-filter finding (a back-door that
# bypasses the documented protection). The inlined literal has no
# importable name, so the only surface is the immutable proxy.
# Keyed by `(profile_id, model)` per DECISIONS.md#056 — the same slug served by two
# hosts bills at two rates. Anthropic-native models sit under `"anthropic"`; GLM 5.2
# under `"baseten"`. Build the key via `pricing_key(profile_id, model)`.
RATE_TABLE: Final[Mapping[tuple[str, str], ModelPricing]] = MappingProxyType(
    {
        ("anthropic", "claude-sonnet-5"): ModelPricing(
            # 2026-06-30 INTRODUCTORY rate (through 2026-08-31; $3.00/$15.00 standard
            # from 2026-09-01 — see the PRICING_VERSION v5 note + FUP-201).
            in_per_token=Decimal("0.000002"),  # 2.00/MTok (intro)
            cache_write_per_token=Decimal("0.0000025"),  # 2.50/MTok (1.25× input)
            cache_read_per_token=Decimal("0.0000002"),  # 0.20/MTok (0.10× input)
            out_per_token=Decimal("0.00001"),  # 10.00/MTok (intro)
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
        # GLM 5.2 on Baseten (PRICING_VERSION v3 added the rates; v4 host-qualified
        # the key). Per-MTok figures confirmed against baseten.co/pricing (not in the
        # docs mirror — re-verify live on a rate change). cache_write_per_token=0:
        # Baseten automatic prefix caching has no cache-write/creation token class;
        # read_usage's PROMPT_INCLUDES_CACHED arm returns cache_write=0, so the write
        # term is inert for this host.
        ("baseten", "zai-org/GLM-5.2"): ModelPricing(
            in_per_token=Decimal("0.0000014"),  # 1.40/MTok
            cache_write_per_token=Decimal("0"),  # no Baseten cache-write class
            cache_read_per_token=Decimal("0.00000026"),  # 0.26/MTok (cached input)
            out_per_token=Decimal("0.0000044"),  # 4.40/MTok
        ),
        # GLM 5.2 on Fireworks (PRICING_VERSION v6; DECISIONS.md#056 amendment 2026-07-06).
        # Verified 2026-07-06 against fireworks.ai/models/fireworks/glm-5p2 AND the 6/30 blog:
        # "$1.40 / $0.14 / $4.40 Per 1M Tokens (input/cached input/output)" — the STANDARD path
        # (the Fast path, $2.10/$0.21/$6.60 at ~2× throughput, is not what this provider calls).
        # cache_read DIVERGES from Baseten's 0.26 (Fireworks dropped cached-input to 0.14 on
        # 6/30) — the "same slug, two hosts, two rates" case the host-qualified key exists for.
        # cache_write_per_token=0: Fireworks automatic prompt caching surfaces no cache-write
        # token class, and the provider sets cache_write_tokens=0 (PROMPT_INCLUDES_CACHED).
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


# >272K-input requests reprice the FULL request (mirror openai-api snapshot
# 2026-07-18, per-model pages: "2x input and 1.5x output for the full request";
# the pricing table's long-context columns are exactly 2×/2×/2×/1.5× of each
# flat row for all three models). Keyed like RATE_TABLE; absence = no
# documented long-context repricing (every non-5.6 key today).
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

    Derives EVERY policy decision internally: the >272K long-context
    determination from the raw `billed_prompt_tokens` wire count, AND whether a
    service-tier echo is expected from `profile_id` membership in the versioned
    `TIER_ECHO_EXPECTED_PROFILE_IDS` — callers supply wire facts only, never
    policy booleans, so an OpenAI caller cannot misprice an absent/auto/scale
    echo as Standard by omitting a flag. Tier-less hosts (Anthropic, the GLM
    hosts) price flat, byte-identical to the pre-v7 behavior. Raises `KeyError`
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

    candidate = LONG_CONTEXT_POLICY.get(key)
    long_policy: LongContextPolicy | None = None
    if (
        candidate is not None
        and billed_prompt_tokens is not None
        and billed_prompt_tokens > candidate.threshold_tokens
    ):
        long_policy = candidate
    if long_policy is not None and expects_tier_echo and service_tier == "priority":
        return Unpriced(CostUnpricedReason.PRIORITY_LONG_CONTEXT)

    with decimal.localcontext() as ctx:
        ctx.prec = 28  # Python's documented default; insulates against caller mutations.
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
    """Compute total cost in USD for one LLM call on `(profile_id, model)`.

    `profile_id` host-qualifies the rate lookup (DECISIONS.md#056): the same model
    slug served by two hosts bills at two rates. At response-derived call sites pass
    `response.profile_id` so the call is priced under the host that served it; the
    parameter is `str | None` to accept that field directly, but a `None` raises —
    an unqualified response can't be priced, and a real provider always stamps the
    host-identity triad, so `None` is a provider/fixture bug, not a billable call.

    Token args are keyword-only so the four same-typed `int` parameters
    can't be swapped by a positional caller. The swap that matters most
    is `cache_write_tokens` ↔ `cache_read_tokens` — they bill at
    OPPOSITE rates (1.25× input premium for writes vs 0.10× input
    discount for reads), so a swap silently overcharges or undercharges
    by an order of magnitude. Same misuse-resistance pattern as
    `coordinates.tree_sitter_to_github`'s keyword-only barrier.

    Four-term sum per spec — earlier designs missed cache-write tokens,
    undercounting Anthropic's actual bill since cache writes are
    premium-rated.

    Returns a `Decimal`; the caller (provider's complete() step 9)
    casts to `float` for `LLMCallEvent.cost_usd`. Raising `KeyError`
    on a missing model is intentional — `AnthropicProvider.__init__`'s
    eager pricing-coverage validation should make this unreachable
    in production (see AC#24).

    The body wraps in `decimal.localcontext()` AND explicitly resets
    `ctx.prec = 28` so the computation is self-contained against
    ambient `decimal.getcontext().prec` mutations. A bare
    `decimal.localcontext()` copies the caller's thread context —
    a caller that set `prec=5` before calling would still run cost
    arithmetic at precision 5 and silently truncate. Resetting prec to
    Python's documented default (28) inside the local context produces
    deterministic 28-digit results regardless of caller state.
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
