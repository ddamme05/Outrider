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
from types import MappingProxyType
from typing import Final, NamedTuple

__all__ = [
    "MIN_CACHEABLE_TOKENS",
    "PRICING_VERSION",
    "RATE_TABLE",
    "ModelPricing",
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
#   2026-09-01 and needs a v6 bump then (FUP-201). The prior claude-sonnet-4-6
#   rates are RETAINED for historical replay + the GLM-vs-Anthropic scorecard
#   baseline. Anthropic pricing values are runtime-enforced and can drift between
#   snapshots — verify live before the v6 bump.
PRICING_VERSION: Final[str] = "v5"
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
        # Baseten automatic prefix caching has no cache-write/creation token class, and
        # the provider always sets cache_write_tokens=0, so the write term is inert.
        ("baseten", "zai-org/GLM-5.2"): ModelPricing(
            in_per_token=Decimal("0.0000014"),  # 1.40/MTok
            cache_write_per_token=Decimal("0"),  # no Baseten cache-write class
            cache_read_per_token=Decimal("0.00000026"),  # 0.26/MTok (cached input)
            out_per_token=Decimal("0.0000044"),  # 4.40/MTok
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
        # min-cacheable floor is 1024 (same as Sonnet 4.6 / Opus 4.8). Runtime-enforced,
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


def compute_cost_usd(
    profile_id: str | None,
    model: str,
    *,
    input_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
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
    import decimal

    if profile_id is None:
        raise ValueError(
            f"compute_cost_usd requires a host-qualified profile_id to price "
            f"model {model!r}; got None. A real provider stamps the host-identity "
            f"triad (DECISIONS.md#056), so an unqualified response reaching pricing "
            f"is a provider/fixture bug, not a billable call."
        )
    with decimal.localcontext() as ctx:
        ctx.prec = 28  # Python's documented default; insulates against caller mutations.
        rates = RATE_TABLE[pricing_key(profile_id, model)]
        return (
            rates.in_per_token * input_tokens
            + rates.cache_write_per_token * cache_write_tokens
            + rates.cache_read_per_token * cache_read_tokens
            + rates.out_per_token * output_tokens
        )
