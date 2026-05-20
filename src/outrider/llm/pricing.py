# Per-model token-cost rate table.
# See specs/2026-05-05-llm-provider-wrapper.md and DECISIONS.md #016 Amended 2026-05-05.
"""Per-model token-cost rate table + `PRICING_VERSION` constant.

Backs `LLMCallEvent.cost_usd` provider-side computation per round 13's
canonical contract: `AnthropicProvider.complete()` step 8 reads this
table, multiplies token counts × rates, and produces `cost_usd`. Step 9
populates `LLMCallEvent.pricing_version` from `PRICING_VERSION` so
historical-row replay reads the version directly rather than depending
on an external version-effective-range map (matches the
`severity-policy-versioned-for-replay` precedent from DECISIONS#001).

Four billable token classes per Anthropic's billing surface (round 14
fold of Codex finding F2):
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
    "PRICING_VERSION",
    "RATE_TABLE",
    "ModelPricing",
    "compute_cost_usd",
    "normalize_to_pricing_key",
]


# Round-27 fold (Copilot) — dated model IDs (e.g., `claude-haiku-4-5-20251001`)
# are accepted by `ModelConfig`'s widened regex (round-21) but `RATE_TABLE` is
# keyed only by undated aliases (`claude-haiku-4-5`). Without normalization,
# `AnthropicProvider`'s eager pricing-coverage check would reject every dated
# pin and step-8 cost computation would `KeyError`. The dated form is the SDK
# catalog's exact-pin shape; pricing is identical to the alias.
#
# Pattern: trailing `-YYYYMMDD` (8 digits, anchored to end). Matches the
# round-21 regex exactly. Does NOT match `-1` or `-2025` (wrong digit count).
_DATED_SUFFIX_PATTERN: Final = re.compile(r"-\d{8}$")


def normalize_to_pricing_key(model: str) -> str:
    """Strip a trailing `-YYYYMMDD` date suffix so dated SDK-catalog pins
    resolve to their undated pricing alias.

    Idempotent: undated input is returned unchanged.

    Examples:
      `claude-haiku-4-5-20251001` → `claude-haiku-4-5`
      `claude-haiku-4-5`          → `claude-haiku-4-5` (unchanged)
      `claude-haiku-4-5-2025`     → `claude-haiku-4-5-2025` (wrong digit count, unchanged)

    `LLMCallEvent.model` records the upstream-returned model identifier
    (`response.model` from the SDK) for audit fidelity — which may be a
    substituted and/or dated ID. Only the pricing-table lookup uses the
    normalized key; the audit row preserves the literal SDK response
    model so replay can reconstruct exactly what executed.
    """
    return _DATED_SUFFIX_PATTERN.sub("", model)


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
# v1 (initial): claude-sonnet-4-7 + claude-haiku-4-5 (round-13)
# v2 (round-20): replaced claude-sonnet-4-7 with claude-sonnet-4-6
#   (the 4-7 model didn't exist in the Anthropic SDK 0.100 catalog;
#   canonical model name correction)
PRICING_VERSION: Final[str] = "v2"
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
# Sources (round-20 update — corrected to current Anthropic 0.100 model
# family per SDK docs; `claude-sonnet-4-7` doesn't exist):
#   - claude-sonnet-4-6: $3.00/MTok input, $15.00/MTok output,
#     $0.30/MTok cache read, $3.75/MTok cache write (1.25× input premium).
#   - claude-haiku-4-5: $1.00/MTok input, $5.00/MTok output,
#     $0.10/MTok cache read, $1.25/MTok cache write.
# Per-token = per-MTok / 1_000_000.
#
# Wrapped in `MappingProxyType` so runtime mutation raises `TypeError`
# (round-16 sharp-edges M2 fold + round-27 defense-in-depth tightening).
# The `Final` annotation alone is a type-checker hint; `MappingProxyType`
# enforces immutability at runtime. A test fixture that does
# `RATE_TABLE["X"] = cheap_pricing` fails loudly instead of silently
# mutating module state for the rest of the pytest session.
#
# Round-27 fold (sweep against round-27 patterns): the dict literal is
# inlined directly into `MappingProxyType(...)` rather than bound to a
# module-level `_RATE_TABLE_RAW` name. The underscore-prefix convention
# alone doesn't prevent `from outrider.llm.pricing import _RATE_TABLE_RAW`
# followed by `_RATE_TABLE_RAW["X"] = cheap_pricing` — same defense-in-
# depth class as Copilot's logging-filter finding (a back-door that
# bypasses the documented protection). The inlined literal has no
# importable name, so the only surface is the immutable proxy.
RATE_TABLE: Final[Mapping[str, ModelPricing]] = MappingProxyType(
    {
        "claude-sonnet-4-6": ModelPricing(
            in_per_token=Decimal("0.000003"),  # 3.00/MTok
            cache_write_per_token=Decimal("0.00000375"),  # 3.75/MTok (1.25× input)
            cache_read_per_token=Decimal("0.0000003"),  # 0.30/MTok (0.10× input)
            out_per_token=Decimal("0.000015"),  # 15.00/MTok
        ),
        "claude-haiku-4-5": ModelPricing(
            in_per_token=Decimal("0.000001"),  # 1.00/MTok
            cache_write_per_token=Decimal("0.00000125"),  # 1.25/MTok
            cache_read_per_token=Decimal("0.0000001"),  # 0.10/MTok
            out_per_token=Decimal("0.000005"),  # 5.00/MTok
        ),
    }
)


def compute_cost_usd(
    model: str,
    *,
    input_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
) -> Decimal:
    """Compute total cost in USD for one LLM call.

    Token args are keyword-only so the four same-typed `int` parameters
    can't be swapped by a positional caller. The swap that matters most
    is `cache_write_tokens` ↔ `cache_read_tokens` — they bill at
    OPPOSITE rates (1.25× input premium for writes vs 0.10× input
    discount for reads), so a swap silently overcharges or undercharges
    by an order of magnitude. Same misuse-resistance pattern as
    `coordinates.tree_sitter_to_github` (round-N keyword-only barrier).

    Four-term sum per spec round-14 fold (Codex finding F2 — earlier
    designs missed cache-write tokens, undercounting Anthropic's actual
    bill since cache writes are premium-rated).

    Returns a `Decimal`; the caller (provider's complete() step 9)
    casts to `float` for `LLMCallEvent.cost_usd`. Raising `KeyError`
    on a missing model is intentional — `AnthropicProvider.__init__`'s
    eager pricing-coverage validation should make this unreachable
    in production (see AC#24).

    **Round-18 fold (variant audit) + round-27 correction (Copilot):**
    wraps in `decimal.localcontext()` AND explicitly resets
    `ctx.prec = 28` so the computation is self-contained against
    ambient `decimal.getcontext().prec` mutations. Round-18 used a bare
    `decimal.localcontext()` which copies the caller's thread context —
    a caller that set `prec=5` before calling would still run cost
    arithmetic at precision 5 and silently truncate. Resetting prec to
    Python's documented default (28) inside the local context produces
    deterministic 28-digit results regardless of caller state.
    """
    import decimal

    with decimal.localcontext() as ctx:
        ctx.prec = 28  # Python's documented default; insulates against caller mutations.
        rates = RATE_TABLE[normalize_to_pricing_key(model)]
        return (
            rates.in_per_token * input_tokens
            + rates.cache_write_per_token * cache_write_tokens
            + rates.cache_read_per_token * cache_read_tokens
            + rates.out_per_token * output_tokens
        )
