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

from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import Final, NamedTuple

__all__ = [
    "PRICING_VERSION",
    "RATE_TABLE",
    "ModelPricing",
    "compute_cost_usd",
]


# Bump on every rate-table change. The digest-pinning test in
# `test_llm_pricing.py` fails if rates change without a version bump,
# preventing silent replay drift.
PRICING_VERSION: Final[str] = "v1"


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
# Sources:
#   - claude-sonnet-4-7: $3.00/MTok input, $15.00/MTok output,
#     $0.30/MTok cache read, $3.75/MTok cache write (1.25× input premium).
#   - claude-haiku-4-5: $1.00/MTok input, $5.00/MTok output,
#     $0.10/MTok cache read, $1.25/MTok cache write.
# Per-token = per-MTok / 1_000_000.
#
# Wrapped in `MappingProxyType` so runtime mutation raises `TypeError`
# (round-16 sharp-edges M2 fold). The `Final` annotation alone is a
# type-checker hint; `MappingProxyType` enforces immutability at runtime.
# A test fixture that does `RATE_TABLE["X"] = cheap_pricing` fails
# loudly instead of silently mutating module state for the rest of the
# pytest session.
_RATE_TABLE_RAW: dict[str, ModelPricing] = {
    "claude-sonnet-4-7": ModelPricing(
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
RATE_TABLE: Final[Mapping[str, ModelPricing]] = MappingProxyType(_RATE_TABLE_RAW)


def compute_cost_usd(
    model: str,
    input_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
) -> Decimal:
    """Compute total cost in USD for one LLM call.

    Four-term sum per spec round-14 fold (Codex finding F2 — earlier
    designs missed cache-write tokens, undercounting Anthropic's actual
    bill since cache writes are premium-rated).

    Returns a `Decimal`; the caller (provider's complete() step 9)
    casts to `float` for `LLMCallEvent.cost_usd`. Raising `KeyError`
    on a missing model is intentional — `AnthropicProvider.__init__`'s
    eager pricing-coverage validation should make this unreachable
    in production (see AC#24).

    **Round-18 fold (variant audit):** wraps in `decimal.localcontext()`
    so the computation is self-contained against ambient
    `decimal.getcontext().prec` mutations. A caller in the same thread
    that sets `prec=5` before calling shouldn't silently truncate cost
    arithmetic; the local context (default 28-digit precision) gives
    deterministic results regardless of caller state.
    """
    import decimal

    with decimal.localcontext():
        rates = RATE_TABLE[model]
        return (
            rates.in_per_token * input_tokens
            + rates.cache_write_per_token * cache_write_tokens
            + rates.cache_read_per_token * cache_read_tokens
            + rates.out_per_token * output_tokens
        )
