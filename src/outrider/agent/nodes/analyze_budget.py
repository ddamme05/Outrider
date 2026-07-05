# Pre-flight budget planner per specs/2026-07-05-parallel-analyze.md.
"""Pure pre-flight budget allocation for the parallel-analyze fan-out.

The sequential analyze loop enforces its cost cap post-hoc: each file's
gate reads the pools as drawn down by every earlier file. N concurrent
workers cannot share that running state, so the fan-out allocates BEFORE
dispatch: `plan_file_budgets` mirrors the sequential gate and drawdown
exactly, once per file in input (triage) order, and each worker later
enforces its own allocation instead of a shared remainder. No-overspend
is structural — sum of allocations can never exceed the pools (the
accounting equation is asserted at construction) and a worker never
exceeds its allocation — so it does not depend on estimate quality.

Sequential semantics mirrored (analyze.py cost gate + drawdown):

- A file is funded iff its estimate fits BOTH the per-file cap and the
  remaining budget for its class (general + reserved for high-risk,
  general only otherwise). The cap REJECTS, it does not clamp.
- A funded file's allocation equals its estimate (the sequential loop
  debits the estimate, not actual usage), drawn reserved-first for
  high-risk files, general-only for benign files.
- An unfunded file gets zero allocation; the worker emits the
  COST_BUDGET_EXHAUSTED skip (single-emission-point-per-file holds).

One dynamic is deliberately NOT mirrored (spec Non-goals): sequential
later-files-benefit-from-underspend. Allocations here are final; unspent
tokens are not redistributed in V1.5.

This module is pure and import-light (no LLM, prompt, or ast_facts
machinery) so planner tests and structural scenarios exercise it
directly — the `decide_degradation` precedent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "BYTES_PER_TOKEN",
    "DUP_FACTOR",
    "BudgetPlan",
    "DuplicatePlannedPathError",
    "FileAllocation",
    "FileBudgetRequest",
    "plan_file_budgets",
    "proxy_estimate_tokens",
]

# Bytes-per-token divisor. Mirrors analyze's `_BYTES_PER_TOKEN` (a
# cross-module test pins the two equal — the planner's proxy and the
# worker's real estimate must divide by the same conservative-up
# constant or the proxy-covers-real calibration property is meaningless).
BYTES_PER_TOKEN: Final[int] = 3

# Duplication factor for the proxy estimator: the rendered user prompt
# embeds scope excerpts that can OVERLAP (same-file caller/callee context
# duplicates content regions), so content + patch bytes alone bounds
# nothing. Deliberately conservative-high at introduction — an
# over-estimate over-reserves (utilization cost, measured via skip
# counters), never overspends. The fan-out increment's calibration
# property test (proxy >= real rendered estimate on every fixture) is the
# regression gate for tightening it; this is a calibrated heuristic, not
# a proven bound (spec: enforcement rests on the worker's
# real-estimate-vs-allocation check, not on this constant).
DUP_FACTOR: Final[float] = 2.0


class DuplicatePlannedPathError(ValueError):
    """A duplicate canonical path in the kept-file list.

    GitHub's files-list contract is one entry per path, so a duplicate is
    vendor-data corruption (the same integrity class as patch/head
    misalignment) — and two workers sharing a `(file, pass)` slot key
    would corrupt the aggregate merge. Fail loud before any Send; never
    silently dedup (that hides the corruption).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"duplicate canonical path in kept-file list: {path!r}")


def proxy_estimate_tokens(
    content_bytes: int,
    patch_bytes: int,
    *,
    fixed_overhead_tokens: int,
    dup_factor: float = DUP_FACTOR,
) -> int:
    """Conservative pre-render token estimate for one file's analyze call.

    `ceil((dup_factor × (content_bytes + patch_bytes)) / BYTES_PER_TOKEN)
    + fixed_overhead_tokens`. The caller supplies `fixed_overhead_tokens`,
    which MUST cover the same fixed terms the worker's real estimate
    carries: system-prompt tokens + user-template scaffolding + the
    output-token reservation (`analyze_prompt.MAX_TOKENS` — the sequential
    gate's estimate is system + user + MAX_TOKENS). Omitting the output
    reservation systematically under-funds every file by that constant,
    and the worker's real-estimate-vs-allocation check converts the gap
    into spurious COST_BUDGET_EXHAUSTED skips. Computed once per review
    from the real prompt constants — this module stays import-light and
    never renders. Heuristic by design; see `DUP_FACTOR`.
    """
    if content_bytes < 0 or patch_bytes < 0:
        raise ValueError("byte counts must be non-negative")
    if fixed_overhead_tokens < 0:
        raise ValueError("fixed_overhead_tokens must be non-negative")
    variable_bytes = dup_factor * (content_bytes + patch_bytes)
    variable_tokens = int(-(-variable_bytes // BYTES_PER_TOKEN))
    return fixed_overhead_tokens + variable_tokens


@dataclass(frozen=True, slots=True)
class FileBudgetRequest:
    """Planner input for one kept file, in triage order."""

    path: str
    estimate_tokens: int
    is_high_risk: bool


@dataclass(frozen=True, slots=True)
class FileAllocation:
    """Planner verdict for one file.

    `funded=False` means the estimate did not fit (cap or class budget);
    the worker emits the COST_BUDGET_EXHAUSTED skip and spends nothing.
    `funded=True` with `allocation_tokens == 0` is the legitimate
    zero-estimate case, distinct from unfunded.
    """

    path: str
    estimate_tokens: int
    is_high_risk: bool
    funded: bool
    from_reserved_tokens: int
    from_general_tokens: int

    @property
    def allocation_tokens(self) -> int:
        return self.from_reserved_tokens + self.from_general_tokens


@dataclass(frozen=True, slots=True)
class BudgetPlan:
    """The full pre-flight plan. The accounting equation is asserted at
    construction (fail loud on a planner bug, before any Send):
    per pool, sum of draws + remainder == pool."""

    allocations: tuple[FileAllocation, ...]
    general_pool_tokens: int
    reserved_pool_tokens: int
    general_remainder_tokens: int
    reserved_remainder_tokens: int

    def __post_init__(self) -> None:
        # Per-allocation validation FIRST: the totals equation alone admits
        # compensating negatives (one allocation drawing -100 while another
        # over-draws +100 balances the books while a worker overspends), so
        # no-overspend is proven locally, then summed.
        for a in self.allocations:
            if a.from_reserved_tokens < 0 or a.from_general_tokens < 0:
                raise ValueError(f"budget accounting violated: negative draw for {a.path!r}")
            if not a.funded and a.allocation_tokens != 0:
                raise ValueError(f"budget accounting violated: unfunded draw for {a.path!r}")
            if a.funded and a.allocation_tokens != a.estimate_tokens:
                raise ValueError(
                    f"budget accounting violated: funded allocation != estimate for {a.path!r}"
                )
            if not a.is_high_risk and a.from_reserved_tokens != 0:
                raise ValueError(
                    f"budget accounting violated: benign file drew reserve for {a.path!r}"
                )
        drawn_general = sum(a.from_general_tokens for a in self.allocations)
        drawn_reserved = sum(a.from_reserved_tokens for a in self.allocations)
        if drawn_general + self.general_remainder_tokens != self.general_pool_tokens:
            raise ValueError(
                f"budget accounting violated (general): drawn {drawn_general} + "
                f"remainder {self.general_remainder_tokens} != pool "
                f"{self.general_pool_tokens}"
            )
        if drawn_reserved + self.reserved_remainder_tokens != self.reserved_pool_tokens:
            raise ValueError(
                f"budget accounting violated (reserved): drawn {drawn_reserved} + "
                f"remainder {self.reserved_remainder_tokens} != pool "
                f"{self.reserved_pool_tokens}"
            )
        if self.general_remainder_tokens < 0 or self.reserved_remainder_tokens < 0:
            raise ValueError("budget accounting violated: negative remainder")


def plan_file_budgets(
    requests: tuple[FileBudgetRequest, ...],
    *,
    general_pool_tokens: int,
    reserved_pool_tokens: int,
    per_file_cap_tokens: int,
) -> BudgetPlan:
    """Allocate per-file budgets in input (triage) order.

    Mirrors the sequential gate exactly: funded iff
    `estimate <= per_file_cap_tokens` AND estimate fits the remaining
    class budget (general + reserved for high-risk; general only
    otherwise); funded files draw their FULL estimate (reserved-first for
    high-risk). Raises `DuplicatePlannedPathError` on a repeated
    canonical path — before any allocation.
    """
    if general_pool_tokens < 0 or reserved_pool_tokens < 0 or per_file_cap_tokens < 0:
        raise ValueError("pools and per-file cap must be non-negative")
    seen: set[str] = set()
    for request in requests:
        if request.path in seen:
            raise DuplicatePlannedPathError(request.path)
        seen.add(request.path)
        if request.estimate_tokens < 0:
            raise ValueError(f"estimate_tokens must be non-negative: {request.path!r}")

    remaining_general = general_pool_tokens
    remaining_reserved = reserved_pool_tokens
    allocations: list[FileAllocation] = []
    for request in requests:
        available = (
            remaining_general + remaining_reserved if request.is_high_risk else remaining_general
        )
        funded = (
            request.estimate_tokens <= per_file_cap_tokens and request.estimate_tokens <= available
        )
        from_reserved = 0
        from_general = 0
        if funded:
            if request.is_high_risk:
                from_reserved = min(request.estimate_tokens, remaining_reserved)
                from_general = request.estimate_tokens - from_reserved
            else:
                from_general = request.estimate_tokens
            remaining_reserved -= from_reserved
            remaining_general -= from_general
        allocations.append(
            FileAllocation(
                path=request.path,
                estimate_tokens=request.estimate_tokens,
                is_high_risk=request.is_high_risk,
                funded=funded,
                from_reserved_tokens=from_reserved,
                from_general_tokens=from_general,
            )
        )
    return BudgetPlan(
        allocations=tuple(allocations),
        general_pool_tokens=general_pool_tokens,
        reserved_pool_tokens=reserved_pool_tokens,
        general_remainder_tokens=remaining_general,
        reserved_remainder_tokens=remaining_reserved,
    )
