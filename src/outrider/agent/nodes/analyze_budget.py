"""Divide the review budget among files before their workers start.

plan_file_budgets follows file order and gives high-risk files access to a
reserved portion. A file that cannot fit its limit receives no allocation.
Unused allocations are not reassigned when another file skips model review.

The estimate helps scheduling; each worker still checks its actual prompt
estimate before calling the model. See specs/2026-07-05-parallel-analyze.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "BudgetPlan",
    "DuplicatePlannedPathError",
    "FileAllocation",
    "FileBudgetRequest",
    "plan_file_budgets",
    "reserve_context_headroom",
]


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
        # Validate each draw before totals: compensating negative and excess
        # draws could balance the pools while assigning an invalid allocation.
        for allocation in self.allocations:
            if allocation.from_reserved_tokens < 0 or allocation.from_general_tokens < 0:
                raise ValueError(
                    f"budget accounting violated: negative draw for {allocation.path!r}"
                )
            if not allocation.funded and allocation.allocation_tokens != 0:
                raise ValueError(
                    f"budget accounting violated: unfunded draw for {allocation.path!r}"
                )
            if allocation.funded and allocation.allocation_tokens != allocation.estimate_tokens:
                raise ValueError(
                    "budget accounting violated: funded allocation != estimate for "
                    f"{allocation.path!r}"
                )
            if not allocation.is_high_risk and allocation.from_reserved_tokens != 0:
                raise ValueError(
                    f"budget accounting violated: benign file drew reserve for {allocation.path!r}"
                )
        drawn_general = sum(allocation.from_general_tokens for allocation in self.allocations)
        drawn_reserved = sum(allocation.from_reserved_tokens for allocation in self.allocations)
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
    """Allocate in caller order, rejecting duplicate paths before funding.

    Order decides which files receive scarce tokens; analyze supplies DEEP before
    STANDARD. A funded estimate must fit both the per-file cap and its class budget.
    High-risk funding draws reserved tokens first. Allocations are not clamped."""
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


def reserve_context_headroom(
    plan: BudgetPlan,
    context_estimates: dict[str, int],
    *,
    per_file_cap_tokens: int,
) -> BudgetPlan:
    """Fund optional context from unallocated tokens without shrinking file budgets.

    Each estimate includes its file's existing reservation. An extension that
    cannot fit its class pool or per-file cap leaves that file unchanged.
    """
    known_paths = {allocation.path for allocation in plan.allocations}
    if context_estimates.keys() - known_paths:
        raise ValueError("context estimates include a file outside the baseline plan")
    if per_file_cap_tokens < 0 or any(estimate < 0 for estimate in context_estimates.values()):
        raise ValueError("context estimates and per-file cap must be non-negative")
    remaining_general = plan.general_remainder_tokens
    remaining_reserved = plan.reserved_remainder_tokens
    allocations: list[FileAllocation] = []
    for allocation in plan.allocations:
        estimate = context_estimates.get(allocation.path, allocation.estimate_tokens)
        extra_tokens = estimate - allocation.estimate_tokens
        available = (
            remaining_general + remaining_reserved if allocation.is_high_risk else remaining_general
        )
        if (
            allocation.funded
            and extra_tokens > 0
            and estimate <= per_file_cap_tokens
            and extra_tokens <= available
        ):
            from_reserved = min(extra_tokens, remaining_reserved) if allocation.is_high_risk else 0
            from_general = extra_tokens - from_reserved
            remaining_reserved -= from_reserved
            remaining_general -= from_general
            allocation = replace(
                allocation,
                estimate_tokens=estimate,
                from_reserved_tokens=allocation.from_reserved_tokens + from_reserved,
                from_general_tokens=allocation.from_general_tokens + from_general,
            )
        allocations.append(allocation)
    return replace(
        plan,
        allocations=tuple(allocations),
        general_remainder_tokens=remaining_general,
        reserved_remainder_tokens=remaining_reserved,
    )
