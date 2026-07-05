# Pre-flight budget planner pins per specs/2026-07-05-parallel-analyze.md.
"""The planner mirrors the sequential gate + drawdown; these pins hold it there.

The load-bearing properties: the accounting equation (structural
no-overspend), reserved-then-general attribution, cap-rejects-not-clamps,
all-or-nothing funding, the duplicate-path fail-loud gate, and divisor
lockstep with the worker's real estimator.
"""

from __future__ import annotations

import pytest

from outrider.agent.nodes.analyze_budget import (
    BYTES_PER_TOKEN,
    BudgetPlan,
    DuplicatePlannedPathError,
    FileAllocation,
    FileBudgetRequest,
    plan_file_budgets,
    proxy_estimate_tokens,
)


def _req(path: str, estimate: int, *, high_risk: bool = False) -> FileBudgetRequest:
    return FileBudgetRequest(path=path, estimate_tokens=estimate, is_high_risk=high_risk)


# ---------------------------------------------------------------------------
# Accounting equation (structural no-overspend).
# ---------------------------------------------------------------------------


def test_accounting_equation_holds_per_pool() -> None:
    """Per pool: sum of draws + remainder == pool. This is THE no-overspend
    property — N workers each bounded by an allocation can never exceed the
    pools if the allocations themselves never do."""
    plan = plan_file_budgets(
        (
            _req("a.py", 300, high_risk=True),
            _req("b.py", 200),
            _req("c.py", 500, high_risk=True),
            _req("d.py", 10_000),  # unfunded (over general remainder)
        ),
        general_pool_tokens=1_000,
        reserved_pool_tokens=400,
        per_file_cap_tokens=5_000,
    )
    drawn_general = sum(a.from_general_tokens for a in plan.allocations)
    drawn_reserved = sum(a.from_reserved_tokens for a in plan.allocations)
    assert drawn_general + plan.general_remainder_tokens == plan.general_pool_tokens
    assert drawn_reserved + plan.reserved_remainder_tokens == plan.reserved_pool_tokens
    assert plan.general_remainder_tokens >= 0
    assert plan.reserved_remainder_tokens >= 0


def test_plan_construction_rejects_broken_accounting() -> None:
    """The equation is asserted at BudgetPlan construction (fail loud on a
    planner bug before any Send), not only in tests."""
    alloc = FileAllocation(
        path="a.py",
        estimate_tokens=100,
        is_high_risk=False,
        funded=True,
        from_reserved_tokens=0,
        from_general_tokens=100,
    )
    with pytest.raises(ValueError, match="accounting violated"):
        BudgetPlan(
            allocations=(alloc,),
            general_pool_tokens=1_000,
            reserved_pool_tokens=0,
            general_remainder_tokens=1_000,  # 100 drawn + 1000 != 1000
            reserved_remainder_tokens=0,
        )


# ---------------------------------------------------------------------------
# Reserved-then-general attribution (sequential-drawdown mirror).
# ---------------------------------------------------------------------------


def test_high_risk_draws_reserved_first_then_general_overflow() -> None:
    plan = plan_file_budgets(
        (_req("hot.py", 600, high_risk=True),),
        general_pool_tokens=1_000,
        reserved_pool_tokens=400,
        per_file_cap_tokens=5_000,
    )
    (alloc,) = plan.allocations
    assert alloc.funded
    assert alloc.from_reserved_tokens == 400  # dedicated reserve exhausted first
    assert alloc.from_general_tokens == 200  # overflow only
    assert alloc.allocation_tokens == 600


def test_benign_draws_general_only_and_never_touches_reserve() -> None:
    plan = plan_file_budgets(
        (_req("cold.py", 900),),
        general_pool_tokens=1_000,
        reserved_pool_tokens=400,
        per_file_cap_tokens=5_000,
    )
    (alloc,) = plan.allocations
    assert alloc.funded
    assert alloc.from_reserved_tokens == 0
    assert alloc.from_general_tokens == 900
    assert plan.reserved_remainder_tokens == 400  # untouched


def test_reserve_survives_early_benign_files_for_a_late_high_risk_file() -> None:
    """The reserved-then-general rationale: benign files cannot starve the
    reserve, so a late high-risk file still funds after benign files drained
    general."""
    plan = plan_file_budgets(
        (_req("a.py", 600), _req("b.py", 400), _req("hot.py", 300, high_risk=True)),
        general_pool_tokens=1_000,
        reserved_pool_tokens=400,
        per_file_cap_tokens=5_000,
    )
    a, b, hot = plan.allocations
    assert a.funded and b.funded
    assert plan.general_remainder_tokens == 0
    assert hot.funded
    assert hot.from_reserved_tokens == 300  # reserve carried it


# ---------------------------------------------------------------------------
# Gate semantics: cap rejects (not clamps); all-or-nothing; zero-estimate.
# ---------------------------------------------------------------------------


def test_over_cap_estimate_is_unfunded_not_clamped() -> None:
    """Sequential mirror: `estimate > per_file_cap` skips the file entirely
    (analyze.py cost gate) — the cap is a reject, never a clamp-to-cap."""
    plan = plan_file_budgets(
        (_req("huge.py", 6_000),),
        general_pool_tokens=100_000,
        reserved_pool_tokens=0,
        per_file_cap_tokens=5_000,
    )
    (alloc,) = plan.allocations
    assert not alloc.funded
    assert alloc.allocation_tokens == 0


def test_unfunded_file_draws_nothing_and_later_files_still_fund() -> None:
    """All-or-nothing: an unfunded file leaves the pools untouched, so a
    smaller later file can still fund (sequential parity)."""
    plan = plan_file_budgets(
        (_req("big.py", 900), _req("small.py", 100)),
        general_pool_tokens=500,
        reserved_pool_tokens=0,
        per_file_cap_tokens=5_000,
    )
    big, small = plan.allocations
    assert not big.funded
    assert small.funded
    assert plan.general_remainder_tokens == 400


def test_zero_estimate_is_funded_with_zero_allocation() -> None:
    """`funded=True, allocation=0` is the legitimate zero-estimate case —
    distinct from unfunded (which the worker turns into a budget skip)."""
    plan = plan_file_budgets(
        (_req("empty.py", 0),),
        general_pool_tokens=0,
        reserved_pool_tokens=0,
        per_file_cap_tokens=0,
    )
    (alloc,) = plan.allocations
    assert alloc.funded
    assert alloc.allocation_tokens == 0


def test_empty_request_list_yields_full_remainders() -> None:
    plan = plan_file_budgets(
        (),
        general_pool_tokens=750,
        reserved_pool_tokens=250,
        per_file_cap_tokens=100,
    )
    assert plan.allocations == ()
    assert plan.general_remainder_tokens == 750
    assert plan.reserved_remainder_tokens == 250


# ---------------------------------------------------------------------------
# Duplicate-path gate (vendor-data integrity, before any allocation).
# ---------------------------------------------------------------------------


def test_duplicate_canonical_path_fails_loud_before_any_allocation() -> None:
    """A duplicate path is vendor-data corruption AND two workers sharing a
    (file, pass) slot key — never silently dedup."""
    with pytest.raises(DuplicatePlannedPathError, match="src/dup.py"):
        plan_file_budgets(
            (_req("src/a.py", 10), _req("src/dup.py", 10), _req("src/dup.py", 20)),
            general_pool_tokens=1_000,
            reserved_pool_tokens=0,
            per_file_cap_tokens=100,
        )


def test_negative_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        plan_file_budgets(
            (_req("a.py", -1),),
            general_pool_tokens=10,
            reserved_pool_tokens=0,
            per_file_cap_tokens=10,
        )
    with pytest.raises(ValueError, match="non-negative"):
        plan_file_budgets((), general_pool_tokens=-1, reserved_pool_tokens=0, per_file_cap_tokens=0)


# ---------------------------------------------------------------------------
# Proxy estimator + divisor lockstep.
# ---------------------------------------------------------------------------


def test_proxy_is_monotone_and_covers_fixed_overhead() -> None:
    base = proxy_estimate_tokens(0, 0, fixed_overhead_tokens=500)
    assert base == 500  # zero variable bytes → overhead only
    bigger = proxy_estimate_tokens(3_000, 300, fixed_overhead_tokens=500)
    assert bigger > base
    # DUP_FACTOR=2.0: ceil(2.0 * 3300 / 3) = 2200 variable tokens.
    assert bigger == 500 + 2_200


def test_proxy_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        proxy_estimate_tokens(-1, 0, fixed_overhead_tokens=0)
    with pytest.raises(ValueError, match="non-negative"):
        proxy_estimate_tokens(0, 0, fixed_overhead_tokens=-1)


def test_bytes_per_token_locksteps_with_the_worker_estimator() -> None:
    """The planner's proxy and the worker's real estimate must divide by the
    same conservative-up divisor, or proxy-covers-real is meaningless.
    Cross-module pin (the literal-target-suffixes precedent)."""
    from outrider.agent.nodes.analyze import _BYTES_PER_TOKEN

    assert BYTES_PER_TOKEN == _BYTES_PER_TOKEN


def test_plan_construction_rejects_compensating_negative_draws() -> None:
    """The totals equation alone admits a plan whose books balance while a
    worker overspends (one draw negative, another over-drawn). Per-allocation
    validation kills the class: negative draws, unfunded draws, funded
    allocation != estimate, and benign reserve-draws each fail locally."""
    negative = FileAllocation(
        path="a.py",
        estimate_tokens=100,
        is_high_risk=False,
        funded=True,
        from_reserved_tokens=0,
        from_general_tokens=-100,
    )
    overdrawn = FileAllocation(
        path="b.py",
        estimate_tokens=200,
        is_high_risk=False,
        funded=True,
        from_reserved_tokens=0,
        from_general_tokens=200,
    )
    with pytest.raises(ValueError, match="negative draw"):
        BudgetPlan(
            allocations=(negative, overdrawn),
            general_pool_tokens=100,
            reserved_pool_tokens=0,
            general_remainder_tokens=0,  # books balance: -100 + 200 + 0 == 100
            reserved_remainder_tokens=0,
        )


def test_plan_construction_rejects_unfunded_draw_and_benign_reserve_draw() -> None:
    unfunded_draw = FileAllocation(
        path="c.py",
        estimate_tokens=50,
        is_high_risk=False,
        funded=False,
        from_reserved_tokens=0,
        from_general_tokens=50,
    )
    with pytest.raises(ValueError, match="unfunded draw"):
        BudgetPlan(
            allocations=(unfunded_draw,),
            general_pool_tokens=50,
            reserved_pool_tokens=0,
            general_remainder_tokens=0,
            reserved_remainder_tokens=0,
        )
    benign_reserve = FileAllocation(
        path="d.py",
        estimate_tokens=50,
        is_high_risk=False,
        funded=True,
        from_reserved_tokens=50,
        from_general_tokens=0,
    )
    with pytest.raises(ValueError, match="benign file drew reserve"):
        BudgetPlan(
            allocations=(benign_reserve,),
            general_pool_tokens=0,
            reserved_pool_tokens=50,
            general_remainder_tokens=0,
            reserved_remainder_tokens=0,
        )
