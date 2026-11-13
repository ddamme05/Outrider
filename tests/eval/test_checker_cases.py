"""Establish authored control mechanisms without a model or repository execution."""

from __future__ import annotations

import pytest

from .checker_cases import CHECKER_CASES

_CASES = {case.id: case for case in CHECKER_CASES}


def _divide(source, *, accepted=None):
    namespace = {} if accepted is None else {"accepted": accepted}
    exec(  # noqa: S102 -- Execute only the tiny authored controls above.
        compile(source, "<authored checker control>", "exec"), namespace
    )
    return namespace["divide"]


@pytest.mark.parametrize(
    ("case_id", "head_error"),
    [
        ("incomplete-zero-guard", ZeroDivisionError),
        ("complete-zero-guard", ValueError),
    ],
)
def test_guard_improvement_closes_the_named_failure_only_when_zero_is_rejected(case_id, head_error):
    case = _CASES[case_id]
    before, after = _divide(case.base_source), _divide(case.head_source)
    with pytest.raises(ZeroDivisionError):
        before(0)
    with pytest.raises(head_error):
        after(0)
    assert before(2) == after(2) == 5


def test_missing_helper_allows_both_zero_outcomes_without_proving_either():
    case = _CASES["unknown-zero-guard"]
    permits_zero = _divide(case.head_source, accepted=lambda _value: True)
    rejects_zero = _divide(case.head_source, accepted=lambda value: value != 0)
    with pytest.raises(ZeroDivisionError):
        permits_zero(0)
    with pytest.raises(ValueError):
        rejects_zero(0)
    assert permits_zero(2) == rejects_zero(2) == 5


def test_control_claim_locations_refer_to_the_same_division():
    assert len({case.id for case in CHECKER_CASES}) == len(CHECKER_CASES)
    for case in CHECKER_CASES:
        assert case.head_source.splitlines()[case.claim.line_start - 1] == case.claim.evidence
