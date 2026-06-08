"""Tests for the deterministic model-tier grading gate (tests/eval/grading.py).

The gate is what stops the analyze STANDARD->Haiku flip from shipping a silent recall
regression (specs/2026-06-08-analyze-tiered-model-routing.md step 2). These tests prove
the machinery itself: the match contract (type+file+line-window+severity), recall /
precision / severity scoring, greedy one-to-one matching (no double-claiming), and that
`compare(...)` FAILS on a recall drop or a false-positive balloon and PASSES on a hold.
Pure — no DB, no LLM, no spend.
"""

from __future__ import annotations

from uuid import uuid4

from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.review_finding import ReviewFinding

from .grading import (
    DEFAULT_LINE_WINDOW,
    ExpectedFinding,
    compare,
    finding_matches,
    grade,
)

# severity -> finding_type, so the `severity-set-by-policy` validator on ReviewFinding is
# satisfied (severity must equal SEVERITY_POLICY[finding_type]).
_TYPE_FOR_SEVERITY = {
    FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
    FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
    FindingSeverity.MEDIUM: FindingType.MISSING_INPUT_VALIDATION,
    FindingSeverity.LOW: FindingType.MISSING_ERROR_HANDLING,
}


def _finding(
    *,
    severity: FindingSeverity = FindingSeverity.CRITICAL,
    file_path: str = "app/db.py",
    line_start: int = 10,
    line_end: int | None = None,
) -> ReviewFinding:
    ft = _TYPE_FOR_SEVERITY[severity]
    le = line_start if line_end is None else line_end  # default to a single line
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=ft,
        severity=severity,
        file_path=file_path,
        line_start=line_start,
        line_end=le,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(ft),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path, line_start=line_start, line_end=le, finding_type=ft
        ),
        proposal_hash="a" * 64,
    )


def _expected(
    *,
    severity: FindingSeverity = FindingSeverity.CRITICAL,
    file_path: str = "app/db.py",
    line_start: int = 10,
    line_end: int | None = None,
) -> ExpectedFinding:
    return ExpectedFinding(
        file_path=file_path,
        line_start=line_start,
        line_end=line_start if line_end is None else line_end,
        finding_type=_TYPE_FOR_SEVERITY[severity],
        severity=severity,
    )


# ---------------------------------------------------------------------------
# finding_matches — the declared match contract
# ---------------------------------------------------------------------------


def test_exact_match() -> None:
    assert finding_matches(_finding(), _expected()) is True


def test_no_match_on_different_type() -> None:
    # HIGH finding (HARDCODED_SECRET) vs CRITICAL expected (SQL_INJECTION) at same place.
    assert finding_matches(_finding(severity=FindingSeverity.HIGH), _expected()) is False


def test_no_match_on_different_file() -> None:
    assert finding_matches(_finding(file_path="app/a.py"), _expected(file_path="app/b.py")) is False


def test_line_within_window_matches() -> None:
    # actual at line 12, expected at line 10, window=2 → 12 <= 10+2 → match.
    assert (
        finding_matches(_finding(line_start=12, line_end=12), _expected(line_start=10, line_end=10))
        is True
    )


def test_line_outside_window_no_match() -> None:
    # actual at line 13, expected at 10, default window 2 → 13 > 12 → no match.
    assert (
        finding_matches(_finding(line_start=13, line_end=13), _expected(line_start=10, line_end=10))
        is False
    )
    # ...but a wider window admits it.
    assert (
        finding_matches(
            _finding(line_start=13, line_end=13),
            _expected(line_start=10, line_end=10),
            line_window=5,
        )
        is True
    )


def test_no_match_on_severity_mismatch() -> None:
    # A ground-truth finding whose declared severity disagrees with what policy gives the
    # type (a deliberately-wrong expected, simulating a policy drift): the severity check
    # rejects it even though type+file+line align.
    actual = _finding(severity=FindingSeverity.CRITICAL)  # SQL_INJECTION -> CRITICAL
    expected = ExpectedFinding(
        file_path="app/db.py",
        line_start=10,
        line_end=10,
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.LOW,  # wrong — policy says CRITICAL
    )
    assert finding_matches(actual, expected) is False


def test_overlapping_ranges_match() -> None:
    # actual [8,14] overlaps expected [10,12] directly (no window needed).
    assert (
        finding_matches(_finding(line_start=8, line_end=14), _expected(line_start=10, line_end=12))
        is True
    )


# ---------------------------------------------------------------------------
# grade — recall / precision / severity / missed / extra
# ---------------------------------------------------------------------------


def test_grade_perfect_recall_and_precision() -> None:
    expected = (_expected(line_start=10), _expected(severity=FindingSeverity.HIGH, line_start=20))
    actual = (_finding(line_start=10), _finding(severity=FindingSeverity.HIGH, line_start=20))
    result = grade(actual, expected)
    assert result.recall.value == 1.0
    assert result.recall.numerator == 2
    assert result.recall.denominator == 2
    assert result.precision.value == 1.0
    assert result.severity_accuracy.value == 1.0
    assert result.missed == ()
    assert result.extra == ()
    assert result.n_false_positives == 0


def test_grade_one_missed_is_a_recall_loss() -> None:
    expected = (_expected(line_start=10), _expected(severity=FindingSeverity.HIGH, line_start=20))
    actual = (_finding(line_start=10),)  # the HIGH one at line 20 is missed
    result = grade(actual, expected)
    assert result.recall.value == 0.5
    assert result.recall.numerator == 1
    assert result.recall.denominator == 2
    assert len(result.missed) == 1
    assert result.missed[0].line_start == 20
    # precision is still perfect — the one finding it DID make was correct.
    assert result.precision.value == 1.0


def test_grade_extra_finding_is_a_false_positive() -> None:
    expected = (_expected(line_start=10),)
    actual = (_finding(line_start=10), _finding(file_path="app/other.py", line_start=99))
    result = grade(actual, expected)
    assert result.recall.value == 1.0  # the expected one was found
    assert result.precision.value == 0.5  # 1 of 2 model findings matched
    assert result.n_false_positives == 1
    assert result.extra[0].file_path == "app/other.py"


def test_grade_greedy_one_to_one_no_double_claim() -> None:
    """Two model findings that both match ONE expected finding don't both count — recall
    is 1/1 (the expected is claimed once), and the second is an extra (false positive).
    Prevents a model from inflating recall by repeating a finding."""
    expected = (_expected(line_start=10),)
    actual = (_finding(line_start=10), _finding(line_start=11))  # both match the one expected
    result = grade(actual, expected)
    assert result.recall.value == 1.0
    assert result.recall.numerator == 1
    assert result.n_false_positives == 1  # the second match-candidate is surplus


def test_grade_empty_expected_is_full_recall() -> None:
    result = grade((), ())
    assert result.recall.value == 1.0
    assert result.precision.value == 1.0


# ---------------------------------------------------------------------------
# compare — the quality gate (recall hold + FP bound)
# ---------------------------------------------------------------------------


def _grade_for(actual: tuple[ReviewFinding, ...], expected: tuple[ExpectedFinding, ...]):
    return grade(actual, expected)


def test_gate_passes_when_recall_holds() -> None:
    expected = (_expected(line_start=10),)
    baseline = _grade_for((_finding(line_start=10),), expected)  # Sonnet: recall 1.0
    candidate = _grade_for((_finding(line_start=10),), expected)  # Haiku: recall 1.0
    cmp = compare(baseline, candidate)
    assert cmp.recall_held is True
    assert cmp.fp_bounded is True
    assert cmp.passes is True


def test_gate_fails_on_recall_regression() -> None:
    """The whole point: Haiku missing a finding Sonnet caught FAILS the gate (default
    zero tolerance)."""
    expected = (_expected(line_start=10), _expected(severity=FindingSeverity.HIGH, line_start=20))
    baseline = _grade_for(
        (_finding(line_start=10), _finding(severity=FindingSeverity.HIGH, line_start=20)),
        expected,
    )  # Sonnet recall 1.0
    candidate = _grade_for((_finding(line_start=10),), expected)  # Haiku recall 0.5 — missed one
    cmp = compare(baseline, candidate)
    assert cmp.recall_held is False
    assert cmp.passes is False


def test_gate_allows_recall_loss_within_declared_tolerance() -> None:
    expected = tuple(
        _expected(line_start=10 * i, severity=FindingSeverity.HIGH) for i in range(1, 11)
    )
    baseline = _grade_for(
        tuple(_finding(line_start=10 * i, severity=FindingSeverity.HIGH) for i in range(1, 11)),
        expected,
    )  # recall 1.0 over 10
    candidate = _grade_for(
        tuple(_finding(line_start=10 * i, severity=FindingSeverity.HIGH) for i in range(1, 10)),
        expected,
    )  # recall 0.9 (missed 1 of 10)
    assert compare(baseline, candidate, recall_tolerance=0.0).passes is False
    assert compare(baseline, candidate, recall_tolerance=0.1).recall_held is True


def test_gate_fails_on_false_positive_balloon() -> None:
    """Recall can hold while precision craters — the gate catches a model that finds
    everything plus a pile of noise."""
    expected = (_expected(line_start=10),)
    baseline = _grade_for((_finding(line_start=10),), expected)  # 0 false positives
    candidate = _grade_for(
        (
            _finding(line_start=10),
            _finding(file_path="x.py", line_start=1),
            _finding(file_path="y.py", line_start=1),
        ),
        expected,
    )  # recall 1.0 but 2 false positives
    cmp = compare(baseline, candidate)
    assert cmp.recall_held is True
    assert cmp.fp_bounded is False
    assert cmp.passes is False
    # ...declared allowance can permit some.
    assert compare(baseline, candidate, fp_allowance=2).fp_bounded is True


def test_default_line_window_is_declared() -> None:
    assert DEFAULT_LINE_WINDOW == 2
