"""Tests for the deterministic model-tier grading gate (tests/eval/grading.py).

The gate is what stops the analyze STANDARD->Haiku flip from shipping a silent recall
regression (specs/2026-06-08-analyze-tiered-model-routing.md step 2). These tests prove
the machinery itself: the match contract (type+file+line-window+severity), recall /
precision / severity scoring, greedy one-to-one matching (no double-claiming), and that
`compare(...)` FAILS on a recall drop or a false-positive balloon and PASSES on a hold.
Pure — no DB, no LLM, no spend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    import pytest

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


def test_no_match_on_same_severity_different_type() -> None:
    """Isolates the finding_type clause of the match contract. The other no-match cases
    differ in severity too, so the severity check co-fires and the type check goes
    unguarded (deleting it from finding_matches would pass every other test). Here actual
    and expected are BOTH CRITICAL but of different types (SQL_INJECTION vs AUTH_BYPASS at
    the same place) — only the type clause can reject it. This is the clause that separates
    'caught the SQL injection' from 'flagged some other CRITICAL issue on this line', so it
    is exactly what a false 'recall held' verdict on the live flip would hinge on."""
    actual = _finding(severity=FindingSeverity.CRITICAL)  # SQL_INJECTION via _TYPE_FOR_SEVERITY
    expected = ExpectedFinding(
        file_path="app/db.py",
        line_start=10,
        line_end=10,
        finding_type=FindingType.AUTH_BYPASS,  # same severity (CRITICAL), different type
        severity=FindingSeverity.CRITICAL,
    )
    assert finding_matches(actual, expected) is False


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


def test_gate_does_not_vacuously_pass_when_baseline_misses() -> None:
    """A scenario the BASELINE can't solve certifies nothing. If Sonnet found nothing,
    candidate-found-nothing trivially 'holds recall' (0.0 >= 0.0) — the baseline-validity
    guard turns that vacuous pass into a fail so a non-discriminating scenario can't green-
    light the flip."""
    expected = (_expected(line_start=10),)
    baseline = _grade_for((), expected)  # baseline missed it → recall 0.0
    candidate = _grade_for((), expected)  # candidate also missed → recall 0.0
    cmp = compare(baseline, candidate)
    assert cmp.recall_held is True  # vacuously: 0.0 >= 0.0 - 0.0
    assert cmp.baseline_valid is False  # but the baseline never established the finding
    assert cmp.passes is False  # so the gate does NOT pass
    # A caller can lower the floor to accept a partial/imperfect baseline, explicitly.
    assert compare(baseline, candidate, baseline_recall_floor=0.0).baseline_valid is True


def test_gate_baseline_valid_when_baseline_perfect() -> None:
    """The common case: baseline catches the known finding (recall 1.0), so the scenario
    is a valid discriminator and the floor doesn't block a genuine candidate pass."""
    expected = (_expected(line_start=10),)
    baseline = _grade_for((_finding(line_start=10),), expected)  # recall 1.0
    candidate = _grade_for((_finding(line_start=10),), expected)  # recall 1.0
    cmp = compare(baseline, candidate)
    assert cmp.baseline_valid is True
    assert cmp.passes is True


def test_gate_precision_catches_overflag_on_safe_code() -> None:
    """Precision dimension (safe code, EMPTY ground truth): any finding is an unambiguous
    FP. A candidate that flags clean code while the baseline stays clean breaks fp_bounded,
    so the gate fails — this is the over-flag signal the vulnerable fixtures can't give."""
    clean = grade((), ())  # baseline: no findings on safe code → fp 0
    overflag = grade((_finding(),), ())  # candidate: 1 finding on safe code → fp 1
    assert clean.n_false_positives == 0
    assert overflag.n_false_positives == 1
    cmp = compare(clean, overflag)
    assert cmp.baseline_valid is True  # empty ground truth is vacuously valid
    assert cmp.fp_bounded is False  # candidate over-flagged
    assert cmp.passes is False
    # Both clean → precision passes.
    assert compare(clean, grade((), ())).passes is True


def test_default_line_window_is_declared() -> None:
    assert DEFAULT_LINE_WINDOW == 2


# ---------------------------------------------------------------------------
# _print_aggregate_metrics — the scorecard cross-scenario metric block
# ---------------------------------------------------------------------------


def test_aggregate_metrics_yield_recall_fp_and_per_type(capsys: pytest.CaptureFixture[str]) -> None:
    """The scorecard aggregate block (FUP-196 + the best-metrics set) over two scenarios:
    yield rate (per model, over ALL scenarios), mean recall + mean severity accuracy (recall
    dimension only — safe fixtures have vacuous recall), the safe-code OVER-FLAG RATE
    (fp_per_safe_scenario — precision-as-a-ratio and F1 are dropped as dishonest on these
    populations), the all-rows extras diagnostic count, and per-finding-type recall. Baseline
    catches the known SQL_INJECTION and stays clean on the safe fixture; candidate MISSES the
    finding, OVER-FLAGS the safe fixture, and one candidate response was structurally REJECTED
    (the yield signal)."""
    from .test_model_comparison import _print_aggregate_metrics

    expected_sqli = _expected(severity=FindingSeverity.CRITICAL)
    # Recall scenario: baseline catches it (recall 1.0); candidate emits nothing (recall 0.0),
    # and that empty output was a REJECTED structured response (not a valid-empty one).
    recall_cmp = compare(
        grade((_finding(severity=FindingSeverity.CRITICAL),), (expected_sqli,)),
        grade((), (expected_sqli,)),
        candidate_rejected=True,
    )
    # Safe scenario: EMPTY ground truth; baseline clean, candidate over-flags with one finding.
    safe_cmp = compare(
        grade((), ()),
        grade((_finding(severity=FindingSeverity.CRITICAL),), ()),
    )
    results = [
        ("recall_fx.json", "recall", recall_cmp),
        ("safe_fx.json", "precision", safe_cmp),
    ]
    ground_truth_by_fixture: dict[str, tuple[ExpectedFinding, ...]] = {
        "recall_fx.json": (expected_sqli,)
    }

    _print_aggregate_metrics(results, ground_truth_by_fixture, "base-model", "cand-model")
    out = capsys.readouterr().out

    # Partition header + both model rows.
    assert "AGGREGATE — baseline (base-model): 2 scenarios (1 recall / 1 safe)" in out
    assert "AGGREGATE — candidate (cand-model): 2 scenarios (1 recall / 1 safe)" in out
    # Baseline: full recall, no rejected responses, no over-flag, per-type recall 1.00. Its
    # one legit finding matched, so the all-rows diagnostic shows 0 extras over 1 finding.
    assert "yield_rate=1.00 (2/2 parsed)" in out
    assert "mean_recall=1.00   mean_severity_acc=1.00   [recall rows only]" in out
    assert "fp_per_safe_scenario=0.00 (0 fp over 1 safe)" in out
    assert "all_row_extras=0/1 findings" in out
    assert "sql_injection=1.00" in out
    # Candidate: missed the finding (recall 0 + per-type 0), 1 safe-code FP (headline over-flag
    # rate 1.00), 1 of 2 responses rejected (yield 0.50). The safe FP is the only extra, so the
    # all-rows diagnostic shows 1 extra over 1 finding. severity_acc is 1.00 on the miss
    # (n_matched=0 → vacuously 1.0), NOT a precision claim — there is no precision number now.
    assert "yield_rate=0.50 (1/2 parsed)" in out
    assert "mean_recall=0.00   mean_severity_acc=1.00   [recall rows only]" in out
    assert "fp_per_safe_scenario=1.00 (1 fp over 1 safe)" in out
    assert "all_row_extras=1/1 findings" in out
    assert "sql_injection=0.00" in out
    # The dropped metrics must NOT reappear: no precision ratio, no F1 in the output.
    assert "precision=" not in out
    assert "F1=" not in out


def test_aggregate_metrics_empty_results_is_noop(capsys: pytest.CaptureFixture[str]) -> None:
    """No completed scenarios (every paid call errored) prints nothing — the aggregate is
    a report add-on, never a hard failure on an empty run."""
    from .test_model_comparison import _print_aggregate_metrics

    _print_aggregate_metrics([], {}, "base-model", "cand-model")
    assert capsys.readouterr().out == ""
