"""Compare the real structural-query results with the saved Juice Shop scorecard.

Check both the complete saved result and key expectations directly so editing
the scorecard cannot hide a regression. No model runs; corpus files are inputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .corpus_grading import ExpectedFindingRow, grade, load_ground_truth

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_DIR = _REPO_ROOT / "tests/eval/corpus/juice_shop"
_GROUND_TRUTH = _CORPUS_DIR / "ground_truth.json"
_SCORECARD = _REPO_ROOT / "tests/eval/scorecard_juice_shop.json"


def _regrade():
    gt = load_ground_truth(_GROUND_TRUTH)
    return gt, grade(gt, repo_root=_REPO_ROOT)


def test_scorecard_matches_checked_in_artifact() -> None:
    """The regenerated scorecard equals the checked-in file byte-for-byte. If
    this fails, the catalog's measured behavior changed: regenerate the scorecard
    (`corpus_grading.grade(...).to_json()`), review the diff, and reconcile ground
    truth — a residual may have closed or a regression may have shipped."""
    _, scorecard = _regrade()
    regenerated = scorecard.to_json()
    checked_in = _SCORECARD.read_text(encoding="utf-8")
    assert regenerated == checked_in, (
        "Juice Shop scorecard drifted from the checked-in artifact. "
        "Regenerate and review the diff before committing."
    )


def test_no_false_positive_no_regression_no_unexpected() -> None:
    """The load-bearing invariants, asserted directly (not only via the file):
    the catalog produces zero false positives on the clean corpus, no true
    positive regressed, and no unlabeled emission appeared."""
    _, scorecard = _regrade()
    t = scorecard.totals
    assert t.false_positive == 0, "catalog emitted a finding on an expected-clean file"
    assert t.regression == 0, "a documented true positive stopped emitting"
    assert t.unexpected_emission == 0, "an emission has no ground-truth row (unlabeled FP)"
    assert t.not_graded == 0, "a corpus file failed to parse cleanly"


def test_known_detection_and_residual_counts() -> None:
    """Pin the structural finding count and the stage where each miss stops."""
    _, scorecard = _regrade()
    t = scorecard.totals
    assert t.true_positive == 1
    assert t.accepted_miss == 4
    assert t.true_negative == 3
    residuals = Counter(
        (row.observed_outcome, row.residual_tag)
        for row in scorecard.row_scores
        if row.grade == "accepted_miss"
    )
    assert residuals == {
        ("denied_at_admission", "module_presence_file_level"): 2,
        ("denied_at_production", "requires_context"): 2,
    }


def test_every_accepted_miss_names_a_residual() -> None:
    """An accepted miss must cite the deferred mechanism that explains it — a
    real vulnerability the catalog drops is never an unexplained gap."""
    gt, scorecard = _regrade()
    accepted = [r for r in scorecard.row_scores if r.grade == "accepted_miss"]
    assert accepted, "expected at least one accepted miss in the current corpus"
    for row in accepted:
        assert row.residual_tag, f"{row.file}:{row.query_match_id} miss lacks a residual tag"

    # And every ground-truth row that is a real vuln not currently emitted must
    # carry a residual tag (the label side of the same rule).
    for row in gt.rows:
        if (
            isinstance(row, ExpectedFindingRow)
            and row.real_vulnerability
            and row.current_outcome != "emitted"
        ):
            assert row.residual_tag, f"{row.file} real-vuln non-emit lacks a residual tag"
