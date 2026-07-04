"""Graded Juice Shop corpus — the LLM-free CI gate for the JS/TS OBSERVED catalog.

Runs the grading harness over the vendored corpus against the live catalog and
asserts (a) the regenerated scorecard equals the checked-in
`scorecard_juice_shop.json` byte-for-byte, and (b) the headline invariants hold
directly, so a scorecard-file edit can't silently move the floor. This is where
a catalog change that introduces a false positive, regresses a true positive, or
silently closes a documented residual fails loudly.

Structural (no LLM): drives `parse_source` -> `registry.match` ->
`run_observed_matches` -> `produce_observed_findings` over checked-in fixtures.
Lives under `tests/eval/` (a package member, relative import) rather than
`scenarios/structural/`, because the harness it drives is an eval-package module
and structural-scenario files are collected as top-level modules that cannot
import it (see docs/testing.md's harness-tests-live-under-tests/eval rule). It is
still collected LLM-free by `pytest tests/eval --is-eval`.
`specs/2026-07-04-juice-shop-graded-corpus.md`.
"""

from __future__ import annotations

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
    """Pin the current measured shape: 3 true positives (weak_crypto + two eval),
    2 accepted misses (both the module_presence file-level residual on SQL
    injection), 3 true negatives. A change here is a real precision/recall shift
    that must be adjudicated, not silently re-baselined."""
    _, scorecard = _regrade()
    t = scorecard.totals
    assert t.true_positive == 3
    assert t.accepted_miss == 2
    assert t.true_negative == 3


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
