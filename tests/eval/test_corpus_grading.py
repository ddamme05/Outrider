"""Unit tests for the corpus grading harness (`corpus_grading`).

Pure-logic tests on SYNTHETIC fixtures — no Juice Shop dependency. Lives under
`tests/eval/` (a package member, relative import) per docs/testing.md's rule
that harness-internal checks live alongside the harness they test; the
Juice-Shop-driven CI gate is `test_juice_shop_corpus_scorecard.py`. Here we pin
the row model, the stage-attribution logic, the classification table, scorecard
determinism, and the not-graded / false-positive paths in isolation.

The `eval(...)` strings below are inert synthetic JS FIXTURES fed to the OBSERVED
detector — parsed for structure, never executed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from . import corpus_grading as cg


# ---------------------------------------------------------------------------
# Discriminated row model.
# ---------------------------------------------------------------------------
def test_ground_truth_loads_both_row_kinds() -> None:
    gt = cg.GroundTruth.model_validate(
        {
            "corpus_root": ".",
            "rows": [
                {
                    "kind": "expected_finding",
                    "file": "a.js",
                    "query_match_id": "javascript.command_injection_eval",
                    "finding_type": "command_injection",
                    "line": 2,
                    "real_vulnerability": True,
                    "current_outcome": "emitted",
                    "rationale": "x",
                },
                {"kind": "expected_clean", "file": "b.js", "rationale": "y"},
            ],
        }
    )
    assert isinstance(gt.rows[0], cg.ExpectedFindingRow)
    assert isinstance(gt.rows[1], cg.ExpectedCleanRow)
    # expected_clean carries no location fields — an absence has no line to invent.
    assert gt.rows[1].query_match_id is None


def test_expected_clean_rejects_finding_fields() -> None:
    """Field-mixing across kinds is rejected — a clean row cannot carry a
    line/finding_type (extra='forbid' on the discriminated variant)."""
    with pytest.raises(ValidationError):
        cg.GroundTruth.model_validate(
            {
                "corpus_root": ".",
                "rows": [
                    {"kind": "expected_clean", "file": "b.js", "line": 3, "rationale": "y"},
                ],
            }
        )


def test_expected_finding_requires_outcome() -> None:
    with pytest.raises(ValidationError):
        cg.GroundTruth.model_validate(
            {
                "corpus_root": ".",
                "rows": [
                    {
                        "kind": "expected_finding",
                        "file": "a.js",
                        "query_match_id": "q",
                        "finding_type": "t",
                        "line": 1,
                        "real_vulnerability": True,
                        "rationale": "x",
                    }
                ],
            }
        )


# ---------------------------------------------------------------------------
# Stage attribution.
# ---------------------------------------------------------------------------
def _obs(*, raw=(), admitted=(), emitted=(), clean=True) -> cg._FileObservation:  # noqa: SLF001
    return cg._FileObservation(  # noqa: SLF001
        parsed_clean=clean,
        raw=frozenset(raw),
        admitted=frozenset(admitted),
        emitted=frozenset(emitted),
    )


def test_outcome_for_attributes_deepest_stage() -> None:
    obs = _obs(
        raw={("q", 5), ("q", 9)},
        admitted={("q", 5)},
        emitted={("q", 5)},
    )
    assert cg._outcome_for(obs, "q", 5) == "emitted"  # noqa: SLF001
    assert cg._outcome_for(obs, "q", 9) == "denied_at_admission"  # raw only  # noqa: SLF001
    assert cg._outcome_for(obs, "q", 1) == "no_raw_match"  # never fired  # noqa: SLF001


def test_outcome_for_admitted_but_not_emitted() -> None:
    obs = _obs(raw={("q", 3)}, admitted={("q", 3)}, emitted=set())
    assert cg._outcome_for(obs, "q", 3) == "denied_at_production"  # noqa: SLF001


# ---------------------------------------------------------------------------
# Classification table.
# ---------------------------------------------------------------------------
def _finding_row(current: str, residual: str | None = None) -> cg.ExpectedFindingRow:
    return cg.ExpectedFindingRow(
        kind="expected_finding",
        file="a.js",
        query_match_id="q",
        finding_type="t",
        line=1,
        real_vulnerability=True,
        current_outcome=current,  # type: ignore[arg-type]
        residual_tag=residual,
        rationale="x",
    )


def test_grade_true_positive() -> None:
    grade, _ = cg._grade_expected_finding(_finding_row("emitted"), "emitted")  # noqa: SLF001
    assert grade == "true_positive"


def test_grade_accepted_miss_when_stage_matches_documented_residual() -> None:
    row = _finding_row("denied_at_admission", "module_presence_file_level")
    grade, detail = cg._grade_expected_finding(row, "denied_at_admission")  # noqa: SLF001
    assert grade == "accepted_miss"
    assert "module_presence_file_level" in detail


def test_grade_regression_when_true_positive_stops_emitting() -> None:
    grade, _ = cg._grade_expected_finding(_finding_row("emitted"), "denied_at_admission")  # noqa: SLF001
    assert grade == "regression"


def test_grade_improvement_when_residual_closes() -> None:
    row = _finding_row("denied_at_admission", "module_presence_file_level")
    grade, detail = cg._grade_expected_finding(row, "emitted")  # noqa: SLF001
    assert grade == "improvement"
    assert "update ground truth" in detail


# ---------------------------------------------------------------------------
# End-to-end grade() over a synthetic corpus (real catalog, no Juice Shop).
# ---------------------------------------------------------------------------
_EVAL_SRC = "export function handler(userInput) {\n  eval(userInput)\n}\n"
_CLEAN_SRC = "export function ok() {\n  return 1 + 2\n}\n"


def _write(tmp_path, name: str, src: str) -> None:
    (tmp_path / name).write_text(src, encoding="utf-8")


def test_grade_end_to_end_true_positive_and_true_negative(tmp_path) -> None:
    _write(tmp_path, "evil.js", _EVAL_SRC)
    _write(tmp_path, "clean.js", _CLEAN_SRC)
    gt = cg.GroundTruth.model_validate(
        {
            "corpus_root": ".",
            "rows": [
                {
                    "kind": "expected_finding",
                    "file": "evil.js",
                    "query_match_id": "javascript.command_injection_eval",
                    "finding_type": "command_injection",
                    "line": 2,
                    "real_vulnerability": True,
                    "current_outcome": "emitted",
                    "rationale": "synthetic eval",
                },
                {"kind": "expected_clean", "file": "clean.js", "rationale": "no sink"},
            ],
        }
    )
    sc = cg.grade(gt, repo_root=tmp_path)
    assert sc.totals.true_positive == 1
    assert sc.totals.true_negative == 1
    assert sc.totals.false_positive == 0
    assert sc.totals.regression == 0


def test_grade_flags_false_positive_on_clean_file(tmp_path) -> None:
    """A file labelled clean that DOES emit grades as a false positive — FP
    detection is positive evidence, not absence of expectations."""
    _write(tmp_path, "evil.js", _EVAL_SRC)
    gt = cg.GroundTruth.model_validate(
        {
            "corpus_root": ".",
            "rows": [{"kind": "expected_clean", "file": "evil.js", "rationale": "wrongly clean"}],
        }
    )
    sc = cg.grade(gt, repo_root=tmp_path)
    assert sc.totals.false_positive == 1
    assert sc.totals.true_negative == 0


def test_grade_not_graded_on_degraded_parse(tmp_path, monkeypatch) -> None:
    """A degraded parse grades `not_graded`, never a silent regression/miss."""
    _write(tmp_path, "broken.ts", "function f( {")

    monkeypatch.setattr(
        cg,
        "_observe_file",
        lambda _root, _f: cg._FileObservation(  # noqa: SLF001
            parsed_clean=False, raw=frozenset(), admitted=frozenset(), emitted=frozenset()
        ),
    )
    gt = cg.GroundTruth.model_validate(
        {
            "corpus_root": ".",
            "rows": [
                {
                    "kind": "expected_finding",
                    "file": "broken.ts",
                    "query_match_id": "javascript.command_injection_eval",
                    "finding_type": "command_injection",
                    "line": 1,
                    "real_vulnerability": True,
                    "current_outcome": "emitted",
                    "rationale": "x",
                }
            ],
        }
    )
    sc = cg.grade(gt, repo_root=tmp_path)
    assert sc.totals.not_graded == 1
    assert sc.totals.regression == 0


# ---------------------------------------------------------------------------
# Scorecard determinism (the checked-in-file equality assert depends on it).
# ---------------------------------------------------------------------------
def test_scorecard_serialization_is_deterministic(tmp_path) -> None:
    _write(tmp_path, "evil.js", _EVAL_SRC)
    gt = cg.GroundTruth.model_validate(
        {
            "corpus_root": ".",
            "rows": [
                {
                    "kind": "expected_finding",
                    "file": "evil.js",
                    "query_match_id": "javascript.command_injection_eval",
                    "finding_type": "command_injection",
                    "line": 2,
                    "real_vulnerability": True,
                    "current_outcome": "emitted",
                    "rationale": "x",
                }
            ],
        }
    )
    a = cg.grade(gt, repo_root=tmp_path).to_json()
    b = cg.grade(gt, repo_root=tmp_path).to_json()
    assert a == b
    assert a.endswith("\n")
    # Keys sorted (stable diffs): the first structural key is `by_query`.
    assert a.index('"by_query"') < a.index('"corpus_root"') < a.index('"row_scores"')
