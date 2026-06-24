"""Tests for deterministic triage-tier grading (tests/eval/triage_grading.py).

Pins the analysis-floor safety rubric: an expected-analyzed file (DEEP or
STANDARD) the candidate pushes below the floor (SKIM/SKIP) counts as
`n_dropped_from_analysis` and fails the gate — the case the initial DEEP-only
framing missed. Also covers the softer DEEP->STANDARD downgrade, over-tiering,
dimension recall/precision, under-risking, the asymmetric `compare_triage` gate,
and the runner end-to-end through the real triage node via scripted providers
(zero-spend, no audit events — the scripted provider does not persist).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel, TriageResult

from .scorecard import Scorecard, TriageScorecardRow
from .test_model_comparison import _build_state, _ScriptedProvider
from .triage_grading import (
    ExpectedTriage,
    compare_triage,
    compare_triage_models_on_scenario,
    grade_triage,
    run_triage_under_model,
)

_BASELINE = "claude-sonnet-4-6"
_CANDIDATE = "claude-haiku-4-5"


def _triage(
    file_tiers: dict[str, ReviewTier],
    *,
    risk: RiskLevel = RiskLevel.HIGH,
    dims: tuple[ReviewDimension, ...] = (ReviewDimension.SECURITY,),
) -> TriageResult:
    return TriageResult(
        file_tiers=file_tiers, overall_risk=risk, relevant_dimensions=dims, reasoning="r"
    )


def _expected(
    file_tiers: dict[str, ReviewTier],
    *,
    risk: RiskLevel = RiskLevel.HIGH,
    dims: tuple[ReviewDimension, ...] = (ReviewDimension.SECURITY,),
) -> ExpectedTriage:
    return ExpectedTriage(
        expected_file_tiers=file_tiers, overall_risk=risk, relevant_dimensions=dims
    )


# --- grade_triage -----------------------------------------------------------


def test_grade_triage_exact_match() -> None:
    g = grade_triage(_triage({"a.py": ReviewTier.DEEP}), _expected({"a.py": ReviewTier.DEEP}))
    assert g.tier_accuracy == 1.0
    assert g.n_dropped_from_analysis == 0
    assert g.n_deep_downgraded == 0
    assert g.n_overtiered == 0
    assert g.risk_correct is True
    assert g.under_risked is False
    assert g.dimension_recall == 1.0
    assert g.dimension_precision == 1.0


def test_grade_triage_standard_to_skim_is_a_drop() -> None:
    # The case the DEEP-only framing missed: expected STANDARD, candidate SKIM ->
    # the file leaves the analysis set entirely.
    g = grade_triage(_triage({"a.py": ReviewTier.SKIM}), _expected({"a.py": ReviewTier.STANDARD}))
    assert g.n_dropped_from_analysis == 1
    assert g.dropped_files == ("a.py",)
    assert g.n_deep_downgraded == 0


def test_grade_triage_deep_to_skip_is_a_drop() -> None:
    g = grade_triage(_triage({"a.py": ReviewTier.SKIP}), _expected({"a.py": ReviewTier.DEEP}))
    assert g.n_dropped_from_analysis == 1


def test_grade_triage_deep_to_standard_is_a_downgrade_not_a_drop() -> None:
    # Still reviewed, by the cheaper model — softer than a drop.
    g = grade_triage(_triage({"a.py": ReviewTier.STANDARD}), _expected({"a.py": ReviewTier.DEEP}))
    assert g.n_dropped_from_analysis == 0
    assert g.n_deep_downgraded == 1


def test_grade_triage_overtiering() -> None:
    # expected SKIM (not analyzed), candidate DEEP -> wasted analysis (cost).
    g = grade_triage(_triage({"a.py": ReviewTier.DEEP}), _expected({"a.py": ReviewTier.SKIM}))
    assert g.n_overtiered == 1
    assert g.n_dropped_from_analysis == 0


def test_grade_triage_missing_actual_file_is_a_drop() -> None:
    # ground truth expects a.py analyzed; actual omits it -> treated as SKIP -> drop.
    g = grade_triage(_triage({"b.py": ReviewTier.DEEP}), _expected({"a.py": ReviewTier.DEEP}))
    assert g.n_dropped_from_analysis == 1


def test_grade_triage_dimension_recall_and_precision() -> None:
    g = grade_triage(
        _triage({"a.py": ReviewTier.DEEP}, dims=(ReviewDimension.SECURITY,)),
        _expected(
            {"a.py": ReviewTier.DEEP},
            dims=(ReviewDimension.SECURITY, ReviewDimension.PERFORMANCE),
        ),
    )
    assert g.dimension_recall == 0.5  # caught 1 of 2 expected
    assert g.dimension_precision == 1.0  # its 1 dimension was expected


def test_grade_triage_under_risked() -> None:
    g = grade_triage(
        _triage({"a.py": ReviewTier.DEEP}, risk=RiskLevel.LOW),
        _expected({"a.py": ReviewTier.DEEP}, risk=RiskLevel.HIGH),
    )
    assert g.under_risked is True
    assert g.risk_correct is False


# --- compare_triage (the asymmetric gate) -----------------------------------


def test_compare_triage_passes_when_candidate_clean() -> None:
    expected = _expected({"a.py": ReviewTier.DEEP})
    base = grade_triage(_triage({"a.py": ReviewTier.DEEP}), expected)
    cand = grade_triage(_triage({"a.py": ReviewTier.DEEP}), expected)
    cmp = compare_triage(base, cand)
    assert cmp.baseline_valid is True
    assert cmp.passes is True


def test_compare_triage_fails_on_new_drop() -> None:
    expected = _expected({"a.py": ReviewTier.STANDARD})
    base = grade_triage(_triage({"a.py": ReviewTier.STANDARD}), expected)  # baseline analyzes it
    cand = grade_triage(_triage({"a.py": ReviewTier.SKIM}), expected)  # candidate drops it
    cmp = compare_triage(base, cand)
    assert cmp.baseline_valid is True
    assert cmp.drop_held is False
    assert cmp.passes is False


def test_compare_triage_vacuous_baseline_fails() -> None:
    # Baseline itself drops the file -> can't discriminate -> baseline_valid False.
    expected = _expected({"a.py": ReviewTier.STANDARD})
    base = grade_triage(_triage({"a.py": ReviewTier.SKIM}), expected)
    cand = grade_triage(_triage({"a.py": ReviewTier.SKIM}), expected)
    cmp = compare_triage(base, cand)
    assert cmp.baseline_valid is False
    assert cmp.passes is False


def test_compare_triage_fails_on_under_risk() -> None:
    expected = _expected({"a.py": ReviewTier.DEEP}, risk=RiskLevel.HIGH)
    base = grade_triage(_triage({"a.py": ReviewTier.DEEP}, risk=RiskLevel.HIGH), expected)
    cand = grade_triage(_triage({"a.py": ReviewTier.DEEP}, risk=RiskLevel.LOW), expected)
    cmp = compare_triage(base, cand)
    assert cmp.risk_safety_held is False
    assert cmp.passes is False


def test_compare_triage_fails_on_dimension_recall_regression() -> None:
    expected = _expected(
        {"a.py": ReviewTier.DEEP},
        dims=(ReviewDimension.SECURITY, ReviewDimension.PERFORMANCE),
    )
    base = grade_triage(
        _triage(
            {"a.py": ReviewTier.DEEP},
            dims=(ReviewDimension.SECURITY, ReviewDimension.PERFORMANCE),
        ),
        expected,
    )  # recall 1.0
    cand = grade_triage(
        _triage({"a.py": ReviewTier.DEEP}, dims=(ReviewDimension.SECURITY,)), expected
    )  # recall 0.5
    cmp = compare_triage(base, cand)
    assert cmp.dimension_recall_held is False
    assert cmp.passes is False


def test_compare_triage_deep_downgrade_does_not_fail_gate() -> None:
    # A DEEP->STANDARD downgrade is reported but not a hard safety fail.
    expected = _expected({"a.py": ReviewTier.DEEP})
    base = grade_triage(_triage({"a.py": ReviewTier.DEEP}), expected)
    cand = grade_triage(_triage({"a.py": ReviewTier.STANDARD}), expected)
    cmp = compare_triage(base, cand)
    assert cmp.candidate.n_deep_downgraded == 1
    assert cmp.passes is True  # still reviewed, not dropped


# --- run_triage_under_model + compare_triage_models_on_scenario (real node) --

_TRIAGE_DEEP = json.dumps(
    {
        "file_tiers": {"src/example.py": "deep"},
        "overall_risk": "high",
        "relevant_dimensions": ["security"],
        "reasoning": "deep-review the changed function",
    }
)
_TRIAGE_SKIM = json.dumps(
    {
        "file_tiers": {"src/example.py": "skim"},
        "overall_risk": "high",
        "relevant_dimensions": ["security"],
        "reasoning": "skim only",
    }
)


async def test_run_triage_under_model_returns_triage_result() -> None:
    result = await run_triage_under_model(
        _build_state(), provider=_ScriptedProvider(_TRIAGE_DEEP), model=_CANDIDATE
    )
    assert result.file_tiers["src/example.py"] is ReviewTier.DEEP
    assert result.overall_risk is RiskLevel.HIGH
    assert ReviewDimension.SECURITY in result.relevant_dimensions


async def test_compare_triage_models_on_scenario_catches_a_drop() -> None:
    # End-to-end through the real triage node: baseline tiers DEEP, candidate SKIM
    # (the node accepts SKIM; only SKIP is policy-rejected) -> the drop trips the gate.
    expected = ExpectedTriage(
        expected_file_tiers={"src/example.py": ReviewTier.DEEP},
        overall_risk=RiskLevel.HIGH,
        relevant_dimensions=(ReviewDimension.SECURITY,),
    )
    cmp = await compare_triage_models_on_scenario(
        _build_state(),
        expected,
        baseline_provider=_ScriptedProvider(_TRIAGE_DEEP),
        baseline_model=_BASELINE,
        candidate_provider=_ScriptedProvider(_TRIAGE_SKIM),
        candidate_model=_CANDIDATE,
    )
    assert cmp.baseline_valid is True
    assert cmp.drop_held is False
    assert cmp.passes is False


# --- TriageScorecardRow + Scorecard.triage_rows -----------------------------


def test_triage_scorecard_row_from_comparison() -> None:
    expected = _expected({"a.py": ReviewTier.DEEP})
    cmp = compare_triage(
        grade_triage(_triage({"a.py": ReviewTier.DEEP}), expected),
        grade_triage(_triage({"a.py": ReviewTier.DEEP}), expected),
    )
    row = TriageScorecardRow.from_comparison(
        scenario="s", model=_CANDIDATE, baseline_model=_BASELINE, comparison=cmp
    )
    assert row.node == "triage"
    assert row.status == "ok"
    assert row.tier_accuracy == 1.0
    assert row.n_dropped_from_analysis == 0
    assert row.gate is not None and row.gate.passes is True
    assert row.triage_source == "run_triage_direct"


def test_triage_scorecard_row_errored_has_null_metrics() -> None:
    row = TriageScorecardRow.errored(
        scenario="s", model=_CANDIDATE, baseline_model=_BASELINE, error="529 overloaded"
    )
    assert row.status == "errored"
    assert row.tier_accuracy is None
    assert row.gate is None


def test_triage_scorecard_row_ok_missing_metric_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TriageScorecardRow(
            node="triage", model=_CANDIDATE, scenario="s", baseline_model=_BASELINE, status="ok"
        )


def test_scorecard_triage_aggregate_counts_and_total_dropped() -> None:
    expected = _expected({"a.py": ReviewTier.STANDARD})
    pass_cmp = compare_triage(
        grade_triage(_triage({"a.py": ReviewTier.STANDARD}), expected),
        grade_triage(_triage({"a.py": ReviewTier.STANDARD}), expected),
    )
    drop_cmp = compare_triage(
        grade_triage(_triage({"a.py": ReviewTier.STANDARD}), expected),
        grade_triage(_triage({"a.py": ReviewTier.SKIM}), expected),  # candidate drops it
    )
    card = Scorecard(
        triage_rows=(
            TriageScorecardRow.from_comparison(
                scenario="s1", model=_CANDIDATE, baseline_model=_BASELINE, comparison=pass_cmp
            ),
            TriageScorecardRow.from_comparison(
                scenario="s2", model=_CANDIDATE, baseline_model=_BASELINE, comparison=drop_cmp
            ),
        )
    )
    aggregates = card.triage_aggregates()
    assert len(aggregates) == 1
    agg = aggregates[0]
    assert agg.n_ok == 2
    assert agg.n_passed == 1  # s1 holds
    assert agg.n_failed == 1  # s2 dropped a file -> gate fails
    assert agg.total_dropped_from_analysis == 1  # s2 dropped one file


def test_scorecard_triage_markdown_and_json_sections() -> None:
    expected = _expected({"a.py": ReviewTier.DEEP})
    cmp = compare_triage(
        grade_triage(_triage({"a.py": ReviewTier.DEEP}), expected),
        grade_triage(_triage({"a.py": ReviewTier.DEEP}), expected),
    )
    card = Scorecard(
        triage_rows=(
            TriageScorecardRow.from_comparison(
                scenario="tri_s", model=_CANDIDATE, baseline_model=_BASELINE, comparison=cmp
            ),
        )
    )
    md = card.to_markdown()
    assert "## Triage" in md
    assert "tri_s" in md
    data = json.loads(card.to_json())
    assert len(data["triage_rows"]) == 1
    assert len(data["triage_aggregates"]) == 1
