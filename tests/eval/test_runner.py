"""Tests for the cross-scenario eval runner (tests/eval/runner.py).

Covers the matrix orchestration that step 1 promotes out of the inline opt-in
loop: `build_scorecard` row assembly (one row per scenario × candidate model),
the ported transient-isolation (`_isolate_transients`) and regression
(`_sqli_fp_count` / `_regression_verdict`) helpers, and the cost-pass guard.

The happy path reuses the zero-spend scripted-provider machinery from
`test_model_comparison` (a SQL-injection scenario whose canned responses find /
miss the known finding); the isolation helper is tested directly with raising
coroutines (decoupled from analyze's degraded-mode behavior, exactly as the
original `_run_scenario_isolating_transients` test is). Pure — no DB, no LLM, no
spend.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from outrider.audit.events import compute_finding_content_hash
from outrider.llm.base import LLMAuthError, LLMTimeoutError
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.review_finding import ReviewFinding

from .grading import grade
from .runner import (
    ScenarioSpec,
    _CellError,
    _isolate_transients,
    _regression_verdict,
    _sqli_fp_count,
    build_scorecard,
)
from .test_model_comparison import (
    _FINDS_RESPONSE,
    _GROUND_TRUTH,
    _MISSES_RESPONSE,
    _build_state,
    _ScriptedProvider,
)

if TYPE_CHECKING:
    from .grading import ModelComparison

_BASELINE = "claude-sonnet-4-6"
_CANDIDATE = "claude-haiku-4-5"

_SEVERITY_FOR_TYPE = {
    FindingType.SQL_INJECTION: FindingSeverity.CRITICAL,
    FindingType.HARDCODED_SECRET: FindingSeverity.HIGH,
}


def _finding(
    *, finding_type: FindingType, file_path: str = "app/db.py", line: int = 10
) -> ReviewFinding:
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=1,
        finding_type=finding_type,
        severity=_SEVERITY_FOR_TYPE[finding_type],
        file_path=file_path,
        line_start=line,
        line_end=line,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path, line_start=line, line_end=line, finding_type=finding_type
        ),
        proposal_hash="a" * 64,
    )


# --- ported regression helpers ----------------------------------------------


def test_sqli_fp_count_counts_only_sql_injection_extras() -> None:
    # An sql_injection extra (matches no ground truth) counts; a non-sqli extra
    # does not; no findings is zero.
    assert _sqli_fp_count(grade([_finding(finding_type=FindingType.SQL_INJECTION)], [])) == 1
    assert _sqli_fp_count(grade([_finding(finding_type=FindingType.HARDCODED_SECRET)], [])) == 0
    assert _sqli_fp_count(grade([], [])) == 0


def test_regression_verdict_three_states() -> None:
    inconclusive = _regression_verdict(1, 0)
    assert inconclusive.ok is False
    assert inconclusive.label == "INCONCLUSIVE"

    reproduced = _regression_verdict(0, 1)
    assert reproduced.ok is False
    assert reproduced.label == "REPRODUCED"
    assert reproduced.candidate_sqli_fp == 1

    clean = _regression_verdict(0, 0)
    assert clean.ok is True
    assert clean.label == "CLEAN"


# --- transient isolation ----------------------------------------------------


async def test_isolate_transients_transient_becomes_cell_error() -> None:
    async def _boom() -> ModelComparison:
        raise LLMTimeoutError

    result = await _isolate_transients(_boom())
    assert isinstance(result, _CellError)
    assert "LLMTimeoutError" in result.message


async def test_isolate_transients_terminal_reraises() -> None:
    async def _boom() -> ModelComparison:
        raise LLMAuthError

    with pytest.raises(LLMAuthError):
        await _isolate_transients(_boom())


# --- build_scorecard matrix orchestration -----------------------------------


def test_build_scorecard_passing_gate_clean_regression() -> None:
    spec = ScenarioSpec(
        scenario="example_sqli", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH)
    )
    card = build_scorecard(
        [spec],
        baseline_provider=_ScriptedProvider(_FINDS_RESPONSE),
        candidate_provider=_ScriptedProvider(_FINDS_RESPONSE),
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
    )
    assert len(card.rows) == 1
    row = card.rows[0]
    assert (row.node, row.model, row.scenario) == ("analyze", _CANDIDATE, "example_sqli")
    assert row.baseline_model == _BASELINE
    assert row.status == "ok"
    assert row.gate is not None and row.gate.passes is True
    assert row.regression is not None and row.regression.ok is True
    assert row.cost is None and row.cost_source == "not_measured"
    assert row.quality_source == "analyze_direct"


def test_build_scorecard_failing_gate_on_recall_drop() -> None:
    spec = ScenarioSpec(
        scenario="example_sqli", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH)
    )
    card = build_scorecard(
        [spec],
        baseline_provider=_ScriptedProvider(_FINDS_RESPONSE),
        candidate_provider=_ScriptedProvider(_MISSES_RESPONSE),  # candidate misses the finding
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
    )
    row = card.rows[0]
    assert row.gate is not None
    assert row.gate.passes is False
    assert row.gate.recall_held is False
    assert row.recall is not None and row.recall.value == 0.0


def test_build_scorecard_emits_row_per_scenario() -> None:
    specs = [
        ScenarioSpec(scenario="s1", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH)),
        ScenarioSpec(scenario="s2", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH)),
    ]
    card = build_scorecard(
        specs,
        baseline_provider=_ScriptedProvider(_FINDS_RESPONSE),
        candidate_provider=_ScriptedProvider(_FINDS_RESPONSE),
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
    )
    assert len(card.rows) == 2
    assert {row.scenario for row in card.rows} == {"s1", "s2"}


def test_build_scorecard_measure_cost_requires_fixture_path() -> None:
    # No fixture_path -> the cost pass can't read the PR; fail fast before any run.
    spec = ScenarioSpec(scenario="s", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH))
    with pytest.raises(ValueError, match="fixture_path"):
        build_scorecard(
            [spec],
            baseline_provider=_ScriptedProvider(_FINDS_RESPONSE),
            candidate_provider=_ScriptedProvider(_FINDS_RESPONSE),
            baseline_model=_BASELINE,
            candidate_models=[_CANDIDATE],
            measure_cost=True,
        )


def test_build_scorecard_measures_full_graph_cost() -> None:
    """DB-backed: exercises the cost pass (run_review + CostProbe + the model_config
    seam) on a real full-graph fixture at zero spend — the fixture's own scripted
    responses drive triage+analyze+synthesize, the probe prices real prompt tokens.
    Quality is incidental (empty ground truth + a no-finding provider); the
    assertion is on the review-level cost join."""
    fixture = Path(__file__).parent / "fixtures" / "mock_github" / "pygoat_sql_injection.json"
    spec = ScenarioSpec.from_fixture("pygoat_sqli", str(fixture), ())
    card = build_scorecard(
        [spec],
        baseline_provider=_ScriptedProvider(_MISSES_RESPONSE),
        candidate_provider=_ScriptedProvider(_MISSES_RESPONSE),
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
        measure_cost=True,
    )
    assert len(card.rows) == 1
    row = card.rows[0]
    assert row.status == "ok"
    assert row.cost is not None
    assert row.cost_source == "full_graph"
    assert row.cost.usd > 0.0  # real prompt tokens priced through the production path
    assert row.latency is None  # deferred: harness-dominated wall-clock, not review latency
