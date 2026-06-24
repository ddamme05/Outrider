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

from outrider.agent.nodes.triage import TriagePolicyViolationError
from outrider.audit.events import compute_finding_content_hash
from outrider.llm.base import LLMAuthError, LLMTimeoutError
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel

from .grading import grade
from .runner import (
    ScenarioSpec,
    TriageScenarioSpec,
    _CellError,
    _isolate_transients,
    _regression_verdict,
    _sqli_fp_count,
    build_scorecard,
    build_triage_scorecard,
)
from .test_model_comparison import (
    _FINDS_RESPONSE,
    _GROUND_TRUTH,
    _MISSES_RESPONSE,
    _build_state,
    _ScriptedProvider,
)
from .test_triage_grading import _TRIAGE_DEEP, _TRIAGE_SKIM
from .triage_grading import ExpectedTriage

if TYPE_CHECKING:
    from outrider.llm.base import LLMRequest, LLMResponse

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


class _ClosingProvider:
    """Wraps a scripted provider, recording whether `aclose()` was called — pins
    the `close_providers=` lifecycle without a real httpx client."""

    def __init__(self, response_text: str) -> None:
        self._inner = _ScriptedProvider(response_text)
        self.closed = False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await self._inner.complete(request)

    async def aclose(self) -> None:
        self.closed = True


def test_build_scorecard_closes_providers_when_requested() -> None:
    baseline = _ClosingProvider(_FINDS_RESPONSE)
    candidate = _ClosingProvider(_FINDS_RESPONSE)
    spec = ScenarioSpec(scenario="s", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH))
    build_scorecard(
        [spec],
        baseline_provider=baseline,
        candidate_provider=candidate,
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
        close_providers=True,
    )
    assert baseline.closed is True
    assert candidate.closed is True


def test_build_scorecard_leaves_providers_open_by_default() -> None:
    baseline = _ClosingProvider(_FINDS_RESPONSE)
    candidate = _ClosingProvider(_FINDS_RESPONSE)
    spec = ScenarioSpec(scenario="s", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH))
    build_scorecard(
        [spec],
        baseline_provider=baseline,
        candidate_provider=candidate,
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
    )
    assert baseline.closed is False
    assert candidate.closed is False


# --- input validation + baseline caching ------------------------------------


class _CountingProvider:
    """Counts complete() calls — pins that the baseline analyze runs once per
    scenario, not once per candidate model."""

    def __init__(self, response_text: str) -> None:
        self._inner = _ScriptedProvider(response_text)
        self.complete_calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_calls += 1
        return await self._inner.complete(request)

    async def aclose(self) -> None:
        return None


def test_build_scorecard_rejects_duplicate_scenario_labels() -> None:
    specs = [
        ScenarioSpec(scenario="dup", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH)),
        ScenarioSpec(scenario="dup", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH)),
    ]
    with pytest.raises(ValueError, match="unique"):
        build_scorecard(
            specs,
            baseline_provider=_ScriptedProvider(_FINDS_RESPONSE),
            candidate_provider=_ScriptedProvider(_FINDS_RESPONSE),
            baseline_model=_BASELINE,
            candidate_models=[_CANDIDATE],
        )


def test_build_scorecard_rejects_duplicate_candidate_models() -> None:
    spec = ScenarioSpec(scenario="s", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH))
    with pytest.raises(ValueError, match="unique"):
        build_scorecard(
            [spec],
            baseline_provider=_ScriptedProvider(_FINDS_RESPONSE),
            candidate_provider=_ScriptedProvider(_FINDS_RESPONSE),
            baseline_model=_BASELINE,
            candidate_models=[_CANDIDATE, _CANDIDATE],
        )


def test_build_scorecard_rejects_malformed_candidate_model() -> None:
    # A non-Anthropic model id fails up front (ModelConfig field-validator) before
    # any quality work — not mid-run in the cost pass after the spend landed.
    spec = ScenarioSpec(scenario="s", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH))
    with pytest.raises(ValueError):  # noqa: PT011 — pydantic ValidationError (a ValueError)
        build_scorecard(
            [spec],
            baseline_provider=_ScriptedProvider(_FINDS_RESPONSE),
            candidate_provider=_ScriptedProvider(_FINDS_RESPONSE),
            baseline_model=_BASELINE,
            candidate_models=["gpt-4o"],
        )


def test_build_scorecard_runs_baseline_once_per_scenario() -> None:
    # 1 scenario × 2 candidate models: the baseline analyze runs ONCE (its grade is
    # invariant across candidates), the candidate analyze once per model — not
    # baseline×2. _build_state has one changed file -> one analyze call per run.
    baseline = _CountingProvider(_FINDS_RESPONSE)
    candidate = _CountingProvider(_FINDS_RESPONSE)
    spec = ScenarioSpec(scenario="s", state=_build_state(), ground_truth=tuple(_GROUND_TRUTH))
    build_scorecard(
        [spec],
        baseline_provider=baseline,
        candidate_provider=candidate,
        baseline_model=_BASELINE,
        candidate_models=["claude-haiku-4-5", "claude-sonnet-4-6"],
    )
    assert baseline.complete_calls == 1  # cached across the 2 candidates
    assert candidate.complete_calls == 2  # once per candidate


# --- build_triage_scorecard matrix orchestration ----------------------------

# _build_state's one changed file is src/example.py; the scripted _TRIAGE_DEEP /
# _TRIAGE_SKIM responses (from test_triage_grading) tier exactly that path, so the
# real triage node's path-coverage gate is satisfied.
_TRIAGE_EXPECTED = ExpectedTriage(
    expected_file_tiers={"src/example.py": ReviewTier.DEEP},
    overall_risk=RiskLevel.HIGH,
    relevant_dimensions=(ReviewDimension.SECURITY,),
)


class _RaisingProvider:
    """Raises a transient on complete() — pins the errored-cell path of the triage
    matrix (a `_CellError` -> `TriageScorecardRow.errored`, not an abort)."""

    def __init__(self) -> None:
        self.closed = False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMTimeoutError

    async def aclose(self) -> None:
        self.closed = True


def test_build_triage_scorecard_passing_gate() -> None:
    spec = TriageScenarioSpec(scenario="example", state=_build_state(), expected=_TRIAGE_EXPECTED)
    card = build_triage_scorecard(
        [spec],
        baseline_provider=_ScriptedProvider(_TRIAGE_DEEP),
        candidate_provider=_ScriptedProvider(_TRIAGE_DEEP),
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
    )
    assert card.rows == ()  # this entrypoint fills triage_rows only; analyze rows untouched
    assert len(card.triage_rows) == 1
    row = card.triage_rows[0]
    assert (row.node, row.model, row.scenario) == ("triage", _CANDIDATE, "example")
    assert row.baseline_model == _BASELINE
    assert row.status == "ok"
    assert row.gate is not None and row.gate.passes is True
    assert row.n_dropped_from_analysis == 0
    assert row.triage_source == "run_triage_direct"


def test_build_triage_scorecard_failing_gate_on_drop() -> None:
    # baseline DEEP, candidate SKIM -> the file drops below the analysis floor.
    spec = TriageScenarioSpec(scenario="example", state=_build_state(), expected=_TRIAGE_EXPECTED)
    card = build_triage_scorecard(
        [spec],
        baseline_provider=_ScriptedProvider(_TRIAGE_DEEP),
        candidate_provider=_ScriptedProvider(_TRIAGE_SKIM),
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
    )
    row = card.triage_rows[0]
    assert row.gate is not None
    assert row.gate.passes is False
    assert row.gate.drop_held is False
    assert row.n_dropped_from_analysis == 1


def test_build_triage_scorecard_emits_row_per_scenario() -> None:
    specs = [
        TriageScenarioSpec(scenario="s1", state=_build_state(), expected=_TRIAGE_EXPECTED),
        TriageScenarioSpec(scenario="s2", state=_build_state(), expected=_TRIAGE_EXPECTED),
    ]
    card = build_triage_scorecard(
        specs,
        baseline_provider=_ScriptedProvider(_TRIAGE_DEEP),
        candidate_provider=_ScriptedProvider(_TRIAGE_DEEP),
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
    )
    assert len(card.triage_rows) == 2
    assert {row.scenario for row in card.triage_rows} == {"s1", "s2"}


def test_build_triage_scorecard_errored_cell_on_transient() -> None:
    # A transient on the candidate triage -> an errored row, not an abort.
    spec = TriageScenarioSpec(scenario="example", state=_build_state(), expected=_TRIAGE_EXPECTED)
    card = build_triage_scorecard(
        [spec],
        baseline_provider=_ScriptedProvider(_TRIAGE_DEEP),
        candidate_provider=_RaisingProvider(),
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
    )
    row = card.triage_rows[0]
    assert row.status == "errored"
    assert row.tier_accuracy is None
    assert row.gate is None
    assert "LLMTimeoutError" in (row.error or "")


def test_build_triage_scorecard_runs_baseline_once_per_scenario() -> None:
    # 1 scenario × 2 candidate models: the baseline triage runs ONCE (its grade is
    # invariant across candidates), the candidate triage once per model.
    baseline = _CountingProvider(_TRIAGE_DEEP)
    candidate = _CountingProvider(_TRIAGE_DEEP)
    spec = TriageScenarioSpec(scenario="s", state=_build_state(), expected=_TRIAGE_EXPECTED)
    build_triage_scorecard(
        [spec],
        baseline_provider=baseline,
        candidate_provider=candidate,
        baseline_model=_BASELINE,
        candidate_models=["claude-haiku-4-5", "claude-sonnet-4-6"],
    )
    assert baseline.complete_calls == 1  # cached across the 2 candidates
    assert candidate.complete_calls == 2  # once per candidate


def test_build_triage_scorecard_rejects_duplicate_scenario_labels() -> None:
    specs = [
        TriageScenarioSpec(scenario="dup", state=_build_state(), expected=_TRIAGE_EXPECTED),
        TriageScenarioSpec(scenario="dup", state=_build_state(), expected=_TRIAGE_EXPECTED),
    ]
    with pytest.raises(ValueError, match="unique"):
        build_triage_scorecard(
            specs,
            baseline_provider=_ScriptedProvider(_TRIAGE_DEEP),
            candidate_provider=_ScriptedProvider(_TRIAGE_DEEP),
            baseline_model=_BASELINE,
            candidate_models=[_CANDIDATE],
        )


def test_build_triage_scorecard_rejects_malformed_candidate_model() -> None:
    # A non-Anthropic model id fails up front (ModelConfig field-validator) before
    # any triage work — same guard as build_scorecard.
    spec = TriageScenarioSpec(scenario="s", state=_build_state(), expected=_TRIAGE_EXPECTED)
    with pytest.raises(ValueError):  # noqa: PT011 — pydantic ValidationError (a ValueError)
        build_triage_scorecard(
            [spec],
            baseline_provider=_ScriptedProvider(_TRIAGE_DEEP),
            candidate_provider=_ScriptedProvider(_TRIAGE_DEEP),
            baseline_model=_BASELINE,
            candidate_models=["gpt-4o"],
        )


def test_build_triage_scorecard_closes_providers_when_requested() -> None:
    baseline = _ClosingProvider(_TRIAGE_DEEP)
    candidate = _ClosingProvider(_TRIAGE_DEEP)
    spec = TriageScenarioSpec(scenario="s", state=_build_state(), expected=_TRIAGE_EXPECTED)
    build_triage_scorecard(
        [spec],
        baseline_provider=baseline,
        candidate_provider=candidate,
        baseline_model=_BASELINE,
        candidate_models=[_CANDIDATE],
        close_providers=True,
    )
    assert baseline.closed is True
    assert candidate.closed is True


# A schema-valid triage result that VIOLATES the node's policy gate (rule (a): no
# SKIP). The node raises TriagePolicyViolationError post-schema — not an
# LLMProviderError, so _isolate_transients does not catch it.
_TRIAGE_SKIP = (
    '{"file_tiers": {"src/example.py": "skip"}, "overall_risk": "high", '
    '"relevant_dimensions": ["security"], "reasoning": "skip it"}'
)


def test_build_triage_scorecard_aborts_on_policy_violation() -> None:
    # A candidate whose triage emits SKIP violates the node's policy gate. Unlike a
    # transient (which becomes an errored cell), this is TERMINAL — it aborts the run,
    # surfacing the unusable candidate rather than silently recording a cell. This is
    # the one behavior that distinguishes the triage matrix from the analyze matrix,
    # and it rests on TriagePolicyViolationError being a ValueError, NOT an
    # LLMProviderError (the only class _isolate_transients converts to a _CellError).
    spec = TriageScenarioSpec(scenario="example", state=_build_state(), expected=_TRIAGE_EXPECTED)
    with pytest.raises(TriagePolicyViolationError):
        build_triage_scorecard(
            [spec],
            baseline_provider=_ScriptedProvider(_TRIAGE_DEEP),
            candidate_provider=_ScriptedProvider(_TRIAGE_SKIP),
            baseline_model=_BASELINE,
            candidate_models=[_CANDIDATE],
        )


def test_triage_scenario_spec_from_fixture_builds_state() -> None:
    # from_fixture derives the state via state_from_eval_fixture (no node run); the
    # fixture's changed files populate pr_context for the real triage matrix.
    fixture = Path(__file__).parent / "fixtures" / "mock_github" / "pygoat_sql_injection.json"
    spec = TriageScenarioSpec.from_fixture("pygoat", str(fixture), _TRIAGE_EXPECTED)
    assert spec.scenario == "pygoat"
    assert len(spec.state.pr_context.changed_files) >= 1
    assert spec.expected is _TRIAGE_EXPECTED
