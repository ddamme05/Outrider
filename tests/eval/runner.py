"""Cross-scenario eval runner: drive a (scenario × model) matrix into a Scorecard.

Productizes the inline three-dimension loop (recall / precision / regression) +
`GATE SUMMARY` print from the opt-in spend test into a reusable batch runner that
emits a typed `Scorecard`. Per `specs/2026-06-23-eval-runner-scorecard.md`:

  - Quality (recall/precision/severity/FP/gate) from the analyze-direct
    `compare_models_on_scenario` path — triage held fixed (the `#041` isolation).
    `quality_source="analyze_direct"`.
  - Cost (opt-in via `measure_cost`) from a full-graph `run_review` pass under
    the candidate model (the `model_config=` seam) with a `CostProbe` —
    review-level, joined onto the analyze row; `cost_source="full_graph"`.
    Latency is deferred (run_review wall-clock is harness-dominated — ephemeral
    DB + full migration per call — not review latency); replay-equivalence is
    not yet wired (the schema supports both, the orchestrator does not drive
    them yet).
  - Transient-failure isolation: a `retry_at_layer="node"` provider error on one
    cell becomes an ERRORED row and the batch continues; a terminal class
    (auth/config) re-raises and aborts — a revoked key must not "complete" a run
    with every cell ERRORED. Ports `_run_scenario_isolating_transients`.
  - Type-scoped `sql_injection` regression verdict per cell (the `#041` caveat
    track). Ports `_sqli_fp_count` / `_regression_verdict`.

One row per `(node, candidate-model, scenario)`; the baseline is the gate's
reference, not its own row.

SYNC by necessity: the cost pass calls `run_review` (a sync wrapper around
`asyncio.run`), which cannot nest inside a running loop. So `build_scorecard`
runs all quality comparisons in ONE `asyncio.run`, then issues the cost-pass
`run_review` calls sequentially. The opt-in entrypoint is therefore a sync test,
not `async def`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from outrider.agent.eval_driver import CostProbe, run_review
from outrider.agent.nodes.analyze import _estimate_tokens
from outrider.llm.base import LLMProviderError
from outrider.llm.config import ModelConfig
from outrider.policy import FindingType
from outrider.schemas.triage_result import ReviewTier

from .grading import DEFAULT_LINE_WINDOW
from .metrics import CostPerReview
from .model_comparison import compare_models_on_scenario, state_from_eval_fixture
from .scorecard import RegressionVerdict, Scorecard, ScorecardRow

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from outrider.llm.base import LLMProvider
    from outrider.schemas.review_state import ReviewState
    from outrider.schemas.triage_result import ReviewDimension

    from .grading import ExpectedFinding, GradeResult, ModelComparison


@dataclass(frozen=True)
class ScenarioSpec:
    """One cell-source in the matrix: a label, the post-triage `ReviewState` the
    analyze comparison runs over (triage held fixed), the hand-authored ground
    truth, and — for the cost pass — the `mock_github` fixture path `run_review`
    reads. Build via `from_fixture` for real scenarios; construct directly with
    an in-memory `state` (and no `fixture_path`) for zero-spend quality tests."""

    scenario: str
    state: ReviewState
    ground_truth: tuple[ExpectedFinding, ...]
    fixture_path: str | None = None

    @classmethod
    def from_fixture(
        cls,
        scenario: str,
        fixture_path: str,
        ground_truth: Sequence[ExpectedFinding],
        *,
        tier: ReviewTier = ReviewTier.STANDARD,
        dimensions: tuple[ReviewDimension, ...] | None = None,
    ) -> ScenarioSpec:
        """Build a spec from a `mock_github/*.json` fixture: derive the analyze
        state via `state_from_eval_fixture` (triage held fixed) AND keep the
        fixture path for the cost pass."""
        state = state_from_eval_fixture(fixture_path, tier=tier, dimensions=dimensions)
        return cls(
            scenario=scenario,
            state=state,
            ground_truth=tuple(ground_truth),
            fixture_path=fixture_path,
        )


@dataclass(frozen=True)
class _CellError:
    """A transient failure on one matrix cell — recorded so the batch continues
    (vs aborting). Carries the operator-facing ERRORED message for the row."""

    message: str


def _sqli_fp_count(grade: GradeResult) -> int:
    """Count ONLY `sql_injection` false positives (extras) in a grade. The
    regression track is type-scoped because the `#041` caveat is type-specific
    (a parameterized query mislabeled as SQL injection); counting any-type FPs
    lets an unrelated baseline over-flag force a wrongly non-discriminating
    verdict."""
    return sum(1 for finding in grade.extra if finding.finding_type == FindingType.SQL_INJECTION)


def _regression_verdict(baseline_sqli_fp: int, candidate_sqli_fp: int) -> RegressionVerdict:
    """Type-scoped sql_injection FP regression verdict (three states): a baseline
    that itself over-flags sql_injection is non-discriminating (INCONCLUSIVE); a
    clean baseline with a candidate over-flag is the `#041` caveat REPRODUCED;
    both clean is CLEAN."""
    if baseline_sqli_fp > 0:
        return RegressionVerdict(
            ok=False,
            label="INCONCLUSIVE",
            detail=(
                f"baseline emitted {baseline_sqli_fp} sql_injection FP on a parameterized "
                "query; the over-flag is not candidate-specific"
            ),
            baseline_sqli_fp=baseline_sqli_fp,
            candidate_sqli_fp=candidate_sqli_fp,
        )
    if candidate_sqli_fp > 0:
        return RegressionVerdict(
            ok=False,
            label="REPRODUCED",
            detail=(f"baseline clean of sql_injection FPs, candidate emitted {candidate_sqli_fp}"),
            baseline_sqli_fp=baseline_sqli_fp,
            candidate_sqli_fp=candidate_sqli_fp,
        )
    return RegressionVerdict(
        ok=True,
        label="CLEAN",
        detail="neither model emitted a sql_injection FP this run",
        baseline_sqli_fp=0,
        candidate_sqli_fp=0,
    )


async def _isolate_transients(coro: Awaitable[ModelComparison]) -> ModelComparison | _CellError:
    """Await one cell's comparison, isolating TRANSIENT provider failures. A
    `retry_at_layer="node"` error (timeout/429/409/5xx) → a `_CellError` so the
    batch continues; terminal classes (auth/config) re-raise to abort the run (a
    revoked key must not 'complete' a run with every cell ERRORED). Ports
    `_run_scenario_isolating_transients`."""
    try:
        return await coro
    except LLMProviderError as exc:
        if exc.retry_at_layer != "node":
            raise
        return _CellError(message=f"ERRORED ({type(exc).__name__}) — rerun")


async def _gather_quality(
    specs: Sequence[ScenarioSpec],
    candidate_models: Sequence[str],
    *,
    baseline_provider: LLMProvider,
    candidate_provider: LLMProvider,
    baseline_model: str,
    line_window: int,
    recall_tolerance: float,
    fp_allowance: int,
    baseline_recall_floor: float,
) -> dict[tuple[str, str], ModelComparison | _CellError]:
    """Run every `(scenario, candidate-model)` quality comparison in one event
    loop, each isolated. Sequential awaits (not gathered) keep the real-provider
    spend ordered and the failure semantics simple."""
    out: dict[tuple[str, str], ModelComparison | _CellError] = {}
    for spec in specs:
        for model in candidate_models:
            out[(spec.scenario, model)] = await _isolate_transients(
                compare_models_on_scenario(
                    spec.state,
                    spec.ground_truth,
                    baseline_provider=baseline_provider,
                    baseline_model=baseline_model,
                    candidate_provider=candidate_provider,
                    candidate_model=model,
                    line_window=line_window,
                    recall_tolerance=recall_tolerance,
                    fp_allowance=fp_allowance,
                    baseline_recall_floor=baseline_recall_floor,
                )
            )
    return out


def _measure_cost(
    fixture_path: str, model: str, token_estimator: Callable[[str], int]
) -> CostPerReview:
    """Measure review-level cost for one scenario under `model` via a full-graph
    `run_review` + `CostProbe` (zero Anthropic spend — the probe counts real
    prompt tokens through the production pricing path). `model` drives BOTH
    analyze tiers via the `model_config=` seam so a STANDARD file routes to it;
    the other `ModelConfig` fields resolve from env-or-defaults, held constant
    across candidate models so only analyze varies.

    Cost only — NOT latency: `run_review` spins an ephemeral DB and runs the full
    Alembic migration per call, so its wall-clock is harness-dominated, not the
    "review latency" `LatencyPerReview` contracts for. Review-latency measurement
    needs a graph-span timer (a future seam exposing `_drive`'s `graph.ainvoke`
    duration); deferred — rows keep `latency=None` in step 1. `total_cost_usd` is
    `float | None`, so a missing aggregate floors to 0.0 rather than crashing the
    `CostPerReview` build."""
    probe = CostProbe(token_estimator=token_estimator)
    model_config = ModelConfig(analyze_model=model, standard_analyze_model=model)
    result = run_review(fixture_path, probe=probe, model_config=model_config)
    metrics = result.review_metrics
    total = (
        metrics.total_cost_usd
        if (metrics is not None and metrics.total_cost_usd is not None)
        else 0.0
    )
    return CostPerReview(usd=total)


def _measure_cost_isolating_transients(
    fixture_path: str, model: str, token_estimator: Callable[[str], int]
) -> CostPerReview | None:
    """`_measure_cost` with the same transient isolation as the quality pass: a
    `retry_at_layer="node"` failure leaves cost unmeasured (the quality row is
    still emitted — a flaked cost pass must not drop a real recall signal); a
    terminal class re-raises and aborts."""
    try:
        return _measure_cost(fixture_path, model, token_estimator)
    except LLMProviderError as exc:
        if exc.retry_at_layer != "node":
            raise
        return None


def build_scorecard(
    specs: Sequence[ScenarioSpec],
    *,
    baseline_provider: LLMProvider,
    candidate_provider: LLMProvider,
    baseline_model: str,
    candidate_models: Sequence[str],
    node: str = "analyze",
    measure_cost: bool = False,
    token_estimator: Callable[[str], int] = _estimate_tokens,
    line_window: int = DEFAULT_LINE_WINDOW,
    recall_tolerance: float = 0.0,
    fp_allowance: int = 0,
    baseline_recall_floor: float = 1.0,
) -> Scorecard:
    """Drive the `(scenario × candidate-model)` matrix into a `Scorecard`.

    Quality per cell from the analyze-direct comparison (triage held fixed); cost
    per cell (when `measure_cost`) from a full-graph `run_review` under the
    candidate model (latency deferred — see `_measure_cost`). One row per
    `(node, candidate-model, scenario)`; the baseline is the gate's reference, not
    its own row. Report-only: rows record gate verdicts, this never raises on a
    failed gate.
    """
    if measure_cost and any(spec.fixture_path is None for spec in specs):
        raise ValueError(
            "measure_cost=True requires every ScenarioSpec to carry a fixture_path "
            "(run_review reads the PR fixture from disk); build specs via "
            "ScenarioSpec.from_fixture(...)"
        )
    quality = asyncio.run(
        _gather_quality(
            specs,
            candidate_models,
            baseline_provider=baseline_provider,
            candidate_provider=candidate_provider,
            baseline_model=baseline_model,
            line_window=line_window,
            recall_tolerance=recall_tolerance,
            fp_allowance=fp_allowance,
            baseline_recall_floor=baseline_recall_floor,
        )
    )
    rows: list[ScorecardRow] = []
    for spec in specs:
        for model in candidate_models:
            outcome = quality[(spec.scenario, model)]
            if isinstance(outcome, _CellError):
                rows.append(
                    ScorecardRow.errored(
                        node=node,
                        scenario=spec.scenario,
                        model=model,
                        baseline_model=baseline_model,
                        error=outcome.message,
                    )
                )
                continue
            regression = _regression_verdict(
                _sqli_fp_count(outcome.baseline), _sqli_fp_count(outcome.candidate)
            )
            cost: CostPerReview | None = None
            if measure_cost and spec.fixture_path is not None:
                cost = _measure_cost_isolating_transients(spec.fixture_path, model, token_estimator)
            rows.append(
                ScorecardRow.from_comparison(
                    node=node,
                    scenario=spec.scenario,
                    model=model,
                    baseline_model=baseline_model,
                    comparison=outcome,
                    regression=regression,
                    cost=cost,
                    # latency stays None: review-latency measurement is deferred
                    # (see _measure_cost). replay-equivalence is likewise not yet
                    # wired — the runner does not drive run_review_with_resume per
                    # HITL scenario, so replay_source defaults to "not_applicable".
                )
            )
    return Scorecard(rows=tuple(rows))


__all__ = [
    "ScenarioSpec",
    "build_scorecard",
]
