"""Cross-scenario eval runner: drive a (scenario × model) matrix into a Scorecard.

Productizes the inline three-dimension loop (recall / precision / regression) +
`GATE SUMMARY` print from the opt-in spend test into a reusable batch runner that
emits a typed `Scorecard`. Per `specs/2026-06-23-eval-runner-scorecard.md`:

  - Quality (recall/precision/severity/FP/gate) from the analyze-direct path —
    triage held fixed (the `#041` isolation). `quality_source="analyze_direct"`.
    The BASELINE analyze runs ONCE per scenario (its grade is invariant across
    candidate models), so a multi-candidate matrix doesn't re-pay for it.
  - Cost (opt-in via `measure_cost`) from a full-graph `run_review` pass under a
    fully-pinned `ModelConfig` (candidate on both analyze tiers; the other nodes
    pinned to declared defaults, NOT ambient env — so $/review is reproducible),
    with a `CostProbe`. Review-level; `cost_source` is `full_graph` (measured),
    `not_measured` (not requested), or `measure_failed` (requested, no number).
    Two honest provenance caveats: cost prices the file's REAL triage tier (DEEP
    for the shipped fixtures) while quality holds triage fixed at STANDARD; and
    OUTPUT tokens come from the fixtures' scripted responses (input tokens are
    real). Latency is deferred (run_review wall-clock is harness-dominated, not
    review latency); replay-equivalence is not yet wired.
  - Transient-failure isolation: a `retry_at_layer="node"` provider error on one
    cell becomes an ERRORED row and the batch continues; a terminal class
    (auth/config) re-raises and aborts — a revoked key must not "complete" a run
    with every cell ERRORED. Ports `_run_scenario_isolating_transients`.
  - Type-scoped `sql_injection` regression verdict per cell (the `#041` caveat
    track). Ports `_sqli_fp_count` / `_regression_verdict`.

One row per `(node, candidate-model, scenario)`; the baseline is the gate's
reference, not its own row. Duplicate scenario labels or candidate models are
rejected up front (they would collide on the `(scenario, model)` result key).

SYNC by necessity: the cost pass calls `run_review` (a sync wrapper around
`asyncio.run`), which cannot nest inside a running loop. So `build_scorecard`
runs all quality comparisons in ONE `asyncio.run`, then issues the cost-pass
`run_review` calls sequentially. The opt-in entrypoint is therefore a sync test.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from outrider.agent.eval_driver import CostProbe, run_review
from outrider.agent.nodes.analyze import _estimate_tokens
from outrider.llm.base import LLMProviderError
from outrider.llm.config import ModelConfig
from outrider.policy import FindingType
from outrider.schemas.triage_result import ReviewTier

from .grading import DEFAULT_LINE_WINDOW, compare, grade
from .metrics import CostPerReview
from .model_comparison import run_analyze_under_model, state_from_eval_fixture
from .scorecard import (
    RegressionVerdict,
    Scorecard,
    ScorecardProvenance,
    ScorecardRow,
    TriageScorecardRow,
)
from .triage_grading import (
    compare_triage,
    grade_triage,
    require_expected_coverage,
    run_triage_under_model,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from outrider.llm.base import LLMProvider
    from outrider.schemas.review_state import ReviewState
    from outrider.schemas.triage_result import ReviewDimension

    from .grading import ExpectedFinding, GradeResult, ModelComparison
    from .scorecard import CostSource
    from .triage_grading import ExpectedTriage, TriageComparison, TriageGrade


@dataclass(frozen=True, eq=False)
class ScenarioSpec:
    """One cell-source in the matrix: a label, the post-triage `ReviewState` the
    analyze comparison runs over (triage held fixed), the hand-authored ground
    truth, and — for the cost pass — the `mock_github` fixture path `run_review`
    reads. Build via `from_fixture` for real scenarios; construct directly with
    an in-memory `state` (and no `fixture_path`) for zero-spend quality tests.

    `eq=False` (identity equality/hash): `state` is an unhashable Pydantic model
    (`frozen=False`), so a `frozen=True`-generated field hash would raise; specs
    are only iterated, never used as set/dict keys, so identity is correct."""

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


@dataclass(frozen=True, eq=False)
class TriageScenarioSpec:
    """One TRIAGE matrix cell-source: a label, the `ReviewState` the triage
    comparison runs over, and the hand-authored `ExpectedTriage` ground truth. No
    `fixture_path` — triage rows are quality-only (no cost pass per the spec). The
    triage node re-tiers from `state.pr_context.changed_files`, ignoring any pre-set
    `triage_result`, so unlike `ScenarioSpec` the state is NOT held fixed here — the
    real triage node runs. `eq=False` for the same reason as `ScenarioSpec` (the
    Pydantic `state` is unhashable)."""

    scenario: str
    state: ReviewState
    expected: ExpectedTriage

    @classmethod
    def from_fixture(
        cls, scenario: str, fixture_path: str, expected: ExpectedTriage
    ) -> TriageScenarioSpec:
        """Build from a `mock_github/*.json` fixture: derive the state via
        `state_from_eval_fixture`. The held-fixed tier/dimensions that helper sets
        are irrelevant — the triage node re-tiers the changed files itself."""
        return cls(
            scenario=scenario, state=state_from_eval_fixture(fixture_path), expected=expected
        )


@dataclass(frozen=True)
class _CellError:
    """A transient failure on one matrix cell — recorded so the batch continues
    (vs aborting). Carries the operator-facing ERRORED message for the row."""

    message: str


def _sqli_fp_count(grade_result: GradeResult) -> int:
    """Count ONLY `sql_injection` false positives (extras) in a grade. The
    regression track is type-scoped because the `#041` caveat is type-specific
    (a parameterized query mislabeled as SQL injection); counting any-type FPs
    lets an unrelated baseline over-flag force a wrongly non-discriminating
    verdict."""
    return sum(
        1 for finding in grade_result.extra if finding.finding_type == FindingType.SQL_INJECTION
    )


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


async def _isolate_transients[T](coro: Awaitable[T]) -> T | _CellError:
    """Await one analyze/grade step, isolating TRANSIENT provider failures. A
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


async def _graded_analyze(
    spec: ScenarioSpec, *, provider: LLMProvider, model: str, line_window: int
) -> GradeResult:
    """Run one analyze pass under `model` over the spec's held-fixed state and
    grade it against the spec's ground truth."""
    # `_` discards the structured-output rejection flag — build_scorecard's Scorecard does
    # not surface yield yet (the GLM scorecard's compare_models_on_scenario does, FUP-196).
    findings, _ = await run_analyze_under_model(spec.state, provider=provider, model=model)
    return grade(findings, spec.ground_truth, line_window=line_window)


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
    loop, each isolated. The BASELINE analyze runs ONCE per scenario (invariant
    across candidate models); a baseline transient errors every candidate cell
    for that scenario, a candidate transient errors only its own cell."""
    out: dict[tuple[str, str], ModelComparison | _CellError] = {}
    for spec in specs:
        baseline = await _isolate_transients(
            _graded_analyze(
                spec, provider=baseline_provider, model=baseline_model, line_window=line_window
            )
        )
        for model in candidate_models:
            if isinstance(baseline, _CellError):
                out[(spec.scenario, model)] = baseline
                continue
            candidate = await _isolate_transients(
                _graded_analyze(
                    spec, provider=candidate_provider, model=model, line_window=line_window
                )
            )
            if isinstance(candidate, _CellError):
                out[(spec.scenario, model)] = candidate
                continue
            out[(spec.scenario, model)] = compare(
                baseline,
                candidate,
                recall_tolerance=recall_tolerance,
                fp_allowance=fp_allowance,
                baseline_recall_floor=baseline_recall_floor,
            )
    return out


def _cost_model_config(model: str) -> ModelConfig:
    """Build a fully-pinned `ModelConfig` for the cost pass: the candidate `model`
    on BOTH analyze tiers, and the non-analyze nodes pinned to their DECLARED
    defaults (NOT ambient `OUTRIDER_MODEL_*` env), so the persisted $/review is
    reproducible across machines and only analyze varies across the matrix.
    Constructing it also VALIDATES `model` up front (the regex field-validator),
    so a malformed/deprecated candidate fails before any paid quality work — not
    mid-run in the cost pass after the quality spend already landed."""
    fields = ModelConfig.model_fields
    return ModelConfig(
        analyze_model=model,
        standard_analyze_model=model,
        triage_model=fields["triage_model"].default,
        synthesize_model=fields["synthesize_model"].default,
        trace_model=fields["trace_model"].default,
        patch_model=fields["patch_model"].default,
    )


def _measure_cost(
    fixture_path: str, model_config: ModelConfig, token_estimator: Callable[[str], int]
) -> CostPerReview | None:
    """Measure review-level cost for one scenario via a full-graph `run_review` +
    `CostProbe` (zero Anthropic spend — the probe counts real prompt tokens
    through the production pricing path). Returns `None` when the run produced no
    usable cost (e.g. a size-gate-skipped review with no metrics) so the caller
    flags `measure_failed` rather than reporting a false $0.00.

    Provenance caveats baked into the design (see the module docstring):
      - Cost is WHOLE-REVIEW under the candidate's REAL triage tier (run_review
        runs the real triage node), whereas the quality half holds triage fixed
        at STANDARD — the row joins a STANDARD-tier quality verdict with the
        file's real-tier cost.
      - OUTPUT tokens come from the fixture's SCRIPTED responses (input/prompt
        tokens ARE real), so $/review is accurate on the input side and
        fixture-author-dependent on the output side.
      - Cost only, NOT latency: run_review's wall-clock is harness-dominated
        (ephemeral DB + full migration per call); rows keep latency=None."""
    probe = CostProbe(token_estimator=token_estimator)
    result = run_review(fixture_path, probe=probe, model_config=model_config)
    metrics = result.review_metrics
    if metrics is None or metrics.total_cost_usd is None:
        return None
    return CostPerReview(usd=metrics.total_cost_usd)


def _measure_cost_isolating_transients(
    fixture_path: str, model_config: ModelConfig, token_estimator: Callable[[str], int]
) -> CostPerReview | None:
    """`_measure_cost` with transient isolation: a `retry_at_layer="node"` failure
    returns None (the quality row is still emitted — a flaked cost must not drop a
    real recall signal); a terminal class re-raises and aborts. A None return
    (transient OR absent-metrics) tells the caller to flag `measure_failed`, kept
    distinct from a cost pass that was never requested (`not_measured`)."""
    try:
        return _measure_cost(fixture_path, model_config, token_estimator)
    except LLMProviderError as exc:
        if exc.retry_at_layer != "node":
            raise
        return None


async def _aclose_providers(
    baseline_provider: LLMProvider, candidate_provider: LLMProvider
) -> None:
    """Close the provider(s) via the `LLMProvider.aclose()` Protocol method. MUST
    run in the loop the providers were used in — a real httpx-backed client can't
    be closed cleanly from a different loop. Closes each distinct instance once;
    `aclose()` is idempotent and scripted test doubles no-op."""
    await baseline_provider.aclose()
    if candidate_provider is not baseline_provider:
        await candidate_provider.aclose()


async def _drive_quality[T](
    coro: Awaitable[T],
    baseline_provider: LLMProvider,
    candidate_provider: LLMProvider,
    *,
    close_providers: bool,
) -> T:
    """Await the quality coroutine (analyze OR triage matrix), then close the
    providers in the SAME event loop (the only safe place for a real httpx client).
    On a body exception (a terminal abort), close best-effort and SUPPRESS any
    close failure so it can't mask the original error; on success, close and let a
    real close failure surface."""
    try:
        result = await coro
    except Exception:
        if close_providers:
            with contextlib.suppress(Exception):
                await _aclose_providers(baseline_provider, candidate_provider)
        raise
    if close_providers:
        await _aclose_providers(baseline_provider, candidate_provider)
    return result


def _capture_git_state() -> tuple[str, bool]:
    """Best-effort git SHA + dirty flag for run provenance. Eval-infra only: a
    FIXED `git` invocation (no GitHub-sourced strings), not a shell-exec boundary
    surface. Falls back to ("unknown", False) when git is unavailable (e.g. a CI
    source tarball with no .git). `git_dirty` reads `git status --porcelain`, so it
    is True for UNTRACKED files too — over-marking dirty is the safe direction."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 — `git` on PATH, eval-infra provenance
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],  # noqa: S607 — `git` on PATH, eval-infra
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return ("unknown", False)
    return (sha, dirty)


def build_provenance(
    *,
    prompt_template_version: str | None,
    scenario_labels: Sequence[str],
    baseline_model: str,
    candidate_models: Sequence[str],
) -> ScorecardProvenance | None:
    """Stamp a scorecard run with its provenance so the artifact self-certifies
    which code + prompt + scenario set produced it (closes the reflog/mtime
    forensics gap). `prompt_template_version` is the node-relevant template
    `VERSION` — analyze for the finding scorecard, triage for the triage one.

    Returns None when `prompt_template_version` is None — the scripted-harness path
    that builds cards without stamping, so `build_scorecard` can call this
    unconditionally and let the caller opt in by passing a version."""
    if prompt_template_version is None:
        return None
    git_sha, git_dirty = _capture_git_state()
    return ScorecardProvenance(
        git_sha=git_sha,
        git_dirty=git_dirty,
        prompt_template_version=prompt_template_version,
        scenario_set=tuple(scenario_labels),
        baseline_model=baseline_model,
        candidate_models=tuple(candidate_models),
        generated_at=datetime.now(UTC),
    )


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
    close_providers: bool = False,
    prompt_template_version: str | None = None,
) -> Scorecard:
    """Drive the `(scenario × candidate-model)` matrix into a `Scorecard`.

    Quality per cell from the analyze-direct comparison (triage held fixed); cost
    per cell (when `measure_cost`) from a full-graph `run_review` under the
    candidate model (latency deferred — see `_measure_cost`). One row per
    `(node, candidate-model, scenario)`; the baseline is the gate's reference, not
    its own row. Report-only: rows record gate verdicts, this never raises on a
    failed gate.

    Rejects duplicate scenario labels / candidate models up front (they collide on
    the result key), and validates every candidate model string before any paid
    quality work (a malformed id fails here, not mid-run in the cost pass).

    `close_providers=True` closes `baseline_provider` / `candidate_provider` (via
    the `LLMProvider.aclose()` Protocol method) INSIDE the quality event loop — the
    only safe place to close a real httpx-backed provider, since this is a sync
    function and a sync caller cannot `await aclose()` cleanly across loops. Default
    `False` leaves provider lifecycle to the caller (scripted test doubles).
    """
    scenario_labels = [spec.scenario for spec in specs]
    if len(set(scenario_labels)) != len(scenario_labels):
        raise ValueError(f"ScenarioSpec labels must be unique; got duplicates in {scenario_labels}")
    if len(set(candidate_models)) != len(candidate_models):
        raise ValueError(f"candidate_models must be unique; got {list(candidate_models)}")
    if measure_cost and any(spec.fixture_path is None for spec in specs):
        raise ValueError(
            "measure_cost=True requires every ScenarioSpec to carry a fixture_path "
            "(run_review reads the PR fixture from disk); build specs via "
            "ScenarioSpec.from_fixture(...)"
        )
    # Validate + pin every candidate model BEFORE the (paid) quality pass: a
    # malformed/deprecated id raises here via ModelConfig's field-validator.
    cost_configs = {model: _cost_model_config(model) for model in candidate_models}

    # Stamp run provenance BEFORE the (paid) matrix runs, so git_sha/dirty pin the
    # checkout that actually produced the numbers — not whatever HEAD exists when the
    # multi-minute run finishes. None version -> None provenance (the scripted path).
    provenance = build_provenance(
        prompt_template_version=prompt_template_version,
        scenario_labels=scenario_labels,
        baseline_model=baseline_model,
        candidate_models=candidate_models,
    )

    quality = asyncio.run(
        _drive_quality(
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
            ),
            baseline_provider,
            candidate_provider,
            close_providers=close_providers,
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
            cost_source: CostSource | None = None  # None -> from_comparison derives not_measured
            if measure_cost and spec.fixture_path is not None:
                cost = _measure_cost_isolating_transients(
                    spec.fixture_path, cost_configs[model], token_estimator
                )
                if cost is None:
                    cost_source = "measure_failed"  # requested but produced no usable number
            rows.append(
                ScorecardRow.from_comparison(
                    node=node,
                    scenario=spec.scenario,
                    model=model,
                    baseline_model=baseline_model,
                    comparison=outcome,
                    regression=regression,
                    cost=cost,
                    cost_source=cost_source,
                    # latency stays None (review-latency deferred); replay-equivalence
                    # is not yet wired, so replay_source defaults to "not_applicable".
                )
            )
    return Scorecard(rows=tuple(rows), provenance=provenance)


# --- Triage matrix ----------------------------------------------------------


async def _graded_triage(
    spec: TriageScenarioSpec, *, provider: LLMProvider, model: str
) -> TriageGrade:
    """Run one triage pass under `model` over the spec's state and grade it against
    the spec's `ExpectedTriage`. Parallel to `_graded_analyze`."""
    result = await run_triage_under_model(spec.state, provider=provider, model=model)
    return grade_triage(result, spec.expected)


async def _gather_triage_quality(
    specs: Sequence[TriageScenarioSpec],
    candidate_models: Sequence[str],
    *,
    baseline_provider: LLMProvider,
    candidate_provider: LLMProvider,
    baseline_model: str,
    overtier_allowance: int,
    dimension_recall_tolerance: float,
) -> dict[tuple[str, str], TriageComparison | _CellError]:
    """Run every `(scenario, candidate-model)` triage comparison in one event loop,
    each isolated. The BASELINE triage runs ONCE per scenario (invariant across
    candidate models); a baseline transient errors every candidate cell for that
    scenario, a candidate transient errors only its own cell. Mirrors
    `_gather_quality`. A triage POLICY violation (the node rejecting a candidate's
    output) is NOT a transient — it propagates and aborts (see
    `build_triage_scorecard`)."""
    out: dict[tuple[str, str], TriageComparison | _CellError] = {}
    for spec in specs:
        baseline = await _isolate_transients(
            _graded_triage(spec, provider=baseline_provider, model=baseline_model)
        )
        for model in candidate_models:
            if isinstance(baseline, _CellError):
                out[(spec.scenario, model)] = baseline
                continue
            candidate = await _isolate_transients(
                _graded_triage(spec, provider=candidate_provider, model=model)
            )
            if isinstance(candidate, _CellError):
                out[(spec.scenario, model)] = candidate
                continue
            out[(spec.scenario, model)] = compare_triage(
                baseline,
                candidate,
                overtier_allowance=overtier_allowance,
                dimension_recall_tolerance=dimension_recall_tolerance,
            )
    return out


def triage_preflight(
    specs: Sequence[TriageScenarioSpec],
    candidate_models: Sequence[str],
    *,
    validate_candidate_model: Callable[[str], None] | None = None,
) -> None:
    """Every zero-spend validation the triage matrix runs BEFORE paid work, in
    one callable so a paid runner can execute it BEFORE constructing providers
    (a raise inside `build_triage_scorecard` happens after the caller built
    providers, where `close_providers` cannot reach them — deterministic
    input problems must surface before any resource exists). Checks: unique
    scenario labels + candidate models (result-key collisions), per-candidate
    model-id validation, and exact ground-truth coverage of each spec's
    changed files.

    `validate_candidate_model` is the HOST-AWARE seam: None (the default)
    validates via the Anthropic `ModelConfig` field-validator
    (`_cost_model_config` — the historical behavior, unchanged for existing
    callers); a host-admission runner passes its host's validator (e.g. a
    closure over `OPENAI_PROFILE.validate_model_slug` + pricing coverage) so
    native non-Anthropic slugs validate against their own catalog instead of
    being rejected by the claude-family regex."""
    scenario_labels = [spec.scenario for spec in specs]
    if len(set(scenario_labels)) != len(scenario_labels):
        raise ValueError(
            f"TriageScenarioSpec labels must be unique; got duplicates in {scenario_labels}"
        )
    if len(set(candidate_models)) != len(candidate_models):
        raise ValueError(f"candidate_models must be unique; got {list(candidate_models)}")
    for model in candidate_models:
        if validate_candidate_model is not None:
            validate_candidate_model(model)
        else:
            # ModelConfig's field-validator (discard the config — triage has
            # no cost pass).
            _cost_model_config(model)
    # Each spec's ground truth must cover exactly its changed files, else a missing
    # key silently drops a file from the grade + the safety gate. Fail fast, no spend.
    for spec in specs:
        require_expected_coverage(
            spec.expected,
            {cf.path for cf in spec.state.pr_context.changed_files},
            scenario=spec.scenario,
        )


def build_triage_scorecard(
    specs: Sequence[TriageScenarioSpec],
    *,
    baseline_provider: LLMProvider,
    candidate_provider: LLMProvider,
    baseline_model: str,
    candidate_models: Sequence[str],
    node: str = "triage",
    overtier_allowance: int = 0,
    dimension_recall_tolerance: float = 0.0,
    close_providers: bool = False,
    prompt_template_version: str | None = None,
    validate_candidate_model: Callable[[str], None] | None = None,
) -> Scorecard:
    """Drive the `(scenario × candidate-model)` TRIAGE matrix into a
    `Scorecard.triage_rows`. The analyze `rows` stay empty — the sibling
    `build_scorecard` fills those; an operator wanting one combined artifact merges
    `Scorecard(rows=analyze.rows, triage_rows=triage.triage_rows)`.

    Quality-only per the spec non-goal: no cost / latency / replay on triage rows
    (triage is one cheap call; whole-review cost lives on the analyze rows). One row
    per `(scenario, candidate-model)`; the baseline is the gate's reference, not its
    own row. Report-only — never raises on a failed gate.

    Rejects duplicate scenario labels / candidate models up front (they collide on
    the result key), and validates every candidate model string before any paid
    triage work — via `triage_preflight`, whose `validate_candidate_model` seam
    defaults to `ModelConfig`'s Anthropic field-validator (exactly as
    `build_scorecard` does; the config itself is discarded — triage has no cost
    pass) and accepts a host-native validator for non-Anthropic admission runners.
    Paid callers should run `triage_preflight` themselves BEFORE constructing
    providers: a preflight raise in here is after construction, out of
    `close_providers`' reach.

    Failure handling mirrors `build_scorecard` for TRANSIENT provider errors (→ a
    `_CellError` errored row). It DIVERGES for a triage POLICY violation: a candidate
    that emits `SKIP` or mis-covers paths makes the node raise
    `TriagePolicyViolationError`, which is terminal here — the run aborts naming the
    bad output rather than recording an errored cell. A policy-violating triage is a
    'this candidate is unusable for triage' signal, not a flake to retry.

    `close_providers` — same loop-bound close semantics as `build_scorecard`.
    """
    triage_preflight(specs, candidate_models, validate_candidate_model=validate_candidate_model)
    scenario_labels = [spec.scenario for spec in specs]

    # Stamp provenance BEFORE the (paid) matrix runs (see build_scorecard).
    provenance = build_provenance(
        prompt_template_version=prompt_template_version,
        scenario_labels=scenario_labels,
        baseline_model=baseline_model,
        candidate_models=candidate_models,
    )

    quality = asyncio.run(
        _drive_quality(
            _gather_triage_quality(
                specs,
                candidate_models,
                baseline_provider=baseline_provider,
                candidate_provider=candidate_provider,
                baseline_model=baseline_model,
                overtier_allowance=overtier_allowance,
                dimension_recall_tolerance=dimension_recall_tolerance,
            ),
            baseline_provider,
            candidate_provider,
            close_providers=close_providers,
        )
    )

    rows: list[TriageScorecardRow] = []
    for spec in specs:
        for model in candidate_models:
            outcome = quality[(spec.scenario, model)]
            if isinstance(outcome, _CellError):
                rows.append(
                    TriageScorecardRow.errored(
                        node=node,
                        scenario=spec.scenario,
                        model=model,
                        baseline_model=baseline_model,
                        error=outcome.message,
                    )
                )
                continue
            rows.append(
                TriageScorecardRow.from_comparison(
                    node=node,
                    scenario=spec.scenario,
                    model=model,
                    baseline_model=baseline_model,
                    comparison=outcome,
                )
            )
    return Scorecard(triage_rows=tuple(rows), provenance=provenance)


__all__ = [
    "ScenarioSpec",
    "TriageScenarioSpec",
    "build_provenance",
    "build_scorecard",
    "build_triage_scorecard",
]
