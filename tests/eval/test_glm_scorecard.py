"""Opt-in, real-spend: Anthropic-vs-GLM scorecard over the analyze node.

The GLM-provider-mode candidate gate (FUP-194 / FUP-196). Runs the SAME recall +
precision scenarios as the Sonnet-vs-Haiku model-tier comparison
(`test_model_comparison.py`), but with the **baseline = AnthropicProvider (Sonnet,
the DEEP-tier quality bar)** and the **candidate = GLMProvider (GLM 5.2 on Baseten)**.
Reusing the same fixtures + grader + report helpers keeps it apples-to-apples.

REPORT-ONLY BY DESIGN (mirrors `test_real_model_comparison_evidence` +
`test_synthesize_summary_comparison`): the per-scenario recall/precision/fp is printed
for the human to read; pytest "passed" means the run COMPLETED, not that a gate passed.
A recall delta vs Anthropic is a MODEL-CAPABILITY signal to interpret, not an automatic
fail — GLM is a different model, not a within-family flip.

The number that matters: GLM's recall (does it catch the known finding?) and precision
(does it over-flag safe code?) vs Anthropic — and, implicitly, GLM's structured-output
YIELD. A GLM analyze response whose JSON doesn't conform parses (after
`strip_outer_json_fence`) to no findings → recall 0 on that scenario, so the recall
dimension also surfaces the conformance signal that decides FUP-196 (shared-API soft
vs self-deploy strict).

Run (both keys resolve from .env via 1Password):
  OUTRIDER_EVAL_REAL_MODELS=1 op run --env-file=.env -- \
    uv run pytest tests/eval/test_glm_scorecard.py --is-eval -v -s

Cost: 2 analyze calls/scenario (Anthropic baseline + GLM candidate) over the recall
(`_GROUND_TRUTH_BY_FIXTURE`, 15 fixtures) + safe-code (`_SAFE_CODE_FIXTURES`, 4 fixtures)
sets — 19 scenarios x 2 models = ~38 small analyze calls. Cost-per-review (the third
scorecard column) is a separate
`build_scorecard` pass; this gate measures quality + yield first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from outrider.llm.glm_provider import GLM_MODEL_ID

from .model_comparison import compare_models_on_scenario, state_from_eval_fixture
from .scorecard import Scorecard, ScorecardRow
from .test_model_comparison import (
    _GROUND_TRUTH_BY_FIXTURE,
    _MISSING_ERROR_HANDLING_FIXTURE,
    _SAFE_CODE_FIXTURES,
    _WEAK_CRYPTO_FIXTURE,
    _CapturingExchangePersister,
    _NoOpExchangePersister,
    _print_aggregate_metrics,
    _print_scenario_report,
    _run_scenario_isolating_transients,
    diagnose_recall_stability,
)

if TYPE_CHECKING:
    from .grading import ExpectedFinding, ModelComparison


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model GLM scorecard spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
@pytest.mark.asyncio
async def test_glm_vs_anthropic_scorecard() -> None:
    """OPT-IN, real spend — the Anthropic-vs-GLM candidate scorecard.

    REPORT-ONLY: asserts only that the run COMPLETED. The per-scenario verdicts are
    printed; the human adjudicates whether GLM's recall/yield clears the bar to (a)
    productionize GLM mode (FUP-194) and (b) accept the shared-API soft path vs
    self-deploy for strict JSON (FUP-196). Two dimensions:

    - RECALL over known-vulnerability fixtures (`_GROUND_TRUTH_BY_FIXTURE`): does GLM
      catch the known finding? A non-conforming GLM JSON response parses to no findings
      → recall 0 here, so this dimension ALSO measures structured-output yield.
    - PRECISION over safe code (`_SAFE_CODE_FIXTURES`, empty ground truth): does GLM
      over-flag clean code more than Anthropic? Gated (advisory) on `fp_bounded`.

    Recall is TYPE-EXACT (same finding_type + policy severity + file/line window), so a
    recall delta can be a true miss OR a classification disagreement — read the printed
    `missed`/`extra` detail before acting.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    baseten_key = os.environ.get("BASETEN_API_KEY")
    if not anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the Anthropic baseline")
    if not baseten_key or baseten_key.startswith("op://"):
        pytest.skip(
            "BASETEN_API_KEY (resolved, not an op:// ref) is required for the GLM candidate; "
            "run under `op run --env-file=.env -- ...`"
        )

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.llm.glm_provider import GLMProvider  # noqa: PLC0415

    # Baseline is Anthropic's DEEP-tier analyze model (the quality bar GLM is measured
    # against), read from config rather than hardcoded.
    cfg = ModelConfig()
    baseline_model = cfg.analyze_model
    persister = _NoOpExchangePersister()
    baseline_provider = AnthropicProvider(
        api_key=SecretStr(anthropic_key), model_config=cfg, persister=persister
    )
    candidate_provider = GLMProvider(api_key=SecretStr(baseten_key), persister=persister)

    # (fixture, dimension, ok, label) — `_run_scenario_isolating_transients` records an
    # ERRORED row + returns None on a retryable provider failure so a transient does not
    # discard the rest of the paid run.
    gate_results: list[tuple[str, str, bool, str]] = []
    # One ScorecardRow per COMPLETED scenario, collected so the paid run persists a
    # structured artifact (json + html) via the same Scorecard serializers
    # test_scorecard.py writes to reports/scorecard/ — the report survives pytest
    # stdout capture regardless of -s.
    rows: list[ScorecardRow] = []
    # (fixture, dimension, comparison) per COMPLETED scenario, fed to
    # `_print_aggregate_metrics` for the cross-scenario metric block (yield rate, mean recall
    # + severity, the safe-code over-flag rate, an all-rows extras diagnostic, per-finding-type
    # recall). Recall is meaningful only on the "recall" dimension; the aggregate partitions on it.
    comparisons: list[tuple[str, str, ModelComparison]] = []
    # The structured aggregate (headline metrics) — persisted alongside the Scorecard so the
    # numbers a human adjudicates GLM on survive into the durable artifact, not just stdout.
    aggregate: dict[str, object] | None = None

    async def _compare_or_errored(
        fixture_path: str, ground_truth: tuple[ExpectedFinding, ...], dimension: str
    ) -> ModelComparison | None:
        async def _compare() -> ModelComparison:
            return await compare_models_on_scenario(
                state_from_eval_fixture(fixture_path),
                ground_truth,
                baseline_provider=baseline_provider,
                baseline_model=baseline_model,
                candidate_provider=candidate_provider,
                candidate_model=GLM_MODEL_ID,
            )

        return await _run_scenario_isolating_transients(
            fixture_path, dimension, gate_results, _compare
        )

    try:
        # RECALL + YIELD — does GLM catch the known finding (and produce parseable,
        # conforming JSON to do it)? `recall_held` (GLM recall >= Anthropic recall) is
        # reported, NOT a hard fail — a cross-model recall delta is a signal to read.
        for fixture_path, ground_truth in _GROUND_TRUTH_BY_FIXTURE.items():
            cmp = await _compare_or_errored(fixture_path, ground_truth, "recall")
            if cmp is None:
                continue
            _print_scenario_report(fixture_path, cmp, baseline_model, GLM_MODEL_ID)
            rows.append(
                ScorecardRow.from_comparison(
                    node="analyze",
                    scenario=fixture_path,
                    model=GLM_MODEL_ID,
                    baseline_model=baseline_model,
                    comparison=cmp,
                )
            )
            # Mirror the existing gate (test_model_comparison.py): recall counts as
            # held ONLY when the BASELINE also cleared the recall floor — else a
            # both-models-miss row would vacuously read green.
            recall_ok = cmp.recall_held and cmp.baseline_valid
            gate_results.append((fixture_path, "recall", recall_ok, "GLM recall < Anthropic"))
            comparisons.append((fixture_path, "recall", cmp))
            assert cmp.baseline is not None  # the run completed
        # PRECISION — does GLM over-flag safe code more than Anthropic? Empty ground
        # truth, so ANY finding is a false positive; advisory gate on `fp_bounded`.
        for fixture_path in _SAFE_CODE_FIXTURES:
            cmp = await _compare_or_errored(fixture_path, (), "precision")
            if cmp is None:
                continue
            _print_scenario_report(fixture_path, cmp, baseline_model, GLM_MODEL_ID)
            rows.append(
                ScorecardRow.from_comparison(
                    node="analyze",
                    scenario=fixture_path,
                    model=GLM_MODEL_ID,
                    baseline_model=baseline_model,
                    comparison=cmp,
                )
            )
            gate_results.append((fixture_path, "precision", cmp.fp_bounded, "GLM over-flags"))
            comparisons.append((fixture_path, "precision", cmp))
            assert cmp.baseline is not None  # the run completed
        # Cross-scenario aggregate metric block (FUP-196 + best-metrics set) — printed
        # before the report-only gate summary so the scorecard leads with the numbers a
        # human adjudicates GLM on (yield rate, mean recall + severity, the safe-code
        # over-flag rate, an all-rows extras diagnostic, per-type recall).
        aggregate = _print_aggregate_metrics(
            comparisons, _GROUND_TRUTH_BY_FIXTURE, baseline_model, GLM_MODEL_ID
        )
        # REPORT-ONLY summary — pytest "passed" means the run completed, NOT a verdict.
        flagged = [(fx, dim, label) for fx, dim, ok, label in gate_results if not ok]
        green = len(gate_results) - len(flagged)
        print(  # noqa: T201 — operator gate summary
            "\n"
            + "=" * 72
            + "\nGLM SCORECARD — REPORT ONLY: pytest 'passed' = the run COMPLETED, not a verdict."
            + "\nGLM is a DIFFERENT model (not a within-family flip), so a recall delta vs"
            + "\nAnthropic is a model-capability signal to interpret, not an automatic fail."
            + f"\n  {green}/{len(gate_results)} scenarios where GLM matched the bar "
            + "(recall held / safe-code FP bounded)."
            + "".join(f"\n  {dim.upper()} {label}: {fx}" for fx, dim, label in flagged)
            + "\n  Read the per-scenario detail above: recall/yield (a recall-0 row may be a"
            + "\n  non-conforming GLM JSON response — the FUP-196 yield signal)"
            + "\n  plus over-flag patterns on safe code."
            + "\n"
            + "=" * 72
        )
    finally:
        # Persist in `finally` so a paid run's partial rows survive a mid-loop failure
        # too (the printed report shows only on success); nested so the provider close
        # runs regardless of the write. `if rows` avoids writing an empty artifact.
        try:
            report_dir = Path("reports/scorecard")
            if rows or aggregate is not None:
                report_dir.mkdir(parents=True, exist_ok=True)
            if rows:
                card = Scorecard(rows=tuple(rows))
                (report_dir / "glm-vs-anthropic-scorecard.json").write_text(
                    card.to_json(), encoding="utf-8"
                )
                (report_dir / "glm-vs-anthropic-scorecard.html").write_text(
                    card.to_html(), encoding="utf-8"
                )
                print(  # noqa: T201 — operator artifact pointer
                    f"\n[scorecard written to {report_dir}/glm-vs-anthropic-scorecard.json + .html]"
                )
            if aggregate is not None:
                # Persist the headline aggregate so it survives into the durable artifact (the
                # printed block is otherwise lost to stdout capture — /code-review finding).
                (report_dir / "glm-vs-anthropic-aggregate.json").write_text(
                    json.dumps(aggregate, indent=2), encoding="utf-8"
                )
                print(  # noqa: T201 — operator artifact pointer
                    f"\n[aggregate metrics written to {report_dir}/glm-vs-anthropic-aggregate.json]"
                )
        finally:
            await baseline_provider.aclose()
            await candidate_provider.aclose()


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model GLM diagnostic spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
@pytest.mark.asyncio
async def test_glm_recall_miss_diagnostic() -> None:
    """OPT-IN, real spend — DIAGNOSTIC, not a gate. Reruns GLM N times (`OUTRIDER_DIAG_RUNS`,
    default 5) on the scorecard's known recall MISSES to separate a STOCHASTIC miss (caught
    some runs) from a SYSTEMATIC one (never caught), capturing GLM's raw response on each miss
    so a 0 is explainable rather than opaque. GLM-only (Sonnet catches both); cost =
    len(fixtures) × N analyze calls (default 2 × 5 = 10). Set `OUTRIDER_DIAG_FIXTURES`
    (comma-separated `_GROUND_TRUTH_BY_FIXTURE` paths) to target others. Report-only: pytest
    'passed' means the run COMPLETED, not a verdict."""
    baseten_key = os.environ.get("BASETEN_API_KEY")
    if not baseten_key or baseten_key.startswith("op://"):
        pytest.skip(
            "BASETEN_API_KEY (resolved, not an op:// ref) is required for the GLM diagnostic; "
            "run under `op run --env-file=.env -- ...`"
        )

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.glm_provider import GLMProvider  # noqa: PLC0415

    runs = int(os.environ.get("OUTRIDER_DIAG_RUNS", "5"))
    env_fixtures = os.environ.get("OUTRIDER_DIAG_FIXTURES")
    fixtures = (
        [f.strip() for f in env_fixtures.split(",") if f.strip()]
        if env_fixtures
        else [_MISSING_ERROR_HANDLING_FIXTURE, _WEAK_CRYPTO_FIXTURE]
    )
    capturing = _CapturingExchangePersister()
    provider = GLMProvider(api_key=SecretStr(baseten_key), persister=capturing)
    results: list[dict[str, object]] = []
    try:
        for fixture_path in fixtures:
            result = await diagnose_recall_stability(
                fixture_path,
                provider=provider,
                model=GLM_MODEL_ID,
                capturing=capturing,
                runs=runs,
            )
            results.append(result)
            print(  # noqa: T201 — operator diagnostic
                f"\n[{fixture_path}] GLM caught {result['catches']}/{runs} → {result['verdict']}"
            )
            misses_raw = result["misses_raw"]
            assert isinstance(misses_raw, list)
            for i, raw in enumerate(misses_raw, 1):
                snippet = raw if len(raw) <= 800 else raw[:800] + " …[truncated]"
                print(f"  miss {i} raw response: {snippet}")  # noqa: T201 — operator diagnostic
        print(  # noqa: T201 — operator interpretation
            "\n"
            + "=" * 72
            + "\nGLM MISS DIAGNOSTIC — REPORT ONLY. 'systematic' (0/N) is a real recall gap;"
            + "\n'stochastic' (0<k<N) is sampling noise — raise OUTRIDER_DIAG_RUNS to tighten."
            + "\nRead the captured raw response to see whether GLM emitted nothing, a different"
            + "\nfinding_type, or the right finding at a wrong line (a window miss)."
            + "\n"
            + "=" * 72
        )
        assert results  # the run completed
    finally:
        try:
            if results:
                report_dir = Path("reports/scorecard")
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "glm-recall-miss-diagnostic.json").write_text(
                    json.dumps(
                        {"model": GLM_MODEL_ID, "runs": runs, "scenarios": results}, indent=2
                    ),
                    encoding="utf-8",
                )
                print(  # noqa: T201 — operator artifact pointer
                    f"\n[diagnostic written to {report_dir}/glm-recall-miss-diagnostic.json]"
                )
        finally:
            await provider.aclose()
