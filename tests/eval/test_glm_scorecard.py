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
(`_GROUND_TRUTH_BY_FIXTURE`) + safe-code (`_SAFE_CODE_FIXTURES`) fixtures — ~24 small
analyze calls. Cost-per-review (the third scorecard column) is a separate
`build_scorecard` pass; this gate measures quality + yield first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from outrider.llm.glm_provider import GLM_MODEL_ID

from .model_comparison import compare_models_on_scenario, state_from_eval_fixture
from .scorecard import Scorecard, ScorecardRow
from .test_model_comparison import (
    _GROUND_TRUTH_BY_FIXTURE,
    _SAFE_CODE_FIXTURES,
    _NoOpExchangePersister,
    _print_scenario_report,
    _run_scenario_isolating_transients,
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
            assert cmp.baseline is not None  # the run completed
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
        # Persist the structured artifact (json + html) alongside the printed report,
        # reusing the same Scorecard serializers test_scorecard.py writes to
        # reports/scorecard/. One row per completed scenario; survives pytest capture.
        report_dir = Path("reports/scorecard")
        report_dir.mkdir(parents=True, exist_ok=True)
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
    finally:
        await baseline_provider.aclose()
        await candidate_provider.aclose()
