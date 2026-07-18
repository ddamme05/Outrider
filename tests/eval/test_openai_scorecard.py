"""Opt-in, real-spend: Anthropic-vs-GPT-5.6 scorecard over the analyze node.

The openai-native-host candidate gate (specs/2026-07-18-openai-native-host.md).
Runs the SAME recall + precision scenarios as the Sonnet-vs-Haiku model-tier
comparison (`test_model_comparison.py`) and the GLM scorecard
(`test_glm_scorecard.py`), with TWO candidate columns matching the spec's
evidence-domain rule — the scorecard canonizes exactly the two ANALYZE fields:

  - `gpt-5.6-sol`  vs the Anthropic DEEP-tier baseline (`cfg.analyze_model`)
  - `gpt-5.6-luna` vs the Anthropic STANDARD-tier baseline
    (`cfg.standard_analyze_model`)

REPORT-ONLY BY DESIGN (the glm-scorecard precedent): pytest "passed" means the
run COMPLETED, not that a gate passed. ADJUDICATION RULE (frozen in the spec's
gates section): the operator reads the report and records the verdict + report
pointer in the spec's Actual Outcome. Canonizing a provisional default requires
BOTH (a) structured-output yield at the #059 bar (zero rejected responses
across the rows — json_object mode + prompt-named fields are the conformance
drivers here) and (b) the `grading.py` baseline recall floor against that
tier's incumbent. A miss on either swaps THAT field to `gpt-5.6-terra` and
reruns this scorecard — never a silent fallback — and a Terra swap first
inherits the full paid-wire probe matrix (spikes/openai/probe.py).

PRECONDITION: the openai host is WIRE-PENDING until spikes/openai/probe.py has
captured the request/usage/refusal fixtures — run the probe BEFORE this
scorecard so a conformance surprise is caught on a one-call capture, not a
40-call scorecard.

Run (keys resolve from .env via 1Password):
  OUTRIDER_EVAL_REAL_MODELS=1 op run --env-file=.env -- \
    uv run pytest tests/eval/test_openai_scorecard.py --is-eval -v -s

Cost: 2 candidate columns x (16 recall + 4 safe) scenarios + the shared
Anthropic baselines — roughly 80 small analyze calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from .model_comparison import compare_models_on_scenario, state_from_eval_fixture
from .scorecard import Scorecard, ScorecardRow
from .test_model_comparison import (
    _GROUND_TRUTH_BY_FIXTURE,
    _SAFE_CODE_FIXTURES,
    _NoOpExchangePersister,
    _print_aggregate_metrics,
    _print_scenario_report,
    _run_scenario_isolating_transients,
)

if TYPE_CHECKING:
    from .grading import ExpectedFinding, ModelComparison

_SOL = "gpt-5.6-sol"
_LUNA = "gpt-5.6-luna"


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model GPT-5.6 scorecard spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1",
)
@pytest.mark.asyncio
async def test_gpt56_vs_anthropic_scorecard() -> None:
    """OPT-IN, real spend — the two-column GPT-5.6 candidate scorecard.

    REPORT-ONLY: asserts only that the run COMPLETED; the operator adjudicates
    per the spec's frozen rule (yield at the #059 bar AND the per-tier baseline
    recall floor; miss → Terra swap + rerun). Recall is TYPE-EXACT, so a delta
    can be a true miss OR a classification disagreement — read the printed
    `missed`/`extra` detail before acting. A non-conforming json_object
    response parses to no findings → recall 0, so the recall dimension also
    carries the yield signal.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the Anthropic baselines")
    if not openai_key or openai_key.startswith("op://"):
        pytest.skip(
            "OPENAI_API_KEY (resolved, not an op:// ref) is required for the GPT-5.6 "
            "candidates; run under `op run --env-file=.env -- ...`"
        )

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.llm.host_profiles import OPENAI_PROFILE  # noqa: PLC0415
    from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider  # noqa: PLC0415

    cfg = ModelConfig()
    persister = _NoOpExchangePersister()
    baseline_provider = AnthropicProvider(
        api_key=SecretStr(anthropic_key), model_config=cfg, persister=persister
    )
    candidate_provider = OpenAICompatibleProvider(
        api_key=SecretStr(openai_key),
        profile=OPENAI_PROFILE,
        persister=persister,
        models=(_SOL, _LUNA),
    )

    # Two candidate columns per the spec's evidence-domain rule: each analyze
    # field is judged against ITS incumbent, never one bar for both.
    columns: tuple[tuple[str, str], ...] = (
        (_SOL, cfg.analyze_model),
        (_LUNA, cfg.standard_analyze_model),
    )

    gate_results: list[tuple[str, str, bool, str]] = []
    rows: list[ScorecardRow] = []
    # Per-column comparison lists: the aggregate printer takes ONE candidate/baseline
    # pair, so each column aggregates separately (the spec judges each analyze field
    # against ITS incumbent).
    comparisons_by_column: dict[str, list[tuple[str, str, ModelComparison]]] = {
        _SOL: [],
        _LUNA: [],
    }
    aggregates: dict[str, dict[str, object] | None] = {}

    async def _compare_or_errored(
        fixture_path: str,
        ground_truth: tuple[ExpectedFinding, ...],
        dimension: str,
        *,
        candidate_model: str,
        baseline_model: str,
    ) -> ModelComparison | None:
        async def _compare() -> ModelComparison:
            return await compare_models_on_scenario(
                state_from_eval_fixture(fixture_path),
                ground_truth,
                baseline_provider=baseline_provider,
                baseline_model=baseline_model,
                candidate_provider=candidate_provider,
                candidate_model=candidate_model,
            )

        return await _run_scenario_isolating_transients(
            fixture_path, dimension, gate_results, _compare
        )

    try:
        for candidate_model, baseline_model in columns:
            for fixture_path, ground_truth in _GROUND_TRUTH_BY_FIXTURE.items():
                cmp = await _compare_or_errored(
                    fixture_path,
                    ground_truth,
                    "recall",
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                )
                if cmp is None:
                    continue
                _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
                rows.append(
                    ScorecardRow.from_comparison(
                        node="analyze",
                        scenario=fixture_path,
                        model=candidate_model,
                        baseline_model=baseline_model,
                        comparison=cmp,
                    )
                )
                recall_ok = cmp.recall_held and cmp.baseline_valid
                gate_results.append(
                    (
                        fixture_path,
                        f"recall:{candidate_model}",
                        recall_ok,
                        f"{candidate_model} recall < {baseline_model}",
                    )
                )
                comparisons_by_column[candidate_model].append((fixture_path, "recall", cmp))
                assert cmp.baseline is not None  # the run completed
            for fixture_path in _SAFE_CODE_FIXTURES:
                cmp = await _compare_or_errored(
                    fixture_path,
                    (),
                    "precision",
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                )
                if cmp is None:
                    continue
                _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
                rows.append(
                    ScorecardRow.from_comparison(
                        node="analyze",
                        scenario=fixture_path,
                        model=candidate_model,
                        baseline_model=baseline_model,
                        comparison=cmp,
                    )
                )
                gate_results.append(
                    (
                        fixture_path,
                        f"precision:{candidate_model}",
                        cmp.fp_bounded,
                        f"{candidate_model} over-flags safe code",
                    )
                )
                comparisons_by_column[candidate_model].append((fixture_path, "precision", cmp))
        for candidate_model, baseline_model in columns:
            aggregates[candidate_model] = _print_aggregate_metrics(
                comparisons_by_column[candidate_model],
                _GROUND_TRUTH_BY_FIXTURE,
                baseline_model,
                candidate_model,
            )
    finally:
        # Persist in `finally` (the glm-scorecard shape) so a paid run's partial
        # rows survive a mid-loop failure; nested so provider close always runs.
        try:
            report_dir = Path("reports/scorecard")
            if rows or any(a is not None for a in aggregates.values()):
                report_dir.mkdir(parents=True, exist_ok=True)
            if rows:
                card = Scorecard(rows=tuple(rows))
                (report_dir / "openai-gpt56-scorecard.json").write_text(
                    card.to_json(), encoding="utf-8"
                )
                (report_dir / "openai-gpt56-scorecard.html").write_text(
                    card.to_html(), encoding="utf-8"
                )
                print(  # noqa: T201 — operator artifact pointer
                    f"\n[scorecard written to {report_dir}/openai-gpt56-scorecard.json + .html]"
                )
            for column, aggregate in aggregates.items():
                if aggregate is not None:
                    (report_dir / f"openai-gpt56-aggregate-{column}.json").write_text(
                        json.dumps(aggregate, indent=2), encoding="utf-8"
                    )
        finally:
            await baseline_provider.aclose()
            await candidate_provider.aclose()

    # Report-only completion pin: every scheduled scenario either produced a row
    # or an explicit ERRORED gate entry — nothing silently dropped.
    assert rows or gate_results
    flagged = [(fx, dim, label) for fx, dim, ok, label in gate_results if not ok]
    print(  # noqa: T201 — operator gate summary (adjudication happens on the report)
        f"\n[openai scorecard: {len(gate_results) - len(flagged)} green / "
        f"{len(flagged)} flagged advisory gates — adjudicate per the spec's frozen rule]"
    )
