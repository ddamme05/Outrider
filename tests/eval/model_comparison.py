"""Run the analyze node under two models on one scenario, grade both, apply the gate.

The opt-in half of the analyze model-tier quality gate
(specs/2026-06-08-analyze-tiered-model-routing.md step 2). The grading + gate is in
`grading.py` (pure, fully tested); this module RUNS a scenario under a model to produce
the findings to grade.

The model run is PROVIDER-INJECTED — this module imports no LLM SDK and takes an
`LLMProvider`:
  - zero-spend machinery test: inject a SCRIPTED provider (canned per-model responses) to
    prove the end-to-end flow catches a recall regression deterministically (CI-safe);
  - real evidence run (SPEND): the caller injects the real `AnthropicProvider`, gated
    behind an env flag + API keys (see `test_model_comparison.py::test_real_model_*`).

Runs ONE analyze pass (pass 0) over the scenario state — no trace, no full graph, no
GitHub — so the comparison isolates the analyze model's finding quality. Both models see
the identical prompt + scope context; only `request.model` differs.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — runtime use: resolver signature
from typing import TYPE_CHECKING, Any

from outrider.agent.nodes.analyze import analyze

from .grading import DEFAULT_LINE_WINDOW, ModelComparison, compare, grade

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.llm.base import LLMProvider
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.review_state import ReviewState

    from .grading import ExpectedFinding


class _NullSink:
    """No-op audit sink — the comparison reads findings from analyze's RETURN, not the
    audit stream, so every emit is discarded. Duck-types every sink analyze touches
    (phase / file-examination / analyze-event)."""

    async def emit_phase(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_file_examination(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_finding(self, finding: Any, *, is_eval: bool) -> None:  # noqa: ARG002
        return None

    async def emit_finding_proposal_rejected(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_analyze_response_rejected(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_analyze_completed(self, event: Any) -> None:  # noqa: ARG002
        return None


class _NoOpImportPathResolver:
    """Resolves nothing (returns `[]`) — the comparison runs pass-0 analyze only, which
    does no cross-file trace resolution. Mirrors `eval_driver._NoOpImportPathResolver`."""

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:  # noqa: ARG002
        return []


async def run_analyze_under_model(
    state: ReviewState, *, provider: LLMProvider, model: str
) -> tuple[ReviewFinding, ...]:
    """Run one analyze pass over `state` with `model` for BOTH tiers (so the scenario's
    file is analyzed by `model` regardless of its triage tier), returning the findings.
    Provider-injected; audit emits are discarded. `model` is passed as both
    `analyze_model` and `standard_analyze_model` so a STANDARD-tier scenario file routes
    to it (the thing the flip changes)."""
    sink = _NullSink()
    result = await analyze(
        state,
        provider=provider,
        analyze_model=model,
        standard_analyze_model=model,
        phase_event_sink=sink,
        file_examination_sink=sink,
        analyze_event_sink=sink,
        import_path_resolver=_NoOpImportPathResolver(),
    )
    rounds = result["analysis_rounds"]
    return tuple(rounds[0].findings) if rounds else ()


async def compare_models_on_scenario(
    state: ReviewState,
    ground_truth: Sequence[ExpectedFinding],
    *,
    baseline_provider: LLMProvider,
    baseline_model: str,
    candidate_provider: LLMProvider,
    candidate_model: str,
    line_window: int = DEFAULT_LINE_WINDOW,
    recall_tolerance: float = 0.0,
    fp_allowance: int = 0,
) -> ModelComparison:
    """Run `state` under the baseline (Sonnet) and candidate (Haiku) models, grade each
    against `ground_truth`, and apply the gate. For the REAL run `baseline_provider` and
    `candidate_provider` are the SAME `AnthropicProvider` (the model differs via
    `*_model`); for the machinery test they are distinct scripted providers so a recall
    divergence can be injected deterministically."""
    baseline_findings = await run_analyze_under_model(
        state, provider=baseline_provider, model=baseline_model
    )
    candidate_findings = await run_analyze_under_model(
        state, provider=candidate_provider, model=candidate_model
    )
    baseline_grade = grade(baseline_findings, ground_truth, line_window=line_window)
    candidate_grade = grade(candidate_findings, ground_truth, line_window=line_window)
    return compare(
        baseline_grade,
        candidate_grade,
        recall_tolerance=recall_tolerance,
        fp_allowance=fp_allowance,
    )
