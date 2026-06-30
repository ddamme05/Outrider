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

import json
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — runtime use: resolver signature
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from outrider.agent.eval_driver import EvalFixture
from outrider.agent.nodes.analyze import analyze
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState
from outrider.schemas.triage_result import (
    ReviewDimension,
    ReviewTier,
    RiskLevel,
    TriageResult,
)

from .grading import DEFAULT_LINE_WINDOW, ModelComparison, compare, grade

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.llm.base import LLMProvider
    from outrider.schemas.review_finding import ReviewFinding

    from .grading import ExpectedFinding


class _NullSink:
    """No-op audit sink — the comparison reads findings from analyze's RETURN, not the
    audit stream, so every emit is discarded. Duck-types every sink analyze touches
    (phase / file-examination / analyze-event). ONE exception: it COUNTS
    `emit_analyze_response_rejected` calls so the structured-output YIELD signal (FUP-196 —
    a rejected/unparseable response vs a valid-empty one) is recoverable; all else discarded."""

    def __init__(self) -> None:
        # Structured-output rejections (fence/schema fail → zero findings) this run emitted.
        # >0 means the model produced UNPARSEABLE output, distinct from a valid-empty one.
        self.n_analyze_rejected = 0

    async def emit_phase(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_file_examination(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_finding(self, finding: Any, *, is_eval: bool) -> None:  # noqa: ARG002
        return None

    async def emit_finding_proposal_rejected(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_analyze_response_rejected(self, event: Any) -> None:  # noqa: ARG002
        self.n_analyze_rejected += 1

    async def emit_analyze_completed(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_scope_exclusion(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_cache_lookup(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_cache_serve(self, event: object) -> None:  # noqa: ARG002
        return None

    async def emit_observed_skip_shadow(self, event: object) -> None:  # noqa: ARG002
        return None

    async def emit_anomaly(self, **_kwargs: object) -> None:
        return None


class _NoOpImportPathResolver:
    """Resolves nothing (returns `[]`) — the comparison runs pass-0 analyze only, which
    does no cross-file trace resolution. Mirrors `eval_driver._NoOpImportPathResolver`."""

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:  # noqa: ARG002
        return []


def _triage_dims_and_risk(
    fixture: EvalFixture,
) -> tuple[tuple[ReviewDimension, ...], RiskLevel]:
    """The `relevant_dimensions` + `overall_risk` the fixture's OWN scripted triage
    assigned — the canonical triage output the driven scenarios feed the real triage node.
    Read here (not hard-coded) so the adapter is triage-faithful: a code-quality /
    performance scenario gets its real dimensions, not a blanket SECURITY. Only the TIER is
    overridden downstream (the single variable under test)."""
    triage_responses = fixture.llm_responses.get("triage")
    if not triage_responses:
        raise ValueError("eval fixture has no scripted triage response to read dimensions from")
    parsed = json.loads(triage_responses[0])
    dimensions = tuple(ReviewDimension(d) for d in parsed["relevant_dimensions"])
    return dimensions, RiskLevel(parsed["overall_risk"])


def state_from_eval_fixture(
    fixture_path: str | Path,
    *,
    tier: ReviewTier = ReviewTier.STANDARD,
    dimensions: tuple[ReviewDimension, ...] | None = None,
) -> ReviewState:
    """Build a post-intake / post-triage `ReviewState` from a `mock_github/*.json`
    eval fixture, every changed file pinned to `tier`.

    Reuses the fixture's REAL PR content — the vulnerable code + patch the driven
    scenarios already exercise — but HOLDS TRIAGE FIXED. The model-tier gate varies
    only the analyze model; if the triage tier were itself model-derived it would
    confound the comparison. `tier` defaults to STANDARD (the tier whose model the
    flip changes), so the scenario reads as "if this file were STANDARD, does the
    candidate model still catch the known finding?". This builder STANDS IN for
    intake+triage: it constructs the enriched `changed_files` that `run_review`'s
    `_seed_state` leaves empty (intake fills it there via the fixture GitHub client),
    because this harness runs the analyze node in isolation — no intake, no GitHub. It
    builds the SAME `ChangedFile` shape intake produces (including `language=None`,
    which intake never sets — analyze infers Python from the path), so the state the
    real run analyzes is faithful to production.

    `relevant_dimensions` and `overall_risk` are read from the fixture's OWN scripted
    triage response (a code-quality / performance scenario gets its real dimensions, not a
    blanket SECURITY) — only the TIER is overridden, the single variable under test. Pass
    `dimensions` to override explicitly.
    """
    with open(fixture_path, encoding="utf-8") as fh:
        fixture = EvalFixture.model_validate(json.load(fh))
    changed_files = tuple(
        ChangedFile(
            path=f.path,
            status=f.status,
            additions=f.additions,
            deletions=f.deletions,
            patch=f.patch,
            content_base=f.content_base,
            content_head=f.content_head,
            # Left None to match intake EXACTLY (intake.py never sets `language`;
            # analyze infers Python from the path via `_is_python_file`, and no
            # production code reads this field). The harness builds the state intake
            # would build, so the real run is faithful.
            language=None,
            # Normalize away a non-renamed `previous_path` exactly as intake does
            # (intake.py: `previous_path = previous_filename if status == "renamed"
            # else None`) — `ChangedFile` rejects a non-renamed file carrying one, so
            # this keeps the adapter no stricter than the production intake path.
            previous_path=f.previous_path if f.status == "renamed" else None,
        )
        for f in fixture.files
    )
    pr_context = PRContext(
        installation_id=fixture.installation_id,
        owner=fixture.owner,
        repo=fixture.repo,
        pr_number=fixture.pr_number,
        base_sha=fixture.base_sha,
        head_sha=fixture.head_sha,
        pr_title=fixture.pr_title,
        pr_body=fixture.pr_body,
        author=fixture.author,
        total_additions=fixture.total_additions,
        total_deletions=fixture.total_deletions,
        changed_files=changed_files,
    )
    fixture_dimensions, fixture_risk = _triage_dims_and_risk(fixture)
    triage_result = TriageResult(
        file_tiers={f.path: tier for f in fixture.files},
        overall_risk=fixture_risk,
        relevant_dimensions=dimensions if dimensions is not None else fixture_dimensions,
        reasoning="model-tier comparison harness: triage held fixed (dimensions + risk from "
        "the fixture's own triage) so the gate varies only the analyze model + STANDARD tier",
        policy_version=ACTIVE_POLICY_VERSION,
    )
    return ReviewState(
        review_id=uuid4(),
        received_at=datetime.now(UTC),
        pr_context=pr_context,
        triage_result=triage_result,
        is_eval=True,
    )


async def run_analyze_under_model(
    state: ReviewState, *, provider: LLMProvider, model: str
) -> tuple[tuple[ReviewFinding, ...], bool]:
    """Run one analyze pass over `state` with `model` for BOTH tiers (so the scenario's
    file is analyzed by `model` regardless of its triage tier). Returns `(findings,
    rejected)` — `rejected` is True iff analyze emitted any AnalyzeResponseRejectedEvent
    (the model's structured output failed to parse), the YIELD signal (FUP-196).
    Provider-injected; non-rejection audit emits are discarded. `model` is passed as both
    `analyze_model` and `standard_analyze_model` so a STANDARD-tier scenario file routes to
    it (the thing the flip changes)."""
    sink = _NullSink()
    result = await analyze(
        state,
        provider=provider,
        analyze_model=model,
        standard_analyze_model=model,
        phase_event_sink=sink,
        file_examination_sink=sink,
        analyze_event_sink=sink,
        anomaly_sink=sink,
        import_path_resolver=_NoOpImportPathResolver(),
    )
    rounds = result["analysis_rounds"]
    findings = tuple(rounds[0].findings) if rounds else ()
    return findings, sink.n_analyze_rejected > 0


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
    baseline_recall_floor: float = 1.0,
) -> ModelComparison:
    """Run `state` under the baseline (Sonnet) and candidate (Haiku) models, grade each
    against `ground_truth`, and apply the gate. For the REAL run `baseline_provider` and
    `candidate_provider` are the SAME `AnthropicProvider` (the model differs via
    `*_model`); for the machinery test they are distinct scripted providers so a recall
    divergence can be injected deterministically. All three declared gate thresholds
    (`recall_tolerance`, `fp_allowance`, `baseline_recall_floor`) forward to `compare()`."""
    baseline_findings, baseline_rejected = await run_analyze_under_model(
        state, provider=baseline_provider, model=baseline_model
    )
    candidate_findings, candidate_rejected = await run_analyze_under_model(
        state, provider=candidate_provider, model=candidate_model
    )
    baseline_grade = grade(baseline_findings, ground_truth, line_window=line_window)
    candidate_grade = grade(candidate_findings, ground_truth, line_window=line_window)
    return compare(
        baseline_grade,
        candidate_grade,
        recall_tolerance=recall_tolerance,
        fp_allowance=fp_allowance,
        baseline_recall_floor=baseline_recall_floor,
        baseline_rejected=baseline_rejected,
        candidate_rejected=candidate_rejected,
    )
