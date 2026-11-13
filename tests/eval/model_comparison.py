"""Run the same analysis scenario under two supplied providers and compare results.

Scripted providers test the workflow without spending. A caller may supply
real providers for an explicitly enabled paid run. This runs one analysis
pass with identical prompts, not tracing, the full graph or GitHub posting."""

from __future__ import annotations

import json
from asyncio import current_task
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path  # noqa: TC003 — runtime use: resolver signature
from typing import TYPE_CHECKING, Any
from unittest.mock import patch
from uuid import uuid4

from outrider.agent.eval_driver import EvalFixture
from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS, analyze, analyze_file
from outrider.agent.nodes.analyze_aggregate import analyze_aggregate
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
    from collections.abc import Iterator, Sequence

    from outrider.agent.nodes.analyze import _FileOutcome
    from outrider.llm.base import LLMProvider
    from outrider.schemas.analyze_worker import AnalyzeWorkerOutcome
    from outrider.schemas.review_finding import ReviewFinding

    from .grading import ExpectedFinding


_producer_coverage_active = False


@contextmanager
def _capture_producer_coverage(
    enabled: bool,
) -> Iterator[list[tuple[ReviewFinding, ...]] | None]:
    """Read original producer findings at the real worker's outcome boundary.

    Capture only this task. Concurrent uninstrumented analysis still delegates
    unchanged; overlapping captures fail before a worker can call its provider.
    """
    if not enabled:
        yield None
        return
    global _producer_coverage_active  # noqa: PLW0603 — protect the temporary module patch
    if _producer_coverage_active:
        raise RuntimeError("producer coverage collection cannot overlap another collection")
    module = import_module("outrider.agent.nodes.analyze")
    original_builder = module._worker_outcome_for
    capture_task = current_task()
    captured: list[tuple[ReviewFinding, ...]] = []

    def record(
        file_outcome: _FileOutcome, *, path: str, pass_index: int, review_tier: ReviewTier
    ) -> AnalyzeWorkerOutcome:
        if current_task() is capture_task:
            if file_outcome.served_result is not None:
                raise RuntimeError("cached findings cannot establish independent producer coverage")
            originals = (
                file_outcome.observed_skip_result.admitted_findings
                if file_outcome.observed_skip_result is not None
                else file_outcome.producer_findings
            )
            captured.append(tuple(finding.model_copy(deep=True) for finding in originals))
        return original_builder(
            file_outcome, path=path, pass_index=pass_index, review_tier=review_tier
        )

    _producer_coverage_active = True
    try:
        with patch.object(module, "_worker_outcome_for", record):
            yield captured
    finally:
        _producer_coverage_active = False


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

    async def emit_finding(  # noqa: ARG002
        self, finding: Any, *, is_eval: bool, phase_key: str | None = None
    ) -> None:
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

    def resolve_specifier_candidate_paths(
        self, specifier: str, importing_file_path: str, import_root: Path
    ) -> list[Path]:  # noqa: ARG002
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
    overall_risk: RiskLevel | None = None,
    analysis_file: str | None = None,
) -> ReviewState:
    """Build an analyzer-only state with enriched files and a fixed review tier.

    Dimensions and risk normally come from the fixture's scripted triage.
    Supply both explicitly for a corpus without scripts. This constructs the
    post-intake/triage data shape without running those nodes or using GitHub.
    An analysis_file keeps the complete PR context and skips its other files.
    """
    with open(fixture_path, encoding="utf-8") as fh:
        fixture = EvalFixture.model_validate(json.load(fh))
    if analysis_file is not None and analysis_file not in {f.path for f in fixture.files}:
        raise ValueError("the selected analysis file is absent from the captured PR")
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
            # analyze infers the language from the path (registry gate via
            # `_language_supported`; Python-only stages via `_is_python_file`), and no
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
    if overall_risk is None:
        fixture_dimensions, fixture_risk = _triage_dims_and_risk(fixture)
    elif dimensions is None:
        raise ValueError("an explicit benchmark risk requires explicit review dimensions")
    else:
        fixture_dimensions, fixture_risk = dimensions, overall_risk
    triage_result = TriageResult(
        file_tiers={
            f.path: tier if analysis_file is None or f.path == analysis_file else ReviewTier.SKIP
            for f in fixture.files
        },
        overall_risk=fixture_risk,
        relevant_dimensions=dimensions if dimensions is not None else fixture_dimensions,
        reasoning=(
            "model-tier comparison harness: triage held fixed (dimensions + risk from "
            "the fixture's own triage) so the gate varies only the analyze model + STANDARD tier"
        )
        if overall_risk is None
        else "Benchmark routing is fixed; model triage is not measured.",
        policy_version=ACTIVE_POLICY_VERSION,
    )
    return ReviewState(
        review_id=uuid4(),
        received_at=datetime.now(UTC),
        pr_context=pr_context,
        triage_result=triage_result,
        is_eval=True,
    )


@dataclass(frozen=True)
class AnalyzeRun:
    """Admitted results and worker provenance from one real analysis pass.

    `producer_findings` retains surviving producer origin for existing metrics.
    `producer_coverage_findings` records the producer's original output when
    requested: None means uncollected; an empty tuple means collected with no
    findings. Coverage is independent of which producer objects survive merging.
    """

    findings: tuple[ReviewFinding, ...]
    outcomes: tuple[AnalyzeWorkerOutcome, ...]
    response_rejections: int
    producer_coverage_findings: tuple[ReviewFinding, ...] | None = None

    @property
    def producer_findings(self) -> tuple[ReviewFinding, ...]:
        return tuple(
            finding
            for outcome in self.outcomes
            for finding in outcome.admitted_findings
            if finding.content_hash in outcome.producer_observed_hashes
        )

    @property
    def proposal_rejections(self) -> int:
        return sum(outcome.n_proposals_rejected for outcome in self.outcomes)


async def run_analyze_instrumented(
    state: ReviewState,
    *,
    provider: LLMProvider,
    model: str,
    collect_producer_coverage: bool = False,
    supplemental_context_enabled: bool = True,
) -> AnalyzeRun:
    """Run the real planner, workers and aggregate with one forced model.

    Both analysis tiers use the supplied model. Opt in to independent producer
    coverage for v2 measurements; this observes the real worker once, without
    another model call or structural scan. Existing metrics retain worker origins.
    """
    sink = _NullSink()
    resolver = _NoOpImportPathResolver()
    # Compose the post-cutover three-vertex pass in-process: planner
    # Command → one analyze_file worker per Send (in Send order) →
    # analyze_aggregate over the merged outcomes. Sequential per-file
    # semantics are identical; only the vertex boundaries moved.
    cmd = await analyze(
        state,
        provider=provider,
        analyze_model=model,
        standard_analyze_model=model,
        phase_event_sink=sink,
        file_examination_sink=sink,
        analyze_event_sink=sink,
        anomaly_sink=sink,
        import_path_resolver=resolver,
        **({"supplemental_context_enabled": False} if not supplemental_context_enabled else {}),
    )
    outcomes = []
    with _capture_producer_coverage(collect_producer_coverage) as captured:
        for send in cmd.goto if isinstance(cmd.goto, list) else []:
            captured_before = len(captured) if captured is not None else 0
            worker_update = await analyze_file(
                send.arg,
                provider=provider,
                analyze_model=model,
                standard_analyze_model=model,
                import_path_resolver=resolver,
                phase_event_sink=sink,
                file_examination_sink=sink,
                analyze_event_sink=sink,
            )
            if captured is not None and len(captured) != captured_before + 1:
                raise RuntimeError("independent producer coverage missed a worker outcome")
            outcomes.extend(worker_update["analyze_worker_outcomes"])
    state_after = state.model_copy(
        update={
            "analyze_worker_outcomes": [*state.analyze_worker_outcomes, *outcomes],
            "analyze_pass_started_at": (cmd.update or {})["analyze_pass_started_at"],
        }
    )
    result = await analyze_aggregate(
        state_after,
        analyze_event_sink=sink,
        phase_event_sink=sink,
        anomaly_sink=sink,
        analyze_model=model,
        standard_analyze_model=model,
        total_review_budget_tokens=DEFAULT_REVIEW_BUDGET_TOKENS,
    )
    rounds = result["analysis_rounds"]
    findings = tuple(rounds[0].findings) if rounds else ()
    producer_coverage = (
        tuple(finding for batch in captured for finding in batch) if captured is not None else None
    )
    return AnalyzeRun(findings, tuple(outcomes), sink.n_analyze_rejected, producer_coverage)


async def run_analyze_under_model(
    state: ReviewState, *, provider: LLMProvider, model: str
) -> tuple[tuple[ReviewFinding, ...], int]:
    """Return admitted findings and structured-response rejection count.

    Keep the existing comparison API; detailed refresh reports use
    run_analyze_instrumented to retain producer origins and parser counters.
    """
    result = await run_analyze_instrumented(state, provider=provider, model=model)
    return result.findings, result.response_rejections


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
    baseline_findings, baseline_n_rejected = await run_analyze_under_model(
        state, provider=baseline_provider, model=baseline_model
    )
    candidate_findings, candidate_n_rejected = await run_analyze_under_model(
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
        baseline_n_rejected=baseline_n_rejected,
        candidate_n_rejected=candidate_n_rejected,
    )
