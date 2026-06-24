"""Deterministic triage-tier grading for the triage model-tier quality gate.

Step 2 of the eval arc — parallel to `grading.py` (analyze finding-grading) +
`model_comparison.py` (the analyze runner). Grades a triage node's `TriageResult`
against hand-authored `ExpectedTriage`, then applies an asymmetric gate. So
grading is STRUCTURAL, not LLM-as-judge — same trust-aligned choice as
`grading.py` (no model validating a model).

The metrics, keyed on the ANALYSIS FLOOR (analyze admits only DEEP/STANDARD;
SKIM/SKIP never reach it — see `agent/nodes/analyze.py`):
  - `n_dropped_from_analysis` — the SAFETY metric: an expected-analyzed file
    (DEEP or STANDARD) the candidate pushed BELOW the floor (to SKIM/SKIP) leaves
    the review set entirely. This is the triage analogue of analyze recall loss.
  - `n_deep_downgraded` — softer: expected DEEP, candidate STANDARD. The file is
    still reviewed, by the cheaper model; tracked, not a hard safety fail.
  - `n_overtiered` — the COST metric: expected SKIM/SKIP, candidate DEEP/STANDARD
    (a file that shouldn't be analyzed is — wasted spend).
  - `dimension_recall` / `dimension_precision` — review-dimension set overlap.
  - `risk_correct` + `under_risked` — `overall_risk` exact match, and whether the
    candidate's risk is BELOW the expected (under-risking — a safety failure).

`compare_triage()` then applies the gate: the candidate passes only if the
BASELINE itself is clean (0 drops, not under-risked — the vacuous-pass guard,
same shape as `grading.compare`'s `baseline_valid`), the candidate adds NO new
drops and is not under-risked, dimension recall does not regress, and over-tiering
stays bounded.

Pure functions over a validated `TriageResult` + `ExpectedTriage`;
`run_triage_under_model` runs the real triage node (provider-injected, audit
discarded via `_NullSink`) to produce the `TriageResult` to grade — the caller
supplies a provider that does not persist (scripted in CI / a no-op
`LLMExchangePersister` on the real path), since the triage node's
`provider.complete()` is what would write LLM events.
"""

from __future__ import annotations

from collections.abc import (
    Mapping,  # noqa: TC003 — runtime use: Pydantic resolves the ExpectedTriage field annotation
)
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from outrider.agent.nodes.triage import triage
from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel, TriageResult

from .model_comparison import _NullSink

if TYPE_CHECKING:
    from outrider.llm.base import LLMProvider
    from outrider.schemas.review_state import ReviewState

# analyze.py's pass-0 worklist admits ONLY these tiers; SKIM/SKIP never reach
# analyze, so a file tiered below this floor leaves the review set.
_ANALYZED_TIERS: frozenset[ReviewTier] = frozenset({ReviewTier.DEEP, ReviewTier.STANDARD})

# RiskLevel is a str enum with no inherent order; rank it for under-risk detection.
_RISK_RANK: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class ExpectedTriage(BaseModel):
    """Hand-authored ground truth for one scenario's triage: the tier each changed
    file SHOULD get, the dimensions that SHOULD apply, and the expected overall
    risk. Frozen — ground truth is fixed. The grader matches a model's
    `TriageResult` against this (the analogue of `ExpectedFinding` for analyze)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_file_tiers: Mapping[str, ReviewTier]
    overall_risk: RiskLevel
    relevant_dimensions: tuple[ReviewDimension, ...]


@dataclass(frozen=True, slots=True)
class TriageGrade:
    """One model's triage graded against ground truth. `n_dropped_from_analysis`
    and `under_risked` are the safety signals the gate keys on; `n_deep_downgraded`
    is the softer depth signal; `n_overtiered` is the cost signal. `dropped_files`
    is kept for the artifact (which files left the review set)."""

    tier_accuracy: float
    n_files: int
    n_dropped_from_analysis: int
    n_deep_downgraded: int
    n_overtiered: int
    dimension_recall: float
    dimension_precision: float
    risk_correct: bool
    under_risked: bool
    dropped_files: tuple[str, ...]


def grade_triage(actual: TriageResult, expected: ExpectedTriage) -> TriageGrade:
    """Grade one triage result against ground truth, over the EXPECTED file set
    (the triage node's policy gate guarantees it tiers exactly the changed files,
    which ground truth mirrors). A file absent from `actual` is treated as `SKIP`
    (below the analysis floor)."""
    expected_tiers = expected.expected_file_tiers
    n_files = len(expected_tiers)
    n_correct = 0
    n_dropped = 0
    n_downgraded = 0
    n_overtiered = 0
    dropped: list[str] = []
    for path, exp_tier in expected_tiers.items():
        act_tier = actual.file_tiers.get(path, ReviewTier.SKIP)
        if act_tier == exp_tier:
            n_correct += 1
        exp_analyzed = exp_tier in _ANALYZED_TIERS
        act_analyzed = act_tier in _ANALYZED_TIERS
        if exp_analyzed and not act_analyzed:
            # SAFETY: an expected-analyzed file pushed below the floor leaves review.
            n_dropped += 1
            dropped.append(path)
        elif exp_tier is ReviewTier.DEEP and act_tier is ReviewTier.STANDARD:
            # Softer: still reviewed, by the cheaper model.
            n_downgraded += 1
        elif not exp_analyzed and act_analyzed:
            # COST: a file that shouldn't be analyzed is.
            n_overtiered += 1
        # An in-floor change (e.g. STANDARD->DEEP, both analyzed) is intentionally
        # uncounted: the file is still reviewed, so it is neither a safety drop nor a
        # cost over-tier — only a minor depth shift the gate does not key on.

    expected_dims = set(expected.relevant_dimensions)
    actual_dims = set(actual.relevant_dimensions)
    matched_dims = expected_dims & actual_dims
    dimension_recall = (len(matched_dims) / len(expected_dims)) if expected_dims else 1.0
    dimension_precision = (len(matched_dims) / len(actual_dims)) if actual_dims else 1.0

    return TriageGrade(
        tier_accuracy=(n_correct / n_files) if n_files else 1.0,
        n_files=n_files,
        n_dropped_from_analysis=n_dropped,
        n_deep_downgraded=n_downgraded,
        n_overtiered=n_overtiered,
        dimension_recall=dimension_recall,
        dimension_precision=dimension_precision,
        risk_correct=actual.overall_risk == expected.overall_risk,
        under_risked=_RISK_RANK[actual.overall_risk] < _RISK_RANK[expected.overall_risk],
        dropped_files=tuple(dropped),
    )


@dataclass(frozen=True, slots=True)
class TriageComparison:
    """Baseline (Sonnet) vs candidate (Haiku) triage on one scenario, with the
    asymmetric gate verdict. Mirrors `grading.ModelComparison`:
      - `baseline_valid` — the BASELINE itself is clean (0 drops, not under-risked).
        A baseline that already drops/under-risks can't discriminate, so a candidate
        'hold' over it is vacuous.
      - `drop_held` — the candidate adds NO new drops vs baseline (the safety hold).
      - `risk_safety_held` — the candidate is not under-risked.
      - `overtier_bounded` — candidate over-tiering ≤ baseline + `overtier_allowance`.
      - `dimension_recall_held` — candidate dimension recall ≥ baseline − tolerance.
    `passes` is the AND of all five. `n_deep_downgraded` is reported on the grades
    but NOT gated here (the file is still reviewed)."""

    baseline: TriageGrade
    candidate: TriageGrade
    overtier_allowance: int
    dimension_recall_tolerance: float
    baseline_valid: bool
    drop_held: bool
    risk_safety_held: bool
    overtier_bounded: bool
    dimension_recall_held: bool

    @property
    def passes(self) -> bool:
        return (
            self.baseline_valid
            and self.drop_held
            and self.risk_safety_held
            and self.overtier_bounded
            and self.dimension_recall_held
        )


def compare_triage(
    baseline: TriageGrade,
    candidate: TriageGrade,
    *,
    overtier_allowance: int = 0,
    dimension_recall_tolerance: float = 0.0,
) -> TriageComparison:
    """Apply the triage quality gate. STRICT defaults (no new drops, no under-risk,
    no extra over-tiering, no recall regression); callers loosen explicitly and the
    chosen values are recorded on the result."""
    baseline_valid = baseline.n_dropped_from_analysis == 0 and not baseline.under_risked
    drop_held = candidate.n_dropped_from_analysis <= baseline.n_dropped_from_analysis
    risk_safety_held = not candidate.under_risked
    overtier_bounded = candidate.n_overtiered <= baseline.n_overtiered + overtier_allowance
    dimension_recall_held = (
        candidate.dimension_recall >= baseline.dimension_recall - dimension_recall_tolerance
    )
    return TriageComparison(
        baseline=baseline,
        candidate=candidate,
        overtier_allowance=overtier_allowance,
        dimension_recall_tolerance=dimension_recall_tolerance,
        baseline_valid=baseline_valid,
        drop_held=drop_held,
        risk_safety_held=risk_safety_held,
        overtier_bounded=overtier_bounded,
        dimension_recall_held=dimension_recall_held,
    )


async def run_triage_under_model(
    state: ReviewState, *, provider: LLMProvider, model: str
) -> TriageResult:
    """Run one triage pass over `state` with `model`, returning the `TriageResult`.
    Provider-injected; the node's phase emits are discarded via `_NullSink`. The
    triage node consumes `state.pr_context.changed_files` and produces `file_tiers`,
    so any pre-set `triage_result` on the state is ignored. The injected provider
    must not persist (scripted double / no-op exchange persister) — its
    `complete()` is what would write LLM events."""
    sink = _NullSink()
    result = await triage(state, provider=provider, triage_model=model, phase_event_sink=sink)
    return result["triage_result"]


async def compare_triage_models_on_scenario(
    state: ReviewState,
    expected: ExpectedTriage,
    *,
    baseline_provider: LLMProvider,
    baseline_model: str,
    candidate_provider: LLMProvider,
    candidate_model: str,
    overtier_allowance: int = 0,
    dimension_recall_tolerance: float = 0.0,
) -> TriageComparison:
    """Run `state` through triage under the baseline and candidate models, grade
    each against `expected`, and apply the gate. The parallel to
    `compare_models_on_scenario` for analyze."""
    baseline_triage = await run_triage_under_model(
        state, provider=baseline_provider, model=baseline_model
    )
    candidate_triage = await run_triage_under_model(
        state, provider=candidate_provider, model=candidate_model
    )
    return compare_triage(
        grade_triage(baseline_triage, expected),
        grade_triage(candidate_triage, expected),
        overtier_allowance=overtier_allowance,
        dimension_recall_tolerance=dimension_recall_tolerance,
    )
