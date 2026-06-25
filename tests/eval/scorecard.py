"""Cross-scenario eval scorecard: typed rows + JSON/HTML emit.

Productizes the inline `(fixture, dimension, ok, fail_label)` aggregation that
lived only in the opt-in spend test (`test_model_comparison.py`'s GATE SUMMARY
print) into reusable typed objects. One `ScorecardRow` per `(node, model,
scenario)`; a `Scorecard` collects rows and emits a JSON + HTML artifact.

Per `specs/2026-06-23-eval-runner-scorecard.md`:
  - Step 1 emits `node="analyze"` rows only; the `node` axis is in the schema
    for forward-compat (step 2 slots `node="triage"` rows in with no schema
    change), not filled with placeholder rows here.
  - Metric provenance is per row: quality from the analyze-direct comparison
    (`quality_source`), cost/latency from the full-graph run (`cost_source`),
    replay from the resume/persisting drivers (`replay_source`). The scorecard
    makes the split-path join explicit instead of pretending one path measures
    everything equally well.
  - `cost_source` is THREE-state so a missing cost is never a false zero:
    `full_graph` (measured), `not_measured` (the cost pass was not requested),
    `measure_failed` (it was requested but produced no usable number — a
    transient flake or a review that completed with no metrics). The aggregate
    surfaces `n_costed` so `total_cost_usd` carries its own denominator.
  - Report-only: a row records its gate verdict (`gate.passes`); nothing here
    raises on a failed gate. Enforcement, if ever wanted, is a separate surface.

Wires the three previously-unconsumed `metrics.py` shapes (`FalsePositiveRate`,
`CostPerReview`, `LatencyPerReview`) into their first consumer. Quality numbers
are pulled OFF a `grading.ModelComparison` (the deterministic gate); this module
computes no recall/precision/severity itself — only the false-positive RATE,
over the same finding population the grader's precision uses.
"""

from __future__ import annotations

import html
import json
from statistics import fmean
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .metrics import (
    CostPerReview,
    FalsePositiveRate,
    FindingPrecision,
    FindingRecall,
    LatencyPerReview,
    SeverityAccuracy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .grading import ModelComparison
    from .triage_grading import TriageComparison

QualitySource = Literal["analyze_direct"]
TriageSource = Literal["run_triage_direct"]
CostSource = Literal["full_graph", "not_measured", "measure_failed"]
ReplaySource = Literal["resume", "persisting", "not_applicable"]
RowStatus = Literal["ok", "errored"]


class GateVerdict(BaseModel):
    """The deterministic gate verdict for one candidate-vs-baseline comparison,
    flattened off `grading.ModelComparison` so the row carries the pass/fail AND
    the three declared-threshold sub-conditions that produced it. Keeping the
    sub-conditions makes a FAIL self-explanatory in the artifact (was it a recall
    drop, an FP balloon, or a non-discriminating baseline?)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passes: bool
    baseline_valid: bool
    recall_held: bool
    fp_bounded: bool
    recall_tolerance: float
    fp_allowance: int
    baseline_recall_floor: float


class RegressionVerdict(BaseModel):
    """Type-scoped `sql_injection` false-positive regression verdict (the
    `DECISIONS.md#041` caveat track), computed by the runner from a comparison's
    baseline vs candidate grades. Three states via `ok` + `label`: a baseline
    that over-flags sql_injection is non-discriminating (INCONCLUSIVE); a clean
    baseline with a candidate over-flag is REPRODUCED; both clean is CLEAN."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    label: str
    detail: str
    baseline_sqli_fp: int
    candidate_sqli_fp: int


class DiagnosticFinding(BaseModel):
    """One finding behind a failed/noisy row — a candidate FALSE POSITIVE (the
    model flagged it; ground truth did not) or a candidate MISS (ground truth had
    it; the candidate did not catch it). `title` is present for false positives (a
    real `ReviewFinding`) and None for misses (an `ExpectedFinding` carries no
    title). Lets the artifact name WHAT the candidate got wrong, not just how
    many."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_path: str
    line_start: int
    line_end: int
    finding_type: str
    severity: str
    title: str | None = None


class RowDiagnostics(BaseModel):
    """Failed-row diagnostics: the candidate's actual false positives + misses,
    plus the baseline's counts for the delta. Populated by `from_comparison` ONLY
    when the candidate has at least one FP or miss (a clean row carries None), so
    the artifact turns a FAIL from "why?" (open the JSON / logs) into a glance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_false_positives: tuple[DiagnosticFinding, ...] = ()
    candidate_missed: tuple[DiagnosticFinding, ...] = ()
    baseline_n_false_positives: int = 0
    baseline_n_missed: int = 0


class ScorecardRow(BaseModel):
    """One `(node, model, scenario)` cell of the scorecard.

    Quality (recall/precision/severity/FP/gate) comes from the analyze-direct
    comparison; cost/latency are review-level, joined from the full-graph run;
    replay-equivalence from the resume/persisting drivers. Each carries a
    provenance field so the split-path join is visible, not implied.

    An `status="errored"` row (transient-failure isolation: the scenario raised,
    so it gets a row instead of aborting the batch) carries an `error` and null
    quality/cost/latency/replay metrics. The `_check_consistency` validator
    enforces the status/metric invariant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Identity — the row key, plus the model the gate compared the candidate to.
    node: str
    model: str
    scenario: str
    baseline_model: str

    status: RowStatus = "ok"
    error: str | None = None

    # Quality — from the analyze-direct path; all None on an errored row.
    recall: FindingRecall | None = None
    precision: FindingPrecision | None = None
    severity_accuracy: SeverityAccuracy | None = None
    false_positive_rate: FalsePositiveRate | None = None
    n_false_positives: int | None = None
    gate: GateVerdict | None = None
    quality_source: QualitySource = "analyze_direct"

    # Type-scoped sql_injection FP regression verdict (the #041 caveat track),
    # set by the runner from the comparison's baseline-vs-candidate grades.
    # Optional: not every scenario is security-relevant, and errored rows omit it.
    regression: RegressionVerdict | None = None

    # Failed-row diagnostics — the candidate's actual FP/missed findings + the
    # baseline delta. Populated by `from_comparison` only when there's something
    # to diagnose (>= 1 FP or miss); None on clean and errored rows.
    diagnostics: RowDiagnostics | None = None

    # Cost/latency — review-level, from the full-graph run. Optional even on an
    # ok row: the cost pass is a separate join that a caller may skip (or that
    # may fail). `cost_source` disambiguates present / not-requested / failed.
    cost: CostPerReview | None = None
    latency: LatencyPerReview | None = None
    cost_source: CostSource = "not_measured"

    # Replay-equivalence — from the resume/persisting drivers where applicable.
    replay_equivalent: bool | None = None
    replay_source: ReplaySource = "not_applicable"

    @model_validator(mode="after")
    def _check_consistency(self) -> ScorecardRow:
        quality = (
            self.recall,
            self.precision,
            self.severity_accuracy,
            self.false_positive_rate,
            self.n_false_positives,
            self.gate,
        )
        if self.status == "ok":
            if any(q is None for q in quality):
                raise ValueError("an 'ok' row must populate every quality metric")
            if self.error is not None:
                raise ValueError("an 'ok' row must not carry an error")
        else:
            if (
                any(q is not None for q in quality)
                or self.regression is not None
                or self.diagnostics is not None
                or self.cost is not None
                or self.latency is not None
                or self.replay_equivalent is not None
                or self.replay_source != "not_applicable"
            ):
                raise ValueError("an 'errored' row must have null quality/cost/replay metrics")
            if self.error is None:
                raise ValueError("an 'errored' row must carry an error message")
        # cost present iff measured (full_graph); a None cost is either
        # not_measured (not requested) or measure_failed (requested, no number).
        if (self.cost is not None) != (self.cost_source == "full_graph"):
            raise ValueError("cost is present iff cost_source == 'full_graph'")
        if self.latency is not None and self.cost is None:
            raise ValueError("latency requires a cost (they are a review-level pair)")
        # replay verdict + its provenance are a pair: a True/False equivalence
        # verdict requires a real source (resume/persisting), and no verdict
        # (None) requires "not_applicable" — mirrors the cost/cost_source invariant.
        if (self.replay_equivalent is not None) != (self.replay_source != "not_applicable"):
            raise ValueError("replay_equivalent is set iff replay_source != 'not_applicable'")
        return self

    @classmethod
    def from_comparison(
        cls,
        *,
        node: str,
        scenario: str,
        model: str,
        baseline_model: str,
        comparison: ModelComparison,
        regression: RegressionVerdict | None = None,
        cost: CostPerReview | None = None,
        cost_source: CostSource | None = None,
        latency: LatencyPerReview | None = None,
        replay_equivalent: bool | None = None,
        replay_source: ReplaySource = "not_applicable",
    ) -> ScorecardRow:
        """Build an 'ok' row from a graded comparison. Pulls the candidate's
        quality off `comparison` and derives the false-positive rate over the
        SAME finding population the grader's precision uses (`precision.denominator`
        = all candidate findings) so the two metrics never diverge. `regression`,
        cost, latency are joined in by the caller (the runner). `cost_source`,
        when omitted, is derived from cost presence (`full_graph`/`not_measured`);
        the caller passes it explicitly to flag `measure_failed`."""
        cand = comparison.candidate
        denom = cand.precision.denominator  # n_actual = all candidate findings
        false_positive_rate = FalsePositiveRate(
            value=(cand.n_false_positives / denom) if denom else 0.0,
            numerator=cand.n_false_positives,
            denominator=denom,
        )
        gate = GateVerdict(
            passes=comparison.passes,
            baseline_valid=comparison.baseline_valid,
            recall_held=comparison.recall_held,
            fp_bounded=comparison.fp_bounded,
            recall_tolerance=comparison.recall_tolerance,
            fp_allowance=comparison.fp_allowance,
            baseline_recall_floor=comparison.baseline_recall_floor,
        )
        effective_cost_source: CostSource = (
            cost_source
            if cost_source is not None
            else ("full_graph" if cost is not None else "not_measured")
        )
        # Failed-row diagnostics: name the candidate's actual FP / missed findings
        # (+ baseline counts for the delta) so a FAIL is diagnosable from the
        # artifact. None when there's nothing to show (clean row).
        fps = tuple(
            DiagnosticFinding(
                file_path=f.file_path,
                line_start=f.line_start,
                line_end=f.line_end,
                finding_type=f.finding_type.value,
                severity=f.severity.value,
                title=f.title,
            )
            for f in cand.extra
        )
        missed = tuple(
            DiagnosticFinding(
                file_path=e.file_path,
                line_start=e.line_start,
                line_end=e.line_end,
                finding_type=e.finding_type.value,
                severity=e.severity.value,
            )
            for e in cand.missed
        )
        diagnostics = (
            RowDiagnostics(
                candidate_false_positives=fps,
                candidate_missed=missed,
                baseline_n_false_positives=comparison.baseline.n_false_positives,
                baseline_n_missed=len(comparison.baseline.missed),
            )
            # Populate on ANY gate failure, not just FP/miss: a FAIL caused solely
            # by a non-discriminating baseline (baseline_valid False) carries no
            # candidate FP/miss, but the row must still reach the Diagnostics
            # section so the HTML's `why` column explains the FAIL instead of
            # leaving a bare, unexplained FAIL that reads as "candidate bad".
            if (fps or missed or not comparison.passes)
            else None
        )
        return cls(
            node=node,
            model=model,
            scenario=scenario,
            baseline_model=baseline_model,
            status="ok",
            recall=cand.recall,
            precision=cand.precision,
            severity_accuracy=cand.severity_accuracy,
            false_positive_rate=false_positive_rate,
            n_false_positives=cand.n_false_positives,
            gate=gate,
            regression=regression,
            diagnostics=diagnostics,
            cost=cost,
            latency=latency,
            cost_source=effective_cost_source,
            replay_equivalent=replay_equivalent,
            replay_source=replay_source,
        )

    @classmethod
    def errored(
        cls,
        *,
        node: str,
        scenario: str,
        model: str,
        baseline_model: str,
        error: str,
    ) -> ScorecardRow:
        """Build an 'errored' row: the scenario raised during measurement, so it
        is recorded with the error and null metrics rather than dropping the
        batch (mirrors `_run_scenario_isolating_transients`)."""
        return cls(
            node=node,
            model=model,
            scenario=scenario,
            baseline_model=baseline_model,
            status="errored",
            error=error,
        )


class AggregateRow(BaseModel):
    """Per-`(node, model)` roll-up across scenarios — the GATE SUMMARY, typed.

    Means/totals are over the 'ok' rows only; errored rows count toward
    `n_scenarios`/`n_errored` but not the metric reductions (a transient failure
    must not silently depress a mean recall). `n_costed` is the denominator for
    `total_cost_usd` — how many ok rows actually carried a measured cost, so a
    partial-cost batch isn't mistaken for a complete total."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str
    model: str
    n_scenarios: int
    n_ok: int
    n_errored: int
    n_passed: int
    n_failed: int
    n_costed: int = 0
    mean_recall: float | None = None
    mean_precision: float | None = None
    total_cost_usd: float | None = None
    mean_latency_seconds: float | None = None


class TriageGateVerdict(BaseModel):
    """The deterministic triage gate verdict for one candidate-vs-baseline
    comparison, flattened off `triage_grading.TriageComparison` — pass/fail plus
    the declared-threshold sub-conditions, so a FAIL is self-explanatory (a new
    drop, an under-risk, an over-tier balloon, or a dimension-recall regression)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passes: bool
    baseline_valid: bool
    drop_held: bool
    risk_safety_held: bool
    overtier_bounded: bool
    dimension_recall_held: bool
    overtier_allowance: int
    dimension_recall_tolerance: float


class TriageScorecardRow(BaseModel):
    """One `(node, model, scenario)` triage cell — the tier-classification parallel
    to `ScorecardRow`, kept cohesive (no finding-level fields). Carries the
    analysis-floor safety metric (`n_dropped_from_analysis`), the softer
    `n_deep_downgraded`, the cost `n_overtiered`, dimension recall/precision, risk
    correctness + under-risk, and the gate. An `errored` row carries an `error` and
    null metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str = "triage"
    model: str
    scenario: str
    baseline_model: str

    status: RowStatus = "ok"
    error: str | None = None

    tier_accuracy: float | None = None
    n_dropped_from_analysis: int | None = None
    n_deep_downgraded: int | None = None
    n_overtiered: int | None = None
    dimension_recall: float | None = None
    dimension_precision: float | None = None
    risk_correct: bool | None = None
    under_risked: bool | None = None
    gate: TriageGateVerdict | None = None
    # The PATHS behind n_dropped_from_analysis (which files left the review set) — a
    # bare count can't name them in a multi-file scenario. Empty tuple = no drops.
    dropped_files: tuple[str, ...] | None = None
    triage_source: TriageSource = "run_triage_direct"

    @model_validator(mode="after")
    def _check_consistency(self) -> TriageScorecardRow:
        metrics = (
            self.tier_accuracy,
            self.n_dropped_from_analysis,
            self.n_deep_downgraded,
            self.n_overtiered,
            self.dimension_recall,
            self.dimension_precision,
            self.risk_correct,
            self.under_risked,
            self.gate,
            self.dropped_files,
        )
        if self.status == "ok":
            if any(m is None for m in metrics):
                raise ValueError("an 'ok' triage row must populate every metric")
            if self.error is not None:
                raise ValueError("an 'ok' triage row must not carry an error")
            # The count and the paths must agree — a decision artifact cannot emit a
            # row that says "dropped=1" with no path, or "dropped=0" with hidden paths.
            if len(self.dropped_files or ()) != self.n_dropped_from_analysis:
                raise ValueError(
                    "dropped_files must name exactly n_dropped_from_analysis paths "
                    f"(got {len(self.dropped_files or ())}, "
                    f"n_dropped_from_analysis={self.n_dropped_from_analysis})"
                )
        else:
            if any(m is not None for m in metrics):
                raise ValueError("an 'errored' triage row must have null metrics")
            if self.error is None:
                raise ValueError("an 'errored' triage row must carry an error message")
        return self

    @classmethod
    def from_comparison(
        cls,
        *,
        scenario: str,
        model: str,
        baseline_model: str,
        comparison: TriageComparison,
        node: str = "triage",
    ) -> TriageScorecardRow:
        """Build an 'ok' triage row from a graded comparison (the candidate's grade
        + the gate verdict)."""
        cand = comparison.candidate
        gate = TriageGateVerdict(
            passes=comparison.passes,
            baseline_valid=comparison.baseline_valid,
            drop_held=comparison.drop_held,
            risk_safety_held=comparison.risk_safety_held,
            overtier_bounded=comparison.overtier_bounded,
            dimension_recall_held=comparison.dimension_recall_held,
            overtier_allowance=comparison.overtier_allowance,
            dimension_recall_tolerance=comparison.dimension_recall_tolerance,
        )
        return cls(
            node=node,
            model=model,
            scenario=scenario,
            baseline_model=baseline_model,
            status="ok",
            tier_accuracy=cand.tier_accuracy,
            n_dropped_from_analysis=cand.n_dropped_from_analysis,
            n_deep_downgraded=cand.n_deep_downgraded,
            n_overtiered=cand.n_overtiered,
            dimension_recall=cand.dimension_recall,
            dimension_precision=cand.dimension_precision,
            risk_correct=cand.risk_correct,
            under_risked=cand.under_risked,
            gate=gate,
            dropped_files=cand.dropped_files,
        )

    @classmethod
    def errored(
        cls,
        *,
        scenario: str,
        model: str,
        baseline_model: str,
        error: str,
        node: str = "triage",
    ) -> TriageScorecardRow:
        """Build an 'errored' triage row (transient-failure isolation)."""
        return cls(
            node=node,
            model=model,
            scenario=scenario,
            baseline_model=baseline_model,
            status="errored",
            error=error,
        )


class TriageAggregateRow(BaseModel):
    """Per-`(node, model)` triage roll-up. `total_dropped_from_analysis` is the
    safety headline (files that left the review set across the matrix); means are
    over the ok rows only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str
    model: str
    n_scenarios: int
    n_ok: int
    n_errored: int
    n_passed: int
    n_failed: int
    total_dropped_from_analysis: int | None = None
    mean_tier_accuracy: float | None = None
    mean_dimension_recall: float | None = None


class Scorecard(BaseModel):
    """A collection of `ScorecardRow`s with per-`(node, model)` aggregates and
    JSON + HTML emitters — the persisted decision artifact. Pure: the
    emitters return strings; writing them to a path is the caller's job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: tuple[ScorecardRow, ...] = ()
    triage_rows: tuple[TriageScorecardRow, ...] = ()

    def aggregates(self) -> tuple[AggregateRow, ...]:
        """Roll rows up per `(node, model)`, sorted by key for a deterministic
        artifact (no wall-clock / random ordering). Single pass over each group's
        ok rows."""
        groups: dict[tuple[str, str], list[ScorecardRow]] = {}
        for row in self.rows:
            groups.setdefault((row.node, row.model), []).append(row)

        out: list[AggregateRow] = []
        for (node, model), group in sorted(groups.items(), key=lambda item: item[0]):
            n_ok = n_passed = 0
            recalls: list[float] = []
            precisions: list[float] = []
            costs: list[float] = []
            latencies: list[float] = []
            for r in group:
                if r.status != "ok":
                    continue
                n_ok += 1
                if r.gate is not None and r.gate.passes:
                    n_passed += 1
                if r.recall is not None:
                    recalls.append(r.recall.value)
                if r.precision is not None:
                    precisions.append(r.precision.value)
                if r.cost is not None:
                    costs.append(r.cost.usd)
                if r.latency is not None:
                    latencies.append(r.latency.seconds)
            out.append(
                AggregateRow(
                    node=node,
                    model=model,
                    n_scenarios=len(group),
                    n_ok=n_ok,
                    n_errored=len(group) - n_ok,
                    n_passed=n_passed,
                    n_failed=n_ok - n_passed,
                    n_costed=len(costs),
                    mean_recall=fmean(recalls) if recalls else None,
                    mean_precision=fmean(precisions) if precisions else None,
                    total_cost_usd=sum(costs) if costs else None,
                    mean_latency_seconds=fmean(latencies) if latencies else None,
                )
            )
        return tuple(out)

    def triage_aggregates(self) -> tuple[TriageAggregateRow, ...]:
        """Roll triage rows up per `(node, model)`, sorted by key. Means over the ok
        rows; `total_dropped_from_analysis` sums the safety metric across them."""
        groups: dict[tuple[str, str], list[TriageScorecardRow]] = {}
        for row in self.triage_rows:
            groups.setdefault((row.node, row.model), []).append(row)

        out: list[TriageAggregateRow] = []
        for (node, model), group in sorted(groups.items(), key=lambda item: item[0]):
            n_ok = n_passed = 0
            total_dropped = 0
            tier_accs: list[float] = []
            dim_recalls: list[float] = []
            for r in group:
                if r.status != "ok":
                    continue
                n_ok += 1
                if r.gate is not None and r.gate.passes:
                    n_passed += 1
                if r.n_dropped_from_analysis is not None:
                    total_dropped += r.n_dropped_from_analysis
                if r.tier_accuracy is not None:
                    tier_accs.append(r.tier_accuracy)
                if r.dimension_recall is not None:
                    dim_recalls.append(r.dimension_recall)
            out.append(
                TriageAggregateRow(
                    node=node,
                    model=model,
                    n_scenarios=len(group),
                    n_ok=n_ok,
                    n_errored=len(group) - n_ok,
                    n_passed=n_passed,
                    n_failed=n_ok - n_passed,
                    total_dropped_from_analysis=total_dropped if n_ok else None,
                    mean_tier_accuracy=fmean(tier_accs) if tier_accs else None,
                    mean_dimension_recall=fmean(dim_recalls) if dim_recalls else None,
                )
            )
        return tuple(out)

    def to_json(self) -> str:
        """JSON artifact: `{rows, aggregates, triage_rows, triage_aggregates}`,
        key-sorted for stable diffs."""
        payload = {
            "rows": [r.model_dump(mode="json") for r in self.rows],
            "aggregates": [a.model_dump(mode="json") for a in self.aggregates()],
            "triage_rows": [r.model_dump(mode="json") for r in self.triage_rows],
            "triage_aggregates": [a.model_dump(mode="json") for a in self.triage_aggregates()],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def to_html(self) -> str:
        """Self-contained HTML artifact (inline CSS, no external deps) — the
        human-glance counterpart to `to_json`. A per-row table + a per-`(node,
        model)` aggregate, for analyze and/or triage. Empty sections are
        SUPPRESSED (a triage-only or analyze-only card renders only its own
        tables); every cell value is HTML-escaped and PASS / FAIL / ERROR are
        badged so the gate verdict reads at a glance."""
        sections: list[str] = []

        if self.rows:
            row_headers = [
                "node",
                "model",
                "scenario",
                "recall",
                "precision",
                "severity",
                "FP",
                "gate",
                "regr",
                "$/review",
                "latency(s)",
                "replay",
                "replay_src",
                "quality_src",
                "cost_src",
            ]
            row_data: list[list[str]] = []
            for r in self.rows:
                if r.status == "errored":
                    gate_cell = f"ERROR: {r.error}"
                else:
                    gate_cell = "PASS" if (r.gate is not None and r.gate.passes) else "FAIL"
                row_data.append(
                    [
                        r.node,
                        r.model,
                        r.scenario,
                        _fmt_ratio(r.recall),
                        _fmt_ratio(r.precision),
                        _fmt_ratio(r.severity_accuracy),
                        str(r.n_false_positives) if r.n_false_positives is not None else "—",
                        gate_cell,
                        _fmt_regression(r.regression),
                        f"{r.cost.usd:.4f}" if r.cost is not None else "—",
                        f"{r.latency.seconds:.2f}" if r.latency is not None else "—",
                        _fmt_bool(r.replay_equivalent),
                        r.replay_source,
                        r.quality_source,
                        r.cost_source,
                    ]
                )

            agg_headers = [
                "node",
                "model",
                "scenarios",
                "ok",
                "errored",
                "passed",
                "failed",
                "mean recall",
                "mean precision",
                "total $",
                "costed",
                "mean latency(s)",
            ]
            agg_data: list[list[str]] = []
            for a in self.aggregates():
                agg_data.append(
                    [
                        a.node,
                        a.model,
                        str(a.n_scenarios),
                        str(a.n_ok),
                        str(a.n_errored),
                        str(a.n_passed),
                        str(a.n_failed),
                        f"{a.mean_recall:.3f}" if a.mean_recall is not None else "—",
                        f"{a.mean_precision:.3f}" if a.mean_precision is not None else "—",
                        f"{a.total_cost_usd:.4f}" if a.total_cost_usd is not None else "—",
                        str(a.n_costed),
                        f"{a.mean_latency_seconds:.2f}"
                        if a.mean_latency_seconds is not None
                        else "—",
                    ]
                )

            sections.append("<h2>Analyze</h2>")
            sections.extend(_html_table(row_headers, row_data))
            sections.append('<h3>Aggregate <span class="muted">(per node × model)</span></h3>')
            sections.extend(_html_table(agg_headers, agg_data))

            # Diagnostics: the candidate findings behind a FAIL (FP titles/paths)
            # plus the per-scenario delta vs baseline. Built only from rows that
            # carry diagnostics (>= 1 FP or miss); a clean batch renders nothing.
            delta_data: list[list[str]] = []
            detail_data: list[list[str]] = []
            for r in self.rows:
                d = r.diagnostics
                if d is None:
                    continue
                gate_cell = "PASS" if (r.gate is not None and r.gate.passes) else "FAIL"
                delta_data.append(
                    [
                        r.scenario,
                        gate_cell,
                        _gate_reason(r.gate),
                        str(len(d.candidate_false_positives)),
                        str(d.baseline_n_false_positives),
                        str(len(d.candidate_missed)),
                        str(d.baseline_n_missed),
                    ]
                )
                detail_data.extend(
                    _diag_detail_row(r.scenario, "false positive", f)
                    for f in d.candidate_false_positives
                )
                detail_data.extend(
                    _diag_detail_row(r.scenario, "missed", f) for f in d.candidate_missed
                )
            if delta_data:
                sections.append("<h2>Diagnostics</h2>")
                sections.append(_DIAG_INTRO)
                sections.append(
                    '<h3>Per-scenario delta <span class="muted">(candidate vs baseline)</span></h3>'
                )
                sections.extend(
                    _html_table(
                        [
                            "scenario",
                            "gate",
                            "why",
                            "cand FP",
                            "base FP",
                            "cand missed",
                            "base missed",
                        ],
                        delta_data,
                    )
                )
                if detail_data:
                    sections.append("<h3>Findings</h3>")
                    sections.extend(
                        _html_table(
                            ["scenario", "kind", "title", "location", "type", "severity"],
                            detail_data,
                        )
                    )

        if self.triage_rows:
            triage_row_headers = [
                "node",
                "model",
                "scenario",
                "tier_acc",
                "dropped",
                "downgrade",
                "overtier",
                "dim_recall",
                "dim_prec",
                "risk_ok",
                "under_risk",
                "gate",
            ]
            triage_row_data: list[list[str]] = []
            for trow in self.triage_rows:
                if trow.status == "errored":
                    gate_cell = f"ERROR: {trow.error}"
                else:
                    gate_cell = "PASS" if (trow.gate is not None and trow.gate.passes) else "FAIL"
                triage_row_data.append(
                    [
                        trow.node,
                        trow.model,
                        trow.scenario,
                        f"{trow.tier_accuracy:.3f}" if trow.tier_accuracy is not None else "—",
                        _fmt_dropped(trow.n_dropped_from_analysis, trow.dropped_files),
                        str(trow.n_deep_downgraded) if trow.n_deep_downgraded is not None else "—",
                        str(trow.n_overtiered) if trow.n_overtiered is not None else "—",
                        f"{trow.dimension_recall:.3f}"
                        if trow.dimension_recall is not None
                        else "—",
                        f"{trow.dimension_precision:.3f}"
                        if trow.dimension_precision is not None
                        else "—",
                        _fmt_bool(trow.risk_correct),
                        _fmt_under_risk(trow.under_risked),
                        gate_cell,
                    ]
                )
            triage_agg_headers = [
                "node",
                "model",
                "scenarios",
                "ok",
                "errored",
                "passed",
                "failed",
                "total dropped",
                "mean tier_acc",
                "mean dim_recall",
            ]
            triage_agg_data: list[list[str]] = []
            for tagg in self.triage_aggregates():
                triage_agg_data.append(
                    [
                        tagg.node,
                        tagg.model,
                        str(tagg.n_scenarios),
                        str(tagg.n_ok),
                        str(tagg.n_errored),
                        str(tagg.n_passed),
                        str(tagg.n_failed),
                        str(tagg.total_dropped_from_analysis)
                        if tagg.total_dropped_from_analysis is not None
                        else "—",
                        f"{tagg.mean_tier_accuracy:.3f}"
                        if tagg.mean_tier_accuracy is not None
                        else "—",
                        f"{tagg.mean_dimension_recall:.3f}"
                        if tagg.mean_dimension_recall is not None
                        else "—",
                    ]
                )
            sections.append("<h2>Triage</h2>")
            sections.extend(_html_table(triage_row_headers, triage_row_data))
            sections.append("<h3>Triage aggregate</h3>")
            sections.extend(_html_table(triage_agg_headers, triage_agg_data))

        footer_notes: list[str] = []
        if self.rows:
            footer_notes.append(_ANALYZE_NOTE)
        if self.triage_rows:
            footer_notes.append(_TRIAGE_NOTE)
        footer_notes.append(_CORPUS_NOTE)
        footer = "<footer>\n" + "\n".join(footer_notes) + "\n</footer>"
        return _HTML_HEAD + "\n".join(sections) + "\n" + footer + _HTML_TAIL


# Self-contained HTML document shell (inline CSS, no external deps) wrapping the
# scorecard tables. Head/foot are split so `to_html` only assembles the body; the
# footer carries the two reading caveats an operator needs to not over-read the
# numbers (cost provenance + strict-gate semantics).
_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Outrider eval scorecard</title>
<style>
  body {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    max-width: 1120px; margin: 2rem auto; padding: 0 1.25rem; line-height: 1.45; color: #1b1f24;
  }
  h1 { font-size: 1.5rem; margin: 0 0 .15rem; }
  h2 { font-size: 1.15rem; margin: 1.8rem 0 .5rem; border-bottom: 2px solid #e3e6ea; }
  h3 { font-size: .95rem; margin: 1.2rem 0 .35rem; font-weight: 600; }
  .muted { color: #6a737d; font-weight: 400; }
  table { border-collapse: collapse; width: 100%; font-size: .82rem; margin-bottom: .4rem; }
  th, td { padding: .38rem .55rem; text-align: left; border-bottom: 1px solid #eceff1; }
  th { background: #24292f; color: #fff; font-weight: 600; white-space: nowrap; }
  td { white-space: nowrap; }
  tbody tr:nth-child(even) { background: #f6f8fa; }
  tbody tr:hover { background: #fff8e1; }
  .badge { display: inline-block; padding: .04rem .45rem; border-radius: .7rem; font-weight: 700; }
  .badge.pass { background: #1a7f37; color: #fff; }
  .badge.fail { background: #cf222e; color: #fff; }
  .badge.err { background: #9a6700; color: #fff; }
  footer { margin-top: 2rem; font-size: .76rem; color: #57606a; border-top: 1px solid #e3e6ea; }
  footer p { margin: .5rem 0; }
  code { background: #f3f4f6; padding: .05rem .25rem; border-radius: .25rem; }
</style>
</head>
<body>
<h1>Outrider eval scorecard</h1>
<p class="muted">
  Baseline vs candidate quality gate — report-only.
  <span class="badge pass">PASS</span>/<span class="badge fail">FAIL</span> is the gate verdict.
</p>
"""

# Footer reading-notes, assembled per the sections ACTUALLY present so a
# triage-only card never claims analyze-only metrics ($/review, the FP gate) and
# an analyze-only card never claims triage metrics. A combined card gets both.
_ANALYZE_NOTE = """<p><strong>Analyze reading notes.</strong> Quality (recall / precision /
severity / FP) is real-model spend through the analyze-direct path. <code>$/review</code>
prices the full-graph run with <em>real input tokens</em> and <em>scripted output tokens</em>
&mdash; an input-accurate estimate, not a measured spend. The gate is strict (zero FP
allowance, zero recall slack), so a <span class="badge fail">FAIL</span> is typically an
added false positive, not a missed finding.</p>"""

_TRIAGE_NOTE = """<p><strong>Triage reading notes.</strong> Metrics are tier accuracy, files
dropped below the analysis floor, DEEP downgrades, over-tiering, and dimension
recall/precision &mdash; there is no cost column (triage rows are quality-only). The gate is
safety-oriented: it fails on a file dropped from analysis, under-risking, or a dimension-recall
regression, not on over-broad tiering.</p>"""

_CORPUS_NOTE = "<p>Small fixture corpus &mdash; read trends, not absolutes.</p>"

_HTML_TAIL = """
</body>
</html>
"""


def _html_cell(value: str) -> str:
    """Escape a value for an HTML table cell, then badge the known gate states
    (PASS / FAIL / ERROR:) with a status span so the verdict reads at a glance.
    The badge match is on the ALREADY-ESCAPED text, so no cell content can
    inject markup."""
    esc = html.escape(value)
    if esc == "PASS":
        return '<span class="badge pass">PASS</span>'
    if esc == "FAIL":
        return '<span class="badge fail">FAIL</span>'
    if esc.startswith("ERROR:"):
        return f'<span class="badge err">{esc}</span>'
    return esc


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render an HTML table. Every header + cell is escaped; cells route through
    `_html_cell` for gate-state badging."""
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{html.escape(h)}</th>" for h in headers)
    out.append("</tr></thead>")
    out.append("<tbody>")
    out.extend("<tr>" + "".join(f"<td>{_html_cell(c)}</td>" for c in row) + "</tr>" for row in rows)
    out.append("</tbody></table>")
    return out


_DIAG_INTRO = (
    '<p class="muted">Every FAIL plus any candidate false positive / miss '
    "&mdash; the failing gate condition (<code>why</code>) and the findings, "
    "not just the count.</p>"
)


def _loc(file_path: str, line_start: int, line_end: int) -> str:
    """`file:line` (or `file:start-end`) location string for a diagnostic finding."""
    if line_start == line_end:
        return f"{file_path}:{line_start}"
    return f"{file_path}:{line_start}-{line_end}"


def _gate_reason(gate: GateVerdict | None) -> str:
    """The failing gate sub-condition(s) for a FAIL row, so a bare FAIL is
    self-explanatory — especially a baseline-invalid (non-discriminating) FAIL,
    which carries no candidate FP/miss to show in the findings table."""
    if gate is None or gate.passes:
        return "—"
    reasons = []
    if not gate.baseline_valid:
        reasons.append("baseline invalid")
    if not gate.recall_held:
        reasons.append("recall")
    if not gate.fp_bounded:
        reasons.append("FP")
    return ", ".join(reasons) or "—"


def _diag_detail_row(scenario: str, kind: str, finding: DiagnosticFinding) -> list[str]:
    """One row of the diagnostics detail table for a candidate FP or miss."""
    return [
        scenario,
        kind,
        finding.title or "—",
        _loc(finding.file_path, finding.line_start, finding.line_end),
        finding.finding_type,
        finding.severity,
    ]


def _fmt_ratio(metric: FindingRecall | FindingPrecision | SeverityAccuracy | None) -> str:
    return f"{metric.value:.3f}" if metric is not None else "—"


def _fmt_bool(value: bool | None) -> str:
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "—"


def _fmt_regression(verdict: RegressionVerdict | None) -> str:
    if verdict is None:
        return "—"
    return "clean" if verdict.ok else f"⚠ {verdict.label}"


def _fmt_under_risk(value: bool | None) -> str:
    if value is True:
        return "⚠ yes"
    if value is False:
        return "no"
    return "—"


def _fmt_dropped(count: int | None, paths: tuple[str, ...] | None) -> str:
    """Render the drop cell: "—" (errored / no metric), "0" (no drops), or
    "N (path, ...)" so a multi-file drop names the files that left review."""
    if count is None:
        return "—"
    if count and paths:
        return f"{count} ({', '.join(paths)})"
    return str(count)


__all__ = [
    "AggregateRow",
    "DiagnosticFinding",
    "GateVerdict",
    "RegressionVerdict",
    "RowDiagnostics",
    "Scorecard",
    "ScorecardRow",
    "TriageAggregateRow",
    "TriageGateVerdict",
    "TriageScorecardRow",
]
