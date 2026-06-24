"""Cross-scenario eval scorecard: typed rows + JSON/Markdown emit.

Productizes the inline `(fixture, dimension, ok, fail_label)` aggregation that
lived only in the opt-in spend test (`test_model_comparison.py`'s GATE SUMMARY
print) into reusable typed objects. One `ScorecardRow` per `(node, model,
scenario)`; a `Scorecard` collects rows and emits a JSON + Markdown artifact.

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
    JSON + Markdown emitters — the persisted decision artifact. Pure: the
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

    def to_markdown(self) -> str:
        """Human-glance artifact: a per-row table + a per-`(node, model)`
        aggregate table. Null cells render as an em dash; cell values are
        pipe/newline-escaped so a stray label can't break column alignment."""
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
                    f"{a.mean_latency_seconds:.2f}" if a.mean_latency_seconds is not None else "—",
                ]
            )

        lines: list[str] = ["# Eval scorecard", ""]
        lines.extend(_md_table(row_headers, row_data))
        lines.extend(["", "## Aggregate (per node × model)", ""])
        lines.extend(_md_table(agg_headers, agg_data))

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
            lines.extend(["", "## Triage (per node × model)", ""])
            lines.extend(_md_table(triage_row_headers, triage_row_data))
            lines.extend(["", "### Triage aggregate", ""])
            lines.extend(_md_table(triage_agg_headers, triage_agg_data))

        lines.append("")
        return "\n".join(lines)


def _md_cell(value: str) -> str:
    """Escape a value for a Markdown table cell — a literal `|` would add a
    phantom column and a newline would split the row, silently misaligning the
    one artifact the operator reads to make a decision."""
    return value.replace("|", "\\|").replace("\n", " ")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a GitHub-flavored Markdown table. The dashed separator is DERIVED
    from the header count (not hand-maintained), so a column add/remove can't
    drift the separator out of alignment."""
    out = ["| " + " | ".join(_md_cell(h) for h in headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows)
    return out


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
    "GateVerdict",
    "RegressionVerdict",
    "Scorecard",
    "ScorecardRow",
    "TriageAggregateRow",
    "TriageGateVerdict",
    "TriageScorecardRow",
]
