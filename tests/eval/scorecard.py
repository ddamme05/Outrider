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
  - Report-only: a row records its gate verdict (`gate.passes`); nothing here
    raises on a failed gate. Enforcement, if ever wanted, is a separate surface.

Wires the three previously-unconsumed `metrics.py` shapes (`FalsePositiveRate`,
`CostPerReview`, `LatencyPerReview`) into their first consumer. Quality numbers
are pulled OFF a `grading.ModelComparison` (the deterministic gate); this module
computes no recall/precision/severity itself — only the false-positive RATE,
which the grader exposes as a raw count.
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
    from .grading import ModelComparison

QualitySource = Literal["analyze_direct"]
CostSource = Literal["full_graph", "not_measured"]
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
    quality metrics. The `_check_consistency` validator enforces the
    status/metric invariant."""

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
    # ok row: the cost pass is a separate join that a caller may skip.
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
            ):
                raise ValueError("an 'errored' row must have null quality/cost metrics")
            if self.error is None:
                raise ValueError("an 'errored' row must carry an error message")
        if (self.cost is not None) != (self.cost_source == "full_graph"):
            raise ValueError("cost is present iff cost_source == 'full_graph'")
        if self.latency is not None and self.cost is None:
            raise ValueError("latency requires a cost (they are a review-level pair)")
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
        latency: LatencyPerReview | None = None,
        replay_equivalent: bool | None = None,
        replay_source: ReplaySource = "not_applicable",
    ) -> ScorecardRow:
        """Build an 'ok' row from a graded comparison. Pulls the candidate's
        quality off `comparison` and derives the false-positive rate (extra
        findings over all candidate findings — the grader exposes only the raw
        count). `regression`, cost, latency are joined in by the caller (the
        runner); `cost_source` is set from whether a cost was supplied."""
        cand = comparison.candidate
        n_findings = cand.n_matched + cand.n_false_positives
        false_positive_rate = FalsePositiveRate(
            value=(cand.n_false_positives / n_findings) if n_findings else 0.0,
            numerator=cand.n_false_positives,
            denominator=n_findings,
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
            cost_source="full_graph" if cost is not None else "not_measured",
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
    must not silently depress a mean recall)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str
    model: str
    n_scenarios: int
    n_ok: int
    n_errored: int
    n_passed: int
    n_failed: int
    mean_recall: float | None = None
    mean_precision: float | None = None
    total_cost_usd: float | None = None
    mean_latency_seconds: float | None = None


class Scorecard(BaseModel):
    """A collection of `ScorecardRow`s with per-`(node, model)` aggregates and
    JSON + Markdown emitters — the persisted decision artifact. Pure: the
    emitters return strings; writing them to a path is the caller's job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: tuple[ScorecardRow, ...] = ()

    def aggregates(self) -> tuple[AggregateRow, ...]:
        """Roll rows up per `(node, model)`, sorted by key for a deterministic
        artifact (no wall-clock / random ordering)."""
        groups: dict[tuple[str, str], list[ScorecardRow]] = {}
        for row in self.rows:
            groups.setdefault((row.node, row.model), []).append(row)

        out: list[AggregateRow] = []
        for (node, model), group in sorted(groups.items(), key=lambda item: item[0]):
            ok = [r for r in group if r.status == "ok"]
            recalls = [r.recall.value for r in ok if r.recall is not None]
            precisions = [r.precision.value for r in ok if r.precision is not None]
            costs = [r.cost.usd for r in ok if r.cost is not None]
            latencies = [r.latency.seconds for r in ok if r.latency is not None]
            n_passed = sum(1 for r in ok if r.gate is not None and r.gate.passes)
            out.append(
                AggregateRow(
                    node=node,
                    model=model,
                    n_scenarios=len(group),
                    n_ok=len(ok),
                    n_errored=len(group) - len(ok),
                    n_passed=n_passed,
                    n_failed=len(ok) - n_passed,
                    mean_recall=fmean(recalls) if recalls else None,
                    mean_precision=fmean(precisions) if precisions else None,
                    total_cost_usd=sum(costs) if costs else None,
                    mean_latency_seconds=fmean(latencies) if latencies else None,
                )
            )
        return tuple(out)

    def to_json(self) -> str:
        """JSON artifact: `{rows, aggregates}`, key-sorted for stable diffs."""
        payload = {
            "rows": [r.model_dump(mode="json") for r in self.rows],
            "aggregates": [a.model_dump(mode="json") for a in self.aggregates()],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        """Human-glance artifact: a per-row table + a per-`(node, model)`
        aggregate table. Null cells render as an em dash."""
        lines: list[str] = ["# Eval scorecard", ""]
        lines.append(
            "| node | model | scenario | recall | precision | severity | FP | gate | regr "
            "| $/review | latency(s) | replay | quality_src | cost_src |"
        )
        lines.append(
            "|------|-------|----------|--------|-----------|----------|----|------|------"
            "|----------|------------|--------|-------------|----------|"
        )
        for r in self.rows:
            if r.status == "errored":
                gate_cell = f"ERROR: {r.error}"
            else:
                gate_cell = "PASS" if (r.gate is not None and r.gate.passes) else "FAIL"
            lines.append(
                "| "
                + " | ".join(
                    (
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
                        _fmt_replay(r.replay_equivalent),
                        r.quality_source,
                        r.cost_source,
                    )
                )
                + " |"
            )
        lines.extend(["", "## Aggregate (per node × model)", ""])
        lines.append(
            "| node | model | scenarios | ok | errored | passed | failed "
            "| mean recall | mean precision | total $ | mean latency(s) |"
        )
        lines.append(
            "|------|-------|-----------|----|---------|--------|--------"
            "|-------------|----------------|---------|-----------------|"
        )
        for a in self.aggregates():
            lines.append(
                "| "
                + " | ".join(
                    (
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
                        f"{a.mean_latency_seconds:.2f}"
                        if a.mean_latency_seconds is not None
                        else "—",
                    )
                )
                + " |"
            )
        lines.append("")
        return "\n".join(lines)


def _fmt_ratio(metric: FindingRecall | FindingPrecision | SeverityAccuracy | None) -> str:
    return f"{metric.value:.3f}" if metric is not None else "—"


def _fmt_replay(value: bool | None) -> str:
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "—"


def _fmt_regression(verdict: RegressionVerdict | None) -> str:
    if verdict is None:
        return "—"
    return "clean" if verdict.ok else f"⚠ {verdict.label}"


__all__ = [
    "AggregateRow",
    "GateVerdict",
    "RegressionVerdict",
    "Scorecard",
    "ScorecardRow",
]
