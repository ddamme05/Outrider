"""Tests for the cross-scenario eval scorecard (tests/eval/scorecard.py).

Covers the typed objects + serialization that step 1 promotes out of the inline
GATE SUMMARY print: `ScorecardRow.from_comparison` (the analyze-direct quality
join + false-positive-rate derivation), the errored-row path (transient
isolation), the status/metric consistency validator, the per-`(node, model)`
aggregate reduction, and the JSON + HTML emitters. Pure — no DB, no LLM, no
spend.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType, lookup_severity
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.review_finding import ReviewFinding

from .grading import ExpectedFinding, compare, grade
from .metrics import (
    CostPerReview,
    FalsePositiveRate,
    FindingPrecision,
    FindingRecall,
    LatencyPerReview,
    SeverityAccuracy,
)
from .scorecard import GateVerdict, RowDiagnostics, Scorecard, ScorecardProvenance, ScorecardRow
from .test_model_comparison import (
    _GROUND_TRUTH_BY_FIXTURE,
    _MISSING_ERROR_HANDLING_FIXTURE,
    _MISSING_INPUT_VALIDATION_FIXTURE,
    _N_PLUS_ONE_FIXTURE,
    _PATH_TRAVERSAL_FIXTURE,
    _PYGOAT_AUTH_FIXTURE,
    _PYGOAT_SQL_FIXTURE,
    _SQLI_HOLDOUT_CONCAT_FIXTURE,
)

_BASELINE_MODEL = "claude-sonnet-4-6"
_CANDIDATE_MODEL = "claude-haiku-4-5"

# severity -> finding_type so the `severity-set-by-policy` validator on
# ReviewFinding is satisfied (severity must equal SEVERITY_POLICY[finding_type]).
_TYPE_FOR_SEVERITY = {
    FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
    FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
}


def _finding(*, file_path: str = "app/db.py", line_start: int = 10) -> ReviewFinding:
    ft = _TYPE_FOR_SEVERITY[FindingSeverity.CRITICAL]
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=ft,
        severity=FindingSeverity.CRITICAL,
        file_path=file_path,
        line_start=line_start,
        line_end=line_start,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(ft),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path, line_start=line_start, line_end=line_start, finding_type=ft
        ),
        proposal_hash="a" * 64,
    )


def _expected(*, file_path: str = "app/db.py", line_start: int = 10) -> ExpectedFinding:
    return ExpectedFinding(
        file_path=file_path,
        line_start=line_start,
        line_end=line_start,
        finding_type=_TYPE_FOR_SEVERITY[FindingSeverity.CRITICAL],
        severity=FindingSeverity.CRITICAL,
    )


# --- direct-construction helpers (no findings needed) -----------------------


def _gate(*, passes: bool) -> GateVerdict:
    return GateVerdict(
        passes=passes,
        baseline_valid=True,
        recall_held=passes,
        fp_bounded=True,
        recall_tolerance=0.0,
        fp_allowance=0,
        baseline_recall_floor=1.0,
    )


def _ok_row(
    scenario: str,
    *,
    node: str = "analyze",
    model: str = _CANDIDATE_MODEL,
    recall: float = 1.0,
    n_fp: int = 0,
    passes: bool = True,
    cost_usd: float | None = None,
    latency_s: float | None = None,
) -> ScorecardRow:
    n_findings = 2 + n_fp
    return ScorecardRow(
        node=node,
        model=model,
        scenario=scenario,
        baseline_model=_BASELINE_MODEL,
        status="ok",
        recall=FindingRecall(value=recall, numerator=round(recall * 2), denominator=2),
        precision=FindingPrecision(value=1.0, numerator=2, denominator=2),
        severity_accuracy=SeverityAccuracy(value=1.0, numerator=2, denominator=2),
        false_positive_rate=FalsePositiveRate(
            value=n_fp / n_findings, numerator=n_fp, denominator=n_findings
        ),
        n_false_positives=n_fp,
        gate=_gate(passes=passes),
        cost=CostPerReview(usd=cost_usd) if cost_usd is not None else None,
        latency=LatencyPerReview(seconds=latency_s) if latency_s is not None else None,
        cost_source="full_graph" if cost_usd is not None else "not_measured",
    )


def _errored_row(scenario: str, *, error: str = "anthropic 529 overloaded") -> ScorecardRow:
    return ScorecardRow.errored(
        node="analyze",
        scenario=scenario,
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        error=error,
    )


# --- from_comparison: the analyze-direct quality join -----------------------


def test_from_comparison_builds_passing_ok_row() -> None:
    expected = [_expected()]
    comparison = compare(grade([_finding()], expected), grade([_finding()], expected))
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="fx/sqli.json",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
        cost=CostPerReview(usd=0.0123),
        latency=LatencyPerReview(seconds=4.2),
    )
    assert row.status == "ok"
    assert row.recall is not None and row.recall.value == 1.0
    assert row.gate is not None and row.gate.passes is True
    assert row.n_false_positives == 0
    assert row.false_positive_rate is not None and row.false_positive_rate.value == 0.0
    assert row.cost is not None and row.cost.usd == pytest.approx(0.0123)
    assert row.cost_source == "full_graph"
    assert row.quality_source == "analyze_direct"


def test_from_comparison_derives_false_positive_rate() -> None:
    expected = [_expected()]
    baseline = grade([_finding()], expected)
    # Candidate catches the expected finding AND emits one finding in a different
    # file (matches nothing) -> 1 FP over 2 findings -> rate 0.5.
    candidate = grade([_finding(), _finding(file_path="app/other.py")], expected)
    comparison = compare(baseline, candidate, fp_allowance=1)
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="fx/sqli.json",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
    )
    assert row.n_false_positives == 1
    assert row.false_positive_rate is not None
    assert row.false_positive_rate.value == pytest.approx(0.5)
    assert row.false_positive_rate.numerator == 1
    assert row.false_positive_rate.denominator == 2
    # No cost supplied -> provenance reflects that, not a phantom full-graph join.
    assert row.cost is None and row.cost_source == "not_measured"


def test_from_comparison_records_failing_gate() -> None:
    expected = [_expected()]
    baseline = grade([_finding()], expected)  # recall 1.0 -> baseline_valid
    candidate = grade([], expected)  # recall 0.0 -> recall_held False
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="fx/sqli.json",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=compare(baseline, candidate),
    )
    assert row.gate is not None
    assert row.gate.passes is False
    assert row.gate.baseline_valid is True
    assert row.gate.recall_held is False
    assert row.recall is not None and row.recall.value == 0.0


def test_from_comparison_carries_replay_provenance() -> None:
    expected = [_expected()]
    comparison = compare(grade([_finding()], expected), grade([_finding()], expected))
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="fx/sqli.json",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
        replay_equivalent=True,
        replay_source="resume",
    )
    assert row.replay_equivalent is True
    assert row.replay_source == "resume"


# --- failed-row diagnostics (FP titles/paths + baseline delta) --------------


def test_from_comparison_populates_diagnostics_with_fp() -> None:
    expected = [_expected()]
    # Candidate catches the expected finding AND adds one FP in another file.
    candidate = grade([_finding(), _finding(file_path="app/evil.py")], expected)
    comparison = compare(grade([_finding()], expected), candidate, fp_allowance=1)
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="s",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
    )
    assert row.diagnostics is not None
    assert len(row.diagnostics.candidate_false_positives) == 1
    fp = row.diagnostics.candidate_false_positives[0]
    assert fp.file_path == "app/evil.py"
    assert fp.title == "t"
    assert fp.finding_type == FindingType.SQL_INJECTION.value
    assert row.diagnostics.candidate_missed == ()
    assert row.diagnostics.baseline_n_false_positives == 0


def test_from_comparison_diagnostics_none_when_clean() -> None:
    expected = [_expected()]
    comparison = compare(grade([_finding()], expected), grade([_finding()], expected))
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="s",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
    )
    assert row.diagnostics is None


def test_from_comparison_diagnostics_captures_missed() -> None:
    expected = [_expected()]
    # Candidate catches nothing -> the expected finding is a recall loss (miss).
    comparison = compare(grade([_finding()], expected), grade([], expected))
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="s",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
    )
    assert row.diagnostics is not None
    assert len(row.diagnostics.candidate_missed) == 1
    assert row.diagnostics.candidate_missed[0].title is None  # ExpectedFinding has no title
    assert row.diagnostics.candidate_missed[0].file_path == "app/db.py"


def test_to_html_renders_diagnostics_section() -> None:
    expected = [_expected()]
    candidate = grade([_finding(), _finding(file_path="app/evil.py")], expected)
    comparison = compare(grade([_finding()], expected), candidate, fp_allowance=1)
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="fx/sqli.json",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
    )
    rendered = Scorecard(rows=(row,)).to_html()
    assert "<h2>Diagnostics</h2>" in rendered
    assert "app/evil.py:10" in rendered  # the FP location is named, not just the count
    assert "false positive" in rendered


def test_to_html_no_diagnostics_section_when_clean() -> None:
    expected = [_expected()]
    comparison = compare(grade([_finding()], expected), grade([_finding()], expected))
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="fx/clean.json",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
    )
    rendered = Scorecard(rows=(row,)).to_html()
    assert "Diagnostics" not in rendered


def test_from_comparison_diagnostics_on_baseline_invalid_fail() -> None:
    expected = [_expected()]
    # Baseline (Sonnet) MISSES the finding -> baseline_valid False -> gate FAILS,
    # even though the candidate caught it (no FP, no miss). Diagnostics must still
    # populate so the FAIL is explained as non-discriminating, not candidate-bad.
    comparison = compare(grade([], expected), grade([_finding()], expected))
    assert comparison.passes is False
    assert comparison.baseline_valid is False
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="s",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
    )
    assert row.diagnostics is not None  # populated despite no candidate FP/miss
    assert row.diagnostics.candidate_false_positives == ()
    assert row.diagnostics.candidate_missed == ()


def test_to_html_explains_baseline_invalid_fail() -> None:
    expected = [_expected()]
    comparison = compare(grade([], expected), grade([_finding()], expected))
    row = ScorecardRow.from_comparison(
        node="analyze",
        scenario="fx/x.json",
        model=_CANDIDATE_MODEL,
        baseline_model=_BASELINE_MODEL,
        comparison=comparison,
    )
    rendered = Scorecard(rows=(row,)).to_html()
    assert "<h2>Diagnostics</h2>" in rendered
    assert "baseline invalid" in rendered  # the why-reason names the cause
    assert "<h3>Findings</h3>" not in rendered  # no FP/miss -> no findings table


# --- errored rows + the consistency validator -------------------------------


def test_errored_row_has_null_metrics() -> None:
    row = _errored_row("fx/boom.json", error="anthropic 529 overloaded")
    assert row.status == "errored"
    assert row.error == "anthropic 529 overloaded"
    assert row.recall is None
    assert row.gate is None
    assert row.n_false_positives is None


def test_errored_row_with_diagnostics_is_rejected() -> None:
    # Diagnostics are null on an errored row, same invariant as the other metrics.
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="errored",
            error="boom",
            diagnostics=RowDiagnostics(),
        )


def test_ok_row_missing_quality_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="ok",  # but every quality field defaults to None
        )


def test_errored_row_with_quality_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="errored",
            error="boom",
            recall=FindingRecall(value=1.0, numerator=1, denominator=1),
        )


def test_errored_row_without_error_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="errored",
        )


def test_cost_present_requires_full_graph_source() -> None:
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="ok",
            recall=FindingRecall(value=1.0, numerator=1, denominator=1),
            precision=FindingPrecision(value=1.0, numerator=1, denominator=1),
            severity_accuracy=SeverityAccuracy(value=1.0, numerator=1, denominator=1),
            false_positive_rate=FalsePositiveRate(value=0.0, numerator=0, denominator=1),
            n_false_positives=0,
            gate=_gate(passes=True),
            cost=CostPerReview(usd=0.01),
            cost_source="not_measured",  # inconsistent: cost present, source says otherwise
        )


def test_errored_row_with_cost_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="errored",
            error="boom",
            cost=CostPerReview(usd=0.01),
            cost_source="full_graph",
        )


def test_latency_without_cost_is_rejected() -> None:
    # latency + cost are a review-level pair; latency alone is nonsensical.
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="ok",
            recall=FindingRecall(value=1.0, numerator=1, denominator=1),
            precision=FindingPrecision(value=1.0, numerator=1, denominator=1),
            severity_accuracy=SeverityAccuracy(value=1.0, numerator=1, denominator=1),
            false_positive_rate=FalsePositiveRate(value=0.0, numerator=0, denominator=1),
            n_false_positives=0,
            gate=_gate(passes=True),
            latency=LatencyPerReview(seconds=1.0),  # latency without a cost
        )


# --- aggregates + serialization ---------------------------------------------


def test_aggregates_counts_and_means_exclude_errored() -> None:
    card = Scorecard(
        rows=(
            _ok_row("s1", recall=1.0, passes=True, cost_usd=0.01, latency_s=2.0),
            _ok_row("s2", recall=0.5, passes=False, cost_usd=0.03, latency_s=4.0),
            _errored_row("s3"),
        )
    )
    aggregates = card.aggregates()
    assert len(aggregates) == 1
    agg = aggregates[0]
    assert (agg.node, agg.model) == ("analyze", _CANDIDATE_MODEL)
    assert agg.n_scenarios == 3
    assert agg.n_ok == 2
    assert agg.n_errored == 1
    assert agg.n_passed == 1
    assert agg.n_failed == 1
    assert agg.mean_recall == pytest.approx(0.75)  # (1.0 + 0.5)/2, errored excluded
    assert agg.total_cost_usd == pytest.approx(0.04)  # 0.01 + 0.03, errored excluded
    assert agg.mean_latency_seconds == pytest.approx(3.0)


def test_aggregates_sorted_by_node_then_model() -> None:
    card = Scorecard(
        rows=(
            _ok_row("s1", model="claude-sonnet-4-6"),
            _ok_row("s1", model="claude-haiku-4-5"),
        )
    )
    assert [agg.model for agg in card.aggregates()] == ["claude-haiku-4-5", "claude-sonnet-4-6"]


def test_to_json_round_trips_rows_and_aggregates() -> None:
    card = Scorecard(rows=(_ok_row("s1", cost_usd=0.01), _errored_row("s2")))
    data = json.loads(card.to_json())
    assert set(data) == {"rows", "aggregates", "triage_rows", "triage_aggregates", "provenance"}
    assert len(data["rows"]) == 2
    assert len(data["aggregates"]) == 1
    assert data["aggregates"][0]["n_errored"] == 1
    assert data["rows"][0]["quality_source"] == "analyze_direct"
    assert data["triage_rows"] == []  # analyze-only card -> empty triage section
    assert data["provenance"] is None  # unset on a directly-constructed card


def test_to_html_renders_rows_and_aggregate() -> None:
    card = Scorecard(
        rows=(
            _ok_row("s_pass", passes=True),
            _ok_row("s_fail", passes=False),
            _errored_row("s_err", error="529 overloaded"),
        )
    )
    rendered = card.to_html()
    assert "<!DOCTYPE html>" in rendered
    assert "<table>" in rendered
    assert "<h2>Analyze</h2>" in rendered
    assert '<span class="badge pass">PASS</span>' in rendered
    assert '<span class="badge fail">FAIL</span>' in rendered
    assert "ERROR: 529 overloaded" in rendered
    assert "s_pass" in rendered and "s_fail" in rendered and "s_err" in rendered


def test_to_html_escapes_markup_in_label() -> None:
    # A label with HTML metacharacters is escaped so it can't inject markup into
    # the artifact (the HTML analogue of the old Markdown pipe-escaping).
    card = Scorecard(rows=(_ok_row("a<b>&c", passes=True),))
    rendered = card.to_html()
    assert "a&lt;b&gt;&amp;c" in rendered
    assert "<b>" not in rendered  # raw markup must never reach the document


# --- provenance stamping (git + prompt version + scenario set) ---------------


def _provenance(*, git_dirty: bool = False) -> ScorecardProvenance:
    return ScorecardProvenance(
        git_sha="6d4caee",
        git_dirty=git_dirty,
        prompt_template_version="analyze-vtest",  # fixed sample; decoupled from live VERSION
        scenario_set=("pygoat_sql_injection", "ssrf_user_host"),
        baseline_model=_BASELINE_MODEL,
        candidate_models=(_CANDIDATE_MODEL,),
        generated_at=datetime(2026, 6, 25, 17, 3, tzinfo=UTC),
    )


def test_to_json_includes_provenance() -> None:
    card = Scorecard(rows=(_ok_row("fx/sqli.json"),), provenance=_provenance())
    prov = json.loads(card.to_json())["provenance"]
    assert prov["git_sha"] == "6d4caee"
    assert prov["git_dirty"] is False
    assert prov["prompt_template_version"] == "analyze-vtest"
    assert prov["scenario_set"] == ["pygoat_sql_injection", "ssrf_user_host"]
    assert prov["candidate_models"] == [_CANDIDATE_MODEL]


def test_to_json_provenance_is_null_when_absent() -> None:
    # The key is always present (stable shape); absent provenance serializes as null.
    assert json.loads(Scorecard(rows=(_ok_row("fx/sqli.json"),)).to_json())["provenance"] is None


def test_to_html_renders_provenance() -> None:
    card = Scorecard(rows=(_ok_row("fx/sqli.json"),), provenance=_provenance(git_dirty=True))
    rendered = card.to_html()
    assert "Provenance" in rendered
    assert "6d4caee" in rendered
    assert "analyze-vtest" in rendered
    assert "dirty tree" in rendered  # git_dirty=True surfaces the marker


def test_to_html_no_provenance_section_when_absent() -> None:
    rendered = Scorecard(rows=(_ok_row("fx/sqli.json"),)).to_html()
    assert 'class="provenance"' not in rendered


def test_build_provenance_captures_git_and_inputs() -> None:
    # Exercises the real build_provenance + git capture the entrypoints call — the
    # entrypoints themselves are real-model-gated (skipped here), so this is the
    # only non-spend coverage of that path.
    from .runner import build_provenance  # noqa: PLC0415

    prov = build_provenance(
        prompt_template_version="analyze-test",
        scenario_labels=["a", "b"],
        baseline_model=_BASELINE_MODEL,
        candidate_models=[_CANDIDATE_MODEL],
    )
    assert prov is not None  # a non-None version -> a stamped ScorecardProvenance
    assert prov.prompt_template_version == "analyze-test"
    assert prov.scenario_set == ("a", "b")
    assert prov.candidate_models == (_CANDIDATE_MODEL,)
    assert prov.generated_at.tzinfo is not None  # AwareDatetime, never naive
    # In-repo: the real HEAD hex (40 for SHA-1, 64 for SHA-256 object format);
    # off-repo (no .git): the "unknown" fallback.
    assert prov.git_sha == "unknown" or len(prov.git_sha) in (40, 64)


def test_build_provenance_returns_none_without_version() -> None:
    # None version is the scripted-harness path — build_scorecard calls this
    # unconditionally and only stamps when a caller opts in with a version.
    from .runner import build_provenance  # noqa: PLC0415

    assert (
        build_provenance(
            prompt_template_version=None,
            scenario_labels=["a"],
            baseline_model=_BASELINE_MODEL,
            candidate_models=[_CANDIDATE_MODEL],
        )
        is None
    )


# --- cost-source 3-state + errored-replay + aggregate denominator ------------


def test_measure_failed_row_is_valid() -> None:
    # A requested-but-failed cost pass: cost=None + cost_source="measure_failed"
    # validates clean (distinct from "not_measured" = never requested).
    row = ScorecardRow(
        node="analyze",
        model=_CANDIDATE_MODEL,
        scenario="s",
        baseline_model=_BASELINE_MODEL,
        status="ok",
        recall=FindingRecall(value=1.0, numerator=1, denominator=1),
        precision=FindingPrecision(value=1.0, numerator=1, denominator=1),
        severity_accuracy=SeverityAccuracy(value=1.0, numerator=1, denominator=1),
        false_positive_rate=FalsePositiveRate(value=0.0, numerator=0, denominator=1),
        n_false_positives=0,
        gate=_gate(passes=True),
        cost=None,
        cost_source="measure_failed",
    )
    assert row.cost is None
    assert row.cost_source == "measure_failed"


def test_cost_present_with_measure_failed_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="ok",
            recall=FindingRecall(value=1.0, numerator=1, denominator=1),
            precision=FindingPrecision(value=1.0, numerator=1, denominator=1),
            severity_accuracy=SeverityAccuracy(value=1.0, numerator=1, denominator=1),
            false_positive_rate=FalsePositiveRate(value=0.0, numerator=0, denominator=1),
            n_false_positives=0,
            gate=_gate(passes=True),
            cost=CostPerReview(usd=0.01),
            cost_source="measure_failed",  # cost present requires full_graph
        )


def test_errored_row_with_replay_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="errored",
            error="boom",
            replay_equivalent=True,
            replay_source="resume",
        )


def test_aggregate_surfaces_n_costed() -> None:
    # n_costed is total_cost_usd's denominator: how many ok rows actually carried
    # a measured cost, so a partial-cost batch isn't read as a complete total.
    card = Scorecard(rows=(_ok_row("s1", cost_usd=0.01), _ok_row("s2")))
    agg = card.aggregates()[0]
    assert agg.n_ok == 2
    assert agg.n_costed == 1  # only s1 carried a measured cost
    assert agg.total_cost_usd == pytest.approx(0.01)


def test_replay_equivalent_without_source_is_rejected() -> None:
    # A replay verdict requires a real source: replay_equivalent set while
    # replay_source is still "not_applicable" is inconsistent (mirrors cost).
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="ok",
            recall=FindingRecall(value=1.0, numerator=1, denominator=1),
            precision=FindingPrecision(value=1.0, numerator=1, denominator=1),
            severity_accuracy=SeverityAccuracy(value=1.0, numerator=1, denominator=1),
            false_positive_rate=FalsePositiveRate(value=0.0, numerator=0, denominator=1),
            n_false_positives=0,
            gate=_gate(passes=True),
            replay_equivalent=True,  # but replay_source defaults to "not_applicable"
        )


def test_replay_source_without_verdict_is_rejected() -> None:
    # The reverse: a real source with no equivalence verdict is also inconsistent.
    with pytest.raises(ValidationError):
        ScorecardRow(
            node="analyze",
            model=_CANDIDATE_MODEL,
            scenario="s",
            baseline_model=_BASELINE_MODEL,
            status="ok",
            recall=FindingRecall(value=1.0, numerator=1, denominator=1),
            precision=FindingPrecision(value=1.0, numerator=1, denominator=1),
            severity_accuracy=SeverityAccuracy(value=1.0, numerator=1, denominator=1),
            false_positive_rate=FalsePositiveRate(value=0.0, numerator=0, denominator=1),
            n_false_positives=0,
            gate=_gate(passes=True),
            replay_source="resume",  # but replay_equivalent is None
        )


# --- opt-in real-model artifact entrypoint ----------------------------------


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model scorecard spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
def test_real_scorecard_evidence() -> None:
    """OPT-IN real API spend — emits the cross-scenario scorecard artifact.

    REPORT-ONLY, BY DESIGN: asserts only that the run COMPLETED (a row per spec).
    The verdict is the JSON + HTML scorecard written to `reports/scorecard/`,
    read by a human — pytest does not gate on a candidate gate failure (the runner
    is report-only). Quality (recall/precision/severity/FP/gate) is REAL spend
    through the analyze-direct path under baseline (Sonnet) vs candidate (Haiku);
    cost is zero-spend (the fixtures' scripted run_review responses priced through
    the production path).

    Sync test on purpose: `build_scorecard` calls `run_review` (asyncio.run inside)
    for the cost pass, which cannot nest in a running loop.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the real-model scorecard")

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.prompts.analyze import VERSION as ANALYZE_PROMPT_VERSION  # noqa: PLC0415

    from .runner import ScenarioSpec, build_scorecard  # noqa: PLC0415

    class _NoOpExchangePersister:
        """No-op `LLMExchangePersister`: `AnthropicProvider.complete()` is
        fail-closed on `persister=None`; the scorecard reads findings from
        analyze's return, so the exchange persist is discarded."""

        async def persist(self, event: object, request: object, response: object) -> None:  # noqa: ARG002
            return None

    cfg = ModelConfig()
    baseline_model = cfg.analyze_model  # today's top-tier analyze (Sonnet), NOT standard_*
    candidate_model = "claude-haiku-4-5"  # the shipped STANDARD default (DECISIONS#041)

    from outrider.llm.pricing import normalize_to_pricing_key  # noqa: PLC0415

    # Guard the meaningless self-comparison (e.g. OUTRIDER_MODEL_ANALYZE_MODEL=Haiku)
    # BEFORE constructing the provider, so a guard-fire can't leak an unclosed client.
    if normalize_to_pricing_key(baseline_model) == normalize_to_pricing_key(candidate_model):
        pytest.fail(
            f"baseline ({baseline_model}) and candidate ({candidate_model}) normalize to the "
            "same model — the scorecard would prove nothing about Sonnet-vs-Haiku. Point "
            "OUTRIDER_MODEL_ANALYZE_MODEL at Sonnet (or unset it) for the evidence run."
        )
    provider = AnthropicProvider(
        api_key=SecretStr(api_key), model_config=cfg, persister=_NoOpExchangePersister()
    )

    mock = Path("tests/eval/fixtures/mock_github")

    def _gt(
        file_path: str, line_start: int, line_end: int, finding_type: FindingType
    ) -> tuple[ExpectedFinding, ...]:
        return (
            ExpectedFinding(
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                finding_type=finding_type,
                severity=lookup_severity(finding_type),
            ),
        )

    # Full-response fixtures (drive triage+analyze+synthesize, so the cost pass's
    # run_review works) with ground truth; safe_refactor carries none (clean code).
    # Curated scorecard corpus (decision-oriented, NOT a fixture dump) — each row
    # is real-model spend per rerun, so the set stays explicit and capped. Tier 1
    # wires existing single-finding fixtures for broader finding-type coverage +
    # precision traps; Tier 2 adds the SSRF recall + fixed-host precision pair
    # (regression coverage for ssrf destination-control precision — a fixed-host
    # fetch must not over-flag). Dual-mode command_injection/weak_crypto fixtures
    # are intentionally excluded: those finding types are deterministically
    # backstopped by OBSERVED tree-sitter queries (the real-model finding collapses
    # to OBSERVED), so they do not discriminate Sonnet-vs-Haiku and would be
    # near-vacuous rows.
    specs = [
        # --- true positives (recall) ---
        # These 7 rows reuse the canonical ground truth from `_GROUND_TRUTH_BY_FIXTURE`
        # (shared with test_model_comparison) — a single source of truth, so a fixture
        # line-shift cannot drift the two eval suites' expected findings apart.
        ScenarioSpec.from_fixture(
            "pygoat_sql_injection",
            _PYGOAT_SQL_FIXTURE,
            _GROUND_TRUTH_BY_FIXTURE[_PYGOAT_SQL_FIXTURE],
        ),
        ScenarioSpec.from_fixture(
            "sqli_concat",
            _SQLI_HOLDOUT_CONCAT_FIXTURE,
            _GROUND_TRUTH_BY_FIXTURE[_SQLI_HOLDOUT_CONCAT_FIXTURE],
        ),
        ScenarioSpec.from_fixture(
            "pygoat_auth_bypass",
            _PYGOAT_AUTH_FIXTURE,
            _GROUND_TRUTH_BY_FIXTURE[_PYGOAT_AUTH_FIXTURE],
        ),
        ScenarioSpec.from_fixture(
            "missing_input_validation",
            _MISSING_INPUT_VALIDATION_FIXTURE,
            _GROUND_TRUTH_BY_FIXTURE[_MISSING_INPUT_VALIDATION_FIXTURE],
        ),
        ScenarioSpec.from_fixture(
            "path_traversal",
            _PATH_TRAVERSAL_FIXTURE,
            _GROUND_TRUTH_BY_FIXTURE[_PATH_TRAVERSAL_FIXTURE],
        ),
        ScenarioSpec.from_fixture(
            "missing_error_handling",
            _MISSING_ERROR_HANDLING_FIXTURE,
            _GROUND_TRUTH_BY_FIXTURE[_MISSING_ERROR_HANDLING_FIXTURE],
        ),
        ScenarioSpec.from_fixture(
            "n_plus_one_query",
            _N_PLUS_ONE_FIXTURE,
            _GROUND_TRUTH_BY_FIXTURE[_N_PLUS_ONE_FIXTURE],
        ),
        # Scorecard-local TPs (not in the shared registry): a weak_password_hash row +
        # an SSRF recall row. ssrf_user_host pins base `ssrf` (HIGH), NOT escalated
        # ssrf_metadata: a real-model run showed both Sonnet and Haiku classify a bare
        # caller-controlled fetch (no literal metadata host) as base `ssrf`, so the
        # escalated pin made the row non-discriminating (baseline-invalid). Base `ssrf`
        # is what the models actually emit, so the row measures SSRF recall again; the
        # metadata-escalation taxonomy gets its own explicit-metadata fixture (Tier 3).
        ScenarioSpec.from_fixture(
            "weak_password_hash",
            str(mock / "weak_password_hash_md5.json"),
            _gt("accounts/auth.py", 5, 5, FindingType.WEAK_PASSWORD_HASH),
        ),
        ScenarioSpec.from_fixture(
            "ssrf_user_host",
            str(mock / "ssrf_user_host.json"),
            _gt("app/fetch.py", 7, 7, FindingType.SSRF),
        ),
        # --- precision traps (clean: NO finding expected) ---
        ScenarioSpec.from_fixture(
            "safe_parameterized_query",
            str(mock / "safe_parameterized_query.json"),
            (),
        ),
        ScenarioSpec.from_fixture(
            "ssrf_fixed_host_safe",
            str(mock / "ssrf_fixed_host_safe.json"),
            (),
        ),
        ScenarioSpec.from_fixture("safe_refactor", str(mock / "safe_refactor.json"), ()),
    ]

    card = build_scorecard(
        specs,
        baseline_provider=provider,
        candidate_provider=provider,
        baseline_model=baseline_model,
        candidate_models=[candidate_model],
        measure_cost=True,
        close_providers=True,  # close the real provider inside build_scorecard's loop
        prompt_template_version=ANALYZE_PROMPT_VERSION,  # stamps run provenance on the card
    )

    out_dir = Path("reports") / "scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(card.to_json(), encoding="utf-8")
    (out_dir / "scorecard.html").write_text(card.to_html(), encoding="utf-8")

    print(  # noqa: T201 — operator artifact pointer
        f"\nSCORECARD — REPORT ONLY: wrote {out_dir}/scorecard.{{json,html}} "
        f"({len(card.rows)} rows, baseline={baseline_model}, candidate={candidate_model})"
    )
    # Report-only: assert only that the run produced a row per spec (it COMPLETED).
    assert len(card.rows) == len(specs)


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model triage scorecard spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
def test_real_triage_scorecard_evidence() -> None:
    """OPT-IN real API spend — emits the TRIAGE scorecard artifact (Sonnet vs Haiku
    triage over the known-vuln fixtures).

    REPORT-ONLY, BY DESIGN: asserts only that the run COMPLETED (a triage row per
    spec). The verdict is the JSON + HTML written to
    `reports/scorecard/triage-scorecard.{json,html}`, read by a human — the runner is
    report-only. Quality (tier accuracy / drop-from-analysis / dimension recall /
    under-risking / gate) is REAL spend through the real triage node; there is NO
    cost pass (triage rows are quality-only per the spec).

    Self-contained: its own provider, closed inside `build_triage_scorecard`'s event
    loop. It does NOT share the analyze entrypoint's client — each `asyncio.run`
    binds the httpx client to its own loop, and a real client can't be reused across
    loops. Sync test for the same reason as `test_real_scorecard_evidence`.

    Ground truth is hand-authored (per spec, no `--regenerate-expected`): each
    fixture's single changed file gets the tier/risk/dimension a human reviewer would
    assign. If the baseline (Sonnet) under-tiers or under-risks against this opinion,
    `baseline_valid` is False and the candidate's hold reads vacuous — that's the
    signal to revisit either the model or the ground truth, surfaced in the artifact.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the real-model triage scorecard")

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.llm.pricing import normalize_to_pricing_key  # noqa: PLC0415
    from outrider.prompts.triage import VERSION as TRIAGE_PROMPT_VERSION  # noqa: PLC0415
    from outrider.schemas.triage_result import (  # noqa: PLC0415
        ReviewDimension,
        ReviewTier,
        RiskLevel,
    )

    from .runner import TriageScenarioSpec, build_triage_scorecard  # noqa: PLC0415
    from .triage_grading import ExpectedTriage  # noqa: PLC0415

    class _NoOpExchangePersister:
        """No-op `LLMExchangePersister`: the triage node's `provider.complete()` is
        fail-closed on `persister=None`; the grader reads the `TriageResult` off the
        node return, so the exchange persist is discarded (no audit events)."""

        async def persist(self, event: object, request: object, response: object) -> None:  # noqa: ARG002
            return None

    cfg = ModelConfig()
    baseline_model = cfg.analyze_model  # Sonnet — the strong reference tier
    candidate_model = "claude-haiku-4-5"  # the cheap tier under test for triage
    if normalize_to_pricing_key(baseline_model) == normalize_to_pricing_key(candidate_model):
        pytest.fail(
            f"baseline ({baseline_model}) and candidate ({candidate_model}) normalize to the "
            "same model — the triage scorecard would prove nothing about Sonnet-vs-Haiku. Point "
            "OUTRIDER_MODEL_ANALYZE_MODEL at Sonnet (or unset it) for the evidence run."
        )
    provider = AnthropicProvider(
        api_key=SecretStr(api_key), model_config=cfg, persister=_NoOpExchangePersister()
    )

    mock = Path("tests/eval/fixtures/mock_github")

    def _exp(
        path: str, tier: ReviewTier, risk: RiskLevel, dimension: ReviewDimension
    ) -> ExpectedTriage:
        return ExpectedTriage(
            expected_file_tiers={path: tier}, overall_risk=risk, relevant_dimensions=(dimension,)
        )

    # One changed file per fixture; the vuln files warrant DEEP + a security lens,
    # the quality/perf changes STANDARD, and the clean billing refactor STANDARD-LOW
    # (totals math is worth a cheap pass even when the diff looks safe).
    specs = [
        TriageScenarioSpec.from_fixture(
            "pygoat_sql_injection",
            str(mock / "pygoat_sql_injection.json"),
            _exp(
                "pygoat/introduction/views.py",
                ReviewTier.DEEP,
                RiskLevel.HIGH,
                ReviewDimension.SECURITY,
            ),
        ),
        TriageScenarioSpec.from_fixture(
            "pygoat_auth_bypass",
            str(mock / "pygoat_auth_bypass.json"),
            _exp(
                "pygoat/introduction/auth_views.py",
                ReviewTier.DEEP,
                RiskLevel.HIGH,
                ReviewDimension.SECURITY,
            ),
        ),
        TriageScenarioSpec.from_fixture(
            "missing_error_handling",
            str(mock / "missing_error_handling.json"),
            _exp(
                "profile/client.py",
                ReviewTier.STANDARD,
                RiskLevel.MEDIUM,
                ReviewDimension.CODE_QUALITY,
            ),
        ),
        TriageScenarioSpec.from_fixture(
            "n_plus_one_query",
            str(mock / "n_plus_one_query.json"),
            _exp(
                "orders/enrich.py",
                ReviewTier.STANDARD,
                RiskLevel.MEDIUM,
                ReviewDimension.PERFORMANCE,
            ),
        ),
        TriageScenarioSpec.from_fixture(
            "safe_refactor",
            str(mock / "safe_refactor.json"),
            _exp(
                "billing/totals.py",
                ReviewTier.STANDARD,
                RiskLevel.LOW,
                ReviewDimension.CODE_QUALITY,
            ),
        ),
    ]

    card = build_triage_scorecard(
        specs,
        baseline_provider=provider,
        candidate_provider=provider,
        baseline_model=baseline_model,
        candidate_models=[candidate_model],
        close_providers=True,  # close the real provider inside build_triage_scorecard's loop
        prompt_template_version=TRIAGE_PROMPT_VERSION,  # stamps run provenance on the card
    )

    out_dir = Path("reports") / "scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "triage-scorecard.json").write_text(card.to_json(), encoding="utf-8")
    (out_dir / "triage-scorecard.html").write_text(card.to_html(), encoding="utf-8")

    print(  # noqa: T201 — operator artifact pointer
        f"\nTRIAGE SCORECARD — REPORT ONLY: wrote {out_dir}/triage-scorecard.{{json,html}} "
        f"({len(card.triage_rows)} rows, baseline={baseline_model}, candidate={candidate_model})"
    )
    # Report-only: assert only that the run produced a triage row per spec.
    assert len(card.triage_rows) == len(specs)
