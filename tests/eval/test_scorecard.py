"""Tests for the cross-scenario eval scorecard (tests/eval/scorecard.py).

Covers the typed objects + serialization that step 1 promotes out of the inline
GATE SUMMARY print: `ScorecardRow.from_comparison` (the analyze-direct quality
join + false-positive-rate derivation), the errored-row path (transient
isolation), the status/metric consistency validator, the per-`(node, model)`
aggregate reduction, and the JSON + Markdown emitters. Pure — no DB, no LLM, no
spend.
"""

from __future__ import annotations

import json
import os
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
from .scorecard import GateVerdict, Scorecard, ScorecardRow

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


# --- errored rows + the consistency validator -------------------------------


def test_errored_row_has_null_metrics() -> None:
    row = _errored_row("fx/boom.json", error="anthropic 529 overloaded")
    assert row.status == "errored"
    assert row.error == "anthropic 529 overloaded"
    assert row.recall is None
    assert row.gate is None
    assert row.n_false_positives is None


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
    assert set(data) == {"rows", "aggregates"}
    assert len(data["rows"]) == 2
    assert len(data["aggregates"]) == 1
    assert data["aggregates"][0]["n_errored"] == 1
    assert data["rows"][0]["quality_source"] == "analyze_direct"


def test_to_markdown_renders_rows_and_aggregate() -> None:
    card = Scorecard(
        rows=(
            _ok_row("s_pass", passes=True),
            _ok_row("s_fail", passes=False),
            _errored_row("s_err", error="529 overloaded"),
        )
    )
    md = card.to_markdown()
    assert "# Eval scorecard" in md
    assert "## Aggregate" in md
    assert "PASS" in md
    assert "FAIL" in md
    assert "ERROR: 529 overloaded" in md
    assert "s_pass" in md and "s_fail" in md and "s_err" in md


# --- opt-in real-model artifact entrypoint ----------------------------------


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model scorecard spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
def test_real_scorecard_evidence() -> None:
    """OPT-IN real API spend — emits the cross-scenario scorecard artifact.

    REPORT-ONLY, BY DESIGN: asserts only that the run COMPLETED (a row per spec).
    The verdict is the JSON + Markdown scorecard written to `reports/scorecard/`,
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
    specs = [
        ScenarioSpec.from_fixture(
            "pygoat_sql_injection",
            str(mock / "pygoat_sql_injection.json"),
            _gt("pygoat/introduction/views.py", 5, 5, FindingType.SQL_INJECTION),
        ),
        ScenarioSpec.from_fixture(
            "pygoat_auth_bypass",
            str(mock / "pygoat_auth_bypass.json"),
            _gt("pygoat/introduction/auth_views.py", 7, 8, FindingType.AUTH_BYPASS),
        ),
        ScenarioSpec.from_fixture(
            "missing_error_handling",
            str(mock / "missing_error_handling.json"),
            _gt("profile/client.py", 5, 5, FindingType.MISSING_ERROR_HANDLING),
        ),
        ScenarioSpec.from_fixture(
            "n_plus_one_query",
            str(mock / "n_plus_one_query.json"),
            _gt("orders/enrich.py", 7, 7, FindingType.N_PLUS_ONE_QUERY),
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
    )

    out_dir = Path("reports") / "scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(card.to_json(), encoding="utf-8")
    (out_dir / "scorecard.md").write_text(card.to_markdown(), encoding="utf-8")

    print(  # noqa: T201 — operator artifact pointer
        f"\nSCORECARD — REPORT ONLY: wrote {out_dir}/scorecard.{{json,md}} "
        f"({len(card.rows)} rows, baseline={baseline_model}, candidate={candidate_model})"
    )
    # Report-only: assert only that the run produced a row per spec (it COMPLETED).
    assert len(card.rows) == len(specs)
