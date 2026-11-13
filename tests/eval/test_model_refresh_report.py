"""Check comparison limits using small, complete repeated-result records."""

from __future__ import annotations

import copy

import pytest

from .model_refresh import REPORT_SCHEMA, TARGETS, build_plan
from .model_refresh_report import compare_saved, comparisons


def evidence():
    plan = build_plan((TARGETS["anthropic-haiku-5-5"],), phase="analyze", reps=3)
    plan["measurement_contract"] = "refresh-judged-origin-excluded-three-reps-v1"
    plan["scenarios"] = [
        {"fixture": "new-sqli", "track": "holdout-recall"},
        {"fixture": "safe-sql", "track": "holdout-safe"},
    ]
    candidates = []
    for role in ("baseline-cheap", "candidate-cheap"):
        calls = [
            {
                "scenario": f"compat:{path}",
                "status": "completed",
                "persisted_exchange": True,
                "usage_based_usd": "0.001",
                "wall_ms": 10,
            }
            for path in plan["compatibility_cases"]
        ]
        cases = []
        for rep in range(3):
            for scenario in plan["scenarios"]:
                positive = scenario["track"] == "holdout-recall"
                cases.append(
                    {
                        "phase": "analyze",
                        "rep": rep,
                        **scenario,
                        "model_judged": {
                            "expected": int(positive),
                            "matched": int(positive),
                            "false_alarms": None if positive else 0,
                            "extras_advisory": 0,
                        },
                        "response_rejections": 0,
                        "proposal_rejections": 0,
                    }
                )
                calls.append(
                    {
                        "scenario": f"analyze:{rep}:{scenario['fixture']}",
                        "status": "completed",
                        "persisted_exchange": True,
                        "usage_based_usd": "0.001",
                        "wall_ms": 10,
                    }
                )
        candidates.append(
            {
                "target": {"host": "anthropic", "model": role, "role": role},
                "status": "completed",
                "cases": cases,
                "calls": calls,
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "evidence_kind": "live-provider",
        "plan": plan,
        "candidates": candidates,
    }


@pytest.mark.parametrize("failure", ["missing", "output-limit"])
def test_incomplete_priced_collection_is_neither_cost_nor_latency_winner(failure: str) -> None:
    report = evidence()
    candidate = report["candidates"][1]
    if failure == "missing":
        candidate["calls"].pop()
    else:
        candidate["status"] = "output-limit"
        candidate["calls"][-1].update(status="output-limit", truncated=True)
    result = comparisons(report)[0]
    assert result["cost"] == result["latency"] == "inconclusive"
    assert candidate["summary"]["collection_complete"] is False
    assert candidate["summary"]["quality_complete"] is False


def test_fresh_sql_safety_is_required_even_when_baseline_also_fails() -> None:
    report = evidence()
    for candidate in report["candidates"]:
        candidate["cases"][0]["model_judged"]["matched"] = 0
        candidate["cases"][2]["model_judged"]["matched"] = 0
        candidate["cases"][1]["model_judged"]["false_alarms"] = 1
    result = comparisons(report)[0]
    assert any("fresh SQL recall failed" in issue for issue in result["issues"])
    assert any("fresh safe SQL" in issue for issue in result["issues"])


def test_saved_comparison_rejects_missing_or_drifted_runtime_inputs() -> None:
    before, after = evidence(), evidence()
    after["plan"]["runtime"]["python"] = "changed runtime"
    with pytest.raises(ValueError, match="runtime"):
        compare_saved(before, after)
    del before["plan"]["inputs"]
    del after["plan"]["inputs"]
    with pytest.raises(ValueError, match="missing saved input identities: inputs"):
        compare_saved(before, after)
    unchanged = evidence()
    assert compare_saved(unchanged, copy.deepcopy(unchanged))["schema"] == REPORT_SCHEMA


@pytest.mark.parametrize("identity", ["unknown-contract", "missing-criteria", "changed-criteria"])
def test_v2_saved_comparison_rejects_unidentified_or_drifted_measurement(identity: str) -> None:
    from .review_quality import MEASUREMENT_CONTRACT_V2

    before, after = evidence(), evidence()
    for report in (before, after):
        report["plan"]["measurement_contract"] = MEASUREMENT_CONTRACT_V2
    if identity == "unknown-contract":
        after["plan"]["measurement_contract"] = "unrecognized-measurement"
    elif identity == "missing-criteria":
        before["plan"].pop("measurement_inputs")
    else:
        after["plan"]["measurement_inputs"] = {"changed-criterion": "0" * 64}
    with pytest.raises(ValueError, match="measurement"):
        compare_saved(before, after)


@pytest.mark.parametrize("identity", ["suite", "dataset"])
def test_v2_saved_comparison_rejects_different_analysis_corpora(identity: str) -> None:
    from .review_quality import MEASUREMENT_CONTRACT_V2

    before, after = evidence(), evidence()
    for report in (before, after):
        report["plan"]["measurement_contract"] = MEASUREMENT_CONTRACT_V2
    if identity == "suite":
        after["plan"]["suite"] = "grounded"
    else:
        after["plan"]["dataset"]["id"] = "different-data-provenance"
    with pytest.raises(ValueError, match="suites|dataset"):
        compare_saved(before, after)


@pytest.mark.parametrize(
    "coverage",
    [
        "complete",
        "unknown-context",
        "pending",
        "missing",
        "duplicate",
        "unpriced",
        "refusal",
        "max_tokens",
        "length",
        "tool_use",
        "unknown",
        "missing-finish",
    ],
)
def test_v2_cost_uses_complete_priced_workloads_without_implying_quality(coverage: str) -> None:
    from .review_quality import MEASUREMENT_CONTRACT_V2

    report = evidence()
    report["plan"]["measurement_contract"] = MEASUREMENT_CONTRACT_V2
    report["plan"]["compatibility_cases"] = []
    for candidate in report["candidates"]:
        candidate["calls"] = [
            call for call in candidate["calls"] if call["scenario"].startswith("analyze:")
        ]
        for call in candidate["calls"]:
            call["finish_reason"] = "end_turn"
        for case in candidate["cases"]:
            case["model_dependent"] = case.pop("model_judged")
            case["measurement_v2"] = {
                "unscored_expected": 0,
                "safe_model_emissions": 0,
                "safe_product_emissions": 0,
            }
    candidate = report["candidates"][1]
    for call in candidate["calls"]:
        call.update(usage_based_usd="0.0005", wall_ms=5)
    if coverage == "missing":
        candidate["calls"].pop()
    elif coverage == "duplicate":
        candidate["calls"].append(dict(candidate["calls"][0]))
    elif coverage == "unpriced":
        candidate["calls"][-1]["usage_based_usd"] = None
    elif coverage == "unknown-context":
        candidate["cases"][0]["measurement_v2"]["unscored_expected"] = 1
        candidate["cases"][0]["model_dependent"].update(expected=0, matched=0)
    elif coverage == "pending":
        candidate["cases"][0]["measurement_v2"].update(pending_model_expected=1, pending_expected=1)
        candidate["cases"][0]["model_dependent"].update(matched=0, pending=1)
    elif coverage in {"refusal", "max_tokens", "length", "tool_use", "unknown"}:
        candidate["calls"][-1].update(finish_reason=coverage, error_type=None, truncated=False)
    elif coverage == "missing-finish":
        candidate["calls"][-1].pop("finish_reason")
    decision = comparisons(report)[0]
    assert decision["quality"] == "requires-adjudication"
    assert candidate["summary"]["quality_complete"] is False
    assert decision["cost"] == (
        "lower" if coverage in {"complete", "unknown-context", "pending"} else "inconclusive"
    )
    assert decision["latency"] == (
        "lower"
        if coverage in {"complete", "unknown-context", "pending", "unpriced"}
        else "inconclusive"
    )
    assert candidate["summary"]["refusals"] == int(coverage == "refusal")
    if coverage == "pending":
        summary = candidate["summary"]["per_fixture"]["new-sqli"]
        assert summary["model_expected"] == 1
        assert summary["unscored_expected"] == 0
        assert summary["pending_model_expected"] == 1
        assert summary["grounded_recall_majority"] is None
