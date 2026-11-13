"""Check recorded-cost comparisons and receipt-bound offline reporting."""

from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal

import pytest

from . import review_costs

ARMS = ("context-off", "context-on", "direct-one-shot")


def _report():
    cases = [
        {"id": name, "changed_files": ["app.py", "tests/test_app.py"]} for name in ("bug", "fix")
    ]
    rows, budgets = [], {}
    for arm_index, arm in enumerate(ARMS):
        total = Decimal(0)
        for index, case in enumerate(cases, 1):
            amount = Decimal(index) * (Decimal("0.10") if arm_index == 2 else Decimal("0.01"))
            total += amount
            call = {
                "index": index,
                "scenario": f"0:{case['id']}:analyze:0",
                "cost_status": "usage-priced",
                "usage_based_usd": str(amount),
                "budget_retained_upper_usd": str(amount),
                "worst_case_reserved_usd": "99",
                "persisted_exchange": True,
                "audit_event_id": f"event:{arm}:{index}",
                "status": "completed",
                "error_type": None,
                "finish_reason": "end_turn",
                "truncated": False,
            }
            rows.append(
                {
                    "arm": arm,
                    "case_id": case["id"],
                    "rep": 0,
                    "status": "completed",
                    "diagnostics_status": "not-applicable" if arm_index == 2 else "captured",
                    "diagnostic_events": [
                        {
                            "event_type": "file_examination",
                            "payload": {
                                "node_id": "analyze",
                                "file_path": "app.py",
                                "skip_reason": "COST_BUDGET_EXHAUSTED",
                            },
                        }
                    ],
                    "calls": [call],
                    "raw_exchange_files": [f"{arm}/calls/{index:06d}.exchange.json"],
                }
            )
        budgets[arm] = {
            "calls": 2,
            "known_usage_based_usd": str(total),
            "uncertain_budget_upper_usd": "0",
        }
    return {
        "schema": "review-comparison-v1",
        "status": "completed",
        "evidence_kind": "injected-provider",
        "plan": {
            "target": {"host": "test", "model": "test-model", "role": "control"},
            "pricing_version": "saved-version",
            "cases": cases,
            "schedule": [{"case_id": case["id"], "rep": 0, "order": list(ARMS)} for case in cases],
        },
        "cases": rows,
        "budgets": budgets,
    }


def _write_report(directory, report):
    data = {"results.json": report, "plan.json": report["plan"]}
    for row in report["cases"]:
        for call in row["calls"]:
            prefix = f"{row['arm']}/calls/{call['index']:06d}"
            data[prefix + ".attempt.json"] = call
            if call["persisted_exchange"]:
                data[prefix + ".exchange.json"] = {
                    "call_index": call["index"],
                    "scenario": call["scenario"],
                    "target": report["plan"]["target"],
                    "request": {"model": "test-model"},
                    "response": {"model": "test-model"},
                    "event": {
                        "event_id": call["audit_event_id"],
                        "model": "test-model",
                        "cost_usd": call["usage_based_usd"],
                    },
                }
    directory.mkdir()
    receipts = {}
    for name, value in data.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        receipts[name] = hashlib.sha256(raw).hexdigest()
    (directory / "receipts.json").write_text(json.dumps(receipts))
    return receipts


def test_costs_use_recorded_usage_not_reservations_and_keep_coverage_visible():
    report = _report()
    untouched = copy.deepcopy(report)
    summary = review_costs.summarize_costs(report)
    assert report == untouched
    assert summary["totals"]["context-off"]["known_usage_based_usd"] == "0.03"
    assert summary["totals"]["direct-one-shot"]["known_usage_based_usd"] == "0.30"
    assert summary["comparisons"]["context-off"]["workflow_cost_divided_by_standalone"] == "0.1"
    assert summary["comparisons"]["context-off"]["workflow_minus_standalone_usd"] == "-0.27"
    pair = summary["paired_reviews"][0]
    assert pair["costs"]["context-off"]["known_usage_based_usd"] == "0.01"
    coverage = pair["coverage"]["context-off"]
    assert coverage["recorded_analysis_skips"] == [
        {"file_path": "app.py", "reason": "COST_BUDGET_EXHAUSTED"}
    ]
    assert coverage["changed_files_without_analysis_event"] == ["tests/test_app.py"]
    assert "no quality winner" in summary["qualification"]
    first = report["cases"][0]["calls"][0]
    later = {
        **first,
        "index": 3,
        "scenario": "0:bug:analyze:1",
        "audit_event_id": "event:parallel",
        "usage_based_usd": "0.04",
        "budget_retained_upper_usd": "0.04",
    }
    report["cases"][0]["calls"] = [later, first]
    report["budgets"]["context-off"].update(calls=3, known_usage_based_usd="0.07")
    parallel = review_costs.summarize_costs(report)
    assert parallel["totals"]["context-off"]["known_usage_based_usd"] == "0.07"


@pytest.mark.parametrize(
    "problem",
    [
        "unpriced",
        "failed",
        "missing",
        "unfinished",
        "no-calls",
        "gapped-partial",
        "gapped-complete",
    ],
)
def test_incomplete_or_unknown_costs_cannot_produce_a_total_ratio(problem):
    report = _report()
    row = report["cases"][0]
    budget = report["budgets"]["context-off"]
    if problem == "unpriced":
        row["calls"][0].update(
            cost_status="uncosted", usage_based_usd=None, budget_retained_upper_usd="1.50"
        )
        budget.update(known_usage_based_usd="0.02", uncertain_budget_upper_usd="1.50")
    elif problem == "failed":
        row["status"] = "failed"
    elif problem == "unfinished":
        row["calls"][0]["finish_reason"] = "max_tokens"
    elif problem == "no-calls":
        row["calls"] = []
        budget.update(calls=1, known_usage_based_usd="0.02")
    elif problem.startswith("gapped-"):
        row["calls"][0]["scenario"] = "0:bug:analyze:2"
        if problem == "gapped-partial":
            row["status"] = "failed"
    else:
        report["cases"].remove(row)
        budget.update(calls=1, known_usage_based_usd="0.02")
    summary = review_costs.summarize_costs(report)
    assert summary["comparisons"]["context-off"]["workflow_cost_divided_by_standalone"] is None
    assert summary["comparisons"]["context-off"]["workflow_minus_standalone_usd"] is None
    if problem == "unpriced":
        assert summary["totals"]["context-off"]["unpriced_upper_usd"] == "1.50"
        assert summary["totals"]["context-off"]["known_usage_based_usd"] == "0.02"


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate-row",
        "duplicate-event",
        "budget",
        "unpersisted-price",
        "nan",
        "boolean-count",
        "scenario",
        "swapped-case-calls",
    ],
)
def test_inconsistent_accounting_fails_explicitly(problem):
    report = _report()
    if problem == "duplicate-row":
        report["cases"].append(copy.deepcopy(report["cases"][0]))
    elif problem == "duplicate-event":
        report["cases"][1]["calls"][0]["audit_event_id"] = report["cases"][0]["calls"][0][
            "audit_event_id"
        ]
    elif problem == "budget":
        report["budgets"]["context-off"]["known_usage_based_usd"] = "0.04"
    elif problem == "unpersisted-price":
        report["cases"][0]["calls"][0]["persisted_exchange"] = False
    elif problem == "nan":
        report["cases"][0]["calls"][0]["usage_based_usd"] = "NaN"
    elif problem == "scenario":
        report["cases"][0]["calls"][0]["scenario"] = "0:other-case:analyze:0"
    elif problem == "swapped-case-calls":
        report["cases"][0]["calls"], report["cases"][1]["calls"] = (
            report["cases"][1]["calls"],
            report["cases"][0]["calls"],
        )
    else:
        report["cases"][0]["rep"] = False
    with pytest.raises(review_costs.CostInputError):
        review_costs.summarize_costs(report)


def test_offline_command_writes_new_derived_output_and_preserves_source(
    tmp_path, capsys, monkeypatch
):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TEST_DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    source, output = tmp_path / "saved", tmp_path / "derived"
    receipts = _write_report(source, _report())
    original = {name: (source / name).read_bytes() for name in receipts}
    assert review_costs.main(["--report", str(source), "--output", str(output)]) == 0
    assert "context-off | $0.03" in capsys.readouterr().out
    summary = json.loads((output / "costs.json").read_text())
    assert summary["model_calls_made"] == 0
    assert summary["provenance"]["input_sha256"]["results.json"] == receipts["results.json"]
    assert "not an invoice or quality verdict" in (output / "costs.txt").read_text()
    assert all((source / name).read_bytes() == raw for name, raw in original.items())
    with pytest.raises(SystemExit) as existing:
        review_costs.main(["--report", str(source), "--output", str(output)])
    assert existing.value.code == 2


@pytest.mark.parametrize(
    "problem", ["receipt", "attempt", "event-price", "outside", "response-model"]
)
def test_saved_artifacts_are_bound_before_cost_comparison(tmp_path, problem):
    report = _report()
    source = tmp_path / "saved"
    _write_report(source, report)
    receipts = json.loads((source / "receipts.json").read_text())
    name = "context-off/calls/000001.exchange.json"
    if problem == "receipt":
        (source / name).write_text("{}")
    elif problem == "outside":
        report["cases"][0]["raw_exchange_files"] = ["../outside.json"]
        name = "results.json"
        (source / name).write_text(json.dumps(report))
    elif problem == "attempt":
        name = "context-off/calls/000001.attempt.json"
        value = json.loads((source / name).read_text())
        value["usage_based_usd"] = "0.99"
        (source / name).write_text(json.dumps(value))
    else:
        exchange = json.loads((source / name).read_text())
        if problem == "event-price":
            exchange["event"]["cost_usd"] = "0.99"
        else:
            exchange["response"]["model"] = "other-model"
        (source / name).write_text(json.dumps(exchange))
    if problem != "receipt":
        receipts[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
        (source / "receipts.json").write_text(json.dumps(receipts))
    if problem == "response-model":
        summary = review_costs.load_costs(source)
        assert summary["comparisons"]["context-off"]["workflow_cost_divided_by_standalone"] is None
    else:
        with pytest.raises(review_costs.CostInputError):
            review_costs.load_costs(source)
