"""Compare recorded standalone and workflow costs without making model calls.

Use saved usage-priced amounts rather than recalculating token prices. Unpriced
reservations remain separate upper bounds. A lower bill is not a quality win.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SCHEMA = "review-costs-v1"
STANDALONE = "direct-one-shot"


class CostInputError(ValueError):
    """Saved cost records are malformed or disagree with their source artifacts."""


def _money(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise CostInputError("cost must be a nonnegative finite amount")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise CostInputError("cost must be a nonnegative finite amount") from exc
    if not amount.is_finite() or amount < 0:
        raise CostInputError("cost must be a nonnegative finite amount")
    return amount


def _integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise CostInputError("counts and repetitions must be nonnegative integers")
    return value


def _coverage(row: dict[str, Any], changed_files: list[str]) -> dict[str, Any]:
    if row["arm"] == STANDALONE:
        return {"scope": "complete captured source in one request; semantic coverage unproven"}
    examined, skipped = set(), []
    for event in row.get("diagnostic_events", []):
        payload = event.get("payload", {})
        if event.get("event_type") != "file_examination" or payload.get("node_id") != "analyze":
            continue
        examined.add(payload["file_path"])
        if payload.get("skip_reason"):
            skipped.append({"file_path": payload["file_path"], "reason": payload["skip_reason"]})
    return {
        "scope": "selected scopes; event presence does not establish whole-file "
        "or semantic coverage",
        "diagnostics_status": row.get("diagnostics_status", "unavailable"),
        "recorded_analysis_skips": skipped,
        "changed_files_without_analysis_event": sorted(set(changed_files) - examined),
    }


def _trial_cost(row: dict[str, Any]) -> dict[str, Any]:
    known, uncertain, unpriced = Decimal(0), Decimal(0), 0
    calls = row["calls"]
    if not isinstance(calls, list):
        raise CostInputError("each review needs a call inventory")
    for call in calls:
        retained = _money(call["budget_retained_upper_usd"])
        reserved = _money(call["worst_case_reserved_usd"])
        if retained > reserved:
            raise CostInputError("retained call cost exceeds its reservation")
        if call["cost_status"] == "usage-priced":
            if call["persisted_exchange"] is not True:
                raise CostInputError("usage-priced calls require a persisted exchange")
            amount = _money(call["usage_based_usd"])
            if amount != retained:
                raise CostInputError("priced call and retained budget disagree")
            known += amount
        elif call["cost_status"] == "uncosted" and call["usage_based_usd"] is None:
            uncertain += retained
            unpriced += 1
        else:
            raise CostInputError("call pricing status and usage cost disagree")
    complete = (
        row["status"] == "completed"
        and row.get("diagnostics_status") in {"captured", "not-applicable"}
        and not row.get("close_error_type")
        and not row.get("response_model_mismatch", False)
        and (len(calls) == 1 if row["arm"] == STANDALONE else bool(calls))
        and all(
            call["status"] == "completed"
            and not call["truncated"]
            and not call.get("error_type")
            and call["finish_reason"] in {"stop", "end_turn"}
            for call in calls
        )
    )
    return {
        "status": row["status"],
        "complete": complete,
        "calls": len(calls),
        "known_usage_based_usd": str(known),
        "unpriced_upper_usd": str(uncertain),
        "unpriced_calls": unpriced,
        "whole_driver_wall_ms": row.get("whole_driver_wall_ms"),
    }


def _difference(workflow: dict[str, Any], standalone: dict[str, Any]) -> dict[str, Any]:
    eligible = all(
        item["complete"] and item["unpriced_calls"] == 0 for item in (workflow, standalone)
    )
    direct_cost = _money(standalone["known_usage_based_usd"])
    cost = _money(workflow["known_usage_based_usd"])
    return {
        "recorded_expense_comparable": eligible,
        "workflow_minus_standalone_usd": str(cost - direct_cost) if eligible else None,
        "workflow_cost_divided_by_standalone": str(cost / direct_cost)
        if eligible and direct_cost > 0
        else None,
    }


def summarize_costs(report: dict[str, Any]) -> dict[str, Any]:
    """Reconcile collector accounting and compare the exact planned case/repetition pairs."""
    if report["schema"] != "review-comparison-v1":
        raise CostInputError("unsupported workflow comparison schema")
    plan = report["plan"]
    if not plan["schedule"]:
        raise CostInputError("comparison schedule is empty")
    arms = tuple(plan.get("arms", plan["schedule"][0]["order"]))
    if len(arms) != 3 or len(set(arms)) != 3 or STANDALONE not in arms:
        raise CostInputError("comparison needs standalone and two distinct workflow arms")
    cases = {case["id"]: case for case in plan["cases"]}
    if len(cases) != len(plan["cases"]):
        raise CostInputError("planned case identifiers must be unique")
    slots = []
    for scheduled in plan["schedule"]:
        slot = (scheduled["case_id"], _integer(scheduled["rep"]))
        if slot in slots or slot[0] not in cases:
            raise CostInputError("schedule has duplicate or unknown cases")
        if len(scheduled["order"]) != len(arms) or set(scheduled["order"]) != set(arms):
            raise CostInputError("scheduled arms do not match the comparison")
        slots.append(slot)
    rows, trial_costs, call_ids, event_ids = {}, {}, set(), set()
    for row in report["cases"]:
        key = (row["arm"], row["case_id"], _integer(row["rep"]))
        if key in rows or key[0] not in arms or key[1:] not in slots:
            raise CostInputError("review rows are duplicate or outside the planned comparison")
        local_ordinals: set[str] = set()
        for call in row["calls"]:
            prefix = f"{row['rep']}:{row['case_id']}:"
            scenario = call.get("scenario", "")
            suffix = scenario[len(prefix) :].split(":") if scenario.startswith(prefix) else []
            if (
                len(suffix) != 2
                or suffix[0] not in {"triage", "analyze", "trace", "synthesize"}
                or re.fullmatch(r"0|[1-9][0-9]*", suffix[1]) is None
                or suffix[1] in local_ordinals
            ):
                raise CostInputError("call scenario does not belong to this case and repetition")
            local_ordinals.add(suffix[1])
            identity = (row["arm"], _integer(call["index"]))
            if identity in call_ids or identity[1] == 0:
                raise CostInputError("call inventory repeats an occurrence")
            call_ids.add(identity)
            if call["persisted_exchange"]:
                event_id = call["audit_event_id"]
                if not isinstance(event_id, str) or not event_id or event_id in event_ids:
                    raise CostInputError("persisted calls must have distinct event identifiers")
                event_ids.add(event_id)
        rows[key] = row
        trial_costs[key] = _trial_cost(row)
        contiguous = local_ordinals == {str(index) for index in range(len(row["calls"]))}
        trial_costs[key]["call_ordinals_contiguous"] = contiguous
        if not contiguous:
            trial_costs[key]["complete"] = False
    totals = {}
    for arm in arms:
        costs = [value for key, value in trial_costs.items() if key[0] == arm]
        known = sum((_money(item["known_usage_based_usd"]) for item in costs), Decimal(0))
        uncertain = sum((_money(item["unpriced_upper_usd"]) for item in costs), Decimal(0))
        calls = sum(item["calls"] for item in costs)
        budget = report["budgets"][arm]
        if (known, uncertain, calls) != (
            _money(budget["known_usage_based_usd"]),
            _money(budget["uncertain_budget_upper_usd"]),
            _integer(budget["calls"]),
        ):
            raise CostInputError("review call inventory does not reconcile with the arm budget")
        totals[arm] = {
            "known_usage_based_usd": str(known),
            "unpriced_upper_usd": str(uncertain),
            "unpriced_calls": sum(item["unpriced_calls"] for item in costs),
            "calls": calls,
            "planned_reviews": len(slots),
            "recorded_reviews": len(costs),
            "completed_reviews": sum(item["complete"] for item in costs),
            "complete": report["status"] == "completed"
            and len(costs) == len(slots)
            and all(item["complete"] for item in costs),
        }
    pairs = []
    for case_id, rep in slots:
        costs = {arm: trial_costs.get((arm, case_id, rep)) for arm in arms}
        comparisons = {}
        for arm in arms:
            if arm == STANDALONE:
                continue
            comparisons[arm] = (
                _difference(costs[arm], costs[STANDALONE])
                if all(costs[item] is not None for item in (arm, STANDALONE))
                else {
                    "recorded_expense_comparable": False,
                    "workflow_minus_standalone_usd": None,
                    "workflow_cost_divided_by_standalone": None,
                }
            )
        pairs.append(
            {
                "case_id": case_id,
                "rep": rep,
                "costs": costs,
                "comparisons": comparisons,
                "coverage": {
                    arm: _coverage(
                        rows[(arm, case_id, rep)], cases[case_id].get("changed_files", [])
                    )
                    for arm in arms
                    if (arm, case_id, rep) in rows
                },
            }
        )
    return {
        "schema": SCHEMA,
        "mode": "offline-recorded-costs",
        "model_calls_made": 0,
        "target": plan["target"],
        "evidence_kind": report.get("evidence_kind", "unknown"),
        "pricing_version": plan.get("pricing_version"),
        "totals": totals,
        "comparisons": {
            arm: _difference(totals[arm], totals[STANDALONE]) for arm in arms if arm != STANDALONE
        },
        "paired_reviews": pairs,
        "qualification": "Recorded expense only; no quality winner or equal-coverage claim.",
        "cost_basis": "Saved usage-priced call amounts, including provider cache billing. "
        "No repricing or token-cost recomputation; unpriced upper bounds are not spend.",
        "timing_note": report.get("timing_note"),
        "configuration": {
            key: plan.get(key)
            for key in (
                "dataset",
                "source_access",
                "scope",
                "latency_configuration",
                "workflow_analyze_concurrency",
            )
        },
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CostInputError("duplicate JSON field")
        result[key] = value
    return result


def _json(data: bytes) -> Any:
    return json.loads(data, object_pairs_hook=_unique_object)


def load_costs(directory: Path) -> dict[str, Any]:
    """Bind derived costs to receipts, attempts and completed exchange event amounts."""
    receipt_bytes = (directory / "receipts.json").read_bytes()
    receipts = _json(receipt_bytes)
    inputs = {"receipts.json": hashlib.sha256(receipt_bytes).hexdigest()}

    def artifact(name: str) -> Any:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or name not in receipts:
            raise CostInputError("cost artifact is not receipt-covered within the report")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != receipts[name]:
            raise CostInputError("cost artifact differs from its receipt")
        inputs[name] = digest
        return _json(data)

    report = artifact("results.json")
    if report["plan"] != artifact("plan.json"):
        raise CostInputError("saved result and canonical plan differ")
    model = report["plan"]["target"]["model"]
    for row in report["cases"]:
        expected_exchanges = set()
        row["response_model_mismatch"] = False
        for call in row["calls"]:
            prefix = f"{row['arm']}/calls/{_integer(call['index']):06d}"
            if artifact(prefix + ".attempt.json") != call:
                raise CostInputError("saved attempt differs from the review call inventory")
            if call["persisted_exchange"]:
                name = prefix + ".exchange.json"
                expected_exchanges.add(name)
                exchange = artifact(name)
                if (
                    exchange["call_index"] != call["index"]
                    or exchange["event"]["event_id"] != call["audit_event_id"]
                    or exchange["scenario"] != call["scenario"]
                ):
                    raise CostInputError("exchange and call identity differ")
                if exchange["request"]["model"] != model:
                    raise CostInputError("call requested a model outside the comparison")
                if row["arm"] == STANDALONE:
                    case = next(
                        case for case in report["plan"]["cases"] if case["id"] == row["case_id"]
                    )
                    expected_user = case.get("direct_user_sha256")
                    if (
                        expected_user is not None
                        and hashlib.sha256(exchange["request"]["user_prompt"].encode()).hexdigest()
                        != expected_user
                    ):
                        raise CostInputError(
                            "standalone source request differs from the planned case"
                        )
                if (
                    exchange["event"]["model"] != model
                    or exchange["target"] != report["plan"]["target"]
                ):
                    raise CostInputError(
                        "exchange host or model identity differs from the comparison"
                    )
                row["response_model_mismatch"] |= exchange["response"]["model"] != model
                if call["cost_status"] == "usage-priced" and _money(
                    exchange["event"]["cost_usd"]
                ) != _money(call["usage_based_usd"]):
                    raise CostInputError("exchange event and recorded usage cost differ")
        if (
            len(row["raw_exchange_files"]) != len(expected_exchanges)
            or set(row["raw_exchange_files"]) != expected_exchanges
        ):
            raise CostInputError("review exchange inventory differs from recorded attempts")
    summary = summarize_costs(report)
    summary["provenance"] = {
        "source_report": str(directory),
        "input_sha256": inputs,
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "trust": "Local hash consistency; receipt authenticity and finding truth "
        "are not established.",
    }
    return summary


def render_text(summary: dict[str, Any]) -> str:
    lines = [
        "Recorded model-call cost; not an invoice or quality verdict.",
        "Arm | Usage-priced USD | Unpriced upper USD | Calls | Complete / planned",
    ]
    for arm, item in summary["totals"].items():
        lines.append(
            f"{arm} | ${item['known_usage_based_usd']} | ${item['unpriced_upper_usd']} "
            f"| {item['calls']} | {item['completed_reviews']}/{item['planned_reviews']}"
        )
    for pair in summary["paired_reviews"]:
        lines.append(f"\n{pair['case_id']} repetition {pair['rep']}:")
        for arm, comparison in pair["comparisons"].items():
            costs, direct = pair["costs"][arm], pair["costs"][STANDALONE]
            workflow_usd = "missing" if costs is None else "$" + costs["known_usage_based_usd"]
            direct_usd = "missing" if direct is None else "$" + direct["known_usage_based_usd"]
            ratio = comparison["workflow_cost_divided_by_standalone"]
            label = (
                f"{Decimal(ratio):.3f}x recorded standalone expense"
                if ratio is not None
                else ("ratio unavailable: incomplete, unpriced or zero-cost baseline")
            )
            lines.append(f"  {arm}: {workflow_usd}; standalone: {direct_usd}; {label}")
            coverage = pair["coverage"].get(arm, {})
            skips = Counter(item["reason"] for item in coverage.get("recorded_analysis_skips", []))
            if skips:
                lines.append(
                    "    Recorded analysis skips: "
                    + ", ".join(f"{key}={value}" for key, value in sorted(skips.items()))
                )
    lines.extend(
        [
            "",
            summary["qualification"],
            "Cheaper selected-scope reviews may omit code shown to standalone Claude.",
            "Upper bounds are headroom for unpriced calls, not observed charges.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="New derived-output directory; source stays unchanged."
    )
    args = parser.parse_args(argv)
    try:
        summary = load_costs(args.report)
        rendered = render_text(summary)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "costs.json").write_text(json.dumps(summary, indent=2) + "\n")
            (args.output / "costs.txt").write_text(rendered)
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        reason = str(exc) if isinstance(exc, CostInputError) else type(exc).__name__
        parser.error(f"cannot compare recorded costs: {reason}")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
