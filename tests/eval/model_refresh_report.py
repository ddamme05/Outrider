"""Summarize repeated refresh observations without turning missing data into wins.

Three complete repetitions give a two-of-three analyze recall verdict. Cost comparisons
need usage-priced calls throughout; qualitative summary review remains separate.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

REPORT_SCHEMA = "model-refresh-v1"


def summarize(candidate: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    from .model_refresh_regrade import V1_CONTRACT
    from .review_quality import MEASUREMENT_CONTRACT_V2

    if plan.get("measurement_contract") == MEASUREMENT_CONTRACT_V2:
        return _summarize_v2(candidate, plan)
    if plan.get("measurement_contract") != V1_CONTRACT:
        raise ValueError("unknown refresh measurement contract")
    return _summarize_v1(candidate, plan)


def _summarize_v1(candidate: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    cases = [row for row in candidate["cases"] if row["phase"] == "analyze"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[row["fixture"]].append(row)
    expected_paths = {row["fixture"] for row in plan["scenarios"]}
    complete = (
        plan["repetitions"] == 3
        and set(grouped) == expected_paths
        and all(
            len(rows) == 3 and {row["rep"] for row in rows} == {0, 1, 2}
            for rows in grouped.values()
        )
    )
    per_fixture = {}
    for path, rows in grouped.items():
        denominators = {row["model_judged"]["expected"] for row in rows}
        if len(denominators) != 1:
            complete = False
        expected = rows[0]["model_judged"]["expected"]
        extras = [row["model_judged"]["extras_advisory"] for row in rows]
        alarms = [row["model_judged"]["false_alarms"] for row in rows]
        per_fixture[path] = {
            "track": rows[0]["track"],
            "model_expected": expected,
            "model_recall_majority": None
            if not expected
            else sum(row["model_judged"]["matched"] == expected for row in rows) >= 2,
            "false_alarm_majority": None
            if alarms[0] is None
            else sum(bool(value) for value in alarms) >= 2,
            "false_alarm_total": sum(value for value in alarms if value is not None),
            "extra_total": sum(extras),
            "extra_max": max(extras),
            "response_rejections": sum(row["response_rejections"] for row in rows),
            "proposal_rejections": sum(row["proposal_rejections"] for row in rows),
        }
    calls = candidate["calls"]
    expected_calls = [f"compat:{path}" for path in plan["compatibility_cases"]]
    for rep in range(plan["repetitions"]):
        if plan["phase"] in {"analyze", "all"}:
            expected_calls.extend(f"analyze:{rep}:{path}" for path in expected_paths)
        if plan["phase"] in {"nodes", "all"}:
            expected_calls.extend(f"nodes:{rep}:{name}" for name in plan["node_cases"])
    collection_complete = (
        candidate["status"] == "completed"
        and Counter(call["scenario"] for call in calls) == Counter(expected_calls)
        and all(call["status"] == "completed" and call["persisted_exchange"] for call in calls)
    )
    truncated = sum(row.get("truncated", False) for row in calls)
    latencies = [row["wall_ms"] for row in calls if row["status"] == "completed"]
    known = [Decimal(row["usage_based_usd"]) for row in calls if row["usage_based_usd"] is not None]
    node_rows = [row for row in candidate["cases"] if row["phase"] == "nodes"]
    nodes = {}
    for name in plan["node_cases"]:
        rows = [row["result"] for row in node_rows if row["name"] == name]
        nodes[name] = {
            "repetitions": len(rows),
            "issues_majority": sum(bool(row["issues"]) for row in rows) >= 2,
            "failed_repetitions": [
                row["rep"] for row in node_rows if row["name"] == name and row["result"]["issues"]
            ],
            "issues_count": sum(bool(row["issues"]) for row in rows),
            "dropped_files": sorted(
                {path for row in rows for path in row["grade"].get("dropped_files", ())}
            ),
            "under_risk_count": sum(bool(row["grade"].get("under_risked")) for row in rows),
            "qualitative_pending": any(row.get("qualitative_review") == "pending" for row in rows),
            "valid_repetitions": {row["rep"] for row in node_rows if row["name"] == name}
            == {0, 1, 2}
            and len(rows) == 3
            and all(row.get("n_calls") == 1 for row in rows),
            "triage": {
                metric: (
                    min
                    if metric in {"tier_accuracy", "dimension_recall", "dimension_precision"}
                    else max
                )(row["grade"][metric] for row in rows)
                for metric in (
                    "tier_accuracy",
                    "dimension_recall",
                    "dimension_precision",
                    "n_dropped_from_analysis",
                    "n_deep_downgraded",
                    "n_overtiered",
                )
            }
            if rows and name.startswith("triage/")
            else None,
        }
    return {
        "quality_complete": complete and collection_complete and not truncated,
        "collection_complete": collection_complete,
        "truncated_calls": truncated,
        "nodes_complete": plan["repetitions"] == 3
        and bool(nodes)
        and all(row["valid_repetitions"] for row in nodes.values())
        and collection_complete
        and not truncated,
        "per_fixture": per_fixture,
        "nodes": nodes,
        "usage_based_usd": str(sum(known, Decimal(0))) if known else None,
        "cost_complete": collection_complete and bool(calls) and len(known) == len(calls),
        "refusals": sum(row.get("finish_reason") == "refusal" for row in calls),
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_complete": collection_complete and bool(calls) and len(latencies) == len(calls),
    }


def comparisons(report: dict[str, Any]) -> list[dict[str, Any]]:
    from .model_refresh_regrade import V1_CONTRACT
    from .review_quality import MEASUREMENT_CONTRACT_V2

    if report["plan"].get("measurement_contract") == MEASUREMENT_CONTRACT_V2:
        return _comparisons_v2(report)
    if report["plan"].get("measurement_contract") != V1_CONTRACT:
        raise ValueError("unknown refresh measurement contract")
    return _comparisons_v1(report)


def _comparisons_v1(report: dict[str, Any]) -> list[dict[str, Any]]:
    plan = report["plan"]
    candidates = report["candidates"]
    for candidate in candidates:
        candidate["summary"] = summarize(candidate, plan)
    decisions = []
    for candidate in candidates:
        role = candidate["target"]["role"]
        if not role.startswith("candidate"):
            continue
        baseline_role = role.replace("candidate", "baseline", 1)
        baselines = [
            row
            for row in candidates
            if row["target"]["role"] == baseline_role
            and (
                baseline_role != "baseline" or row["target"]["host"] == candidate["target"]["host"]
            )
        ]
        decision: dict[str, Any] = {
            "candidate": candidate["target"],
            "status": "inconclusive",
            "issues": [],
            "limits": [],
        }
        decisions.append(decision)
        if len(baselines) != 1:
            decision["issues"].append("exactly one matching explicit baseline role is required")
            continue
        baseline = baselines[0]
        decision["baseline"] = baseline["target"]
        a, b = baseline["summary"], candidate["summary"]
        if report.get("evidence_kind") != "live-provider":
            decision["issues"].append(
                "injected providers establish instrumentation, not model quality"
            )
            continue
        decision["cost"] = "inconclusive"
        if a["cost_complete"] and b["cost_complete"]:
            decision["cost"] = (
                "lower"
                if Decimal(b["usage_based_usd"]) < Decimal(a["usage_based_usd"])
                else "not-lower"
            )
        decision["latency"] = "inconclusive"
        if a["latency_complete"] and b["latency_complete"]:
            decision["latency"] = (
                "lower" if b["latency_ms_median"] < a["latency_ms_median"] else "not-lower"
            )
        decision["node_quality"] = "inconclusive"
        if a["nodes_complete"] and b["nodes_complete"]:
            node_issues = []
            for name, cell in a["nodes"].items():
                other = b["nodes"][name]
                if other["issues_count"] > cell["issues_count"]:
                    node_issues.append(name)
                if other["under_risk_count"] > cell["under_risk_count"] or set(
                    other["dropped_files"]
                ) - set(cell["dropped_files"]):
                    node_issues.append(f"{name}: lost routing safety")
                if cell["triage"] is not None:
                    for metric in ("tier_accuracy", "dimension_recall", "dimension_precision"):
                        if other["triage"][metric] < cell["triage"][metric]:
                            node_issues.append(f"{name}: {metric}")
                    for metric in ("n_dropped_from_analysis", "n_deep_downgraded", "n_overtiered"):
                        if other["triage"][metric] > cell["triage"][metric]:
                            node_issues.append(f"{name}: {metric}")
            decision["node_quality"] = "held" if not node_issues else "regressed"
            decision["node_issues"] = node_issues
            decision["limits"].append("summary prose still needs qualitative review")
        if not a["quality_complete"] or not b["quality_complete"]:
            decision["issues"].append(
                "quality requires three complete repetitions over the same fixture set"
            )
            if a["nodes_complete"] and b["nodes_complete"]:
                decision["status"] = "node-review-required"
            continue
        for path, cell in a["per_fixture"].items():
            other = b["per_fixture"][path]
            if cell["model_expected"] != other["model_expected"]:
                decision["issues"].append(f"model-only denominator changed: {path}")
            if cell["model_recall_majority"] is False:
                decision["limits"].append(
                    f"baseline missed a known model-dependent finding: {path}"
                )
            if cell["model_recall_majority"] and not other["model_recall_majority"]:
                decision["issues"].append(f"model recall regressed: {path}")
            if not cell["false_alarm_majority"] and other["false_alarm_majority"]:
                decision["issues"].append(f"safe-code false alarms increased: {path}")
            if other["track"] == "holdout-recall" and not other["model_recall_majority"]:
                decision["issues"].append(f"fresh SQL recall failed: {path}")
            if other["track"] == "holdout-safe" and other["false_alarm_total"]:
                decision["issues"].append(f"fresh safe SQL produced a false alarm: {path}")
            for metric in ("response_rejections", "proposal_rejections"):
                if other[metric] > cell[metric]:
                    decision["issues"].append(f"{metric} increased: {path}")
        decision["quality"] = "regressed-or-unproven" if decision["issues"] else "held"
        decision["status"] = "review-required"
        decision["promotion"] = (
            "No automatic default change; node coverage and qualitative summary review "
            "are separate."
        )
    return decisions


def compare_saved(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    from .model_refresh_regrade import V1_CONTRACT
    from .review_quality import MEASUREMENT_CONTRACT_V2

    versions = (
        first["plan"].get("measurement_contract"),
        second["plan"].get("measurement_contract"),
    )
    if any(version not in {V1_CONTRACT, MEASUREMENT_CONTRACT_V2} for version in versions):
        raise ValueError("unknown refresh measurement contract")
    if MEASUREMENT_CONTRACT_V2 in versions:
        if first["plan"].get("suite", "legacy") != second["plan"].get("suite", "legacy"):
            raise ValueError("incomparable saved analysis suites")
        if first["plan"].get("dataset") != second["plan"].get("dataset"):
            raise ValueError("incomparable saved dataset identities")
        first_inputs, second_inputs = (
            report["plan"].get("measurement_inputs") for report in (first, second)
        )
        if not first_inputs or not second_inputs or first_inputs != second_inputs:
            raise ValueError("missing or incomparable v2 measurement_inputs")
    return _compare_saved_v1(first, second)


def _compare_saved_v1(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare separately collected runs only when their actual input contracts agree."""
    if first.get("schema") != REPORT_SCHEMA or second.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported saved refresh schema")
    identities = (
        "schema",
        "measurement_contract",
        "phase",
        "repetitions",
        "runtime",
        "inputs",
        "scenarios",
        "node_cases",
        "prompt_version",
        "prompt_digest",
        "prompt_identity_recipe",
        "pricing_version",
    )
    missing = [
        name for name in identities if not first["plan"].get(name) or not second["plan"].get(name)
    ]
    if missing:
        raise ValueError("missing saved input identities: " + ", ".join(missing))
    differences = [
        name for name in identities if first["plan"].get(name) != second["plan"].get(name)
    ]
    if differences:
        raise ValueError("incomparable saved inputs: " + ", ".join(differences))
    if first.get("evidence_kind") != second.get("evidence_kind"):
        raise ValueError("incomparable evidence kinds")
    merged = {
        "schema": first["schema"],
        "evidence_kind": first["evidence_kind"],
        "plan": first["plan"],
        "candidates": [*first["candidates"], *second["candidates"]],
    }
    merged["comparisons"] = comparisons(merged)
    return merged


def markdown(report: dict[str, Any]) -> str:
    from .model_refresh_regrade import V1_CONTRACT
    from .review_quality import MEASUREMENT_CONTRACT_V2

    if report["plan"].get("measurement_contract") == MEASUREMENT_CONTRACT_V2:
        return _markdown_v2(report)
    if report["plan"].get("measurement_contract") != V1_CONTRACT:
        raise ValueError("unknown refresh measurement contract")
    return _markdown_v1(report)


def _markdown_v1(report: dict[str, Any]) -> str:
    lines = [
        "# Model refresh",
        "",
        (
            "Usage-based costs are not invoices. Missing or unpriced calls cannot establish "
            "a cost winner."
        ),
        "",
        "| Model | Role | Collection | Usage USD | Cost complete | Median ms |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in report["candidates"]:
        summary = row["summary"]
        lines.append(
            f"| {row['target']['model']} | {row['target']['role']} | {row['status']} | "
            f"{summary['usage_based_usd']} | {summary['cost_complete']} | "
            f"{summary['latency_ms_median']} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for row in report["comparisons"]:
        lines.append(
            f"- {row['candidate']['model']}: {row['status']}; "
            f"quality={row.get('quality', 'inconclusive')}, "
            f"cost={row.get('cost', 'inconclusive')}. {'; '.join(row['issues'])}"
        )
    lines.extend(
        [
            "",
            (
                "Review raw findings and summary prose before changing defaults. "
                "Deterministic producer coverage is separated from JUDGED model quality."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _summarize_v2(candidate: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Summarize complete slots without making unknown context a quality pass."""
    from .model_refresh_regrade import case_slot, planned_slots

    expected_calls = planned_slots(plan)
    calls = candidate["calls"]
    complete = (
        candidate["status"] == "completed"
        and Counter(call["scenario"] for call in calls) == Counter(expected_calls)
        and Counter(case_slot(case) for case in candidate["cases"]) == Counter(expected_calls)
        and all(
            call["status"] == "completed"
            and call["persisted_exchange"]
            and not call.get("truncated", False)
            and call.get("finish_reason") in {"stop", "end_turn"}
            for call in calls
        )
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in candidate["cases"]:
        if case["phase"] == "analyze":
            grouped[case["fixture"]].append(case)
    paths = {row["fixture"] for row in plan["scenarios"]}
    repeated = (
        plan["repetitions"] == 3
        and set(grouped) == paths
        and all(
            len(rows) == 3 and {row["rep"] for row in rows} == {0, 1, 2}
            for rows in grouped.values()
        )
    )
    per_fixture = {}
    for path, rows in grouped.items():
        expected = {row["model_dependent"]["expected"] for row in rows}
        unknown = sum(row.get("measurement_v2", {}).get("unscored_expected", 1) for row in rows)
        pending = sum(
            row.get("measurement_v2", {}).get("pending_model_expected", 0) for row in rows
        )
        per_fixture[path] = {
            "track": rows[0]["track"],
            "model_expected": next(iter(expected)) if len(expected) == 1 else None,
            "grounded_recall_majority": None
            if unknown
            or pending
            or len(rows) != 3
            or {row["rep"] for row in rows} != {0, 1, 2}
            or len(expected) != 1
            or not next(iter(expected))
            else sum(
                row["model_dependent"]["matched"] == row["model_dependent"]["expected"]
                for row in rows
            )
            >= 2,
            "unscored_expected": unknown,
            "pending_expected": sum(
                row.get("measurement_v2", {}).get("pending_expected", 0) for row in rows
            ),
            "pending_model_expected": pending,
            "safe_model_emissions": sum(
                row.get("measurement_v2", {}).get("safe_model_emissions") or 0 for row in rows
            ),
            "safe_product_emissions": sum(
                row.get("measurement_v2", {}).get("safe_product_emissions") or 0 for row in rows
            ),
            "response_rejections": sum(row["response_rejections"] for row in rows),
            "proposal_rejections": sum(row["proposal_rejections"] for row in rows),
            "semantic_adjudication": "pending",
        }
    known_costs = [
        Decimal(call["usage_based_usd"]) for call in calls if call["usage_based_usd"] is not None
    ]
    latencies = [call["wall_ms"] for call in calls if call["status"] == "completed"]
    nodes = [case for case in candidate["cases"] if case["phase"] == "nodes"]
    node_assessments = {}
    for name in plan["node_cases"]:
        rows = [case for case in nodes if case["name"] == name]
        assessments = [case["result"].get("measurement_v2") for case in rows]
        node_assessments[name] = {
            "repetitions": len(rows),
            "failed_repetitions": [
                case["rep"]
                for case, assessment in zip(rows, assessments, strict=True)
                if assessment and assessment["issues"]
            ],
            "issues": [
                issue for assessment in assessments if assessment for issue in assessment["issues"]
            ],
            "unscored": [
                gap for assessment in assessments if assessment for gap in assessment["unscored"]
            ],
            "missing_assessments": sum(assessment is None for assessment in assessments),
            "qualitative_review": "pending"
            if any(
                assessment and assessment["qualitative_review"] == "pending"
                for assessment in assessments
            )
            else None,
        }
    node_complete = plan["repetitions"] == 3 and all(
        Counter(case["rep"] for case in nodes if case["name"] == name)
        == Counter({0: 1, 1: 1, 2: 1})
        for name in plan["node_cases"]
    )
    return {
        "collection_complete": complete,
        "quality_complete": False,
        "analyze_repetitions_complete": complete and repeated,
        "node_repetitions_complete": complete and bool(nodes) and node_complete,
        "nodes": node_assessments,
        "per_fixture": per_fixture,
        "usage_based_usd": str(sum(known_costs, Decimal(0))) if known_costs else None,
        "cost_complete": complete and len(known_costs) == len(calls),
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_complete": complete and bool(calls) and len(latencies) == len(calls),
        "refusals": sum(
            call.get("error_type") == "LLMRefusalError" or call.get("finish_reason") == "refusal"
            for call in calls
        ),
        "semantic_adjudication": "pending",
        "precision_scope": (
            "Safe-case emissions are observations, not automatically proven false positives."
        ),
    }


def _comparisons_v2(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose bounded evidence while leaving semantic/default decisions unresolved."""
    for candidate in report["candidates"]:
        candidate["summary"] = _summarize_v2(candidate, report["plan"])
    decisions = []
    for candidate in report["candidates"]:
        role = candidate["target"]["role"]
        if not role.startswith("candidate"):
            continue
        baseline_role = role.replace("candidate", "baseline", 1)
        baselines = [
            row
            for row in report["candidates"]
            if row["target"]["role"] == baseline_role
            and (
                baseline_role != "baseline" or row["target"]["host"] == candidate["target"]["host"]
            )
        ]
        decision = {
            "candidate": candidate["target"],
            "status": "inconclusive",
            "quality": "requires-adjudication",
            "cost": "inconclusive",
            "latency": "inconclusive",
            "node_quality": "requires-adjudication",
            "issues": [
                "Grounded matches are bounded evidence; semantic quality and emitted "
                "safe-case findings require adjudication."
            ],
            "limits": [],
            "promotion": "No automatic model or default qualification.",
        }
        if len(baselines) == 1:
            baseline = baselines[0]
            decision["baseline"] = baseline["target"]
            actual_evidence = report.get("evidence_kind") in {
                "live-provider",
                "saved-response-reanalysis",
            }
            if (
                actual_evidence
                and candidate["summary"]["cost_complete"]
                and baseline["summary"]["cost_complete"]
            ):
                current_cost = Decimal(candidate["summary"]["usage_based_usd"])
                old_cost = Decimal(baseline["summary"]["usage_based_usd"])
                decision["cost"] = (
                    "lower"
                    if current_cost < old_cost
                    else "higher"
                    if current_cost > old_cost
                    else "equal"
                )
            if (
                actual_evidence
                and candidate["summary"]["latency_complete"]
                and baseline["summary"]["latency_complete"]
            ):
                current_ms = candidate["summary"]["latency_ms_median"]
                old_ms = baseline["summary"]["latency_ms_median"]
                decision["latency"] = (
                    "lower" if current_ms < old_ms else "higher" if current_ms > old_ms else "equal"
                )
            if (
                not candidate["summary"]["collection_complete"]
                or not baseline["summary"]["collection_complete"]
            ):
                decision["issues"].append(
                    "Requested call coverage is partial or duplicated; "
                    "no quality, cost or latency winner."
                )
            for path, current in candidate["summary"]["per_fixture"].items():
                previous = baseline["summary"]["per_fixture"].get(path)
                if not previous:
                    continue
                if (
                    previous["grounded_recall_majority"] is True
                    and current["grounded_recall_majority"] is False
                ):
                    decision["issues"].append(f"grounded recall decreased: {path}")
                if current["unscored_expected"]:
                    decision["limits"].append(f"unscored visible-context expectation: {path}")
        else:
            decision["issues"].append("Exactly one comparable role baseline is required.")
        decisions.append(decision)
    return decisions


def _markdown_v2(report: dict[str, Any]) -> str:
    plan = report["plan"]
    dataset = plan.get("dataset", {})
    lines = [
        "# Model refresh v2",
        "",
        f"Analysis suite: {plan.get('suite', 'legacy')}; "
        f"dataset: {dataset.get('id', 'original recorded corpus')}; "
        f"exposure: {dataset.get('exposure', 'not recorded')}.",
        "Fixed grounded routing: STANDARD tier, medium risk and all review dimensions; "
        "triage is not measured. Compatibility uses the three legacy cases."
        if plan.get("suite") == "grounded"
        else "Fixed PR routing: one selected changed file at STANDARD tier, medium risk "
        "and all review dimensions; other changed files SKIP. The full capture is retained, "
        "but only rendered scopes are model-visible. Compatibility uses the three legacy cases. "
        "Counterpart labels cover specified types and source ranges, not the entire PR."
        if plan.get("suite") == "real-pr"
        else "",
        "Bounded issue grounding, quote/location accuracy and visible context are separate. "
        "Semantic quality remains subject to adjudication; no automatic default qualification.",
        "",
        "Original recorded usage costs and timing are retained without new calls or repricing."
        if report.get("evidence_kind") == "saved-response-reanalysis"
        else "Usage-based costs are not invoices. "
        "Incomplete or unpriced coverage cannot win cost comparisons.",
        "",
        "| Model | Collection | Matches / supported expectations | Pending matches | "
        "Grounded majority / supported fixtures | Unscored context | "
        "Safe emissions model / product | Usage USD | Median ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        summary = candidate["summary"]
        fixtures = list(summary["per_fixture"].values())
        supported = [row for row in fixtures if row["model_expected"]]
        detected = sum(row["grounded_recall_majority"] is True for row in supported)
        observations = [
            case["model_dependent"] for case in candidate["cases"] if case["phase"] == "analyze"
        ]
        expected = sum(row["expected"] for row in observations)
        matched = sum(row["matched"] for row in observations)
        pending = sum(row.get("pending", 0) for row in observations)
        unknown = sum(row["unscored_expected"] for row in fixtures)
        model_emissions = sum(row["safe_model_emissions"] for row in fixtures)
        product_emissions = sum(row["safe_product_emissions"] for row in fixtures)
        cost = summary["usage_based_usd"] if summary["usage_based_usd"] is not None else "unknown"
        latency = (
            summary["latency_ms_median"] if summary["latency_ms_median"] is not None else "unknown"
        )
        lines.append(
            f"| {candidate['target']['model']} | {candidate['status']} | "
            f"{matched}/{expected} | {pending} | {detected}/{len(supported)} | "
            f"{unknown} | {model_emissions}/{product_emissions} | "
            f"{cost} | {latency} |"
        )
        for name, node in summary["nodes"].items():
            if node["repetitions"]:
                lines.append(
                    f"\n{name}: failed repetitions {node['failed_repetitions']}; "
                    f"unscored checks {len(node['unscored'])}; "
                    f"qualitative review {node['qualitative_review'] or 'not required'}."
                )
    lines.extend(
        [
            "",
            "Safe-case emissions are not automatically classified as false positives. "
            "Unknown context and unsupported claims do not become passes.",
            "",
            "## Comparisons",
            "",
        ]
    )
    for decision in report["comparisons"]:
        lines.append(
            f"- {decision['candidate']['model']}: quality={decision['quality']}, "
            f"cost={decision['cost']}, latency={decision['latency']}. "
            + " ".join([*decision["issues"], *decision["limits"]])
        )
    return "\n".join(lines) + "\n"
