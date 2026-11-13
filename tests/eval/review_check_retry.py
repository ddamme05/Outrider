"""Collect only missing candidate checks from an unfinished saved comparison.

Planning is free. Paid collection is explicit, serial, and never repeats original
discovery or a prior retry directory. This produces assessments, not a resumed
full-workflow review, summary, publication or quality verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from outrider.agent.candidate_checks import VERSION, assess_candidate_response
from outrider.eval_support import require_eval_mode
from outrider.llm.host_profiles import resolve_host_identity
from outrider.llm.pricing import PRICING_VERSION
from outrider.policy.publish_eligibility import verification_hold_finding_ids
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.native_candidate import CandidateAssessmentRef

from .model_refresh import (
    ROOT,
    Budget,
    BudgetedProvider,
    ExchangeRecorder,
    create_provider,
    input_hashes,
    maximum_rates,
    reservation_usd,
    write_json,
)
from .review_check_inputs import SavedCheckInputs, load_saved_checks, verify_saved_inputs
from .review_diagnostics_inputs import fields

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.llm.base import LLMProvider, LLMRequest

SCHEMA = "review-check-retry-v1"


def _request_digest(request: LLMRequest) -> str:
    return hashlib.sha256(json.dumps(fields(request), sort_keys=True).encode()).hexdigest()


def build_plan(source: SavedCheckInputs) -> dict[str, Any]:
    """Prepare the same missing questions without constructing a provider."""
    if source.policy_version != ACTIVE_POLICY_VERSION:
        raise ValueError("historical policy permits saved-answer replay, not new check generation")
    if source.target.host != "anthropic":
        raise ValueError("check-only retry currently supports captured Anthropic comparisons")
    profile, reasoning, contract = resolve_host_identity(
        source.target.host, reasoning=False, models=(source.target.model,)
    )
    original = json.loads((source.directory / "plan.json").read_bytes())
    if (profile, reasoning, contract) != tuple(
        original[name] for name in ("profile_id", "reasoning_enabled", "profile_contract_digest")
    ):
        raise ValueError("current provider identity differs from saved discovery")
    candidates = {ref.candidate_id: ref for case in source.cases for ref in case.candidates}
    checks: list[dict[str, Any]] = [
        {
            "case_id": case.case_id,
            "rep": case.rep,
            "candidate_id": prepared.check.candidate_id,
            "file_path": candidates[prepared.check.candidate_id].file_path,
            "original_finding_type": candidates[prepared.check.candidate_id].finding_type,
            "original_priority": candidates[prepared.check.candidate_id].severity,
            "attempt_id": str(prepared.check.attempt_id),
            "request_sha256": _request_digest(prepared.request),
            "snippet_count": len(prepared.check.snippets),
            "request_text_bytes": sum(
                len(text.encode())
                for text in (
                    prepared.request.system_prompt,
                    prepared.request.user_prompt,
                    prepared.request.response_schema_json or "",
                )
            ),
            "max_output_tokens": prepared.request.max_tokens,
            "reservation_usd": str(reservation_usd(source.target, prepared.request.max_tokens)),
        }
        for case in source.cases
        for prepared in case.prepared_checks
    ]
    if not checks or len({item["attempt_id"] for item in checks}) != len(checks):
        raise ValueError("retry requires distinct missing prepared attempts")
    retry_id = hashlib.sha256(json.dumps(sorted(candidates)).encode()).hexdigest()
    input_rate, output_rate = maximum_rates(source.target)
    illustration = sum(
        (
            (Decimal(item["request_text_bytes"]) / 4 + 512) * input_rate
            + item["max_output_tokens"] * output_rate
            for item in checks
        ),
        Decimal(0),
    )
    return {
        "schema": SCHEMA,
        "mode": "free-plan",
        "model_calls_made": 0,
        "source_report": str(source.directory),
        "retry_id": retry_id,
        "source_hashes": source.hashes,
        "target": asdict(source.target),
        "profile_id": profile,
        "reasoning_enabled": reasoning,
        "profile_contract_digest": contract,
        "pricing_version": PRICING_VERSION,
        "check_version": VERSION,
        "planned_calls": len(checks),
        "retained_candidates": sum(len(case.candidates) for case in source.cases),
        "unselected_candidates": sorted(
            candidates.keys() - {item["candidate_id"] for item in checks}
        ),
        "checks": checks,
        "historical_total_comparison_usd": str(source.original_total_cost_usd),
        "reused_discovery_usd": str(sum(case.original_cost_usd for case in source.cases)),
        "historical_cost_note": "Previously incurred costs; never charged again by this runner.",
        "cost_illustration": {
            "usd_for_all_checks": str(illustration),
            "basis": (
                "Assume UTF-8 request bytes / 4 + 512 input tokens per call, maximum "
                "table input rates and the full output ceiling. Token count is an "
                "assumption, not measured usage, an invoice or a spending guarantee. "
                "This illustration never replaces the conservative reservation."
            ),
        },
        "reservation_note": (
            "Existing full-context worst-case headroom per serial call, not expected spend. "
            "Actual priced usage settles before the next call. Smaller caps may refuse "
            "before transport; no heuristic replaces the spending guard."
        ),
        "scope": "Checks only; no triage, discovery, direct review, synthesis, DB or publication.",
        "quality": (
            "Provisional assessments of selected warnings only; no completed-workflow or "
            "parity claim. Original selection stays fixed; current bounded BASE/HEAD "
            "change and dependency context retains explicit omissions. These checks "
            "cannot establish every PR defect."
        ),
        "repeat_policy": "An existing retry directory is refused before provider construction.",
        "inputs": input_hashes(()),
    }


def _case_results(
    source: SavedCheckInputs, assessments: list[CandidateAssessmentRef]
) -> list[dict[str, Any]]:
    results = []
    for case in source.cases:
        triage = case.state.triage_result
        if triage is None:
            raise ValueError("saved checks require their original triage policy")
        candidate_ids = {ref.candidate_id for ref in case.candidates}
        selected = [item for item in assessments if item.candidate_id in candidate_ids]
        holds = verification_hold_finding_ids(
            case.original_findings,
            case.candidates,
            selected,
            policy_version=triage.policy_version,
        )
        results.append(
            {
                "case_id": case.case_id,
                "rep": case.rep,
                "original_findings": [
                    item.model_dump(mode="json") for item in case.original_findings
                ],
                "native_candidates": [item.model_dump(mode="json") for item in case.candidates],
                "assessments": [item.model_dump(mode="json") for item in selected],
                "verification_hold_finding_ids": [str(item) for item in holds],
                "unassessed_candidates": sorted(
                    candidate_ids - {item.candidate_id for item in selected}
                ),
            }
        )
    return results


async def collect_checks(
    source: SavedCheckInputs,
    plan: dict[str, Any],
    *,
    maximum_usd: Decimal,
    maximum_calls: int,
    output: Path,
    provider_factory: Callable[..., LLMProvider] = create_provider,
    validate_control: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Issue each missing question at most once; refuse any existing output."""
    require_eval_mode()
    if build_plan(source) != plan:
        raise ValueError("prepared retry changed before collection")
    verify_saved_inputs(source)
    budget = Budget(maximum_usd, maximum_calls)
    if maximum_calls > plan["planned_calls"]:
        raise ValueError("call cap cannot exceed the prepared missing checks")
    first_reservation = Decimal(plan["checks"][0]["reservation_usd"])
    if maximum_usd < first_reservation:
        raise ValueError("dollar cap cannot fit the existing conservative guard; zero calls made")
    output.mkdir(parents=True, exist_ok=False)
    calls = output / "calls"
    calls.mkdir()
    write_json(output / "plan.json", plan)
    recorder = ExchangeRecorder(calls, source.target)
    provider = None
    assessments: list[CandidateAssessmentRef] = []
    attempts: list[dict[str, Any]] = []
    status, error_type = "incomplete", None

    def result() -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "status": status,
            "error_type": error_type,
            "evidence_kind": "live-provider"
            if provider_factory is create_provider
            else "injected-provider",
            "plan": plan,
            "new_check_budget": {
                "authorized_maximum_usd": str(budget.maximum_usd),
                "maximum_calls": budget.maximum_calls,
                "calls": budget.calls,
                "known_usage_based_usd": str(budget.known_usd),
                "uncertain_budget_upper_usd": str(budget.uncertain_usd),
            },
            "calls": attempts,
            "cases": _case_results(source, assessments),
            "quality": plan["quality"],
        }

    try:
        provider = BudgetedProvider(
            provider_factory(source.target, recorder),
            source.target,
            recorder,
            budget,
            plan["inputs"],
        )
        schedule = [(case, check) for case in source.cases for check in case.prepared_checks]
        for case, prepared in schedule:
            triage = case.state.triage_result
            if triage is None:
                raise ValueError("saved checks require their original triage policy")
            verify_saved_inputs(source)
            if validate_control is not None:
                validate_control()
            reserve = reservation_usd(source.target, prepared.request.max_tokens)
            if (
                budget.calls >= budget.maximum_calls
                or budget.known_usd + budget.uncertain_usd + reserve > budget.maximum_usd
            ):
                status = "budget-stopped-before-call"
                break
            index = budget.calls + 1
            recorder.scenario = (
                f"{case.rep}:{case.case_id}:candidate-check:{prepared.check.candidate_id}"
            )
            write_json(
                calls / f"{index:06d}.started.json",
                {
                    "check": prepared.check.model_dump(mode="json"),
                    "request_sha256": _request_digest(prepared.request),
                },
            )
            response = await provider.complete(prepared.request)
            try:
                assessment = assess_candidate_response(
                    prepared, response, policy_version=triage.policy_version
                )
            except (ValueError, TypeError):
                assessment = CandidateAssessmentRef(
                    candidate_id=prepared.check.candidate_id,
                    attempt_id=prepared.check.attempt_id,
                    assessment_exchange_event_id=response.exchange_event_id,
                    status="uncertain",
                    outcome="invalid_response",
                    policy_version=triage.policy_version,
                )
            assessments.append(assessment)
            write_json(calls / f"{index:06d}.assessment.json", assessment.model_dump(mode="json"))
        else:
            status = (
                "collected"
                if all(item.outcome == "completed" for item in assessments)
                else "invalid-assessments"
            )
    except Exception as error:
        status, error_type = "incomplete", type(error).__name__
    finally:
        if provider is not None:
            attempts = provider.attempts
            try:
                await provider.aclose()
            except Exception as error:
                status, error_type = "incomplete", type(error).__name__
        write_json(output / "results.json", result())
        write_json(
            output / "receipts.json",
            {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*.json"))
            },
        )

    return result()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--i-authorize-spend", action="store_true")
    parser.add_argument("--max-cost-usd", type=Decimal)
    parser.add_argument("--max-calls", type=int)
    args = parser.parse_args(argv)
    source = load_saved_checks(args.report)
    plan = build_plan(source)
    display = {key: value for key, value in plan.items() if key not in {"inputs", "source_hashes"}}
    print(json.dumps(display, indent=2))
    if args.dry_run or not (args.paid or args.i_authorize_spend):
        return 0
    if not args.paid or not args.i_authorize_spend:
        parser.error("paid collection requires both --paid and --i-authorize-spend")
    if args.max_cost_usd is None or args.max_calls is None:
        parser.error("paid collection requires explicit --max-cost-usd and --max-calls")
    try:
        Budget(args.max_cost_usd, args.max_calls)
    except ValueError as error:
        parser.error(str(error))
    if args.max_calls > plan["planned_calls"]:
        parser.error("call cap cannot exceed the prepared missing checks")
    minimum = Decimal(plan["checks"][0]["reservation_usd"])
    if args.max_cost_usd < minimum:
        parser.error(
            f"No calls made: the existing guard requires ${minimum} of headroom. "
            "That is not expected spend; the requested lower cap remains in force."
        )
    require_eval_mode()
    output = ROOT / "reports/review-check-retry" / plan["retry_id"][:24]
    try:
        result = asyncio.run(
            collect_checks(
                source,
                plan,
                maximum_usd=args.max_cost_usd,
                maximum_calls=args.max_calls,
                output=output,
            )
        )
    except FileExistsError:
        parser.error("A retry for these saved warnings already exists; no repeat calls were made.")
    print(
        json.dumps(
            {
                "report": str(output.relative_to(ROOT)),
                "status": result["status"],
                "budget": result["new_check_budget"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "collected" else 2


if __name__ == "__main__":
    sys.exit(main())
