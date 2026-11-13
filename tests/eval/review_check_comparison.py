"""Compare revised checks with retained paid answers, without repeating discovery."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from outrider.eval_support import require_eval_mode

from . import review_check_retry as retry
from .model_refresh import ROOT, Budget, create_provider, write_json
from .revised_check_inputs import load_previous_checks, verify_previous_checks

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.llm.base import LLMProvider

    from .revised_check_inputs import PreviousChecks

SCHEMA = "review-check-comparison-v1"


def build_plan(previous: PreviousChecks) -> dict[str, Any]:
    """Match the same claims and model while requiring genuinely changed questions."""
    current = retry.build_plan(previous.source)
    old = previous.results["plan"]
    identity_fields = ("target", "profile_id", "reasoning_enabled", "profile_contract_digest")
    if any(current[field] != old[field] for field in identity_fields):
        raise ValueError("revised checks must keep the historical model and provider identity")
    old_checks = {item["candidate_id"]: item for item in old["checks"]}
    new_checks = {item["candidate_id"]: item for item in current["checks"]}
    if old_checks.keys() != new_checks.keys():
        raise ValueError("revised checks must preserve the original selected claims")
    pairs = []
    for candidate_id, new in new_checks.items():
        prior = old_checks[candidate_id]
        if (new["case_id"], new["rep"]) != (prior["case_id"], prior["rep"]):
            raise ValueError("revised check differs from its historical case")
        if (
            new["request_sha256"] == prior["request_sha256"]
            or new["attempt_id"] == prior["attempt_id"]
        ):
            raise ValueError("unchanged questions cannot be billed as a revised comparison")
        pairs.append(
            {
                "candidate_id": candidate_id,
                "case_id": new["case_id"],
                "rep": new["rep"],
                "previous_attempt_id": prior["attempt_id"],
                "new_attempt_id": new["attempt_id"],
                "previous_request_sha256": prior["request_sha256"],
                "new_request_sha256": new["request_sha256"],
            }
        )
    comparison_id = hashlib.sha256(
        json.dumps({"pairs": pairs, "target": current["target"]}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "mode": "free-plan",
        "model_calls_made": 0,
        "comparison_id": comparison_id,
        "previous_report": str(previous.directory),
        "previous_hashes": previous.hashes,
        "new_plan": current,
        "pairs": pairs,
        "previous_check_cost_usd": previous.results["new_check_budget"]["known_usage_based_usd"],
        "control": (
            "Historical answers on the same claims and model. Context and instructions change "
            "together; provider timing/cache state is uncontrolled. No causal isolation or "
            "full-workflow parity claim."
        ),
        "repeat_policy": (
            "The same previous attempts and revised requests share a deterministic output. "
            "An existing directory refuses before provider construction."
        ),
    }


def _assessments(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        assessment["candidate_id"]: assessment
        for case in results["cases"]
        for assessment in case["assessments"]
    }


async def collect_comparison(
    previous: PreviousChecks,
    plan: dict[str, Any],
    *,
    maximum_usd: Decimal,
    maximum_calls: int,
    output: Path,
    provider_factory: Callable[..., LLMProvider] = create_provider,
) -> dict[str, Any]:
    """Collect only revised questions and preserve both sets of answers separately."""
    require_eval_mode()
    verify_previous_checks(previous)
    if build_plan(previous) != plan:
        raise ValueError("revised comparison changed before collection")
    Budget(maximum_usd, maximum_calls)
    current = plan["new_plan"]
    if maximum_calls > current["planned_calls"]:
        raise ValueError("call cap cannot exceed the prepared revised checks")
    if maximum_usd < Decimal(current["checks"][0]["reservation_usd"]):
        raise ValueError("dollar cap cannot fit the existing conservative guard; zero calls made")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    collected: dict[str, Any] | None = None
    result: dict[str, Any] = {"schema": SCHEMA, "status": "incomplete"}
    try:
        collected = await retry.collect_checks(
            previous.source,
            current,
            maximum_usd=maximum_usd,
            maximum_calls=maximum_calls,
            output=output / "revised-checks",
            provider_factory=provider_factory,
            validate_control=lambda: verify_previous_checks(previous),
        )
        verify_previous_checks(previous)
        old_answers = _assessments(previous.results)
        new_answers = _assessments(collected)
        result = {
            "schema": SCHEMA,
            "status": collected["status"],
            "plan": plan,
            "previous_check_cost_usd": plan["previous_check_cost_usd"],
            "new_check_budget": collected["new_check_budget"],
            "pairs": [
                {
                    **pair,
                    "previous_assessment": old_answers[pair["candidate_id"]],
                    "revised_assessment": new_answers.get(pair["candidate_id"]),
                }
                for pair in plan["pairs"]
            ],
            "quality": "Unadjudicated; model assessments are provisional, not quality scores.",
        }
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        if collected is not None and "new_check_budget" not in result:
            result["new_check_budget"] = collected["new_check_budget"]
        write_json(output / "results.json", result)
        write_json(
            output / "receipts.json",
            {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*.json"))
                if path != output / "receipts.json"
            },
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--i-authorize-spend", action="store_true")
    parser.add_argument("--max-cost-usd", type=Decimal)
    parser.add_argument("--max-calls", type=int)
    args = parser.parse_args(argv)
    previous = load_previous_checks(args.previous)
    plan = build_plan(previous)
    current = plan["new_plan"]
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "mode": "free-plan",
                "model_calls_made": 0,
                "comparison_id": plan["comparison_id"],
                "previous_report": plan["previous_report"],
                "target": current["target"],
                "planned_new_calls": current["planned_calls"],
                "retained_candidates": current["retained_candidates"],
                "historical_check_cost_usd": plan["previous_check_cost_usd"],
                "new_cost_illustration": current["cost_illustration"],
                "reservation_note": current["reservation_note"],
                "control": plan["control"],
                "repeat_policy": plan["repeat_policy"],
            },
            indent=2,
        )
    )
    if args.dry_run or not (args.paid or args.i_authorize_spend):
        return 0
    if not args.paid or not args.i_authorize_spend:
        parser.error("paid collection requires both spending flags")
    if args.max_cost_usd is None or args.max_calls is None:
        parser.error("paid collection requires explicit dollar and call caps")
    output = ROOT / "reports/review-check-comparison" / plan["comparison_id"][:24]
    try:
        result = asyncio.run(
            collect_comparison(
                previous,
                plan,
                maximum_usd=args.max_cost_usd,
                maximum_calls=args.max_calls,
                output=output,
            )
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "report": str(output.relative_to(ROOT)),
                "status": result["status"],
                "budget": result.get("new_check_budget"),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "collected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
