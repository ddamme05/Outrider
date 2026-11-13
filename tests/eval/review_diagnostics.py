"""Compare review instructions or context using verified saved requests.

Free planning verifies the capture without keys or a database. Paid collection
is serial and bounded by one shared ceiling. Raw proposals require independent
adjudication; this command does not change production or declare a winner.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from outrider.eval_support import require_eval_mode
from outrider.llm.base import INCLUDE_TEXT_OPT_IN
from outrider.llm.host_profiles import resolve_host_identity, validate_available_models
from outrider.llm.parsing import strip_outer_json_fence
from outrider.llm.pricing import PRICING_VERSION

from .direct_review import assess_response
from .model_refresh import (
    ROOT,
    Budget,
    BudgetedProvider,
    BudgetLimitError,
    ExchangeRecorder,
    Target,
    create_provider,
    input_hashes,
    reservation_usd,
    write_json,
)
from .review_adjudication import write_packet
from .review_diagnostics_inputs import (
    APPEND_RECIPE,
    ARMS,
    CLARIFIED_CLAUSE,
    CLARIFIED_VERSION,
    DISCOVERY_ARMS,
    SOURCE_CLAUSE,
    artifact,
    case_plan,
    discovery_identity,
    fields,
    load_source,
    request_identity,
    requests_for_case,
    sha,
    source_identity,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.agent.eval_driver import EvalFixture
    from outrider.llm.base import LLMProvider, LLMRequest

SCHEMA = "captured-review-diagnostics-v1"


def build_plan(
    report: Path, file_path: str, *, reps: int = 3, compare_discovery: bool = False
) -> dict[str, Any]:
    if not 1 <= reps <= 10:
        raise ValueError("use 1–10 repetitions")
    source = load_source(report, file_path, compare_discovery=compare_discovery)
    validate_available_models(source.target.host, (source.target.model,))
    profile, reasoning, contract = resolve_host_identity(
        source.target.host, reasoning=False, models=(source.target.model,)
    )
    if compare_discovery and source.profile_identity != {
        "profile_id": profile,
        "reasoning_enabled": reasoning,
        "profile_contract_digest": contract,
    }:
        raise ValueError("current provider profile differs from the captured discovery control")
    arms = DISCOVERY_ARMS if compare_discovery else ARMS
    cases = []
    schedule = []
    for index, case in enumerate(source.cases):
        metadata = case_plan(case, compare_discovery=compare_discovery)
        metadata["original_fixture"] = metadata["fixture"]
        metadata["fixture"] = f"captured-fixtures/{index:06d}.json"
        cases.append(metadata)
        for rep in range(reps):
            offset = (index * reps + rep) % len(arms)
            schedule.extend(
                {"case_id": metadata["id"], "rep": rep, "arm": arm}
                for arm in (*arms[offset:], *arms[:offset])
            )
    inputs = input_hashes(())
    if (ROOT / "tests/__init__.py").is_file():
        inputs["tests/__init__.py"] = sha(ROOT / "tests/__init__.py")
    plan = {
        "schema": SCHEMA,
        "mode": "free-plan",
        "model_calls_made": 0,
        **source_identity(source),
        "profile_id": profile,
        "reasoning_enabled": reasoning,
        "profile_contract_digest": contract,
        "pricing_version": PRICING_VERSION,
        "analysis_file": file_path,
        "cases": cases,
        "repetitions": reps,
        "arms": list(arms),
        "schedule": schedule,
        "planned_calls": len(schedule),
        "call_caps_per_arm": dict.fromkeys(arms, len(cases) * reps),
        "budget_policy": (
            "One serial shared USD/call ceiling; equal scheduled call caps per arm. "
            "No independent USD caps."
        ),
        "per_call_reservation_upper_usd": str(reservation_usd(source.target, 8192)),
        "transform_recipes": {
            "instructions": (
                "Preserve the receipt-bound captured control; replace only system text and "
                "template version with the registered current concise prompt."
            ),
            "context": APPEND_RECIPE,
            "clarified": {
                "version": CLARIFIED_VERSION,
                "replace_once": SOURCE_CLAUSE,
                "replacement": CLARIFIED_CLAUSE,
            },
            "direct": (
                "Regenerate original direct request; compare every field including original "
                "review UUID before new metadata UUIDs."
            ),
        },
        "scope": (
            "Single-call generation diagnostics; no triage, trace, synthesis, native admission "
            "or full-workflow timing."
        ),
        "quality": (
            "Independent semantic adjudication pending; no automatic winner "
            "or default qualification."
        ),
        "context_limit": (
            "Full-reference arms append complete captured PR data. They do not widen primary "
            "finding bounds or isolate arbitrary repackaging."
        ),
        "settings_note": (
            "Normalized request settings are fixed; the common current provider adapter controls "
            "the actual wire shape."
        ),
        "order": (
            "Rotate first arm by global case/repetition block; "
            "provider cache state is uncontrolled."
        ),
        "runtime": {
            "python": sys.version,
            "packages": {name: version(name) for name in ("anthropic", "openai", "pydantic")},
        },
        "inputs": inputs,
    }
    if compare_discovery:
        versions = {case.native.prompt_template_version for case in source.cases}
        if len(versions) != 1:
            raise ValueError("discovery cases must share one captured prompt version")
        plan.update(
            comparison_mode="discovery",
            transform_recipes=discovery_identity(versions.pop()),
            context_limit=(
                "Every arm retains the exact captured bounded user source, manifests and "
                "primary finding-location instructions. No full-PR reference is appended."
            ),
            restriction_scope=(
                "The category arm changes only the allowed-type instruction. The wire schema "
                "already permits finding_type strings and stays identical; decoder, admission "
                "and location restrictions are not changed or evaluated."
            ),
        )
    return plan


def _copy_pinned(source: Path, hashes: dict[str, str], destination: Path) -> None:
    for name, digest in hashes.items():
        path = artifact(source, name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"frozen diagnostic input changed: {name}")
        target = artifact(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)


def _proposal_diagnostics(
    text: str, fixture: EvalFixture, request: LLMRequest, file_path: str | None
) -> dict[str, Any]:
    try:
        response = json.loads(strip_outer_json_fence(text))
        proposals = response["findings"]
        if not isinstance(proposals, list):
            raise ValueError("findings must be an array")
    except (ValueError, TypeError, KeyError, RecursionError):
        return {"status": "unparseable-raw-findings", "proposals": []}
    heads = {file.path: file.content_head for file in fixture.files}
    heads.update(fixture.repository_contents_head)
    diagnostics = []
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            diagnostics.append({"index": index, "status": "non-object-proposal"})
            continue
        path = proposal.get("file_path", file_path)
        content = heads.get(path) if isinstance(path, str) else None
        evidence = proposal.get("evidence")
        start, end = proposal.get("line_start"), proposal.get("line_end")
        ordered = type(start) is int and type(end) is int and 1 <= start <= end
        lines = None if content is None else content.splitlines(keepends=True)
        valid_span = ordered and lines is not None and end <= len(lines)
        quote = evidence if isinstance(evidence, str) and evidence else None
        diagnostics.append(
            {
                "index": index,
                "file_path": path,
                "location_in_captured_head": None if content is None else valid_span,
                "location_in_served_manifest": ordered
                and any(
                    entry.file_path == path and entry.line_start <= start <= end <= entry.line_end
                    for entry in request.context_summary
                ),
                "exact_quote_in_captured_head": None
                if content is None or not isinstance(evidence, str)
                else quote is not None and quote in content,
                "exact_quote_in_claimed_span": None
                if content is None or not isinstance(evidence, str)
                else quote is not None and valid_span and quote in "".join(lines[start - 1 : end]),
                "exact_quote_in_request_text": None
                if not isinstance(evidence, str)
                else quote is not None and quote in request.user_prompt,
                "evidence_has_literal_backslash_n": isinstance(evidence, str) and "\\n" in evidence,
                "evidence_has_newline": isinstance(evidence, str) and "\n" in evidence,
                "semantic_verdict": "pending-independent-adjudication",
            }
        )
    return {
        "status": "malformed-raw-findings"
        if any(item.get("status") == "non-object-proposal" for item in diagnostics)
        else "parsed-raw-findings",
        "proposals": diagnostics,
        "interpretation": (
            "Literal source/span checks, not semantic verdicts. Numbered reference source can "
            "break a literal multiline request match even when the source was supplied."
        ),
    }


async def collect(
    plan: dict[str, Any],
    *,
    maximum_usd: Decimal,
    maximum_calls: int,
    output: Path,
    provider_factory: Callable[[Target, ExchangeRecorder], LLMProvider] | None = None,
) -> dict[str, Any]:
    require_eval_mode()
    budget = Budget(maximum_usd, maximum_calls)
    compare_discovery = plan.get("comparison_mode") == "discovery"
    fresh = build_plan(
        Path(plan["source_report"]),
        plan["analysis_file"],
        reps=plan["repetitions"],
        compare_discovery=compare_discovery,
    )
    if fresh != plan:
        raise ValueError("diagnostic plan or frozen inputs changed before provider construction")
    source = load_source(
        Path(plan["source_report"]), plan["analysis_file"], compare_discovery=compare_discovery
    )
    if output.resolve().is_relative_to(source.directory):
        raise ValueError("diagnostics must not write inside the original paid report")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    _copy_pinned(ROOT, plan["inputs"], output / "generation")
    _copy_pinned(source.directory, plan["source_inputs"], output / "source-report")
    prepared = {
        case.metadata["id"]: requests_for_case(case, compare_discovery=compare_discovery)
        for case in source.cases
    }
    cases = {case.metadata["id"]: case for case in source.cases}
    for metadata in plan["cases"]:
        original = artifact(source.directory, "generation/" + metadata["original_fixture"])
        target = artifact(output / "generation", metadata["fixture"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        if sha(target) != metadata["fixture_sha256"]:
            raise ValueError("copied captured fixture differs")
    calls_dir = output / "calls"
    calls_dir.mkdir()
    (output / "prepared").mkdir()
    recorder = ExchangeRecorder(calls_dir, source.target)
    factory = create_provider if provider_factory is None else provider_factory
    rows, setup_error, close_error = [], None, None
    provider = None
    budgeted = None
    arm_calls = dict.fromkeys(plan["arms"], 0)
    try:
        provider = factory(source.target, recorder)
        budgeted = BudgetedProvider(provider, source.target, recorder, budget, plan["inputs"])
        for sequence, slot in enumerate(plan["schedule"]):
            arm, case_id, rep = slot["arm"], slot["case_id"], slot["rep"]
            request = prepared[case_id][arm].model_copy(update={"review_id": uuid4()})
            expected = next(case for case in plan["cases"] if case["id"] == case_id)
            if request_identity(request) != expected["request_identities"][arm]:
                raise ValueError("prepared request differs from the frozen diagnostic plan")
            row = {
                **slot,
                "status": "prepared",
                "semantic_adjudication": "pending",
                "diagnostic_events": [],
                "native_admission": "not-run; raw generation diagnostic",
                "admission_status": "not-evaluated",
                "primary_context_summary": expected["primary_context_summary"],
                "prepared_request": f"prepared/{sequence:06d}.json",
            }
            if arm != "direct-control":
                row["analysis_file"] = plan["analysis_file"]
            rows.append(row)
            write_json(output / row["prepared_request"], fields(request))
            before = len(recorder.exchanges)
            attempts = len(budgeted.attempts)
            reserved_calls_before = budget.calls
            started = time.perf_counter()
            try:
                if arm_calls[arm] >= plan["call_caps_per_arm"][arm]:
                    raise BudgetLimitError("equal per-arm call cap reached")
                for name, digest in plan["source_inputs"].items():
                    if sha(artifact(source.directory, name)) != digest:
                        raise ValueError("original captured input changed before generation")
                recorder.scenario = f"{case_id}:{rep}:{arm}"
                response = await budgeted.complete(request)
                if len(recorder.exchanges) - before != 1:
                    row["unrecorded_response"] = f"prepared/{sequence:06d}.returned-response.json"
                    write_json(
                        output / row["unrecorded_response"],
                        response.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
                    )
                    raise RuntimeError("completed response lacks exactly one persisted exchange")
                row["proposal_diagnostics"] = _proposal_diagnostics(
                    response.text, cases[case_id].fixture, request, row.get("analysis_file")
                )
                row["status"] = "completed"
                if arm == "direct-control":
                    row["direct_assessment"] = assess_response(
                        response.text, cases[case_id].fixture
                    )
                    if row["direct_assessment"]["status"] != "parsed":
                        row["status"] = "invalid-response"
                elif row["proposal_diagnostics"]["status"] != "parsed-raw-findings":
                    row["status"] = "invalid-response"
            except Exception as exc:
                row.update(
                    status="budget-stopped-before-call"
                    if isinstance(exc, BudgetLimitError)
                    else "failed",
                    error_type=type(exc).__name__,
                )
            finally:
                arm_calls[arm] += budget.calls - reserved_calls_before
                row["whole_call_wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
                row["calls"] = budgeted.attempts[attempts:]
                row["raw_exchange_files"] = [
                    f"calls/{item['call_index']:06d}.exchange.json"
                    for item in recorder.exchanges[before:]
                ]
            if row["status"] != "completed":
                break
    except Exception as exc:
        setup_error = type(exc).__name__
    finally:
        if provider is not None:
            try:
                await provider.aclose()
            except Exception as exc:
                close_error = type(exc).__name__
    report = {
        "schema": SCHEMA,
        "status": "completed"
        if len(rows) == plan["planned_calls"]
        and all(row["status"] == "completed" for row in rows)
        and setup_error is None
        and close_error is None
        else "incomplete",
        "plan": plan,
        "evidence_kind": "live-provider" if factory is create_provider else "injected-provider",
        "cases": rows,
        "setup_error_type": setup_error,
        "close_error_type": close_error,
        "unattempted_calls": plan["schedule"][len(rows) :],
        "arm_call_counts": arm_calls,
        "budget": {
            "authorized_maximum_usd": str(budget.maximum_usd),
            "maximum_calls": budget.maximum_calls,
            "calls": budget.calls,
            "known_usage_based_usd": str(budget.known_usd),
            "uncertain_budget_upper_usd": str(budget.uncertain_usd),
        },
        "quality": (
            "Independent adjudication pending; operational completion is not a semantic verdict."
        ),
    }
    write_json(output / "results.json", report)
    write_packet(report, output)
    write_json(
        output / "receipts.json",
        {
            str(path.relative_to(output)): sha(path)
            for path in output.rglob("*")
            if path.is_file() and path != output / "adjudication/review.json"
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--compare-discovery",
        action="store_true",
        help="Compare opening instructions and allowed categories separately on captured source.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--i-authorize-spend", action="store_true")
    parser.add_argument("--max-cost-usd", type=Decimal)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/review-diagnostics"))
    args = parser.parse_args(argv)
    if args.paid != args.i_authorize_spend:
        parser.error("paid collection requires both spending flags")
    if args.max_cost_usd is not None and (
        not args.max_cost_usd.is_finite() or args.max_cost_usd <= 0
    ):
        parser.error("--max-cost-usd must be finite and positive")
    if args.max_calls is not None and args.max_calls < 1:
        parser.error("--max-calls must be positive")
    try:
        plan = build_plan(
            args.report, args.file, reps=args.reps, compare_discovery=args.compare_discovery
        )
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(plan, indent=2), flush=True)
    else:
        description = (
            "Compare the exact captured control with an alternative review opening and, "
            "separately, an unrestricted category instruction. "
            "Source, schema and bounds stay fixed."
            if args.compare_discovery
            else "Compare current/shorter instructions with selected/full reference source, "
            "one source-use clarification, and direct review."
        )
        print(
            f"Free plan: {len(plan['cases'])} cases × {args.reps} repetitions × "
            f"{len(plan['arms'])} arms "
            f"= {plan['planned_calls']} calls. Model: {plan['target']['model']}.\n"
            f"{description}\n"
            "No database. No calls made by planning. Quality judgments remain pending.\n"
            "Scope: analyzer-only raw generation; native admission is not evaluated.\n"
            f"Serial reservation upper bound per call: ${plan['per_call_reservation_upper_usd']} "
            "(headroom, not expected cost).",
            flush=True,
        )
    if not args.paid or args.dry_run:
        return 0
    if args.max_cost_usd is None or args.max_calls is None:
        parser.error("paid collection requires explicit --max-cost-usd and --max-calls")
    destination = args.output_dir / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    )
    print(
        f"Starting collection: maximum ${args.max_cost_usd}, at most {args.max_calls} calls.",
        flush=True,
    )
    report = asyncio.run(
        collect(
            plan,
            maximum_usd=args.max_cost_usd,
            maximum_calls=args.max_calls,
            output=destination,
        )
    )
    print(
        json.dumps(
            {"report": str(destination), "status": report["status"], "budget": report["budget"]},
            indent=2,
        )
    )
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
