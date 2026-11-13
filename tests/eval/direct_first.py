"""Prepare or exercise a direct-first review without changing the production graph.

The command reuses saved direct responses and makes no model calls. The async
runner accepts an explicitly supplied provider for scripted tests or a separately
authorized, budgeted experiment. Its local draft is not native Draft/HITL delivery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from outrider.agent.eval_driver import EvalFixture
from outrider.llm.base import INCLUDE_TEXT_OPT_IN, LLMProviderError

from . import direct_review
from .direct_first_candidates import (
    CandidateAssessment,
    CandidateLedger,
    CandidateLedgerError,
    build_ledger,
    record_assessment,
)
from .direct_first_context import captured_heads, prepare_context, prepare_followup
from .model_refresh import BudgetLimitError, OutputLimitError, write_json

if TYPE_CHECKING:
    from outrider.llm.base import LLMProvider, LLMRequest, LLMResponse

SCHEMA = "direct-first-review-v1"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _review(ledger: CandidateLedger, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "scope": "local experimental draft; production Draft/HITL is not exercised",
        "status": "incomplete"
        if ledger.diagnostics
        or any(candidate.errors for candidate in ledger.candidates)
        or any(attempt["status"] == "failed" for attempt in attempts)
        else "prepared-unadjudicated",
        "quality": "independent adjudication required; preservation is not review-quality parity",
        "native_delivery": "not-evaluated",
        "shared_initial_response": True,
        "candidate_count": len(ledger.candidates),
        "local_draft_candidate_count": len(ledger.candidates),
        "ledger": ledger.model_dump(mode="json"),
        "investigations": attempts,
    }


def _markdown(report: dict[str, Any]) -> str:
    ledger = report["ledger"]
    sections = [
        "# Experimental direct-first review",
        report["scope"],
        "Every initial candidate is included below. Investigator assessments are claims, "
        "not independently verified verdicts. No severity is assigned and nothing is posted.",
        "Status: " + report["status"],
        "Discovery diagnostics: " + json.dumps(ledger["diagnostics"]),
    ]
    for candidate in ledger["candidates"]:
        sections.append(f"## Candidate {candidate['index'] + 1}")
        proposal = candidate["proposal"]
        if proposal is None:
            display = {
                "state": "Invalid proposal; original retained for inspection",
                "errors": candidate["errors"],
                "original": candidate["raw_proposal_json"],
            }
        else:
            display = {
                "state": "Awaiting independent review",
                **proposal,
                "investigator_assessments": candidate["assessments"],
            }
        data = json.dumps(display, ensure_ascii=False, indent=2)
        fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", data)), default=0))
        sections.append(f"{fence}json\n{data}\n{fence}")
    if not ledger["candidates"]:
        sections.append(
            "No decoded candidates. Check discovery diagnostics before interpreting this."
        )
    sections.append(
        "## Investigation attempts\n\n" + json.dumps(report["investigations"], indent=2)
    )
    return "\n\n".join(sections) + "\n"


def _save_review(directory: Path, ledger: CandidateLedger, attempts: list[dict[str, Any]]) -> None:
    report = _review(ledger, attempts)
    # The numbered snapshots retain earlier assessments; these two files show the latest draft.
    (directory / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (directory / "review.md").write_text(_markdown(report))
    write_json(directory / f"stage-{len(attempts):04d}.json", report)


def _append_assessment(
    ledger: CandidateLedger,
    candidate_id: str,
    response: LLMResponse,
    request: LLMRequest,
    fixture: EvalFixture,
) -> CandidateLedger:
    if response.model != request.model:
        raise CandidateLedgerError("investigation response uses a different model")
    if response.finish_reason not in {"stop", "end_turn"}:
        raise CandidateLedgerError("investigation did not finish normally")
    raw = _json(response.text)
    if not isinstance(raw, dict) or set(raw) != {"candidate_id", "status", "reason", "citations"}:
        raise CandidateLedgerError("investigation must return the complete assessment shape")
    assessment = CandidateAssessment.model_validate_json(
        json.dumps({**raw, "origin": "model"}), strict=True
    )
    for citation in assessment.citations:
        if not any(
            item.file_path == citation.file_path
            and item.line_start <= citation.line_start <= citation.line_end <= item.line_end
            for item in request.context_summary
        ):
            raise CandidateLedgerError("citation was not supplied to this investigation")
    return record_assessment(ledger, candidate_id, assessment, contents=captured_heads(fixture))


async def run_direct_first(
    fixture: EvalFixture,
    *,
    provider: LLMProvider,
    model: str,
    output: Path,
    max_followups: int = 4,
) -> dict[str, Any]:
    """Share one discovery response with the baseline, then investigate bounded candidates.

    The caller owns provider authorization, USD reservations, exchange persistence
    and closing. There are at most 1 + max_followups complete() invocations, without
    retries. The CLI deliberately exposes no live-provider or paid mode.
    """
    if type(max_followups) is not int or not 0 <= max_followups <= 100:
        raise ValueError("max_followups must be an integer from 0 to 100")
    context = prepare_context(fixture)
    output.mkdir(parents=True, exist_ok=False)
    request = direct_review.render_request(fixture, model=model, review_id=uuid4())
    request_data = request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
    write_json(output / "discovery.request.json", request_data)
    try:
        response = await provider.complete(request)
    except (LLMProviderError, BudgetLimitError, OutputLimitError) as exc:
        write_json(
            output / "failure.json",
            {
                "schema": SCHEMA,
                "status": "incomplete",
                "stage": "discovery",
                "error_type": type(exc).__name__,
                "response_returned": False,
            },
        )
        raise
    write_json(
        output / "discovery.response.json",
        response.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
    )
    ledger = build_ledger(
        response.text,
        source_id=_sha(json.dumps(request_data, sort_keys=True).encode()),
        finish_reason=response.finish_reason,
    )
    write_json(output / "seed.json", ledger.model_dump(mode="json"))
    attempts: list[dict[str, Any]] = []
    _save_review(output, ledger, attempts)
    if response.model != model:
        attempts.append(
            {"stage": "discovery", "status": "failed", "error_type": "ModelIdentityMismatch"}
        )
        _save_review(output, ledger, attempts)
        return _review(ledger, attempts)
    if ledger.diagnostics:
        return _review(ledger, attempts)
    calls = 0
    for candidate in ledger.candidates:
        if calls >= max_followups:
            break
        followup = prepare_followup(
            fixture, candidate, model=model, review_id=request.review_id, context=context
        )
        attempt = {"candidate_id": candidate.candidate_id, "status": "prepared"}
        if followup is None:
            attempt.update(status="not-requested", reason="invalid-or-unavailable-primary-source")
        else:
            calls += 1
            write_json(
                output / f"investigation-{calls:04d}.request.json",
                followup.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
            )
            try:
                answer = await provider.complete(followup)
                write_json(
                    output / f"investigation-{calls:04d}.response.json",
                    answer.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
                )
                ledger = _append_assessment(
                    ledger, candidate.candidate_id, answer, followup, fixture
                )
                attempt["status"] = "assessment-recorded"
            except (
                LLMProviderError,
                BudgetLimitError,
                OutputLimitError,
                ValueError,
                RecursionError,
            ) as exc:
                attempt.update(status="failed", error_type=type(exc).__name__)
        attempts.append(attempt)
        _save_review(output, ledger, attempts)
        if attempt["status"] == "failed":
            break
    return _review(ledger, attempts)


def _artifact(directory: Path, name: str, receipts: dict[str, str]) -> bytes:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or name not in receipts:
        raise ValueError("input must be a receipt-covered file within the saved report")
    data = path.read_bytes()
    if _sha(data) != receipts[name]:
        raise ValueError("saved artifact differs from its receipt")
    return data


def prepare_saved(report: Path, *, output: Path, case_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    """Prepare all selected saved candidates and follow-up requests without a provider."""
    receipts = _json((report / "receipts.json").read_text())
    results = _json(_artifact(report, "results.json", receipts).decode())
    plan = _json(_artifact(report, "plan.json", receipts).decode())
    if results["schema"] != "review-comparison-v1" or results["plan"] != plan:
        raise ValueError("saved workflow comparison and canonical plan must agree")
    cases = {case["id"]: case for case in plan["cases"]}
    if set(case_ids) - cases.keys():
        raise ValueError("selected case is absent from the saved comparison")
    prepared = []
    for row in results["cases"]:
        if row["arm"] != "direct-one-shot" or (case_ids and row["case_id"] not in case_ids):
            continue
        if len(row["raw_exchange_files"]) != 1:
            raise ValueError("one original discovery exchange is required per review")
        case = cases[row["case_id"]]
        fixture_name = "generation/" + case["fixture"]
        fixture_bytes = _artifact(report, fixture_name, {fixture_name: case["fixture_sha256"]})
        fixture = EvalFixture.model_validate_json(fixture_bytes)
        exchange_bytes = _artifact(report, row["raw_exchange_files"][0], receipts)
        exchange = _json(exchange_bytes.decode())
        request_data = exchange["request"]
        request = direct_review.render_request(
            fixture, model=plan["target"]["model"], review_id=UUID(request_data["review_id"])
        )
        if request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN) != request_data:
            raise ValueError("current direct request does not reproduce this captured baseline")
        ledger = build_ledger(
            exchange["response"]["text"],
            source_id=_sha(exchange_bytes),
            finish_reason=exchange["response"]["finish_reason"],
        )
        prepared.append((row, fixture, request, ledger, exchange_bytes, exchange["response"]))
    if not prepared:
        raise ValueError("no selected saved direct reviews")
    output.mkdir(parents=True, exist_ok=False)
    reviews = []
    for index, (row, fixture, request, ledger, exchange_bytes, response) in enumerate(prepared):
        directory = output / f"review-{index:04d}"
        directory.mkdir()
        (directory / "original.exchange.json").write_bytes(exchange_bytes)
        write_json(directory / "original.result.json", row)
        write_json(directory / "seed.json", ledger.model_dump(mode="json"))
        attempts = []
        if row.get("status") != "completed" or response.get("model") != request.model:
            attempts.append(
                {"stage": "discovery", "status": "failed", "error_type": "SavedDiscoveryIncomplete"}
            )
        context = prepare_context(fixture)
        incomplete = bool(ledger.diagnostics or attempts)
        for candidate in ledger.candidates:
            followup = (
                None
                if incomplete
                else prepare_followup(
                    fixture,
                    candidate,
                    model=request.model,
                    review_id=request.review_id,
                    context=context,
                )
            )
            attempt = {
                "candidate_id": candidate.candidate_id,
                "status": "prepared" if followup else "not-requested",
                "reason": "no generation requested"
                if followup
                else "discovery-or-source-incomplete",
            }
            if followup:
                write_json(
                    directory / f"candidate-{candidate.index:04d}.request.json",
                    followup.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
                )
            attempts.append(attempt)
        _save_review(directory, ledger, attempts)
        reviews.append(
            {
                "case_id": row["case_id"],
                "rep": row["rep"],
                "directory": directory.name,
                "status": _review(ledger, attempts)["status"],
                "candidate_count": len(ledger.candidates),
                "followup_requests_prepared": sum(
                    item["status"] == "prepared" for item in attempts
                ),
            }
        )
    summary = {
        "schema": SCHEMA,
        "mode": "offline-preparation",
        "model_calls_made": 0,
        "source_report": str(report),
        "source_results_sha256": receipts["results.json"],
        "quality": "unadjudicated; local preservation is not native delivery or a quality pass",
        "reviews": reviews,
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local output directory.")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = prepare_saved(args.report, output=args.output, case_ids=tuple(args.case))
    except FileExistsError:
        parser.error("--output must be a new directory")
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        reason = str(exc) if type(exc) is ValueError else type(exc).__name__
        parser.error(f"cannot prepare saved direct review: {reason}")
    print(json.dumps(result, indent=2))
    return 0 if all(row["status"] != "incomplete" for row in result["reviews"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
