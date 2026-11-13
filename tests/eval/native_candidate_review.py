"""Prepare bounded checks of saved native analyzer candidates, without new discovery.

The free command verifies a diagnostic report, retains every decoded proposal and
prepares source-backed questions. Optional scripted answers exercise bookkeeping;
they do not prove model quality. There is no provider, database or publication path.
Raw content stays in explicit local eval artifacts, outside production checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from outrider.agent.eval_driver import EvalFixture
from outrider.coordinates import validate_diff_path
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.policy import SEVERITY_POLICY, FindingType
from outrider.policy.severity import ACTIVE_POLICY_VERSION

from .direct_first_candidates import (
    CandidateAssessment,
    CandidateLedger,
    CandidateLedgerError,
    build_ledger,
    decode_candidate_json,
    record_assessment,
)
from .direct_first_context import (
    PreparedFollowupContext,
    captured_heads,
    prepare_context,
    prepare_followup,
)
from .model_refresh import write_json
from .review_diagnostics_inputs import artifact, fields, request_identity, sha

SCHEMA = "native-candidate-review-v1"
CHECK_VERSION = "native-candidate-check-v1"


@dataclass(frozen=True)
class NativeCandidateReview:
    fixture: EvalFixture
    request: LLMRequest
    ledger: CandidateLedger
    context: PreparedFollowupContext


def prepare_native_review(
    fixture: EvalFixture,
    request: LLMRequest,
    response: LLMResponse,
    *,
    context: PreparedFollowupContext | None = None,
) -> NativeCandidateReview:
    """Retain one existing clean eval exchange; the request owns its source path."""
    if request.node_id != "analyze" or not request.is_eval or request.degraded_mode:
        raise ValueError("native preparation requires a clean analyzer eval request")
    if response.model != request.model:
        raise ValueError("native response model differs from its request")
    headers = re.findall(r"(?m)^File: ([^\r\n]+)$", request.user_prompt)
    if len(headers) != 1:
        raise ValueError("native request must name exactly one primary file")
    source_file = validate_diff_path(headers[0])
    context = prepare_context(fixture) if context is None else context
    if context.contents != captured_heads(fixture):
        raise ValueError("native context belongs to different captured heads")
    if source_file not in context.contents:
        raise ValueError("native request primary source is absent from captured heads")
    ledger = build_ledger(
        response.text,
        source_id=hashlib.sha256(json.dumps(fields(request), sort_keys=True).encode()).hexdigest(),
        finish_reason=response.finish_reason,
        source_file_path=source_file,
    )
    return NativeCandidateReview(fixture, request, ledger, context)


def prepare_candidate_check(review: NativeCandidateReview, candidate_id: str) -> LLMRequest | None:
    """Ask about a retained candidate even when native admission would reject it."""
    candidates = [item for item in review.ledger.candidates if item.candidate_id == candidate_id]
    if len(candidates) != 1:
        raise CandidateLedgerError("check must name exactly one existing candidate")
    if review.ledger.diagnostics:
        return None
    request = prepare_followup(
        review.fixture,
        candidates[0],
        model=review.request.model,
        review_id=review.request.review_id,
        context=review.context,
    )
    if request is None:
        return None
    schema = json.loads(request.response_schema_json or "{}")
    schema["properties"]["suggested_finding_type"] = {
        "anyOf": [
            {"type": "string", "enum": [item.value for item in FindingType]},
            {"type": "null"},
        ]
    }
    schema["required"].append("suggested_finding_type")
    return request.model_copy(
        update={
            "system_prompt": request.system_prompt
            + "\nAlso return suggested_finding_type: an existing allowed type only when it "
            "fits the claim and supplied evidence, otherwise null. Do not force a claim "
            "into a nearby category. Do not assign severity. A category suggestion is "
            "provisional and never replaces the original category.\n",
            "response_schema_json": json.dumps(schema, separators=(",", ":")),
            "prompt_template_version": CHECK_VERSION,
        }
    )


def record_candidate_check(
    review: NativeCandidateReview,
    candidate_id: str,
    request: LLMRequest,
    response: LLMResponse,
    *,
    origin: Literal["model", "scripted"] = "model",
) -> NativeCandidateReview:
    """Append a provisional assessment after request, shape and exact citation checks."""
    expected = prepare_candidate_check(review, candidate_id)
    if expected is None or fields(request) != fields(expected):
        raise CandidateLedgerError("check request differs from its bounded preparation")
    if response.model != request.model or response.finish_reason not in {"end_turn", "stop"}:
        raise CandidateLedgerError("check response must use the requested model and finish")
    raw = decode_candidate_json(response.text)
    if not isinstance(raw, dict) or set(raw) != {
        "candidate_id",
        "status",
        "reason",
        "citations",
        "suggested_finding_type",
    }:
        raise CandidateLedgerError("check must return the complete assessment shape")
    assessment = CandidateAssessment.model_validate_json(
        json.dumps({**raw, "origin": origin}), strict=True
    )
    for citation in assessment.citations:
        if not any(
            entry.file_path == citation.file_path
            and entry.line_start <= citation.line_start <= citation.line_end <= entry.line_end
            for entry in request.context_summary
        ):
            raise CandidateLedgerError("citation was not supplied to this check")
    ledger = record_assessment(
        review.ledger, candidate_id, assessment, contents=review.context.contents
    )
    return replace(review, ledger=ledger)


def _policy_category(finding_type: str | None) -> dict[str, str] | None:
    if finding_type is None:
        return None
    try:
        category = FindingType(finding_type)
    except (ValueError, TypeError):
        return None
    # Indexing fails if the policy omits a type; no default priority is invented.
    return {
        "finding_type": category.value,
        "policy_severity": SEVERITY_POLICY[category].value,
        "policy_version": ACTIVE_POLICY_VERSION,
    }


def review_summary(review: NativeCandidateReview) -> dict[str, Any]:
    """Keep original labels, policy suggestions and investigation status separate."""
    candidates = []
    for candidate in review.ledger.candidates:
        category = None if candidate.proposal is None else candidate.proposal.category
        original = _policy_category(category)
        suggestions = [
            {"assessment_index": index, **classification}
            for index, assessment in enumerate(candidate.assessments)
            if (classification := _policy_category(assessment.suggested_finding_type)) is not None
        ]
        labels = {item["finding_type"] for item in suggestions}
        if original is not None:
            labels.add(original["finding_type"])
        state = "needs-classification" if original is None else "unreviewed"
        if len(labels) > 1:
            state = "disputed"
        elif candidate.assessments:
            state = (
                "suggested"
                if all(item.suggested_finding_type is not None for item in candidate.assessments)
                else "uncertain"
            )
        candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "index": candidate.index,
                "original_category": category,
                "original_policy_category": original,
                "classification_state": state,
                "classification_suggestions": suggestions,
                "proposal_shape": "malformed" if candidate.errors else "readable",
                "check_prepared": prepare_candidate_check(review, candidate.candidate_id)
                is not None,
                "assessment_count": len(candidate.assessments),
                "native_admission": "not-evaluated",
            }
        )
    return {"source_id": review.ledger.source_id, "candidates": candidates}


def _verified_artifact(directory: Path, name: str, receipts: dict[str, str]) -> bytes:
    if name not in receipts:
        raise ValueError("native input is not covered by report receipts")
    path = artifact(directory, name)
    if sha(path) != receipts[name]:
        raise ValueError(f"saved native artifact differs from its receipt: {name}")
    return path.read_bytes()


def prepare_saved_native(report: Path) -> dict[str, NativeCandidateReview]:
    """Verify frozen diagnostic inputs and prepare native exchanges without a provider."""
    receipts = json.loads((report / "receipts.json").read_bytes())
    for name in receipts:
        _verified_artifact(report, name, receipts)
    plan = json.loads(_verified_artifact(report, "plan.json", receipts))
    results = json.loads(_verified_artifact(report, "results.json", receipts))
    if (
        results.get("schema") != "captured-review-diagnostics-v1"
        or results.get("status") != "completed"
        or results.get("plan") != plan
    ):
        raise ValueError("native preparation requires a completed, matching diagnostic report")
    for name, digest in plan["inputs"].items():
        _verified_artifact(report, "generation/" + name, receipts)
        if receipts["generation/" + name] != digest:
            raise ValueError("native generation input differs from the saved plan")
    cases = {}
    for case in plan["cases"]:
        name = "generation/" + case["fixture"]
        data = _verified_artifact(report, name, receipts)
        if receipts[name] != case["fixture_sha256"] or case["id"] in cases:
            raise ValueError("native fixture identity is ambiguous or changed")
        fixture = EvalFixture.model_validate_json(data)
        cases[case["id"]] = (case, fixture, prepare_context(fixture))
    reviews = {}
    for row in results["cases"]:
        if "analysis_file" not in row:
            continue
        if row["status"] != "completed" or len(row["raw_exchange_files"]) != 1:
            raise ValueError("native row needs exactly one completed saved exchange")
        name = row["raw_exchange_files"][0]
        if name in reviews:
            raise ValueError("native exchange occurs more than once")
        exchange = json.loads(_verified_artifact(report, name, receipts))
        prepared = json.loads(_verified_artifact(report, row["prepared_request"], receipts))
        request = LLMRequest.model_validate(exchange["request"])
        metadata, fixture, context = cases[row["case_id"]]
        if (
            fields(request) != prepared
            or request_identity(request) != metadata["request_identities"][row["arm"]]
            or request.model != plan["target"]["model"]
        ):
            raise ValueError("native exchange differs from its prepared request or saved plan")
        review = prepare_native_review(
            fixture, request, LLMResponse.model_validate(exchange["response"]), context=context
        )
        if review.ledger.source_file_path != row["analysis_file"]:
            raise ValueError("native row source differs from the verified request")
        reviews[name] = review
    if not reviews:
        raise ValueError("report contains no native exchanges")
    return reviews


def prepare_saved(
    report: Path, *, output: Path, scripted_assessments: Path | None = None
) -> dict[str, Any]:
    """Write local retained candidates, bounded requests and optional scripted replay."""
    reviews = prepare_saved_native(report)
    if output.resolve().is_relative_to(report.resolve()):
        raise ValueError("native preparation must not overwrite its source report")
    scripts = [] if scripted_assessments is None else json.loads(scripted_assessments.read_bytes())
    if not isinstance(scripts, list):
        raise ValueError("scripted assessments must be an array")
    for script in scripts:
        review = reviews[script["source_exchange"]]
        candidate_id = script["assessment"]["candidate_id"]
        request = prepare_candidate_check(review, candidate_id)
        if request is None:
            raise ValueError("scripted assessment names an unavailable check")
        response = LLMResponse(
            model=request.model,
            text=json.dumps(script["assessment"]),
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=0,
        )
        reviews[script["source_exchange"]] = record_candidate_check(
            review, candidate_id, request, response, origin="scripted"
        )
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (name, review) in enumerate(reviews.items()):
        directory = output / f"exchange-{index:04d}"
        directory.mkdir()
        write_json(directory / "ledger.json", review.ledger.model_dump(mode="json"))
        summary = review_summary(review)
        write_json(directory / "summary.json", summary)
        for candidate in review.ledger.candidates:
            request = prepare_candidate_check(review, candidate.candidate_id)
            if request is not None:
                write_json(directory / f"check-{candidate.index:04d}.request.json", fields(request))
        rows.append({"source_exchange": name, "directory": directory.name, **summary})
    result = {
        "schema": SCHEMA,
        "model_calls_made": 0,
        "standalone_discovery_calls": 0,
        "database_calls": 0,
        "candidate_count": sum(len(item.ledger.candidates) for item in reviews.values()),
        "unknown_category_count": sum(
            candidate["original_policy_category"] is None
            for row in rows
            for candidate in row["candidates"]
        ),
        "scripted_assessment_count": len(scripts),
        "quality": "Unproven; exact citations and retained candidates are not semantic verdicts.",
        "privacy": "Raw eval artifacts only; production state and retention are unchanged.",
        "source_report": str(report),
        "source_receipts_sha256": sha(report / "receipts.json"),
        "scripted_assessments_sha256": (
            None if scripted_assessments is None else sha(scripted_assessments)
        ),
        "policy_snapshot": {
            "version": ACTIVE_POLICY_VERSION,
            "categories": {key.value: value.value for key, value in SEVERITY_POLICY.items()},
        },
        "reviews": rows,
    }
    write_json(output / "review.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scripted-assessments", type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare_saved(
            args.report, output=args.output, scripted_assessments=args.scripted_assessments
        )
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in result.items() if key != "reviews"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
