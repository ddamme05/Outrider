"""Validate completed check-only answers before measuring revised questions.

The prior answers are historical controls, not new quality judgments. Loading
checks receipt, source and exchange lineage without constructing a provider.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid5

from outrider.agent.candidate_checks import (
    _ATTEMPT_NAMESPACE,
    _restore_check_evidence,
    assess_candidate_response,
    candidate_text,
)
from outrider.audit.events import LLMCallEvent
from outrider.coordinates import extract_line_range_body
from outrider.llm.base import (
    LLMRequest,
    LLMResponse,
    _canonical_prompt_hash,
    _canonical_system_prompt_hash,
)
from outrider.policy.canonical import compute_identity_hash
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.native_candidate import (
    CandidateAssessmentRef,
    CandidateCheckStarted,
    candidate_check_contract_for_version,
)

from .model_refresh import Budget
from .review_check_inputs import SavedCheckInputs, load_saved_checks, verify_saved_inputs
from .review_check_retry import _case_results, _request_digest
from .review_diagnostics_inputs import artifact, sha


@dataclass(frozen=True)
class PreviousChecks:
    directory: Path
    hashes: dict[str, str]
    results: dict[str, Any] = field(repr=False)
    source: SavedCheckInputs = field(repr=False)


def _receipt_hashes(directory: Path) -> dict[str, str]:
    receipts = json.loads(artifact(directory, "receipts.json").read_bytes())
    if not isinstance(receipts, dict) or not {"plan.json", "results.json"} <= receipts.keys():
        raise ValueError("previous checks require receipt-bound plan and results")
    existing = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "receipts.json"
    }
    if set(receipts) != existing:
        raise ValueError("previous check artifact inventory differs from its receipts")
    hashes = {"receipts.json": sha(artifact(directory, "receipts.json"))}
    for name, digest in receipts.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("previous check receipt digest is invalid")
        if sha(artifact(directory, name)) != digest:
            raise ValueError(f"previous check receipt differs: {name}")
        hashes[name] = digest
    return hashes


def verify_previous_checks(previous: PreviousChecks) -> None:
    """Refuse original discovery or prior-answer drift after free preparation."""
    verify_saved_inputs(previous.source)
    if _receipt_hashes(previous.directory) != previous.hashes:
        raise ValueError("previous check artifacts changed after preparation")


def _amount(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("previous check amount is invalid") from error
    if not result.is_finite() or result < 0:
        raise ValueError("previous check amount must be finite and nonnegative")
    return result


def _validate_served_snippets(
    request: LLMRequest, check: CandidateCheckStarted, source: SavedCheckInputs
) -> None:
    candidates = {ref.candidate_id: (case, ref) for case in source.cases for ref in case.candidates}
    if check.candidate_id not in candidates:
        raise ValueError("previous check names an unknown discovery candidate")
    case, ref = candidates[check.candidate_id]
    restored = _restore_check_evidence(case.state, ref, check)
    if restored is None:
        raise ValueError("previous check source or text hashes differ from captured evidence")
    packet = json.loads(request.user_prompt)
    if not isinstance(packet, dict) or packet.get("candidate_id") != check.candidate_id:
        raise ValueError("previous check request names a different candidate")
    claim = candidate_text(case.source_responses[ref.source_exchange_event_id].text, ref)
    if claim is None or packet.get("claim") != claim.model_dump():
        raise ValueError("previous check question differs from its original discovery claim")
    served = packet.get("reference_snippets")
    if not isinstance(served, list) or len(served) != len(check.snippets):
        raise ValueError("previous check source inventory differs from its request")
    for item, snippet in zip(served, check.snippets, strict=True):
        if not isinstance(item, dict):
            raise ValueError("previous check supplied an invalid source snippet")
        sources = restored.base_sources if snippet.source_side == "base" else restored.sources
        text = extract_line_range_body(
            snippet.line_start, snippet.line_end, sources[snippet.file_path]
        ).removesuffix("\n")
        expected = {
            "snippet_id": snippet.snippet_id,
            "file_path": snippet.file_path,
            "line_start": snippet.line_start,
            "line_end": snippet.line_end,
            "source": "\n".join(
                f"{number} | {line}"
                for number, line in enumerate(text.split("\n"), snippet.line_start)
            ),
        }
        if "source_side" in item or snippet.source_side != "head":
            expected["source_side"] = snippet.source_side
        if item != expected:
            raise ValueError("previous check served snippets differ from their captured evidence")


def _load_assessment(
    directory: Path,
    index: int,
    selected: dict[str, Any],
    call: dict[str, Any],
    plan: dict[str, Any],
    source: SavedCheckInputs,
) -> CandidateAssessmentRef:
    prefix = f"calls/{index:06d}"
    started = json.loads(artifact(directory, prefix + ".started.json").read_bytes())
    check = CandidateCheckStarted.model_validate(started["check"])
    exchange = json.loads(artifact(directory, prefix + ".exchange.json").read_bytes())
    request = LLMRequest.model_validate(exchange["request"])
    event = LLMCallEvent.model_validate(exchange["event"])
    response = LLMResponse.model_validate(exchange["response"])
    candidates = {ref.candidate_id: (case, ref) for case in source.cases for ref in case.candidates}
    if check.candidate_id not in candidates:
        raise ValueError("previous check names an unknown discovery candidate")
    case, ref = candidates[check.candidate_id]
    digest = _request_digest(request)
    question_hash = compute_identity_hash(
        {
            "candidate_id": check.candidate_id,
            "prompt_hash": check.prompt_hash,
            "version": request.prompt_template_version,
            "model": check.model,
        }
    )
    expected_scenario = f"{case.rep}:{case.case_id}:candidate-check:{ref.candidate_id}"
    if (
        selected["candidate_id"] != ref.candidate_id
        or (selected["case_id"], selected["rep"]) != (case.case_id, case.rep)
        or selected["file_path"] != ref.file_path
        or selected["original_finding_type"] != ref.finding_type
        or selected["original_priority"] != ref.severity
        or selected["attempt_id"] != str(check.attempt_id)
        or selected["request_sha256"] != digest
        or started["request_sha256"] != digest
        or check.question_hash != question_hash
        or check.attempt_id != uuid5(_ATTEMPT_NAMESPACE, question_hash)
    ):
        raise ValueError("previous check differs from its selected question identity")
    if (
        request.candidate_attempt_id != check.attempt_id
        or event.candidate_attempt_id != check.attempt_id
        or event.event_id != response.exchange_event_id
        or str(event.event_id) != call["audit_event_id"]
        or event.timestamp != response.exchange_recorded_at
        or request.review_id != case.state.review_id
        or event.review_id != case.state.review_id
        or not request.is_eval
        or not event.is_eval
        or request.node_id != "trace"
        or event.node_id != "trace"
        or {request.model, event.model, response.model, check.model} != {source.target.model}
        or exchange["target"] != plan["target"]
        or exchange["call_index"] != index
        or exchange["scenario"] != expected_scenario
        or call["index"] != index
        or call["scenario"] != expected_scenario
        or call["target"] != source.target.name
    ):
        raise ValueError("previous check exchange differs in review, call or model scope")
    if (
        request.prompt_template_version != plan["check_version"]
        or check.response_contract
        != candidate_check_contract_for_version(request.prompt_template_version)
        or event.prompt_template_version != request.prompt_template_version
        or event.prompt_hash != check.prompt_hash
        or event.prompt_hash
        != _canonical_prompt_hash(
            system_prompt=request.system_prompt, user_prompt=request.user_prompt
        )
        or event.system_prompt_hash != _canonical_system_prompt_hash(request.system_prompt)
        or event.response_format_digest != request.response_format_digest
        or event.context_summary != request.context_summary
        or event.pricing_version != plan["pricing_version"]
        or selected["snippet_count"] != len(check.snippets)
        or selected["max_output_tokens"] != request.max_tokens
        or selected["request_text_bytes"]
        != sum(
            len(text.encode())
            for text in (
                request.system_prompt,
                request.user_prompt,
                request.response_schema_json or "",
            )
        )
    ):
        raise ValueError("previous check audit does not bind its exact request")
    for key in ("profile_id", "reasoning_enabled", "profile_contract_digest"):
        if getattr(event, key) != plan[key] or getattr(response, key) != plan[key]:
            raise ValueError("previous check provider identity differs from its plan")
    if (event.input_tokens, event.output_tokens, event.cached_tokens) != (
        response.input_tokens,
        response.output_tokens,
        response.cache_read_tokens,
    ):
        raise ValueError("previous check recorded usage differs from its response")
    if (
        call["status"] != "completed"
        or call["error_type"] is not None
        or call["persisted_exchange"] is not True
        or call["truncated"] is not False
        or call["cost_status"] != "usage-priced"
        or response.finish_reason not in {"end_turn", "stop"}
        or event.finish_reason != response.finish_reason
        or call["finish_reason"] != response.finish_reason
        or _amount(call["usage_based_usd"]) != _amount(event.cost_usd)
        or _amount(call["budget_retained_upper_usd"]) != _amount(call["usage_based_usd"])
        or _amount(selected["reservation_usd"]) != _amount(call["worst_case_reserved_usd"])
        or _amount(call["usage_based_usd"]) > _amount(call["worst_case_reserved_usd"])
    ):
        raise ValueError("previous check requires completed usage-priced calls")
    _validate_served_snippets(request, check, source)
    restored = _restore_check_evidence(case.state, ref, check)
    assert restored is not None and case.state.triage_result is not None
    assessment = assess_candidate_response(
        restored,
        response,
        policy_version=case.state.triage_result.policy_version,
        historical_policy=source.severity_policy,
    )
    if assessment.model_dump(mode="json") != json.loads(
        artifact(directory, prefix + ".assessment.json").read_bytes()
    ):
        raise ValueError("previous check assessment differs from its recorded answer")
    return assessment


def load_previous_checks(directory: Path) -> PreviousChecks:
    """Validate every prior check before constructing a revised paid experiment."""
    directory = directory.resolve()
    hashes = _receipt_hashes(directory)
    plan = json.loads(artifact(directory, "plan.json").read_bytes())
    results = json.loads(artifact(directory, "results.json").read_bytes())
    if (
        plan.get("schema") != "review-check-retry-v1"
        or results.get("schema") != "review-check-retry-v1"
        or results.get("plan") != plan
        or results.get("status") != "collected"
        or results.get("error_type") is not None
        or results.get("evidence_kind") not in {"live-provider", "injected-provider"}
    ):
        raise ValueError("comparison requires a completed check-only retry")
    source = load_saved_checks(Path(plan["source_report"]), prepare_new_checks=False)
    if source.policy_version == ACTIVE_POLICY_VERSION:
        source = load_saved_checks(Path(plan["source_report"]))
    if plan["source_hashes"] != source.hashes or plan["target"] != asdict(source.target):
        raise ValueError("previous checks differ from their original discovery lineage")
    original = json.loads(artifact(source.directory, "plan.json").read_bytes())
    if any(
        plan[key] != original[key]
        for key in ("profile_id", "reasoning_enabled", "profile_contract_digest")
    ):
        raise ValueError("previous check provider differs from original discovery")
    checks, calls = plan["checks"], results["calls"]
    count = len(checks)
    if (
        not checks
        or count != plan["planned_calls"]
        or count != len(calls)
        or len({item["candidate_id"] for item in checks}) != count
        or len({item["attempt_id"] for item in checks}) != count
        or len({item["audit_event_id"] for item in calls}) != count
    ):
        raise ValueError("previous check call inventory is incomplete or duplicated")
    expected = {"receipts.json", "plan.json", "results.json"} | {
        f"calls/{index:06d}.{kind}.json"
        for index in range(1, count + 1)
        for kind in ("started", "exchange", "attempt", "assessment")
    }
    if set(hashes) != expected:
        raise ValueError("previous checks contain unaccounted call artifacts")
    for index, call in enumerate(calls, 1):
        if json.loads(artifact(directory, f"calls/{index:06d}.attempt.json").read_bytes()) != call:
            raise ValueError("previous check attempt differs from its result call")
    assessments = [
        _load_assessment(directory, index, selected, call, plan, source)
        for index, (selected, call) in enumerate(zip(checks, calls, strict=True), 1)
    ]
    if _case_results(source, assessments) != results["cases"]:
        raise ValueError("previous checks changed original findings, candidates or hold results")
    candidate_ids = {ref.candidate_id for case in source.cases for ref in case.candidates}
    if plan["retained_candidates"] != len(candidate_ids) or plan["unselected_candidates"] != sorted(
        candidate_ids - {item["candidate_id"] for item in checks}
    ):
        raise ValueError("previous checks changed their selected candidate inventory")
    if _amount(
        plan["historical_total_comparison_usd"]
    ) != source.original_total_cost_usd or _amount(plan["reused_discovery_usd"]) != sum(
        (case.original_cost_usd for case in source.cases), Decimal()
    ):
        raise ValueError("previous check historical costs differ from original discovery")
    budget = results["new_check_budget"]
    Budget(_amount(budget["authorized_maximum_usd"]), budget["maximum_calls"])
    cost = sum((_amount(call["usage_based_usd"]) for call in calls), Decimal())
    if (
        budget["calls"] != count
        or budget["maximum_calls"] < count
        or _amount(budget["known_usage_based_usd"]) != cost
        or _amount(budget["uncertain_budget_upper_usd"]) != 0
        or cost > _amount(budget["authorized_maximum_usd"])
    ):
        raise ValueError("previous check budget differs from its recorded calls")
    return PreviousChecks(directory, hashes, results, source)
