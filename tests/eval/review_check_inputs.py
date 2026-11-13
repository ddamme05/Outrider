"""Load unfinished candidate checks from receipt-bound saved discovery.

This is a check-only evaluation input. It preserves historical admissions and
their identities, but does not resume a workflow, qualify claims or publish them.
Captured Python files are data; loading never imports or executes them.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import UUID

from outrider.agent.candidate_checks import (
    CANDIDATE_CHECK_CALL_LIMIT,
    PreparedCandidateCheck,
    prepare_candidate_check,
)
from outrider.agent.eval_driver import EvalFixture
from outrider.agent.native_candidates import candidate_event_id
from outrider.audit.events import AnalyzeCompletedEvent, LLMCallEvent, NativeCandidateEvent
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER
from outrider.llm.base import (
    LLMRequest,
    LLMResponse,
    _canonical_prompt_hash,
    _canonical_system_prompt_hash,
)
from outrider.llm.parsing import strip_outer_json_fence
from outrider.policy.canonical import compute_proposal_hash, compute_response_hash
from outrider.policy.dimensions import FINDING_TYPE_TO_DIMENSION
from outrider.policy.severity import (
    ACTIVE_POLICY_VERSION,
    BARE_SEMVER_PATTERN,
    FindingSeverity,
    FindingType,
)
from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.review_state import ReviewState
from outrider.schemas.triage_result import TriageResult

from .model_refresh import Target
from .review_diagnostics_inputs import artifact, sha

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from pathlib import Path

    from outrider.schemas.native_candidate import NativeCandidateRef


@dataclass(frozen=True)
class SavedCheckCase:
    case_id: str
    rep: int
    state: ReviewState = field(repr=False)
    candidates: tuple[NativeCandidateRef, ...]
    source_responses: dict[UUID, LLMResponse] = field(repr=False)
    original_findings: tuple[ReviewFinding, ...] = field(repr=False)
    prepared_checks: tuple[PreparedCandidateCheck, ...] = field(repr=False)
    original_cost_usd: Decimal


@dataclass(frozen=True)
class SavedCheckInputs:
    directory: Path
    target: Target
    hashes: dict[str, str]
    cases: tuple[SavedCheckCase, ...]
    original_total_cost_usd: Decimal
    policy_version: str
    severity_policy: Mapping[FindingType, FindingSeverity]


def _captured_policy(
    directory: Path, hashes: dict[str, str]
) -> tuple[str, Mapping[FindingType, FindingSeverity]]:
    """Read the receipt-bound policy literals as data, never importing captured code."""
    name = "generation/src/outrider/policy/severity.py"
    path = artifact(directory, name)
    if name not in hashes or sha(path) != hashes[name]:
        raise ValueError("saved discovery lacks its captured severity policy")
    assignments = {
        node.target.id: node.value
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    try:
        version_node = assignments["ACTIVE_POLICY_VERSION"]
        if version_node is None:
            raise ValueError("saved severity policy has no literal version")
        version = ast.literal_eval(version_node)
        mapping = assignments["SEVERITY_POLICY"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "saved severity policy is missing its literal version or mapping"
        ) from error
    if (
        not isinstance(version, str)
        or re.fullmatch(BARE_SEMVER_PATTERN, version) is None
        or not isinstance(mapping, ast.Call)
        or not isinstance(mapping.func, ast.Name)
        or mapping.func.id != "MappingProxyType"
        or len(mapping.args) != 1
        or not isinstance(mapping.args[0], ast.Dict)
    ):
        raise ValueError("saved severity policy has an unsupported literal shape")
    policy = {}
    for key, value in zip(mapping.args[0].keys, mapping.args[0].values, strict=True):
        if (
            not isinstance(key, ast.Attribute)
            or not isinstance(key.value, ast.Name)
            or key.value.id != "FindingType"
            or not isinstance(value, ast.Attribute)
            or not isinstance(value.value, ast.Name)
            or value.value.id != "FindingSeverity"
        ):
            raise ValueError("saved severity policy contains a nonliteral category mapping")
        finding_type, severity = FindingType[key.attr], FindingSeverity[value.attr]
        if finding_type in policy:
            raise ValueError("saved severity policy repeats a category")
        policy[finding_type] = severity
    if not policy:
        raise ValueError("saved severity policy is empty")
    return version, MappingProxyType(policy)


def verify_saved_inputs(source: SavedCheckInputs) -> None:
    """Fail before generation if any pinned artifact changed since preparation."""
    for name, digest in source.hashes.items():
        if sha(artifact(source.directory, name)) != digest:
            raise ValueError(f"saved check input changed: {name}")


def _read_report(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    receipts_path = artifact(directory, "receipts.json")
    receipts = json.loads(receipts_path.read_bytes())
    if not isinstance(receipts, dict) or not {"plan.json", "results.json"} <= receipts.keys():
        raise ValueError("saved retry requires receipt-bound plan and results")
    hashes = {"receipts.json": sha(receipts_path)}
    for name, digest in receipts.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("saved receipt digest is invalid")
        if sha(artifact(directory, name)) != digest:
            raise ValueError(f"original report receipt differs: {name}")
        hashes[name] = digest
    plan = json.loads(artifact(directory, "plan.json").read_bytes())
    result = json.loads(artifact(directory, "results.json").read_bytes())
    if (
        result.get("schema") != "review-comparison-v1"
        or result.get("status") != "incomplete"
        or result.get("plan") != plan
        or plan.get("comparison_mode") != "candidate-checks"
    ):
        raise ValueError("retry requires an incomplete candidate-checks comparison")
    _check_call_inventory(directory, result, receipts)
    for name, digest in plan["inputs"].items():
        saved_name = "generation/" + name
        if sha(artifact(directory, saved_name)) != digest:
            raise ValueError(f"original generation input differs: {name}")
        hashes[saved_name] = digest
    return plan, result, hashes


def _check_call_inventory(
    directory: Path, result: dict[str, Any], receipts: dict[str, str]
) -> None:
    """Do not overlook a billed check hidden by an incomplete result export."""
    expected: set[str] = set()
    for row in result["cases"]:
        names = row["raw_exchange_files"]
        if len(names) != len(row["calls"]):
            raise ValueError("saved source has an incomplete model-call inventory")
        for name, call in zip(names, row["calls"], strict=True):
            if not name.endswith(".exchange.json") or name in expected:
                raise ValueError("saved source repeats or misnames an exchange")
            attempt = name.removesuffix(".exchange.json") + ".attempt.json"
            if name not in receipts or attempt not in receipts:
                raise ValueError("saved call lacks paired exchange and attempt receipts")
            if json.loads(artifact(directory, attempt).read_bytes()) != call:
                raise ValueError("saved attempt differs from its result-row call")
            expected.update((name, attempt))
    recorded = {
        name
        for name in receipts
        if not name.startswith("generation/") and name.endswith((".exchange.json", ".attempt.json"))
    }
    existing = {
        str(path.relative_to(directory))
        for suffix in ("*.exchange.json", "*.attempt.json")
        for path in directory.rglob(suffix)
        if path.relative_to(directory).parts[0] != "generation"
    }
    if recorded != expected or existing != expected:
        raise ValueError("saved source contains unaccounted model-call artifacts")


def _load_exchanges(
    directory: Path,
    row: dict[str, Any],
    hashes: dict[str, str],
    target: Target,
    plan: dict[str, Any],
) -> tuple[tuple[LLMCallEvent, LLMRequest, LLMResponse], ...]:
    exchanges = []
    names = row["raw_exchange_files"]
    if len(names) != len(set(names)) or len(names) != len(row["calls"]):
        raise ValueError("saved discovery has duplicate or missing exchanges")
    call_ids = {
        call["audit_event_id"]
        for call in row["calls"]
        if call["status"] == "completed" and call["persisted_exchange"] is True
    }
    if len(call_ids) != len(names):
        raise ValueError("retry requires completed saved discovery calls")
    for name in names:
        if name not in hashes:
            raise ValueError("saved discovery exchange lacks a receipt")
        raw = json.loads(artifact(directory, name).read_bytes())
        event = LLMCallEvent.model_validate(raw["event"])
        request = LLMRequest.model_validate(raw["request"])
        response = LLMResponse.model_validate(raw["response"])
        if (
            str(event.event_id) not in call_ids
            or raw.get("target") != plan["target"]
            or event.event_id != response.exchange_event_id
            or event.timestamp != response.exchange_recorded_at
            or event.review_id != request.review_id
            or not event.is_eval
            or not request.is_eval
            or request.node_id not in {"triage", "analyze"}
            or event.node_id != request.node_id
            or request.candidate_attempt_id is not None
            or event.candidate_attempt_id is not None
            or request.model != target.model
            or event.model != target.model
            or response.model != target.model
            or response.finish_reason not in {"end_turn", "stop"}
            or event.finish_reason != response.finish_reason
            or event.prompt_hash
            != _canonical_prompt_hash(
                system_prompt=request.system_prompt, user_prompt=request.user_prompt
            )
            or event.system_prompt_hash != _canonical_system_prompt_hash(request.system_prompt)
            or event.response_format_digest != request.response_format_digest
            or event.context_summary != request.context_summary
        ):
            raise ValueError("saved discovery exchange identity differs from its request")
        for key in ("profile_id", "reasoning_enabled", "profile_contract_digest"):
            if getattr(event, key) != plan[key] or getattr(response, key) != plan[key]:
                raise ValueError("saved discovery provider identity differs from its plan")
        exchanges.append((event, request, response))
    if len({event.event_id for event, _, _ in exchanges}) != len(exchanges):
        raise ValueError("saved discovery repeats an exchange identity")
    return tuple(exchanges)


def _original_finding(
    ref: NativeCandidateRef,
    response: LLMResponse,
    installation_id: int,
    *,
    policy_version: str,
    severity_policy: Mapping[FindingType, FindingSeverity],
) -> ReviewFinding | None:
    """Restore a recorded admission without changing its original classification."""
    raw = json.loads(strip_outer_json_fence(response.text))["findings"][ref.proposal_index]
    if ref.initial_disposition == "malformed":
        return None
    proposal = AnalyzeFindingProposalRaw.model_validate(raw)
    digest = compute_proposal_hash(
        source_file_path=ref.file_path,
        source_side=ref.source_side,
        **proposal.model_dump(exclude={"trace_candidates"}),
    )
    if digest != ref.proposal_hash:
        raise ValueError("saved proposal differs from its recorded occurrence")
    if ref.initial_disposition != "admitted":
        return None
    if (
        proposal.finding_type != ref.finding_type
        or proposal.evidence_tier != ref.evidence_tier
        or ref.policy_version != policy_version
        or ref.finding_id is None
        or ref.finding_type is None
        or ref.severity is None
        or ref.evidence_tier is None
        or ref.content_hash is None
        or ref.proposal_hash is None
        or severity_policy.get(ref.finding_type) != ref.severity
    ):
        raise ValueError("saved admission differs from its category, tier or policy")
    return ReviewFinding(
        finding_id=ref.finding_id,
        review_id=ref.review_id,
        installation_id=installation_id,
        policy_version=ref.policy_version,
        finding_type=ref.finding_type,
        dimension=FINDING_TYPE_TO_DIMENSION[ref.finding_type],
        severity=ref.severity,
        evidence_tier=ref.evidence_tier,
        file_path=ref.file_path,
        source_side=ref.source_side,
        content_hash=ref.content_hash,
        proposal_hash=ref.proposal_hash,
        **proposal.model_dump(exclude={"finding_type", "evidence_tier", "trace_candidates"}),
    )


def _load_case(
    directory: Path,
    row: dict[str, Any],
    metadata: dict[str, Any],
    hashes: dict[str, str],
    target: Target,
    plan: dict[str, Any],
    policy_version: str,
    severity_policy: Mapping[FindingType, FindingSeverity],
    prepare_new_checks: bool,
) -> SavedCheckCase:
    if (
        row.get("status") != "failed"
        or row.get("error_type") != "ValidationError"
        or row.get("diagnostics_status") != "captured"
    ):
        raise ValueError("retry supports only failed pretransport discovery")
    events = row["diagnostic_events"]
    if any(
        item.get("event_type") in {"candidate_check_started", "candidate_assessment"}
        for item in events
    ):
        raise ValueError("recorded check attempts must not be billed again")
    fixture_name = "generation/" + metadata["fixture"]
    if hashes.get(fixture_name) != metadata["fixture_sha256"]:
        raise ValueError("saved fixture is not bound to its recorded case")
    fixture = EvalFixture.model_validate_json(artifact(directory, fixture_name).read_bytes())
    context = PRContext(
        **fixture.model_dump(include=set(PRContext.model_fields) - {"changed_files"}),
        changed_files=tuple(
            ChangedFile.model_validate(item.model_dump()) for item in fixture.files
        ),
    )
    exchanges = _load_exchanges(directory, row, hashes, target, plan)
    review_ids = {event.review_id for event, _, _ in exchanges}
    triage_responses = [
        response for _, request, response in exchanges if request.node_id == "triage"
    ]
    if len(review_ids) != 1 or len(triage_responses) != 1:
        raise ValueError("saved case requires one review and one completed triage")
    review_id = next(iter(review_ids))
    raw_triage = json.loads(strip_outer_json_fence(triage_responses[0].text))
    if not isinstance(raw_triage, dict):
        raise ValueError("saved triage must be a JSON object")
    triage = TriageResult.model_validate({"policy_version": policy_version, **raw_triage})
    if triage.policy_version != policy_version or set(triage.file_tiers) != {
        file.path for file in fixture.files
    }:
        raise ValueError("saved triage differs from captured files or captured policy")
    refs = []
    source_timestamps: dict[str, datetime] = {}
    completions = []
    for item in events:
        if UUID(item["review_id"]) != review_id or item["is_eval"] is not True:
            raise ValueError("saved diagnostic belongs to another review or evaluation scope")
        if item["event_type"] == "native_candidate":
            candidate_event = NativeCandidateEvent.model_validate(item["payload"])
            if (
                candidate_event.review_id != review_id
                or not candidate_event.is_eval
                or candidate_event.event_id != UUID(item["event_id"])
                or candidate_event.event_id
                != candidate_event_id(candidate_event.candidate.candidate_id)
            ):
                raise ValueError("saved candidate event identity differs from its occurrence")
            refs.append(candidate_event.candidate)
            source_timestamps[candidate_event.candidate.candidate_id] = candidate_event.timestamp
        elif item["event_type"] == "analyze_completed":
            completions.append(AnalyzeCompletedEvent.model_validate(item["payload"]))
    if (
        not refs
        or len({ref.candidate_id for ref in refs}) != len(refs)
        or any(ref.pass_index != 0 for ref in refs)
        or len(completions) != 1
        or completions[0].pass_index != 0
        or completions[0].review_id != review_id
        or not completions[0].is_eval
        or completions[0].policy_version != triage.policy_version
        or completions[0].selected_finding_ids is None
    ):
        raise ValueError("saved pass-zero candidate inventory is incomplete")
    responses = {}
    findings = {}
    for event, request, response in exchanges:
        if request.node_id != "analyze":
            continue
        headers = re.findall(r"(?m)^File: ([^\r\n]+)$", request.user_prompt)
        if len(headers) != 1 or headers[0] not in triage.file_tiers or request.degraded_mode:
            raise ValueError("saved analyzer must identify one captured clean source")
        occurrences = [ref for ref in refs if ref.source_exchange_event_id == event.event_id]
        raw = json.loads(strip_outer_json_fence(response.text))
        if (
            not isinstance(raw, dict)
            or set(raw) != {"findings"}
            or not isinstance(raw["findings"], list)
        ):
            raise ValueError("saved analyzer response has no complete candidate inventory")
        if {ref.proposal_index for ref in occurrences} != set(range(len(raw["findings"]))):
            raise ValueError("saved analyzer occurrences do not cover its original response")
        for ref in occurrences:
            if (
                ref.file_path != headers[0]
                or ref.response_hash != compute_response_hash(response.text)
                or source_timestamps[ref.candidate_id] != event.timestamp
            ):
                raise ValueError("saved candidate differs from its analyzer source")
            finding = _original_finding(
                ref,
                response,
                fixture.installation_id,
                policy_version=policy_version,
                severity_policy=severity_policy,
            )
            if finding is not None:
                findings[finding.finding_id] = finding
        responses[event.event_id] = response
    if set(ref.source_exchange_event_id for ref in refs) - responses.keys():
        raise ValueError("saved candidate source exchange is missing")
    selected = completions[0].selected_finding_ids
    if set(selected) - findings.keys():
        raise ValueError("saved selected findings require unsupported non-native reconstruction")
    state = ReviewState(
        review_id=review_id,
        pr_context=context,
        received_at=min(event.timestamp for event, _, _ in exchanges),
        is_eval=True,
        triage_result=triage,
        native_candidates=refs,
    )
    prepared: list[PreparedCandidateCheck] = []
    for ref in sorted(
        refs,
        key=lambda item: (
            item.initial_disposition == "admitted",
            item.pass_index,
            item.file_path,
            item.proposal_index,
            item.candidate_id,
        ),
    ):
        if not prepare_new_checks:
            break
        if len(prepared) >= CANDIDATE_CHECK_CALL_LIMIT:
            break
        check = prepare_candidate_check(
            state,
            ref,
            responses[ref.source_exchange_event_id].text,
            model=target.model,
            import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
        )
        if check is not None:
            prepared.append(check)
    return SavedCheckCase(
        row["case_id"],
        row["rep"],
        state,
        tuple(refs),
        responses,
        tuple(findings[key] for key in selected),
        tuple(prepared),
        sum((Decimal(call["usage_based_usd"]) for call in row["calls"]), Decimal()),
    )


def load_saved_checks(
    directory: Path, *, case_ids: tuple[str, ...] = (), prepare_new_checks: bool = True
) -> SavedCheckInputs:
    """Prepare only unfinished checks; saved discovery is never sent to a provider."""
    directory = directory.resolve()
    plan, result, hashes = _read_report(directory)
    policy_version, severity_policy = _captured_policy(directory, hashes)
    if prepare_new_checks and policy_version != ACTIVE_POLICY_VERSION:
        raise ValueError("historical policy permits saved-answer replay, not new check generation")
    target = Target(**plan["target"])
    metadata = {case["id"]: case for case in plan["cases"]}
    rows = [row for row in result["cases"] if row["arm"] == "concise-with-checks"]
    if len({(row["case_id"], row["rep"]) for row in rows}) != len(rows):
        raise ValueError("saved retry repeats a case or repetition")
    if case_ids:
        if len(set(case_ids)) != len(case_ids) or set(case_ids) - {row["case_id"] for row in rows}:
            raise ValueError("select distinct saved candidate-check case IDs")
        rows = [row for row in rows if row["case_id"] in case_ids]
    if not rows or any(row["case_id"] not in metadata for row in rows):
        raise ValueError("saved retry has no matching failed cases")
    cases = tuple(
        _load_case(
            directory,
            row,
            metadata[row["case_id"]],
            hashes,
            target,
            plan,
            policy_version,
            severity_policy,
            prepare_new_checks,
        )
        for row in rows
    )
    return SavedCheckInputs(
        directory,
        target,
        hashes,
        cases,
        sum(
            (Decimal(value["known_usage_based_usd"]) for value in result["budgets"].values()),
            Decimal(),
        ),
        policy_version,
        severity_policy,
    )
