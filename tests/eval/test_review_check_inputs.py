"""Keep saved check preparation independent of new discovery and provider calls."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from outrider.agent.native_candidates import candidate_event_id
from outrider.audit.events import (
    AnalyzeCompletedEvent,
    ContextManifestEntry,
    LLMCallEvent,
    NativeCandidateEvent,
    compute_finding_content_hash,
)
from outrider.llm.base import (
    INCLUDE_TEXT_OPT_IN,
    LLMRequest,
    LLMResponse,
    _canonical_prompt_hash,
    _canonical_system_prompt_hash,
)
from outrider.llm.host_profiles import ANTHROPIC_CONTRACT_DIGEST
from outrider.llm.pricing import PRICING_VERSION
from outrider.policy import FindingType
from outrider.policy.canonical import compute_proposal_hash, compute_response_hash
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.native_candidate import NativeCandidateRef, native_candidate_id

from .model_refresh import TARGETS
from .review_check_inputs import load_saved_checks, verify_saved_inputs
from .review_diagnostics_inputs import sha

_SOURCE = (
    "def load(user):\n    return checked_name(user)\n\n"
    "def checked_name(user):\n    return user.name\n"
)
_PROFILE = {
    "profile_id": "anthropic",
    "reasoning_enabled": False,
    "profile_contract_digest": ANTHROPIC_CONTRACT_DIGEST,
}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _receipts(directory):
    _write(
        directory / "receipts.json",
        {
            str(path.relative_to(directory)): sha(path)
            for path in directory.rglob("*.json")
            if path.name != "receipts.json"
        },
    )


def _saved_report(directory, *, policy_version=ACTIVE_POLICY_VERSION):
    now, review_id = datetime.now(UTC), uuid4()
    model = TARGETS["anthropic-sonnet-5-5"].model
    fixture = {
        "installation_id": 1,
        "owner": "example",
        "repo": "sample",
        "pr_number": 1,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "pr_title": "Read a record",
        "author": "local",
        "total_additions": 5,
        "total_deletions": 0,
        "files": [
            {
                "path": "app.py",
                "status": "added",
                "additions": 5,
                "deletions": 0,
                "content_head": _SOURCE,
            }
        ],
        "llm_responses": {},
    }
    fixture_path = directory / "generation/fixture.json"
    _write(fixture_path, fixture)
    policy_name = "src/outrider/policy/severity.py"
    policy_path = directory / "generation" / policy_name
    policy_path.parent.mkdir(parents=True)
    policy_source = (Path(__file__).resolve().parents[2] / policy_name).read_text()
    if policy_version != ACTIVE_POLICY_VERSION:
        policy_source = (
            policy_source.replace(
                f'ACTIVE_POLICY_VERSION: Final[str] = "{ACTIVE_POLICY_VERSION}"',
                f'ACTIVE_POLICY_VERSION: Final[str] = "{policy_version}"',
            )
            .replace('    CORRECTNESS_BUG = "correctness_bug"\n', "")
            .replace("        FindingType.CORRECTNESS_BUG: FindingSeverity.MEDIUM,\n", "")
        )
    policy_path.write_text(policy_source)
    proposal = {
        "finding_type": "auth_bypass",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "Check the account",
        "description": "Caller identity is unclear.",
        "evidence": "    return checked_name(user)",
        "line_start": 2,
        "line_end": 2,
    }
    payload = {"findings": [proposal, {**proposal, "finding_type": "unknown_account_category"}]}
    triage = {
        "file_tiers": {"app.py": "deep"},
        "overall_risk": "medium",
        "relevant_dimensions": ["security"],
        "reasoning": "Local source inventory",
    }
    exchanges = []
    for node, text in (("triage", json.dumps(triage)), ("analyze", json.dumps(payload))):
        request = LLMRequest(
            model=model,
            system_prompt="Review the supplied source.",
            user_prompt=f"File: app.py\n{_SOURCE}",
            max_tokens=2048,
            temperature=0,
            review_id=review_id,
            node_id=node,
            is_eval=True,
            prompt_template_version="test-v1",
            degraded_mode=False,
            context_summary=()
            if node == "triage"
            else (
                ContextManifestEntry(
                    file_path="app.py",
                    scope_unit_name="load",
                    line_start=1,
                    line_end=2,
                    inclusion_reason="changed_scope",
                ),
            ),
        )
        event = LLMCallEvent(
            review_id=review_id,
            timestamp=now,
            is_eval=True,
            model=model,
            node_id=node,
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            cost_usd=0,
            pricing_version=PRICING_VERSION,
            latency_ms=0,
            prompt_hash=_canonical_prompt_hash(
                system_prompt=request.system_prompt, user_prompt=request.user_prompt
            ),
            system_prompt_hash=_canonical_system_prompt_hash(request.system_prompt),
            cache_hit=False,
            context_summary=request.context_summary,
            prompt_template_version=request.prompt_template_version,
            degraded_mode=False,
            finish_reason="end_turn",
            **_PROFILE,
        )
        response = LLMResponse(
            text=text,
            model=model,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=0,
            exchange_event_id=event.event_id,
            exchange_recorded_at=now,
            **_PROFILE,
        )
        name = f"calls/{node}.exchange.json"
        _write(
            directory / name,
            {
                "target": asdict(TARGETS["anthropic-sonnet-5-5"]),
                "event": event.model_dump(mode="json"),
                "request": request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
                "response": response.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
            },
        )
        exchanges.append((name, event, response))
    _, source_event, source_response = exchanges[-1]
    finding_id = uuid4()
    candidates = []
    for index, entry in enumerate(payload["findings"]):
        admitted = index == 0
        ref = NativeCandidateRef(
            candidate_id=native_candidate_id(
                review_id=review_id,
                source_exchange_event_id=source_event.event_id,
                response_hash=compute_response_hash(source_response.text),
                proposal_index=index,
                file_path="app.py",
                pass_index=0,
            ),
            review_id=review_id,
            source_exchange_event_id=source_event.event_id,
            response_hash=compute_response_hash(source_response.text),
            proposal_index=index,
            file_path="app.py",
            pass_index=0,
            initial_disposition="admitted" if admitted else "rejected",
            rejection_reason=None if admitted else "finding_type_not_in_enum",
            proposal_hash=compute_proposal_hash(source_file_path="app.py", **entry),
            finding_id=finding_id if admitted else None,
            content_hash=compute_finding_content_hash(
                "app.py",
                line_start=2,
                line_end=2,
                finding_type=FindingType.AUTH_BYPASS,
            )
            if admitted
            else None,
            finding_type="auth_bypass" if admitted else None,
            severity="critical" if admitted else None,
            evidence_tier="judged" if admitted else None,
            policy_version=policy_version if admitted else None,
        )
        candidates.append(
            NativeCandidateEvent(
                event_id=candidate_event_id(ref.candidate_id),
                review_id=review_id,
                timestamp=now,
                is_eval=True,
                candidate=ref,
            )
        )
    completed = AnalyzeCompletedEvent(
        review_id=review_id,
        timestamp=now,
        is_eval=True,
        pass_index=0,
        n_files_analyzed=1,
        n_files_skipped=0,
        n_llm_calls=1,
        n_proposals_seen=2,
        n_findings_emitted=1,
        n_proposals_rejected=1,
        n_responses_rejected=0,
        n_trace_candidates_emitted=0,
        total_input_tokens=0,
        total_cache_read_tokens=0,
        total_cache_write_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0,
        pricing_version=PRICING_VERSION,
        policy_version=policy_version,
        analyze_model=model,
        selected_finding_ids=(finding_id,),
        **_PROFILE,
    )
    diagnostics = [
        {
            "event_type": event.event_type,
            "event_id": str(event.event_id),
            "review_id": str(review_id),
            "is_eval": True,
            "payload": event.model_dump(mode="json"),
        }
        for event in [*candidates, completed]
    ]
    plan = {
        "comparison_mode": "candidate-checks",
        "target": asdict(TARGETS["anthropic-sonnet-5-5"]),
        "inputs": {"fixture.json": sha(fixture_path), policy_name: sha(policy_path)},
        "cases": [{"id": "local", "fixture": "fixture.json", "fixture_sha256": sha(fixture_path)}],
        **_PROFILE,
    }
    _write(directory / "plan.json", plan)
    _write(
        directory / "results.json",
        {
            "schema": "review-comparison-v1",
            "status": "incomplete",
            "plan": plan,
            "budgets": {"concise-with-checks": {"known_usage_based_usd": "0"}},
            "cases": [
                {
                    "arm": "concise-with-checks",
                    "case_id": "local",
                    "rep": 0,
                    "status": "failed",
                    "error_type": "ValidationError",
                    "diagnostics_status": "captured",
                    "diagnostic_events": diagnostics,
                    "raw_exchange_files": [name for name, _, _ in exchanges],
                    "calls": [
                        {
                            "audit_event_id": str(event.event_id),
                            "status": "completed",
                            "persisted_exchange": True,
                            "usage_based_usd": "0",
                        }
                        for _, event, _ in exchanges
                    ],
                }
            ],
        },
    )
    result = json.loads((directory / "results.json").read_text())
    for name, call in zip(
        result["cases"][0]["raw_exchange_files"], result["cases"][0]["calls"], strict=True
    ):
        _write(directory / name.replace(".exchange.json", ".attempt.json"), call)
    _receipts(directory)
    return directory, finding_id


def test_saved_inputs_preserve_originals_and_prepare_only_missing_checks(tmp_path):
    directory, finding_id = _saved_report(tmp_path / "source")
    source = load_saved_checks(directory)
    [case] = source.cases
    assert len(case.candidates) == len(case.prepared_checks) == 2
    assert [finding.finding_id for finding in case.original_findings] == [finding_id]
    assert case.prepared_checks[0].check.candidate_id == case.candidates[1].candidate_id
    assert all(check.request.node_id == "trace" for check in case.prepared_checks)
    assert all(check.request.review_id == case.state.review_id for check in case.prepared_checks)
    assert case.state.pr_context.installation_id == 1
    assert len(case.state.native_candidates) == 2
    verify_saved_inputs(source)


def test_historical_policy_binds_saved_admissions_without_generating_new_checks(tmp_path):
    from .review_check_retry import build_plan

    directory, finding_id = _saved_report(tmp_path / "source", policy_version="1.2.0")
    source = load_saved_checks(directory, prepare_new_checks=False)
    (case,) = source.cases
    assert case.state.triage_result.policy_version == "1.2.0"
    assert [(finding.finding_id, finding.policy_version) for finding in case.original_findings] == [
        (finding_id, "1.2.0")
    ]
    assert FindingType.CORRECTNESS_BUG not in source.severity_policy
    assert case.prepared_checks == ()
    with pytest.raises(ValueError, match="not new check generation"):
        load_saved_checks(directory)
    with pytest.raises(ValueError, match="not new check generation"):
        build_plan(source)

    record_path = directory / "results.json"
    record = json.loads(record_path.read_text())
    completed = next(
        event
        for event in record["cases"][0]["diagnostic_events"]
        if event["event_type"] == "analyze_completed"
    )
    completed["payload"]["policy_version"] = ACTIVE_POLICY_VERSION
    _write(record_path, record)
    _receipts(directory)
    with pytest.raises(ValueError, match="inventory is incomplete"):
        load_saved_checks(directory, prepare_new_checks=False)


@pytest.mark.parametrize("mutation", ["triage_version", "candidate_version", "captured_severity"])
def test_historical_policy_cannot_drift_from_its_captured_admission(tmp_path, mutation):
    directory, _ = _saved_report(tmp_path / "source", policy_version="1.2.0")
    if mutation == "triage_version":
        path = directory / "calls/triage.exchange.json"
        record = json.loads(path.read_text())
        payload = json.loads(record["response"]["text"])
        payload["policy_version"] = ACTIVE_POLICY_VERSION
        record["response"]["text"] = json.dumps(payload)
        _write(path, record)
    elif mutation == "candidate_version":
        path = directory / "results.json"
        record = json.loads(path.read_text())
        candidate = next(
            item
            for item in record["cases"][0]["diagnostic_events"]
            if item["event_type"] == "native_candidate"
            and item["payload"]["candidate"]["initial_disposition"] == "admitted"
        )
        candidate["payload"]["candidate"]["policy_version"] = ACTIVE_POLICY_VERSION
        _write(path, record)
    else:
        policy_name = "src/outrider/policy/severity.py"
        path = directory / "generation" / policy_name
        path.write_text(
            path.read_text().replace(
                "FindingType.AUTH_BYPASS: FindingSeverity.CRITICAL",
                "FindingType.AUTH_BYPASS: FindingSeverity.LOW",
            )
        )
        plan = json.loads((directory / "plan.json").read_text())
        plan["inputs"][policy_name] = sha(path)
        _write(directory / "plan.json", plan)
        result = json.loads((directory / "results.json").read_text())
        result["plan"] = plan
        _write(directory / "results.json", result)
    _receipts(directory)
    with pytest.raises(ValueError, match="saved (triage|admission) differs"):
        load_saved_checks(directory, prepare_new_checks=False)


@pytest.mark.parametrize("mutation", ["source_scope", "existing_start", "generation_drift"])
def test_saved_inputs_refuse_scope_restart_and_source_drift(tmp_path, mutation):
    directory, _ = _saved_report(tmp_path / "source")
    if mutation == "source_scope":
        path = directory / "calls/analyze.exchange.json"
        record = json.loads(path.read_text())
        record["event"]["review_id"] = str(uuid4())
        _write(path, record)
        _receipts(directory)
    elif mutation == "existing_start":
        path = directory / "results.json"
        record = json.loads(path.read_text())
        record["cases"][0]["diagnostic_events"].append({"event_type": "candidate_check_started"})
        _write(path, record)
        _receipts(directory)
    else:
        path = directory / "generation/fixture.json"
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        load_saved_checks(directory)


def test_saved_inputs_recheck_the_complete_pinned_source_before_reuse(tmp_path):
    directory, _ = _saved_report(tmp_path / "source")
    source = load_saved_checks(directory)
    path = directory / "calls/analyze.exchange.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="saved check input changed"):
        verify_saved_inputs(source)


def test_saved_inputs_reject_receipt_path_escape(tmp_path):
    directory, _ = _saved_report(tmp_path / "source")
    receipts = json.loads((directory / "receipts.json").read_text())
    receipts["../outside.json"] = "a" * 64
    _write(directory / "receipts.json", receipts)
    with pytest.raises(ValueError, match="inside the report"):
        load_saved_checks(directory)


@pytest.mark.parametrize("receipted", [False, True])
def test_saved_inputs_refuse_unlisted_call_artifacts_before_rebilling(tmp_path, receipted):
    directory, _ = _saved_report(tmp_path / "source")
    _write(directory / "calls/unlisted-check.exchange.json", {"note": "interrupted call"})
    if receipted:
        _receipts(directory)
    with pytest.raises(ValueError, match="unaccounted model-call artifacts"):
        load_saved_checks(directory)
