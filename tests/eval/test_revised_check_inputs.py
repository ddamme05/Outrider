"""Refuse broken historical controls before any revised-check provider exists."""

from __future__ import annotations

import json
from decimal import Decimal
from uuid import uuid5

import pytest

from outrider.agent.candidate_checks import _ATTEMPT_NAMESPACE
from outrider.llm.base import LLMRequest, _canonical_prompt_hash
from outrider.policy.canonical import compute_identity_hash

from . import review_check_retry as retry
from .review_diagnostics_inputs import sha
from .revised_check_inputs import load_previous_checks, verify_previous_checks
from .test_review_check_retry import RecordingCheckProvider, _source


async def _completed(tmp_path, monkeypatch):
    source, plan, _ = _source(tmp_path, monkeypatch)
    directory = tmp_path / "completed"
    await retry.collect_checks(
        source,
        plan,
        maximum_usd=Decimal("3"),
        maximum_calls=2,
        output=directory,
        provider_factory=lambda _target, recorder: RecordingCheckProvider(recorder),
    )
    return directory


def _write(path, value):
    path.write_text(json.dumps(value))


def _refresh_receipts(directory):
    _write(
        directory / "receipts.json",
        {
            str(path.relative_to(directory)): sha(path)
            for path in directory.rglob("*.json")
            if path.name != "receipts.json"
        },
    )


def _reseal_question(directory, exchange, results):
    """Keep forged request/audit/plan bindings consistent to test source semantics."""
    request = exchange["request"]
    prompt_hash = _canonical_prompt_hash(
        system_prompt=request["system_prompt"], user_prompt=request["user_prompt"]
    )
    started_path = directory / "calls/000001.started.json"
    started = json.loads(started_path.read_text())
    check = started["check"]
    check["prompt_hash"] = prompt_hash
    check["question_hash"] = compute_identity_hash(
        {
            "candidate_id": check["candidate_id"],
            "prompt_hash": prompt_hash,
            "version": request["prompt_template_version"],
            "model": request["model"],
        }
    )
    attempt = str(uuid5(_ATTEMPT_NAMESPACE, check["question_hash"]))
    check["attempt_id"] = request["candidate_attempt_id"] = attempt
    exchange["event"]["candidate_attempt_id"] = attempt
    exchange["event"]["prompt_hash"] = prompt_hash
    digest = retry._request_digest(LLMRequest.model_validate(request))
    started["request_sha256"] = digest
    selected = results["plan"]["checks"][0]
    selected["attempt_id"], selected["request_sha256"] = attempt, digest
    selected["request_text_bytes"] = sum(
        len(text.encode())
        for text in (
            request["system_prompt"],
            request["user_prompt"],
            request["response_schema_json"] or "",
        )
    )
    assessment_path = directory / "calls/000001.assessment.json"
    assessment = json.loads(assessment_path.read_text())
    assessment["attempt_id"] = attempt
    results["cases"][0]["assessments"][0]["attempt_id"] = attempt
    _write(assessment_path, assessment)
    _write(started_path, started)
    _write(directory / "calls/000001.exchange.json", exchange)
    _write(directory / "plan.json", results["plan"])
    _write(directory / "results.json", results)


@pytest.mark.asyncio
@pytest.mark.parametrize("receipted", [False, True])
async def test_orphan_call_refuses_even_when_it_has_a_receipt(tmp_path, monkeypatch, receipted):
    directory = await _completed(tmp_path, monkeypatch)
    (directory / "calls/999999.exchange.json").write_bytes(
        (directory / "calls/000001.exchange.json").read_bytes()
    )
    if receipted:
        _refresh_receipts(directory)
    with pytest.raises(ValueError, match="inventory|unaccounted"):
        load_previous_checks(directory)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage", ["model", "served_source", "claim", "assessment", "findings", "cost"]
)
async def test_rehashed_control_still_requires_exchange_source_and_cost_lineage(
    tmp_path, monkeypatch, damage
):
    directory = await _completed(tmp_path, monkeypatch)
    exchange_path = directory / "calls/000001.exchange.json"
    exchange = json.loads(exchange_path.read_text())
    results_path = directory / "results.json"
    results = json.loads(results_path.read_text())
    if damage == "model":
        exchange["event"]["model"] = "another-model"
        _write(exchange_path, exchange)
    elif damage in {"served_source", "claim"}:
        packet = json.loads(exchange["request"]["user_prompt"])
        if damage == "claim":
            packet["claim"]["description"] = "A different assumption about this code."
        else:
            packet["reference_snippets"][0]["source"] = "1 | invented source"
        exchange["request"]["user_prompt"] = json.dumps(packet)
        _reseal_question(directory, exchange, results)
    elif damage == "assessment":
        assessment_path = directory / "calls/000001.assessment.json"
        assessment = json.loads(assessment_path.read_text())
        assessment["reason_length"] += 1
        _write(assessment_path, assessment)
    elif damage == "findings":
        results["cases"][0]["original_findings"] = []
        _write(results_path, results)
    else:
        results["calls"][0]["usage_based_usd"] = "0"
        results["calls"][0]["budget_retained_upper_usd"] = "0"
        results["new_check_budget"]["known_usage_based_usd"] = "0.0007"
        _write(results_path, results)
        _write(directory / "calls/000001.attempt.json", results["calls"][0])
    _refresh_receipts(directory)
    with pytest.raises(ValueError):
        load_previous_checks(directory)


@pytest.mark.asyncio
async def test_prepared_control_refuses_later_artifact_or_discovery_changes(tmp_path, monkeypatch):
    directory = await _completed(tmp_path, monkeypatch)
    previous = load_previous_checks(directory)
    verify_previous_checks(previous)
    results_path = directory / "results.json"
    original = results_path.read_bytes()
    results_path.write_bytes(original + b"\n")
    _refresh_receipts(directory)
    with pytest.raises(ValueError, match="changed after preparation"):
        verify_previous_checks(previous)
    results_path.write_bytes(original)
    _refresh_receipts(directory)
    source = previous.source.directory / "calls/analyze.exchange.json"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="saved check input changed"):
        verify_previous_checks(previous)
