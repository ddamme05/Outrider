"""Check native retention and bounded scripted follow-up without a discovery call."""

from __future__ import annotations

import hashlib
import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.agent.eval_driver import EvalFixture, EvalFixtureFile
from outrider.audit.events import ContextManifestEntry
from outrider.llm.base import LLMRequest, LLMResponse

from .direct_first_candidates import CandidateLedgerError, build_ledger
from .native_candidate_review import (
    main,
    prepare_candidate_check,
    prepare_native_review,
    record_candidate_check,
    review_summary,
)
from .review_diagnostics_inputs import fields, request_identity


def _inputs():
    fixture = EvalFixture(
        installation_id=1,
        owner="example",
        repo="reviewed",
        pr_number=1,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title="Read a protected record",
        pr_body="PRIVATE_BODY",
        author="PRIVATE_AUTHOR",
        total_additions=1,
        total_deletions=0,
        files=(
            EvalFixtureFile(
                path="app.py",
                status="added",
                additions=1,
                deletions=0,
                content_head=(
                    "from guards import allowed\ndef read(user):\n    return allowed(user)\n"
                ),
            ),
        ),
        repository_contents_head={
            "guards.py": "def allowed(user):\n    return user.is_admin\n",
            "tests/test_read.py": "from app import read\ndef test_read(user):\n"
            "    assert read(user) is False\n",
        },
        llm_responses={},
    )
    request = LLMRequest(
        system_prompt="Review supplied code.",
        user_prompt="File: app.py\nCaptured source.",
        model="test-model",
        max_tokens=8192,
        temperature=0.0,
        review_id=uuid4(),
        node_id="analyze",
        is_eval=True,
        prompt_template_version="saved-native-test",
        degraded_mode=False,
        context_summary=(
            ContextManifestEntry(
                file_path="app.py",
                scope_unit_name="read",
                line_start=2,
                line_end=3,
                inclusion_reason="changed_scope",
            ),
        ),
    )
    return fixture, request


def _proposal(**changes):
    return {
        "finding_type": "authorization_gap",
        "evidence_tier": "judged",
        "title": "Check authorization",
        "description": "The caller may skip authorization.",
        "evidence": "    return allowed(user)",
        "line_start": 3,
        "line_end": 3,
        **changes,
    }


def _response(request, text):
    return LLMResponse(
        text=text,
        model=request.model,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        finish_reason="end_turn",
        latency_ms=0,
    )


def _refutation(candidate_id, **changes):
    return {
        "candidate_id": candidate_id,
        "status": "refutes",
        "reason": "The captured helper checks the admin flag; still a provisional assessment.",
        "citations": [
            {
                "file_path": "guards.py",
                "line_start": 2,
                "line_end": 2,
                "quote": "    return user.is_admin",
            }
        ],
        "suggested_finding_type": "auth_bypass",
        **changes,
    }


@pytest.mark.parametrize("fenced", [False, True])
def test_native_unknown_and_malformed_siblings_remain_visible_and_investigable(fenced):
    fixture, request = _inputs()
    entries = [_proposal(), _proposal(file_path="untrusted.py"), None]
    raw = json.dumps({"findings": entries}, indent=2)
    if fenced:
        raw = "```json\n" + raw + "\n```"
    review = prepare_native_review(fixture, request, _response(request, raw))
    ledger = review.ledger
    assert ledger.raw_response == raw
    assert len(ledger.candidates) == 3
    assert len({item.candidate_id for item in ledger.candidates}) == 3
    assert ledger.source_file_path == "app.py"
    assert not ledger.candidates[0].errors
    assert ledger.candidates[1].errors
    assert ledger.candidates[1].proposal.file_path == "app.py"
    assert ledger.candidates[2].proposal is None
    for item, original in zip(ledger.candidates, entries, strict=True):
        assert json.loads(item.raw_proposal_json) == original
    checks = [prepare_candidate_check(review, item.candidate_id) for item in ledger.candidates]
    assert checks[0] is not None and checks[1] is not None and checks[2] is None
    assert {entry.file_path for entry in checks[0].context_summary} == {
        "app.py",
        "guards.py",
        "tests/test_read.py",
    }
    assert "PRIVATE_BODY" not in checks[0].user_prompt
    assert review_summary(review)["candidates"][0]["classification_state"] == "needs-classification"
    ordinary = build_ledger('{"findings":[]}', source_id="old-direct", finish_reason="stop")
    assert "source_file_path" not in ordinary.model_dump(mode="json")


def test_cited_checks_append_category_suggestions_without_rewriting_or_admitting():
    fixture, request = _inputs()
    raw = json.dumps({"findings": [_proposal()]})
    original = prepare_native_review(fixture, request, _response(request, raw))
    candidate = original.ledger.candidates[0]
    check = prepare_candidate_check(original, candidate.candidate_id)
    answer = _refutation(candidate.candidate_id)
    updated = record_candidate_check(
        original,
        candidate.candidate_id,
        check,
        _response(check, json.dumps(answer)),
        origin="scripted",
    )
    assert not original.ledger.candidates[0].assessments
    assert updated.ledger.raw_response == raw
    assert updated.ledger.candidates[0].raw_proposal_json == candidate.raw_proposal_json
    assert updated.ledger.candidates[0].proposal.category == "authorization_gap"
    summary = review_summary(updated)["candidates"][0]
    assert summary["original_policy_category"] is None
    suggestion = summary["classification_suggestions"][0]
    assert suggestion["finding_type"] == "auth_bypass"
    assert suggestion["policy_severity"] == "critical"
    assert summary["native_admission"] == "not-evaluated"

    for change, error in (
        ({"suggested_finding_type": "approximately_auth"}, ValidationError),
        ({"severity": "low"}, CandidateLedgerError),
        (
            {
                "citations": [
                    {"file_path": "app.py", "line_start": 3, "line_end": 3, "quote": "invented()"}
                ]
            },
            CandidateLedgerError,
        ),
    ):
        with pytest.raises(error):
            record_candidate_check(
                updated,
                candidate.candidate_id,
                check,
                _response(check, json.dumps({**answer, **change})),
            )
    altered = check.model_copy(update={"user_prompt": check.user_prompt + "Extra instructions"})
    with pytest.raises(CandidateLedgerError, match="bounded preparation"):
        record_candidate_check(
            updated, candidate.candidate_id, altered, _response(check, json.dumps(answer))
        )
    uncertain = _refutation(
        candidate.candidate_id,
        status="uncertain",
        citations=[],
        reason="Runtime caller is unavailable.",
        suggested_finding_type=None,
    )
    final = record_candidate_check(
        updated, candidate.candidate_id, check, _response(check, json.dumps(uncertain))
    )
    assert review_summary(final)["candidates"][0]["classification_state"] == "uncertain"
    assert len(final.ledger.candidates[0].assessments) == 2


def test_free_command_verifies_receipts_prepares_and_replays_scripted_answers(tmp_path, capsys):
    fixture, request = _inputs()
    response = _response(request, json.dumps({"findings": [_proposal()]}))
    review = prepare_native_review(fixture, request, response)
    source = tmp_path / "source"
    source.mkdir()
    fixture_data = fixture.model_dump_json().encode()
    plan = {
        "inputs": {},
        "target": {"model": "test-model"},
        "cases": [
            {
                "id": "case",
                "fixture": "fixture.json",
                "fixture_sha256": hashlib.sha256(fixture_data).hexdigest(),
                "request_identities": {"native": request_identity(request)},
            }
        ],
    }
    row = {
        "case_id": "case",
        "arm": "native",
        "analysis_file": "app.py",
        "prepared_request": "prepared.json",
        "status": "completed",
        "raw_exchange_files": ["call.json"],
    }
    contents = {
        "plan.json": json.dumps(plan).encode(),
        "results.json": json.dumps(
            {
                "schema": "captured-review-diagnostics-v1",
                "status": "completed",
                "plan": plan,
                "cases": [row],
            }
        ).encode(),
        "generation/fixture.json": fixture_data,
        "prepared.json": json.dumps(fields(request)).encode(),
        "call.json": json.dumps(
            {
                "request": fields(request),
                "response": {
                    "text": response.text,
                    "model": response.model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "finish_reason": "end_turn",
                    "latency_ms": 0,
                },
            }
        ).encode(),
    }
    for name, data in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (source / "receipts.json").write_text(
        json.dumps({name: hashlib.sha256(data).hexdigest() for name, data in contents.items()})
    )
    scripted = tmp_path / "scripted.json"
    scripted.write_text(
        json.dumps(
            [
                {
                    "source_exchange": "call.json",
                    "assessment": _refutation(review.ledger.candidates[0].candidate_id),
                }
            ]
        )
    )
    output = tmp_path / "prepared"
    assert (
        main(
            [
                "--report",
                str(source),
                "--output",
                str(output),
                "--scripted-assessments",
                str(scripted),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["model_calls_made"] == summary["standalone_discovery_calls"] == 0
    assert summary["candidate_count"] == summary["unknown_category_count"] == 1
    assert summary["scripted_assessment_count"] == 1
    ledger = json.loads((output / "exchange-0000/ledger.json").read_bytes())
    assert ledger["raw_response"] == response.text
    assert ledger["candidates"][0]["assessments"][0]["origin"] == "scripted"
    (source / "call.json").write_text("changed")
    with pytest.raises(SystemExit):
        main(["--report", str(source), "--output", str(tmp_path / "must-not-exist")])
    assert not (tmp_path / "must-not-exist").exists()
