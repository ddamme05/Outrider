"""Exercise the saved-review command and scripted direct-first review end to end."""

from __future__ import annotations

import hashlib
import json
from uuid import uuid4

import pytest

from outrider.agent.eval_driver import EvalFixture, EvalFixtureFile
from outrider.llm.base import INCLUDE_TEXT_OPT_IN, LLMResponse, LLMTimeoutError

from . import direct_first, direct_review
from .model_refresh import ROOT


def _fixture() -> EvalFixture:
    fixture = EvalFixture.model_validate_json(
        (ROOT / "tests/eval/fixtures/mock_github/safe_refactor.json").read_bytes()
    )
    return fixture.model_copy(
        update={
            "files": (
                EvalFixtureFile(
                    path="app.py",
                    status="added",
                    additions=4,
                    deletions=0,
                    content_head=(
                        "def lookup(value):\n    return value + 1\n\n"
                        "def caller():\n    return lookup(0)\n"
                    ),
                ),
            ),
            "repository_contents_head": {},
            "pr_body": "PRIVATE_BODY",
            "author": "PRIVATE_AUTHOR",
        }
    )


def _proposal(**updates):
    return {
        "file_path": "app.py",
        "line_start": 2,
        "line_end": 2,
        "title": "A `lookup` compatibility claim",
        "description": "Inspect the changed calculation.",
        "evidence": "    return value + 1",
        "category": "unclassified compatibility",
        **updates,
    }


def _response(request, text, finish="end_turn"):
    return LLMResponse(
        text=text,
        model=request.model,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        finish_reason=finish,
        latency_ms=1,
    )


def _saved_report(directory, fixture, raw):
    request = direct_review.render_request(fixture, model="test-model", review_id=uuid4())
    plan = {
        "target": {"model": "test-model"},
        "cases": [
            {
                "id": "sample",
                "fixture": "fixtures/pr.json",
                "fixture_sha256": hashlib.sha256(fixture.model_dump_json().encode()).hexdigest(),
            }
        ],
    }
    exchange = {
        "request": request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN),
        "response": {"text": raw, "finish_reason": "end_turn", "model": "test-model"},
    }
    row = {
        "arm": "direct-one-shot",
        "case_id": "sample",
        "rep": 0,
        "raw_exchange_files": ["direct/call.json"],
        "status": "completed",
    }
    files = {
        "plan.json": json.dumps(plan).encode(),
        "results.json": json.dumps(
            {"schema": "review-comparison-v1", "plan": plan, "cases": [row]}
        ).encode(),
        "generation/fixtures/pr.json": fixture.model_dump_json().encode(),
        "direct/call.json": json.dumps(exchange).encode(),
    }
    directory.mkdir()
    for name, data in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (directory / "receipts.json").write_text(
        json.dumps({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    )
    return files


@pytest.mark.asyncio
@pytest.mark.parametrize("followups", [1, 2])
async def test_shared_discovery_survives_refutation_provider_failure_and_call_cap(
    tmp_path, followups
):
    fixture = _fixture()
    raw = json.dumps({"findings": [_proposal(), _proposal(description="A distinct second claim.")]})

    class Provider:
        requests = []

        async def complete(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                expected = direct_review.render_request(
                    fixture, model="test-model", review_id=request.review_id
                )
                assert request == expected
                return _response(request, raw)
            if len(self.requests) == 3:
                raise LLMTimeoutError("scripted timeout")
            candidate_id = json.loads(request.user_prompt)["candidate"]["candidate_id"]
            return _response(
                request,
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "status": "refutes",
                        "reason": "A model assessment only.",
                        "citations": [
                            {
                                "file_path": "app.py",
                                "line_start": 2,
                                "line_end": 2,
                                "quote": "    return value + 1",
                            }
                        ],
                    }
                ),
            )

    provider = Provider()
    output = tmp_path / "run"
    result = await direct_first.run_direct_first(
        fixture, provider=provider, model="test-model", output=output, max_followups=followups
    )
    seed = json.loads((output / "seed.json").read_text())
    assert seed["raw_response"] == raw
    assert len(provider.requests) == 1 + followups
    assert result["candidate_count"] == result["local_draft_candidate_count"] == 2
    assert result["native_delivery"] == "not-evaluated"
    assert result["status"] == ("incomplete" if followups == 2 else "prepared-unadjudicated")
    assert result["ledger"]["candidates"][0]["assessments"][0]["status"] == "refutes"
    assert not result["ledger"]["candidates"][1]["assessments"]
    assert (
        result["ledger"]["candidates"][0]["raw_proposal_json"]
        == seed["candidates"][0]["raw_proposal_json"]
    )
    assert "scripted timeout" not in json.dumps(result)
    draft = (output / "review.md").read_text()
    assert draft.count("## Candidate ") == 2
    assert "```json" in draft
    assert "Awaiting independent review" in draft
    assert (output / "stage-0000.json").is_file()
    assert (output / f"stage-{followups:04d}.json").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem", ["invented-citation", "wrong-id", "unfinished", "wrong-model", "unshown-citation"]
)
async def test_failed_investigation_cannot_replace_the_original(tmp_path, problem):
    fixture = _fixture().model_copy(
        update={"repository_contents_head": {"unrelated.py": "unused_value = 1\n"}}
    )
    raw = json.dumps({"findings": [_proposal()]})

    class Provider:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                return _response(request, raw)
            candidate_id = json.loads(request.user_prompt)["candidate"]["candidate_id"]
            answer = _response(
                request,
                json.dumps(
                    {
                        "candidate_id": "0" * 64 if problem == "wrong-id" else candidate_id,
                        "status": "supports",
                        "reason": "Unsupported investigator claim.",
                        "citations": [
                            {
                                "file_path": "unrelated.py"
                                if problem == "unshown-citation"
                                else "app.py",
                                "line_start": 1 if problem == "unshown-citation" else 2,
                                "line_end": 1 if problem == "unshown-citation" else 2,
                                "quote": "unused_value = 1"
                                if problem == "unshown-citation"
                                else (
                                    "invented()"
                                    if problem == "invented-citation"
                                    else "    return value + 1"
                                ),
                            }
                        ],
                    }
                ),
                finish="max_tokens" if problem == "unfinished" else "end_turn",
            )
            return (
                answer.model_copy(update={"model": "different-model"})
                if problem == "wrong-model"
                else answer
            )

    result = await direct_first.run_direct_first(
        fixture, provider=Provider(), model="test-model", output=tmp_path / "run"
    )
    assert result["status"] == "incomplete"
    assert result["ledger"]["raw_response"] == raw
    assert not result["ledger"]["candidates"][0]["assessments"]
    assert result["investigations"][0]["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["provider-error", "wrong-model", "invalid-json", "unfinished"])
async def test_discovery_failures_remain_incomplete_without_followup_calls(tmp_path, problem):
    raw = "not-json" if problem == "invalid-json" else json.dumps({"findings": [_proposal()]})

    class Provider:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            if problem == "provider-error":
                raise LLMTimeoutError("private provider detail")
            answer = _response(
                request, raw, finish="max_tokens" if problem == "unfinished" else "end_turn"
            )
            return (
                answer.model_copy(update={"model": "different"})
                if problem == "wrong-model"
                else answer
            )

    provider, output = Provider(), tmp_path / "run"
    if problem == "provider-error":
        with pytest.raises(LLMTimeoutError):
            await direct_first.run_direct_first(
                _fixture(), provider=provider, model="test-model", output=output
            )
        failure = json.loads((output / "failure.json").read_text())
        assert failure["status"] == "incomplete"
        assert failure["response_returned"] is False
        assert "private provider detail" not in json.dumps(failure)
    else:
        result = await direct_first.run_direct_first(
            _fixture(), provider=provider, model="test-model", output=output
        )
        assert result["status"] == "incomplete"
        assert result["ledger"]["raw_response"] == raw
        assert not any(item["assessments"] for item in result["ledger"]["candidates"])
    assert provider.calls == 1


def test_saved_command_prepares_every_occurrence_without_provider_or_database(
    tmp_path, capsys, monkeypatch
):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TEST_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    source, output = tmp_path / "saved", tmp_path / "prepared"
    raw = json.dumps({"findings": [_proposal(), _proposal(), None]})
    files = _saved_report(source, _fixture(), raw)
    arguments = ["--report", str(source), "--output", str(output)]
    assert direct_first.main(arguments) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["model_calls_made"] == 0
    assert summary["reviews"][0]["candidate_count"] == 3
    review = json.loads((output / "review-0000/review.json").read_text())
    candidates = review["ledger"]["candidates"]
    assert len({candidate["candidate_id"] for candidate in candidates}) == 3
    assert review["ledger"]["raw_response"] == raw
    assert (output / "review-0000/original.exchange.json").read_bytes() == files["direct/call.json"]
    assert "Invalid proposal" in (output / "review-0000/review.md").read_text()
    with pytest.raises(SystemExit) as existing:
        direct_first.main(arguments)
    assert existing.value.code == 2
    assert (output / "review-0000/original.exchange.json").read_bytes() == files["direct/call.json"]


@pytest.mark.parametrize("problem", ["tampered", "outside-report", "different-request"])
def test_saved_command_rejects_unbound_or_different_inputs(tmp_path, problem):
    source = tmp_path / "saved"
    _saved_report(source, _fixture(), json.dumps({"findings": [_proposal()]}))
    if problem == "tampered":
        (source / "direct/call.json").write_text("{}")
    elif problem == "outside-report":
        receipts = json.loads((source / "receipts.json").read_text())
        receipts["../outside.json"] = receipts.pop("direct/call.json")
        (source / "receipts.json").write_text(json.dumps(receipts))
    else:
        path = source / "direct/call.json"
        exchange = json.loads(path.read_text())
        exchange["request"]["user_prompt"] += "different context"
        path.write_text(json.dumps(exchange))
        receipts = json.loads((source / "receipts.json").read_text())
        receipts["direct/call.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (source / "receipts.json").write_text(json.dumps(receipts))
    output = tmp_path / "prepared"
    with pytest.raises((ValueError, KeyError)):
        direct_first.prepare_saved(source, output=output)
    assert not output.exists()
