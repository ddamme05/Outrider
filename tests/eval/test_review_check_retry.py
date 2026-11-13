"""Pay only for prepared checks, with explicit authorization and no repeat billing."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from outrider.audit.events import LLMCallEvent
from outrider.llm.base import LLMResponse, _canonical_prompt_hash, _canonical_system_prompt_hash
from outrider.llm.pricing import PRICING_VERSION

from . import review_check_retry as retry
from .review_check_inputs import load_saved_checks
from .review_diagnostics_inputs import sha
from .test_review_check_inputs import _PROFILE, _saved_report


def _source(tmp_path, monkeypatch):
    directory, finding_id = _saved_report(tmp_path / "source")
    source = load_saved_checks(directory)
    monkeypatch.setattr(retry, "input_hashes", lambda _: {})
    monkeypatch.setenv("OUTRIDER_IS_EVAL", "1")
    return source, retry.build_plan(source), finding_id


class RecordingCheckProvider:
    def __init__(self, recorder, *, fail=False):
        self.recorder = recorder
        self.fail = fail
        self.requests = []
        self.closed = False

    async def complete(self, request):
        self.requests.append(request)
        assert request.node_id == "trace"
        assert request.candidate_attempt_id is not None
        if self.fail:
            raise RuntimeError("scripted terminal transport failure")
        now, exchange_id = datetime.now(UTC), uuid4()
        body = {
            "candidate_id": json.loads(request.user_prompt)["candidate_id"],
            **(
                {"head_assessment": "uncertain", "change_assessment": "uncertain"}
                if "head_assessment" in json.loads(request.response_schema_json)["properties"]
                else {"status": "uncertain"}
            ),
            "reason": "The caller contract is unavailable.",
            "suggested_finding_type": None,
            "citations": [],
        }
        if "impact_assessment" in json.loads(request.response_schema_json)["properties"]:
            body["impact_assessment"] = "not_claimed"
        if "repair_assessment" in json.loads(request.response_schema_json)["properties"]:
            body["repair_assessment"] = "not_proposed"
        response = LLMResponse(
            text=json.dumps(body),
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=1,
            exchange_event_id=exchange_id,
            exchange_recorded_at=now,
            **_PROFILE,
        )
        event = LLMCallEvent(
            event_id=exchange_id,
            review_id=request.review_id,
            timestamp=now,
            is_eval=True,
            model=request.model,
            node_id="trace",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=0.0007,
            pricing_version=PRICING_VERSION,
            latency_ms=1,
            prompt_hash=_canonical_prompt_hash(
                system_prompt=request.system_prompt, user_prompt=request.user_prompt
            ),
            system_prompt_hash=_canonical_system_prompt_hash(request.system_prompt),
            cache_hit=False,
            context_summary=request.context_summary,
            prompt_template_version=request.prompt_template_version,
            response_format_digest=request.response_format_digest,
            degraded_mode=False,
            finish_reason="end_turn",
            candidate_attempt_id=request.candidate_attempt_id,
            **_PROFILE,
        )
        await self.recorder.persist(event, request, response)
        return response

    async def aclose(self):
        self.closed = True


def test_retry_defaults_to_free_and_renaming_source_keeps_repeat_identity(tmp_path, monkeypatch):
    source, plan, _ = _source(tmp_path, monkeypatch)
    monkeypatch.setattr(retry, "ROOT", tmp_path)

    def no_collection(*_args, **_kwargs):
        raise AssertionError("free planning must not construct or invoke a provider")

    monkeypatch.setattr(retry, "collect_checks", no_collection)
    assert retry.main(["--report", str(source.directory)]) == 0
    for flag in ("--paid", "--i-authorize-spend"):
        with pytest.raises(SystemExit) as error:
            retry.main(["--report", str(source.directory), flag])
        assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        retry.main(
            [
                "--report",
                str(source.directory),
                "--paid",
                "--i-authorize-spend",
                "--max-cost-usd",
                "0.25",
                "--max-calls",
                "2",
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "reports").exists()
    renamed = tmp_path / "renamed-source"
    shutil.copytree(source.directory, renamed)
    assert retry.build_plan(load_saved_checks(renamed))["retry_id"] == plan["retry_id"]


@pytest.mark.asyncio
async def test_small_authorized_cap_refuses_before_any_provider_or_retry_directory(
    tmp_path, monkeypatch
):
    source, plan, _ = _source(tmp_path, monkeypatch)
    output = tmp_path / "retry"

    def forbidden_factory(*_args):
        raise AssertionError("an insufficient cap must not construct a provider")

    with pytest.raises(ValueError, match="conservative guard"):
        await retry.collect_checks(
            source,
            plan,
            maximum_usd=Decimal("0.25"),
            maximum_calls=2,
            output=output,
            provider_factory=forbidden_factory,
        )
    assert not output.exists()


@pytest.mark.asyncio
async def test_paid_retry_preserves_discovery_and_bills_only_once_for_trace_checks(
    tmp_path, monkeypatch
):
    source, plan, finding_id = _source(tmp_path, monkeypatch)
    original = {name: sha(source.directory / name) for name in source.hashes}
    providers = []

    def factory(_target, recorder):
        provider = RecordingCheckProvider(recorder)
        providers.append(provider)
        return provider

    output = tmp_path / "retry"
    result = await retry.collect_checks(
        source,
        plan,
        maximum_usd=Decimal("3"),
        maximum_calls=2,
        output=output,
        provider_factory=factory,
    )
    assert result["status"] == "collected"
    assert result["new_check_budget"]["calls"] == len(providers[0].requests) == 2
    assert Decimal(result["new_check_budget"]["known_usage_based_usd"]) == Decimal("0.0014")
    assert result["new_check_budget"]["uncertain_budget_upper_usd"] == "0E-8"
    [case] = result["cases"]
    assert len(case["native_candidates"]) == len(case["assessments"]) == 2
    assert [item["finding_id"] for item in case["original_findings"]] == [str(finding_id)]
    assert case["verification_hold_finding_ids"] == [str(finding_id)]
    assert not case["unassessed_candidates"]
    assert providers[0].closed
    receipts = json.loads((output / "receipts.json").read_text())
    assert receipts and all(sha(output / name) == digest for name, digest in receipts.items())
    assert {name: sha(source.directory / name) for name in source.hashes} == original
    with pytest.raises(FileExistsError):
        await retry.collect_checks(
            source,
            plan,
            maximum_usd=Decimal("3"),
            maximum_calls=2,
            output=output,
            provider_factory=factory,
        )
    assert len(providers) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["call_cap", "transport", "source_drift"])
async def test_retry_stops_without_automatic_extra_calls(tmp_path, monkeypatch, failure):
    source, plan, _ = _source(tmp_path, monkeypatch)
    providers = []

    def factory(_target, recorder):
        provider = RecordingCheckProvider(recorder, fail=failure == "transport")
        providers.append(provider)
        return provider

    if failure == "source_drift":
        path = source.directory / "calls/analyze.exchange.json"
        path.write_text(path.read_text() + " ")
        with pytest.raises(ValueError, match="input changed"):
            await retry.collect_checks(
                source,
                plan,
                maximum_usd=Decimal("3"),
                maximum_calls=2,
                output=tmp_path / "retry",
                provider_factory=factory,
            )
        assert not providers
        return
    result = await retry.collect_checks(
        source,
        plan,
        maximum_usd=Decimal("3"),
        maximum_calls=1 if failure == "call_cap" else 2,
        output=tmp_path / "retry",
        provider_factory=factory,
    )
    assert len(providers[0].requests) == result["new_check_budget"]["calls"] == 1
    assert providers[0].closed
    if failure == "call_cap":
        assert result["status"] == "budget-stopped-before-call"
        assert len(result["cases"][0]["unassessed_candidates"]) == 1
    else:
        assert result["status"] == "incomplete"
        assert not result["cases"][0]["assessments"]
        assert Decimal(result["new_check_budget"]["uncertain_budget_upper_usd"]) > 0
