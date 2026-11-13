"""Exercise prompt pairing, immutable captures and archive replay without a paid call."""

from __future__ import annotations

import hashlib
import json
import shutil
from decimal import Decimal

import pytest
from pydantic import ValidationError

from outrider.llm.base import LLMRequest
from outrider.policy import EvidenceTier, FindingType
from outrider.prompts import analyze
from outrider.schemas.llm.analyze import (
    ANALYZE_RESPONSE_SCHEMA,
    AnalyzeFindingProposal,
    AnalyzeFindingProposalRaw,
)

from . import model_refresh as refresh
from . import prompt_control as control
from .test_model_refresh import RecordingProvider, request


def test_concise_contract_supplies_admitted_values_missing_from_provider_schema() -> None:
    fields = ANALYZE_RESPONSE_SCHEMA["properties"]["findings"]["items"]["properties"]
    expected_values = {
        "finding_type": {kind.value for kind in FindingType},
        "evidence_tier": {EvidenceTier.JUDGED.value},
    }
    proposal = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "title": "Untrusted query text",
        "description": "The query contains request data.",
        "evidence": "cursor.execute(query)",
        "line_start": 1,
        "line_end": 1,
    }
    AnalyzeFindingProposal.model_validate(proposal)
    for field, values in expected_values.items():
        assert fields[field] == {"type": "string"}
        prefix = (
            f"Allowed {field} values: "
            if field == "finding_type"
            else "Allowed model evidence_tier value on this first pass: "
        )
        declaration = next(
            line for line in control.CONCISE_PROMPT.splitlines() if line.startswith(prefix)
        )
        declared = json.loads(declaration.removeprefix(prefix).removesuffix("."))
        assert (set(declared) if isinstance(declared, list) else {declared}) == values
        invalid = {**proposal, field: "JUDGED" if field == "evidence_tier" else "unknown_bug"}
        AnalyzeFindingProposalRaw.model_validate(invalid)
        with pytest.raises(ValidationError):
            AnalyzeFindingProposal.model_validate(invalid)


@pytest.mark.parametrize("suite,quality_cases", [("real-pr", 3), ("grounded", 10)])
def test_free_plan_discloses_extra_calls_and_rejects_changed_experiment_scope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], suite, quality_cases
) -> None:
    def unexpected(*_args: object) -> None:
        pytest.fail("free prompt-control planning constructed a provider")

    monkeypatch.setattr(refresh, "create_provider", unexpected)
    calls_per_arm = 3 + 3 * quality_cases
    args = [
        "--target",
        "anthropic-haiku-5-5",
        "--suite",
        suite,
        "--phase",
        "analyze",
        "--compare-prompts",
        "--reps",
        "3",
        "--max-calls",
        str(2 * calls_per_arm + 1),
    ]
    assert refresh.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_calls_made"] == 0 and plan["planned_calls"] == 2 * calls_per_arm
    assert plan["prompt_control"]["planned_calls_per_arm"] == calls_per_arm
    assert plan["prompt_control"]["quality_calls_per_arm"] == 3 * quality_cases
    assert plan["prompt_control"]["compatibility_calls_per_arm"] == 3
    assert plan["per_arm_caps"] == {
        "maximum_usd": None,
        "maximum_calls": calls_per_arm,
        "unallocated_calls": 1,
        "unused_cap_transferred": False,
    }
    assert refresh.main([*args, "--max-cost-usd", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["per_arm_caps"]["maximum_usd"] == "2"
    for extra in (
        ["--target", "anthropic-sonnet-5-5"],
        ["--suite", "legacy"],
        ["--phase", "compat"],
    ):
        with pytest.raises(SystemExit):
            refresh.main([*args, *extra])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [{"user_prompt": "different code"}, {"temperature": 0.5}, {"max_tokens": 101}]
)
async def test_changed_context_or_settings_stop_before_native_provider(tmp_path, change) -> None:
    original = request().model_copy(
        update={
            "node_id": "analyze",
            "system_prompt": analyze.SYSTEM_PROMPT_STABLE_PREFIX,
            "prompt_template_version": analyze.VERSION,
        }
    )
    contexts = control.ContextPairs()
    contexts.check("analyze:0:fixture", original, control.variant_identity("outrider"))
    recorder = refresh.ExchangeRecorder(tmp_path, refresh.TARGETS["anthropic-haiku-5-5"])
    recorder.scenario = "analyze:0:fixture"
    native = RecordingProvider(recorder)
    provider = control.PromptControlProvider(
        native, control.variant_identity("concise-control"), recorder, contexts
    )
    with pytest.raises(ValueError, match="context or settings differ"):
        await provider.complete(original.model_copy(update=change))
    assert native.calls == 0 and not recorder.exchanges


@pytest.mark.asyncio
@pytest.mark.parametrize("suite,quality_cases", [("real-pr", 3), ("grounded", 10)])
async def test_matching_arms_keep_raw_requests_and_replay_both_archived_prompts(
    tmp_path, suite, quality_cases
) -> None:
    from .model_refresh_regrade import ReplayIntegrityError, regrade_saved

    plan = control.prepare_plan(
        refresh.build_plan(
            (refresh.TARGETS["anthropic-haiku-5-5"],), phase="analyze", reps=1, suite=suite
        )
    )
    calls_per_arm = 3 + quality_cases
    output = tmp_path / "control"
    report = await control.run_prompt_control(
        plan,
        maximum_usd=Decimal("4"),
        maximum_calls=2 * calls_per_arm,
        output=output,
        provider_factory=lambda _target, recorder: RecordingProvider(recorder),
    )
    assert report["status"] == "completed"
    assert report["complete_request_pairs"] == report["expected_request_pairs"] == calls_per_arm
    assert [arm["evidence_kind"] for arm in report["arms"]] == ["injected-provider"] * 2
    original_plan = json.loads((output / "plan.json").read_text())
    assert original_plan == plan
    for name, digest in plan["inputs"].items():
        assert hashlib.sha256((output / "generation" / name).read_bytes()).hexdigest() == digest
    requests = []
    for arm in report["arms"]:
        saved = output / arm["variant"]["name"]
        assert arm["budget"]["authorized_maximum_usd"] == "2"
        assert arm["budget"]["maximum_calls"] == calls_per_arm
        assert arm["compatibility"]["calls"] == 3
        assert arm["quality"]["calls"] == quality_cases
        exchanges = [
            json.loads(path.read_text())
            for path in sorted((saved / "calls").glob("*.exchange.json"))
        ]
        assert all(exchange["response"]["text"] == '{"findings":[]}' for exchange in exchanges)
        requests.append([LLMRequest.model_validate(exchange["request"]) for exchange in exchanges])
        assert all(
            hashlib.sha256(item.system_prompt.encode()).hexdigest()
            == arm["variant"]["system_prompt_sha256"]
            for item in requests[-1]
        )
        with pytest.raises(ReplayIntegrityError, match="original live refresh report"):
            regrade_saved(
                saved,
                original_source=output / "generation",
                receipt_manifest=output / "receipts.json",
                output=tmp_path / (arm["variant"]["name"] + "-rejected"),
            )

        # An explicitly synthetic, anchored copy tests the live-capture replay path.
        replay = tmp_path / (arm["variant"]["name"] + "-synthetic")
        shutil.copytree(saved, replay)
        capture = json.loads((replay / "results.json").read_text())
        if suite == "grounded":
            cases = [
                case for case in capture["candidates"][0]["cases"] if case["phase"] == "analyze"
            ]
            assert sum(case["producer_covered_expected"] for case in cases) == 1
            assert sum(case["model_dependent"]["expected"] for case in cases) == 4
        capture["evidence_kind"] = "live-provider"
        (replay / "results.json").write_text(json.dumps(capture))
        receipts = tmp_path / (arm["variant"]["name"] + "-receipts.json")
        receipts.write_text(
            json.dumps(
                {
                    str(replay): {
                        str(path.relative_to(replay)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in replay.rglob("*")
                        if path.is_file()
                    }
                }
            )
        )
        derived = regrade_saved(
            replay,
            original_source=output / "generation",
            receipt_manifest=receipts,
            output=tmp_path / (arm["variant"]["name"] + "-regraded"),
        )
        assert derived["candidates"][0]["status"] == "completed"
        assert all(
            slot["replay_status"] == "reproduced"
            for slot in derived["candidates"][0]["replay_slots"]
        )
        assert (
            json.loads((saved / "results.json").read_text())["evidence_kind"] == "injected-provider"
        )
    for original, concise in zip(*requests, strict=True):
        assert original.user_prompt == concise.user_prompt
        assert original.system_prompt != concise.system_prompt
        assert original.response_schema_json == concise.response_schema_json
