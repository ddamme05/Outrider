"""Check the paid runner's boundaries with recording providers and local HTTP only."""

from __future__ import annotations

import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from outrider.llm.base import LLMRefusalError, LLMRequest, LLMResponse, LLMUnknownError

from . import model_refresh as refresh
from .model_comparison import AnalyzeRun, run_analyze_instrumented, state_from_eval_fixture


def request(model: str = "claude-haiku-5-5") -> LLMRequest:
    return LLMRequest(
        system_prompt="Review the supplied code.",
        user_prompt="fixture",
        model=model,
        max_tokens=100,
        temperature=0,
        review_id=uuid4(),
        node_id="triage",
        is_eval=True,
        prompt_template_version="test",
        degraded_mode=False,
    )


class RecordingProvider:
    def __init__(
        self,
        recorder: refresh.ExchangeRecorder,
        *,
        persist: bool = True,
        fail: bool = False,
        priced: bool = True,
    ) -> None:
        self.recorder, self.persist, self.fail, self.priced = recorder, persist, fail, priced
        self.calls = 0
        self.finish_reason = "stop"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        profile_id, reasoning, contract = refresh.resolve_host_identity(
            self.recorder.target.host, reasoning=False, models=(request.model,)
        )
        response = LLMResponse(
            text='{"findings":[]}',
            model=request.model,
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason=self.finish_reason,
            latency_ms=1,
            profile_id=profile_id,
            reasoning_enabled=reasoning,
            profile_contract_digest=contract,
            billed_prompt_tokens=10,
            service_tier_actual="default",
        )
        if self.persist:
            event = SimpleNamespace(
                model_dump=lambda **_kwargs: {
                    "event_id": str(uuid4()),
                    "model": request.model,
                    "cost_usd": 0.001 if self.priced else None,
                    "pricing_version": "test",
                }
            )
            await self.recorder.persist(event, request, response)
        if self.fail:
            raise RuntimeError("recording failure")
        return response

    async def aclose(self) -> None:
        pass


@pytest.mark.parametrize("value", ["NaN", "Infinity", "0", "-1"])
def test_nonfinite_or_nonpositive_caps_cannot_authorize_calls(value: str) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        refresh.Budget(Decimal(value), 3)


def test_nonlegacy_plans_do_not_depend_on_legacy_holdout_novelty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from . import test_model_comparison as fixtures

    def stale_review() -> str:
        raise ValueError("legacy holdout novelty review is stale")

    monkeypatch.setattr(fixtures, "_holdout_preamble", stale_review)
    target = refresh.TARGETS["anthropic-haiku-5-5"]
    for suite in ("grounded", "real-pr"):
        plan = refresh.build_plan((target,), phase="analyze", reps=1, suite=suite)
        assert [Path(name).name for name in plan["compatibility_cases"]] == [
            "sqli_unseen_template.json",
            "safe_unseen_psycopg_composition.json",
            "cmd_injection_eval_indirect.json",
        ]
    with pytest.raises(ValueError, match="holdout novelty review is stale"):
        refresh.build_plan((target,), phase="analyze", reps=1, suite="legacy")


@pytest.mark.parametrize("phase", ["compat", "analyze", "nodes", "all"])
def test_grounded_plan_is_explicit_development_analysis_with_legacy_compatibility(
    phase: str,
) -> None:
    target = refresh.TARGETS["anthropic-haiku-5-5"]
    if phase in {"nodes", "all"}:
        with pytest.raises(ValueError, match="compat or analyze only"):
            refresh.build_plan((target,), phase=phase, reps=3, suite="grounded")
        return
    plan = refresh.build_plan((target,), phase=phase, reps=3, suite="grounded")
    assert plan["suite"] == "grounded" and plan["fixture_count"] == 10
    assert plan["dataset"]["exposure"] == "development-exposed"
    assert plan["dataset"]["held_out"] is False
    assert plan["compatibility_cases"] == [row.path for row in refresh.compatibility_scenarios()]
    assert plan["analysis_calls_per_target"] == (30 if phase == "analyze" else 0)
    assert plan["planned_calls"] == (33 if phase == "analyze" else 3)
    assert plan["dataset"]["inputs"].items() <= plan["inputs"].items()
    assert plan["dataset"]["inputs"].items() <= plan["measurement_inputs"].items()


@pytest.mark.parametrize("phase", ["compat", "analyze", "nodes", "all"])
def test_real_pr_plan_bounds_selected_file_analysis_without_claiming_holdout(phase: str) -> None:
    targets = (
        refresh.TARGETS["anthropic-haiku-5-5"],
        refresh.TARGETS["anthropic-haiku-4-5"],
    )
    if phase != "analyze":
        with pytest.raises(ValueError, match="analyze only"):
            refresh.build_plan(targets, phase=phase, reps=3, suite="real-pr")
        return
    plan = refresh.build_plan(targets, phase=phase, reps=3, suite="real-pr")
    assert plan["planned_calls"] == 24
    assert [target["planned_calls"] for target in plan["targets"]] == [12, 12]
    assert plan["analysis_calls_per_target"] == 9
    assert plan["dataset"]["held_out"] is False
    assert plan["dataset"]["exposure"] == "development-selected-historical-prs"
    assert plan["analysis_state"]["other_changed_files"] == "skip"
    assert plan["analysis_state"]["full_pr_context_retained"] is True
    assert plan["compatibility_cases"] == [row.path for row in refresh.compatibility_scenarios()]
    assert plan["dataset"]["inputs"].items() <= plan["inputs"].items()
    assert plan["dataset"]["inputs"].items() <= plan["measurement_inputs"].items()


@pytest.mark.parametrize("change", ["capture-bytes", "selected-file"])
def test_real_pr_capture_drift_or_unsupported_selection_fails_free_preflight(
    tmp_path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from . import real_pr_cases

    directory = tmp_path / real_pr_cases.DIRECTORY
    shutil.copytree(refresh.ROOT / real_pr_cases.DIRECTORY, directory)
    monkeypatch.setattr(real_pr_cases, "ROOT", tmp_path)
    manifest_path = tmp_path / real_pr_cases.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if change == "capture-bytes":
        fixture = tmp_path / manifest["cases"][0]["fixture"]
        fixture.write_bytes(fixture.read_bytes() + b"\n")
    else:
        manifest["cases"][0]["analysis_file"] = "not-in-the-captured-pr.py"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed after review|changed Python head"):
        real_pr_cases.load_cases()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suite,model_positive_cases,counterpart_cases", [("grounded", 4, 5), ("real-pr", 2, 1)]
)
async def test_empty_recordings_measure_supported_misses_and_replay_original_suite(
    tmp_path,
    suite,
    model_positive_cases,
    counterpart_cases,
) -> None:
    from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel

    from .model_refresh_regrade import regrade_saved

    rows = refresh.scenarios(suite)
    for case in rows:
        captured = json.loads((refresh.ROOT / case.path).read_text())
        assert not captured["llm_responses"]
        state = refresh.state_for_scenario(case, suite=suite)
        assert state.triage_result.overall_risk is RiskLevel.MEDIUM
        assert set(state.triage_result.relevant_dimensions) == set(ReviewDimension)
        assert [file.path for file in state.pr_context.changed_files] == [
            file["path"] for file in captured["files"]
        ]
        assert state.triage_result.file_tiers == {
            file["path"]: ReviewTier.STANDARD
            if case.analysis_file is None or file["path"] == case.analysis_file
            else ReviewTier.SKIP
            for file in captured["files"]
        }
    target = refresh.TARGETS["anthropic-haiku-5-5"]
    plan = refresh.build_plan((target,), phase="analyze", reps=3, suite=suite)
    call_count = 3 + 3 * len(rows)
    saved = tmp_path / "saved"
    report = await refresh.run_refresh(
        plan,
        maximum_usd=Decimal("1"),
        maximum_calls=call_count,
        output=saved,
        provider_factory=lambda _target, recorder: RecordingProvider(recorder),
    )
    candidate = report["candidates"][0]
    assert candidate["status"] == "completed"
    assert len(candidate["calls"]) == call_count
    cases = [case for case in candidate["cases"] if case["phase"] == "analyze"]
    assert sum(case["model_dependent"]["expected"] for case in cases) == 3 * model_positive_cases
    assert sum(case["model_dependent"]["matched"] for case in cases) == 0
    assert sum(case["measurement_v2"]["unscored_expected"] for case in cases) == 0
    assert sum(case["measurement_v2"]["safe_model_emissions"] == 0 for case in cases) == (
        3 * counterpart_cases
    )

    # Construct an anchored synthetic capture to test replay wiring, not model quality or prices.
    report["evidence_kind"] = "live-provider"
    (saved / "results.json").write_text(json.dumps(report))
    manifest = tmp_path / "receipts.json"
    manifest.write_text(
        json.dumps(
            {
                str(saved): {
                    str(path.relative_to(saved)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in saved.rglob("*")
                    if path.is_file()
                }
            }
        )
    )
    archive = tmp_path / "source"
    for name in plan["inputs"]:
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(refresh.ROOT / name, destination)
    package_init = refresh.ROOT / "tests/__init__.py"
    if package_init.is_file():
        shutil.copyfile(package_init, archive / "tests/__init__.py")
    derived = regrade_saved(
        saved, original_source=archive, receipt_manifest=manifest, output=tmp_path / "regraded"
    )
    assert derived["original_plan"] == plan
    assert derived["plan"]["suite"] == suite
    assert derived["candidates"][0]["status"] == "completed"
    assert len(derived["candidates"][0]["replay_slots"]) == call_count
    assert all(
        slot["replay_status"] == "reproduced" for slot in derived["candidates"][0]["replay_slots"]
    )
    assert (
        sum(
            case["model_dependent"]["expected"]
            for case in derived["candidates"][0]["cases"]
            if case["phase"] == "analyze"
        )
        == 3 * model_positive_cases
    )


def test_default_and_explicit_dry_plan_never_construct_a_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected(*_args: object) -> None:
        pytest.fail("free planning constructed a provider")

    monkeypatch.setattr(refresh, "create_provider", unexpected)
    base = ["--target", "anthropic-haiku-5-5", "--max-cost-usd", "1"]
    assert refresh.main(base) == 0
    assert refresh.main([*base, "--paid", "--i-authorize-spend", "--dry-run"]) == 0
    assert '"model_calls_made": 0' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        refresh.main([*base, "--paid"])


@pytest.mark.asyncio
async def test_uncertain_failure_retains_headroom_and_stops_next_call(tmp_path) -> None:
    target = refresh.TARGETS["anthropic-haiku-5-5"]
    recorder = refresh.ExchangeRecorder(tmp_path, target)
    inner = RecordingProvider(recorder, persist=False, fail=True)
    budget = refresh.Budget(Decimal("1"), 3)
    provider = refresh.BudgetedProvider(inner, target, recorder, budget, {})
    with pytest.raises(RuntimeError, match="recording failure"):
        await provider.complete(request())
    assert budget.known_usd == 0
    assert budget.uncertain_usd == refresh.reservation_usd(target, 100)
    with pytest.raises(refresh.BudgetLimitError):
        await provider.complete(request())
    assert inner.calls == budget.calls == 1


@pytest.mark.asyncio
async def test_unpriced_usage_releases_only_a_conservative_bound(tmp_path) -> None:
    target = refresh.TARGETS["openai-sol"]
    recorder = refresh.ExchangeRecorder(tmp_path, target)
    inner = RecordingProvider(recorder, priced=False)
    budget = refresh.Budget(Decimal("20"), 3)
    provider = refresh.BudgetedProvider(inner, target, recorder, budget, {})
    await provider.complete(request(target.model))
    await provider.complete(request(target.model))
    assert budget.known_usd == 0
    assert 0 < budget.uncertain_usd < refresh.reservation_usd(target, 100)
    assert all(
        row["usage_based_usd"] is None and row["cost_status"] == "uncosted"
        for row in provider.attempts
    )


@pytest.mark.parametrize("finish_reason", ["end_turn", "refusal", "max_tokens", "length"])
@pytest.mark.asyncio
async def test_real_native_completion_refusal_and_output_limit_use_budgeted_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch, finish_reason: str
) -> None:
    import outrider.llm.anthropic_provider as native

    payloads = []

    def respond(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        text = '{"findings":[' if finish_reason in {"max_tokens", "length"} else '{"findings":[]}'
        return httpx.Response(
            200,
            json={
                "id": "msg_refresh_fixture",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [] if finish_reason == "refusal" else [{"type": "text", "text": text}],
                "stop_reason": finish_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = native.anthropic.DefaultAsyncHttpxClient
    monkeypatch.setattr(
        native.anthropic,
        "DefaultAsyncHttpxClient",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-refresh-key")
    target = refresh.TARGETS["anthropic-haiku-5-5"]
    recorder = refresh.ExchangeRecorder(tmp_path, target)
    budget = refresh.Budget(Decimal("1"), 3)
    inner = refresh.create_provider(target, recorder)
    inner._client._platform = "Linux"  # Keep SDK platform probing outside this local HTTP test.
    provider = refresh.BudgetedProvider(inner, target, recorder, budget, {})
    state = state_from_eval_fixture(
        refresh.ROOT / "tests/eval/fixtures/mock_github/safe_unseen_psycopg_composition.json"
    )
    try:
        if finish_reason == "refusal":
            with pytest.raises(LLMRefusalError):
                await run_analyze_instrumented(state, provider=provider, model=target.model)
        elif finish_reason in {"max_tokens", "length"}:
            with pytest.raises(refresh.OutputLimitError):
                await run_analyze_instrumented(state, provider=provider, model=target.model)
            assert provider.attempts[0]["status"] == "output-limit"
            assert provider.attempts[0]["truncated"] is True
        else:
            result = await run_analyze_instrumented(state, provider=provider, model=target.model)
            assert result.findings == () and result.response_rejections == 0
    finally:
        await provider.aclose()
    assert len(payloads) == len(recorder.exchanges) == budget.calls == 1
    assert payloads[0]["model"] == target.model
    assert budget.known_usd > 0 and budget.uncertain_usd == 0
    assert provider.attempts[0]["finish_reason"] == finish_reason
    assert "test-refresh-key" not in "".join(path.read_text() for path in tmp_path.glob("*.json"))


@pytest.mark.asyncio
async def test_v2_compatible_refusal_after_compatibility_stops_and_preserves_usage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import SecretStr

    from outrider.llm import openai_compatible_provider as compatible
    from outrider.llm.host_profiles import resolve_host_profile

    target = refresh.TARGETS["fireworks-glm-5-3-flash"]
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        refused = len(requests) == 4
        return httpx.Response(
            200,
            json={
                "id": "local-refusal-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": target.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None if refused else '{"findings":[]}',
                        },
                        "finish_reason": "content_filter" if refused else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
            },
        )

    client = compatible.openai.AsyncOpenAI

    def local_client(**kwargs):
        assert kwargs["max_retries"] == 0
        result = client(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)), **kwargs
        )
        result._platform = "Linux"  # Keep platform probing outside the local HTTP boundary.
        return result

    monkeypatch.setattr(compatible.openai, "AsyncOpenAI", local_client)

    def provider_factory(target, recorder):
        return compatible.OpenAICompatibleProvider(
            api_key=SecretStr("local-refusal-key"),
            profile=resolve_host_profile(target.host),
            models=(target.model,),
            persister=recorder,
        )

    result = await refresh.run_refresh(
        refresh.build_plan((target,), phase="analyze", reps=3),
        maximum_usd=Decimal("1"),
        maximum_calls=111,
        output=tmp_path / "run",
        provider_factory=provider_factory,
    )
    candidate = result["candidates"][0]
    assert len(requests) == result["budget"]["calls"] == 4
    assert len(candidate["cases"]) == 3
    assert candidate["status"] == "incompatible-or-failed"
    assert candidate["error_type"] == "LLMRefusalError"
    refused = candidate["calls"][-1]
    assert refused["finish_reason"] == "refusal" and refused["persisted_exchange"] is True
    assert refused["error_type"] == "LLMRefusalError"
    assert Decimal(refused["usage_based_usd"]) > 0
    assert Decimal(result["budget"]["known_usage_based_usd"]) > 0
    assert Decimal(result["budget"]["uncertain_budget_upper_usd"]) == 0
    assert candidate["summary"]["refusals"] == 1
    assert candidate["summary"]["collection_complete"] is False
    assert candidate["summary"]["cost_complete"] is False


@pytest.mark.parametrize("bad_case", ["zero", "multiple", "length", "max_tokens"])
@pytest.mark.asyncio
async def test_each_compatibility_case_needs_one_call_and_failure_isolates_target(
    tmp_path, monkeypatch: pytest.MonkeyPatch, bad_case: str
) -> None:
    targets = (
        refresh.Target("anthropic", "claude-haiku-5-5", "baseline-cheap"),
        refresh.TARGETS["anthropic-haiku-5-5"],
    )
    plan = refresh.build_plan(targets, phase="compat", reps=1)
    counts = {}

    async def analyze(_state, *, provider, model, collect_producer_coverage=False):
        name = provider.target.role
        counts[name] = counts.get(name, 0) + 1
        n = (
            {"zero": 0, "multiple": 2}.get(bad_case, 1)
            if name == "baseline-cheap" and counts[name] == 2
            else 1
        )
        if name == "baseline-cheap" and counts[name] == 2 and bad_case in {"length", "max_tokens"}:
            provider.provider.finish_reason = bad_case
        for _ in range(n):
            await provider.complete(request(model))
        return AnalyzeRun((), (), 0, () if collect_producer_coverage else None)

    monkeypatch.setattr(refresh, "run_analyze_instrumented", analyze)
    result = await refresh.run_refresh(
        plan,
        maximum_usd=Decimal("1"),
        maximum_calls=10,
        output=tmp_path / "run",
        provider_factory=lambda _target, recorder: RecordingProvider(recorder),
    )
    assert [row["status"] for row in result["candidates"]] == [
        "output-limit" if bad_case in {"length", "max_tokens"} else "incompatible-or-failed",
        "completed",
    ]
    if bad_case in {"length", "max_tokens"}:
        failed = result["candidates"][0]
        assert failed["failed_scenario"] == failed["calls"][-1]["scenario"]
        assert failed["calls"][-1]["status"] == "output-limit"
        assert failed["calls"][-1]["persisted_exchange"] is True
        assert failed["calls"][-1]["usage_based_usd"] == "0.001"
    assert len(result["candidates"][1]["calls"]) == 3
    assert result["evidence_kind"] == "injected-provider"
    assert result["comparisons"][0]["status"] == "inconclusive"


@pytest.mark.parametrize("known_status", [None, 404])
@pytest.mark.asyncio
async def test_provider_failure_saves_only_bounded_diagnostics(
    tmp_path, known_status: int | None
) -> None:
    target = refresh.TARGETS["anthropic-haiku-5-5"]
    plan = refresh.build_plan((target,), phase="compat", reps=1)

    class FailedProvider(RecordingProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise LLMUnknownError("private vendor body and credential").attach_diagnostics(
                http_status=known_status,
                request_id="req_fixture" if known_status else None,
            )

    result = await refresh.run_refresh(
        plan,
        maximum_usd=Decimal("1"),
        maximum_calls=3,
        output=tmp_path / "run",
        provider_factory=lambda _target, recorder: FailedProvider(recorder),
    )
    failed = result["candidates"][0]
    call = failed["calls"][0]
    assert failed["status"] == "incompatible-or-failed"
    assert failed["error_type"] == call["error_type"] == "LLMUnknownError"
    assert (
        failed["provider_diagnostics"]
        == call["provider_diagnostics"]
        == {
            "http_status": known_status,
            "request_id": "req_fixture" if known_status else None,
            "error_category": None,
        }
    )
    assert call["persisted_exchange"] is False and call["usage_based_usd"] is None
    assert Decimal(result["budget"]["uncertain_budget_upper_usd"]) > 0
    assert "private vendor body" not in "".join(
        path.read_text() for path in (tmp_path / "run").rglob("*.json")
    )


@pytest.mark.parametrize("tier", ["judged", "observed"])
@pytest.mark.parametrize("producer_covered", [False, True])
@pytest.mark.parametrize("prose", ["grounded", "vague"])
@pytest.mark.asyncio
async def test_v2_model_contribution_rejects_model_proof_and_excludes_producer_floor(
    tier: str, producer_covered: bool, prose: str
) -> None:
    from .model_refresh_regrade import score_case_v2
    from .test_model_comparison import _ScriptedProvider

    name = "sqli_holdout_fstring.json" if producer_covered else "pygoat_sql_injection.json"
    scenario = next(row for row in refresh.scenarios() if row.path.endswith(name))
    fixture_path = refresh.ROOT / scenario.path
    payload = json.loads(json.loads(fixture_path.read_text())["llm_responses"]["analyze"][0])
    payload["findings"][0].update(
        evidence_tier=tier,
        query_match_id="python.function_definition" if tier == "observed" else None,
    )
    if prose == "vague":
        payload["findings"][0].update(
            title="Possible issue", description="Please investigate this change."
        )
    provider = _ScriptedProvider(json.dumps(payload))
    result = await run_analyze_instrumented(
        state_from_eval_fixture(fixture_path),
        provider=provider,
        model="claude-haiku-4-5",
        collect_producer_coverage=True,
    )
    assert len(provider.calls) == 1
    assert result.response_rejections == 0
    assert result.proposal_rejections == int(tier == "observed")
    score = score_case_v2(result, scenario, fixture_path=fixture_path, request=provider.calls[0])
    assert score["producer_covered_expected"] == int(producer_covered)
    assert score["model_dependent"]["expected"] == int(not producer_covered)
    model_emitted = not producer_covered and tier == "judged"
    assert score["model_dependent"]["matched"] == int(model_emitted and prose == "grounded")
    assert score["model_dependent"]["pending"] == int(model_emitted and prose == "vague")
    if not producer_covered:
        assert score["measurement_v2"]["unscored_expected"] == 0
        assert score["product_grounded"]["expected"] == 1
        assert score["product_grounded"]["pending"] == int(model_emitted and prose == "vague")
    observational = score_case_v2(
        result,
        refresh.Scenario(scenario.path, (), "safe"),
        fixture_path=fixture_path,
        request=provider.calls[0],
    )
    assert observational["measurement_v2"]["safe_model_emissions"] == int(model_emitted)
    assert observational["measurement_v2"]["safe_product_emissions"] == len(result.findings)
    assert observational["model_dependent"]["false_alarms"] is None


@pytest.mark.parametrize(
    "mutation", ["unlisted-report", "source-drift", "duplicate-call", "slot-drift"]
)
def test_offline_receipts_reject_untrusted_reports_inputs_and_slots(
    tmp_path, mutation: str
) -> None:
    from .model_refresh_regrade import V1_CONTRACT, ReplayIntegrityError, _verify_original

    source, directory = tmp_path / "source", tmp_path / "saved"
    source.mkdir()
    (directory / "calls").mkdir(parents=True)
    script = source / "app.py"
    script.write_text("raise RuntimeError('archive execution is forbidden in this check')\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    target = {"host": "host", "model": "model", "role": "baseline"}
    call = {
        "index": 1,
        "target": "host:model:baseline",
        "scenario": "compat:case",
        "persisted_exchange": False,
    }
    plan = {
        "measurement_contract": V1_CONTRACT,
        "phase": "compat",
        "repetitions": 1,
        "inputs": {"app.py": digest(script)},
        "scenarios": [{"fixture": "case"}],
        "node_cases": [],
        "compatibility_cases": ["case"],
    }
    report = {
        "schema": refresh.REPORT_SCHEMA,
        "evidence_kind": "live-provider",
        "plan": plan,
        "candidates": [
            {"target": target, "status": "incompatible-or-failed", "cases": [], "calls": [call]}
        ],
    }
    for name, value in (
        ("results.json", report),
        ("plan.json", plan),
        ("calls/000001.attempt.json", call),
    ):
        (directory / name).write_text(json.dumps(value))
    receipts = {
        name: digest(directory / name)
        for name in ("results.json", "plan.json", "calls/000001.attempt.json")
    }
    manifest = tmp_path / "receipts.json"
    manifest.write_text(json.dumps({str(directory): receipts}))
    selected = directory / "results.json"
    assert _verify_original(selected, source, manifest)[0] == report
    if mutation == "unlisted-report":
        selected = directory / "unlisted.json"
        selected.write_text(json.dumps(report | {"budget": {"known_usage_based_usd": "0"}}))
    elif mutation == "source-drift":
        script.write_text("pass\n")
    else:
        if mutation == "duplicate-call":
            report["candidates"][0]["calls"].append(dict(call))
        else:
            call["scenario"] = "compat:not-declared"
            (directory / "calls/000001.attempt.json").write_text(json.dumps(call))
        (directory / "results.json").write_text(json.dumps(report))
        receipts = {name: digest(directory / name) for name in receipts}
        manifest.write_text(json.dumps({str(directory): receipts}))
    with pytest.raises(ReplayIntegrityError):
        _verify_original(selected, source, manifest)


def test_retired_target_stops_in_plan_and_paid_execution_before_provider_construction(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected(*_args: object) -> None:
        pytest.fail("retired target constructed a provider")

    monkeypatch.setattr(refresh, "create_provider", unexpected)
    assert refresh.main(["--target", "fireworks-glm-5-2"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_calls_made"] == 0
    assert plan["targets"][0]["ready"] is False
    assert "retired" in plan["targets"][0]["issue"]
    assert (
        refresh.main(
            [
                "--target",
                "fireworks-glm-5-2",
                "--paid",
                "--i-authorize-spend",
                "--max-cost-usd",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    result = json.loads(next(tmp_path.glob("*/results.json")).read_text())
    assert result["budget"]["calls"] == 0
    assert result["candidates"][0]["status"] == "not-ready"


def test_incomplete_paid_collection_returns_nonzero(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def incomplete(*_args, **_kwargs):
        return {
            "budget": {},
            "candidates": [{"target": {}, "status": "budget-stopped-before-call"}],
        }

    monkeypatch.setattr(refresh, "run_refresh", incomplete)
    assert (
        refresh.main(
            [
                "--target",
                "anthropic-haiku-5-5",
                "--paid",
                "--i-authorize-spend",
                "--max-cost-usd",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
