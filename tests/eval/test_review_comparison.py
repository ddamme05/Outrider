"""Verify free planning, generic discoveries and the real workflow without paid calls."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from outrider.agent import eval_driver
from outrider.agent.eval_driver import EvalFixture, _FixtureScriptedProvider
from outrider.llm.base import INCLUDE_TEXT_OPT_IN
from outrider.prompts import analyze, triage
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA_JSON

from . import direct_review
from . import model_refresh as refresh
from . import review_comparison as comparison
from .discovery_control import SIMPLE_VERSION, discovery_identity, requests_for_discovery
from .prompt_control import CONCISE_PROMPT, CONCISE_VERSION
from .test_model_refresh import RecordingProvider


def test_free_plan_needs_neither_provider_nor_database(monkeypatch, capsys) -> None:
    def unexpected(*_args, **_kwargs):
        pytest.fail("free planning attempted provider construction or a workflow run")

    monkeypatch.setattr(comparison, "create_provider", unexpected)
    monkeypatch.setattr(comparison, "run_review", unexpected)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert comparison.main(["--target", "anthropic-sonnet-5-5", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_calls_made"] == 0
    assert plan["workflow_analyze_concurrency"] == 4
    assert Decimal(plan["concurrent_reservation_upper_usd_per_workflow_arm"]) == 4 * Decimal(
        plan["per_call_reservation_upper_usd"]
    )
    assert plan["call_upper_bounds"] == {
        "context-off": 384,
        "context-on": 384,
        "direct-one-shot": 12,
        "total": 780,
    }
    assert Counter(row["order"][0] for row in plan["schedule"]) == {
        "context-off": 4,
        "context-on": 4,
        "direct-one-shot": 4,
    }
    for case in plan["cases"]:
        fixture = EvalFixture.model_validate_json((comparison.ROOT / case["fixture"]).read_bytes())
        request = direct_review.render_request(
            fixture, model=plan["target"]["model"], review_id=uuid4()
        )
        metadata = json.loads(request.user_prompt.split("\n\n", 2)[1])
        assert metadata == {"pr_title": fixture.pr_title}
        assert not {"pr_body", "author", "expected", "scoped_counterparts"} & metadata.keys()
        assert [file.path for file in fixture.files] == case["changed_files"]
        for file in fixture.files:
            assert f"File: {json.dumps(file.path)}" in request.user_prompt
            if file.patch:
                assert file.patch.rstrip("\n") in request.user_prompt
    assert (
        comparison.main(
            ["--target", "anthropic-sonnet-5-5", "--analyze-concurrency", "1", "--dry-run"]
        )
        == 0
    )
    serial = json.loads(capsys.readouterr().out)
    assert serial["workflow_analyze_concurrency"] == 1
    assert "not production latency" in serial["latency_configuration"]
    with pytest.raises(SystemExit) as rejected:
        comparison.main(["--target", "anthropic-sonnet-5-5", "--supplemental-context", "--dry-run"])
    assert rejected.value.code == 2
    assert "requires --compare-instructions" in capsys.readouterr().err


def test_direct_keeps_unrestricted_categories_and_unknown_locations_unadjudicated() -> None:
    fixture = EvalFixture.model_validate_json(
        (comparison.ROOT / "tests/eval/fixtures/mock_github/safe_refactor.json").read_bytes()
    )
    proposal = {
        "file_path": "unavailable/helper.py",
        "line_start": 600,
        "line_end": 600,
        "category": "resource lifetime and ownership",
        "title": "Example candidate",
        "description": "A candidate for independent review.",
        "evidence": "close(resource)",
    }
    assessed = direct_review.assess_response(
        json.dumps({"findings": [proposal, {**proposal, "line_start": 0}]}), fixture
    )
    assert assessed["status"] == "partly-invalid-response"
    assert assessed["raw_proposal_count"] == 2
    assert assessed["findings"][0]["proposal"] == proposal
    assert assessed["findings"][0]["location_in_captured_head"] is None
    assert assessed["findings"][0]["semantic_verdict"] == "needs-independent-adjudication"
    assert len(assessed["invalid_entries"]) == 1
    assert "false_positives" not in assessed
    assert "invalid_response" in direct_review.assess_response("not-json", fixture)


def test_candidate_check_plan_holds_instructions_context_and_caps_fixed(monkeypatch, capsys):
    def unexpected(*_args, **_kwargs):
        pytest.fail("free planning attempted a model or database call")

    monkeypatch.setattr(comparison, "create_provider", unexpected)
    monkeypatch.setattr(comparison, "run_review", unexpected)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    arguments = [
        "--target",
        "anthropic-sonnet-5-5",
        "--compare-candidate-checks",
        "--case",
        "flask-security-1216",
        "--case",
        "flask-security-1183",
        "--reps",
        "1",
        "--analyze-concurrency",
        "1",
        "--calls-per-review",
        "32",
        "--dry-run",
    ]
    assert comparison.main(arguments) == 0
    plan = json.loads(capsys.readouterr().out)
    arms = comparison.CANDIDATE_CHECK_ARMS
    assert plan["comparison_mode"] == "candidate-checks"
    assert plan["model_calls_made"] == 0
    assert plan["arms"] == list(arms)
    assert plan["instruction_control"]["current"] == plan["instruction_control"]["candidate"]
    assert plan["instruction_control"]["current"]["prompt_version"] == CONCISE_VERSION
    assert plan["module_context_control"]["enabled_by_arm"] == {arms[0]: True, arms[1]: True}
    assert (
        plan["module_context_control"]["allocation_policy"] == "baseline-first-module-headroom-v1"
    )
    assert plan["candidate_check_control"]["enabled_by_arm"] == {arms[0]: False, arms[1]: True}
    assert plan["candidate_check_control"]["retention_enabled_by_arm"] == {
        arms[0]: True,
        arms[1]: True,
    }
    assert plan["candidate_check_control"]["maximum_checks_per_review"] == 2
    assert plan["call_upper_bounds"] == {arms[0]: 64, arms[1]: 64, arms[2]: 2, "total": 130}
    for flag in (
        "--compare-instructions",
        "--compare-called-functions",
        "--compare-module-context",
        "--compare-review-opening",
        "--supplemental-context",
    ):
        with pytest.raises(SystemExit) as incompatible:
            comparison.main([*arguments, flag])
        assert incompatible.value.code == 2


@pytest.mark.asyncio
async def test_candidate_check_collection_uses_existing_workflow_budget_and_exports_metadata(
    tmp_path,
):
    plan = comparison.build_plan(
        refresh.TARGETS["anthropic-sonnet-5-5"],
        reps=1,
        calls_per_review=4,
        case_ids=("starlette-933",),
        compare_candidate_checks=True,
    )
    received = []

    def scripted_workflow(path, **kwargs):
        received.append(kwargs)
        fixture = EvalFixture.model_validate_json(path.read_bytes())
        provider = kwargs["provider_factory"]()

        async def run():
            request = direct_review.render_request(
                fixture, model=plan["target"]["model"], review_id=uuid4()
            ).model_copy(
                update={
                    "node_id": "analyze",
                    "system_prompt": analyze.SYSTEM_PROMPT_STABLE_PREFIX,
                    "prompt_template_version": analyze.VERSION,
                    "response_schema_json": ANALYZE_RESPONSE_SCHEMA_JSON,
                }
            )
            await provider.complete(request)
            if kwargs["candidate_checks_enabled"]:
                await provider.complete(
                    request.model_copy(
                        update={
                            "node_id": "trace",
                            "context_summary": (),
                            "prompt_template_version": "native-candidate-check-v1",
                            "candidate_attempt_id": uuid4(),
                        }
                    )
                )
            await provider.aclose()

        asyncio.run(run())
        kwargs["diagnostic_events"].append(
            {
                "kind": "scripted-metadata",
                "event_type": "native_candidate",
                "payload": {},
            }
        )
        return SimpleNamespace(
            findings=(),
            trace_decisions=(),
            hitl_gated=False,
            automated_review_wall_ms=1,
            database_setup_wall_ms=0,
            native_candidates=(),
            candidate_assessments=(),
        )

    report = await comparison.collect(
        plan,
        maximum_usd=Decimal("9"),
        maximum_calls=5,
        output=tmp_path / "checks",
        provider_factory=lambda _target, persister: RecordingProvider(persister),
        workflow_runner=scripted_workflow,
    )
    assert report["status"] == "completed"
    assert [options["candidate_checks_enabled"] for options in received] == [False, True]
    assert all(
        options["module_context_enabled"] and options["supplemental_context_enabled"]
        for options in received
    )
    assert [report["budgets"][arm]["calls"] for arm in comparison.CANDIDATE_CHECK_ARMS] == [1, 2, 1]
    for row in report["cases"][:2]:
        assert row["native_candidates"] == row["candidate_assessments"] == []
        assert row["diagnostics_status"] == "captured"
        assert row["diagnostic_events"][0]["event_type"] == "native_candidate"
        assert row["automated_review_wall_ms"] == 1
    requests = []
    for row in report["cases"][:2]:
        exchange = json.loads((tmp_path / "checks" / row["raw_exchange_files"][0]).read_text())
        requests.append(exchange["request"])
    assert requests[0]["system_prompt"] == requests[1]["system_prompt"] == CONCISE_PROMPT


def test_candidate_check_flag_reaches_public_driver_and_real_graph_builder(monkeypatch):
    fixture_path = comparison.ROOT / "tests/eval/fixtures/mock_github/safe_refactor.json"
    fixture = EvalFixture.model_validate_json(fixture_path.read_bytes())
    received = []
    marker = object()

    async def capture_driver(_fixture, _base_url, **kwargs):
        received.append(kwargs["candidate_checks_enabled"])
        return marker

    monkeypatch.setenv("OUTRIDER_IS_EVAL", "1")
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://user@localhost:5433/outrider_test")
    monkeypatch.setattr(eval_driver, "_arun_review", capture_driver)
    for enabled in (False, True):
        assert eval_driver.run_review(fixture_path, candidate_checks_enabled=enabled) is marker
    assert received == [False, True]

    def capture_graph(**kwargs):
        received.append(kwargs["candidate_checks_enabled"])
        return marker

    received.clear()
    monkeypatch.setattr(eval_driver, "build_graph", capture_graph)
    for enabled in (False, True):
        assert (
            eval_driver._build_eval_graph(
                fixture=fixture,
                session_factory=object(),
                persister=object(),
                provider=object(),
                publisher=object(),
                checkpointer=object(),
                candidate_checks_enabled=enabled,
            )
            is marker
        )
    assert received == [False, True]


def test_scripted_full_workflow_candidate_checks_retain_unknowns_in_both_modes(tmp_path):
    payload = json.loads(
        (comparison.ROOT / "tests/eval/fixtures/mock_github/safe_refactor.json").read_text()
    )
    responses = payload.pop("llm_responses")
    payload["llm_responses"] = {}
    source = payload["files"][0]
    source["content_head"] = (
        source["content_head"].replace("sum(item.price for item in items)", "subtotal(items)")
        + "\n\ndef subtotal(items):\n    return sum(item.price for item in items)\n"
    )
    source["patch"] = (
        "@@ -1,4 +1,8 @@\n"
        + "".join(f"-{line}\n" for line in source["content_base"].splitlines())
        + "".join(f"+{line}\n" for line in source["content_head"].splitlines())
    )
    source["additions"] = payload["total_additions"] = 8
    fixture_path = tmp_path / "native-check.json"
    fixture_path.write_text(json.dumps(payload))
    unknown = json.dumps(
        {
            "findings": [
                {
                    "finding_type": "iteration_precondition_not_in_enum",
                    "evidence_tier": "judged",
                    "title": "Possible invalid iterator",
                    "description": "The caller could supply None.",
                    "evidence": "    running_total = subtotal(items)",
                    "line_start": 2,
                    "line_end": 2,
                }
            ]
        }
    )

    class NativeScript(_FixtureScriptedProvider):
        async def complete(self, request):
            if request.node_id == "analyze":
                self._responses["analyze"] = [unknown]
            elif request.node_id == "trace":
                snippets = json.loads(request.user_prompt)["reference_snippets"]
                assert sum(item["source_side"] == "head" for item in snippets) == 2
                assert any(item["source_side"] == "base" for item in snippets)
                assert {item["file_path"] for item in snippets} == {"billing/totals.py"}
                assert [
                    (entry.line_start, entry.line_end) for entry in request.context_summary
                ] == [(item["line_start"], item["line_end"]) for item in snippets]
                assert len({entry.scope_unit_name for entry in request.context_summary}) == len(
                    snippets
                )
                self._responses["trace"] = [
                    json.dumps(
                        {
                            "candidate_id": json.loads(request.user_prompt)["candidate_id"],
                            "head_assessment": "uncertain",
                            "impact_assessment": "not_claimed",
                            "repair_assessment": "not_proposed",
                            "change_assessment": "uncertain",
                            "reason": "The relevant callers are not captured.",
                            "suggested_finding_type": None,
                            "citations": [],
                        }
                    )
                ]
            return await super().complete(request)

    for enabled in (False, True):
        diagnostics = []
        result = eval_driver.run_review(
            fixture_path,
            provider_factory=lambda persister: NativeScript(
                responses.copy(),
                persister=persister,
                triage_paths=tuple(file["path"] for file in payload["files"]),
            ),
            import_path_resolver=comparison.COORDINATES_IMPORT_PATH_RESOLVER,
            candidate_checks_enabled=enabled,
            diagnostic_events=diagnostics,
        )
        assert result.findings == ()
        assert len(result.native_candidates) == 1
        assert result.native_candidates[0].rejection_reason == "finding_type_not_in_enum"
        assert len(result.candidate_assessments) == int(enabled)
        assert sum(event["event_type"] == "native_candidate" for event in diagnostics) == 1
        if enabled:
            assert result.candidate_assessments[0].outcome == "completed"
            assert result.candidate_assessments[0].status == "uncertain"
            assert result.candidate_assessments[0].repair_assessment == "not_proposed"
            assert result.candidate_assessments[0].outcome == "completed"
            assert result.candidate_assessments[0].suggested_finding_type is None
            assert sum(event["event_type"] == "candidate_assessment" for event in diagnostics) == 1
        assert result.review_metrics.llm_calls_made == 3 + int(enabled)


@pytest.mark.parametrize("supplemental_context", [False, True])
def test_instruction_plan_declares_fixed_context_and_named_arms(
    monkeypatch, capsys, supplemental_context
) -> None:
    def unexpected(*_args, **_kwargs):
        pytest.fail("free planning attempted paid or workflow work")

    monkeypatch.setattr(comparison, "create_provider", unexpected)
    monkeypatch.setattr(comparison, "run_review", unexpected)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    arguments = [
        "--target",
        "anthropic-sonnet-5-5",
        "--compare-instructions",
        "--case",
        "starlette-2962",
        "--case",
        "starlette-933",
        "--reps",
        "1",
        "--calls-per-review",
        "4",
        "--max-cost-usd",
        "9",
        "--max-calls",
        "18",
        "--dry-run",
    ]
    if supplemental_context:
        arguments.append("--supplemental-context")
    assert comparison.main(arguments) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_calls_made"] == 0
    assert plan["comparison_mode"] == "instructions"
    assert [case["id"] for case in plan["cases"]] == ["starlette-2962", "starlette-933"]
    assert [row["case_id"] for row in plan["schedule"]] == ["starlette-2962", "starlette-933"]
    assert plan["arms"] == list(comparison.INSTRUCTION_ARMS)
    assert plan["schedule"][0]["order"] == list(comparison.INSTRUCTION_ARMS)
    assert plan["instruction_control"]["supplemental_context_enabled"] is supplemental_context
    assert plan["instruction_control"]["candidate"]["prompt_version"] == CONCISE_VERSION
    assert "after planning" in plan["instruction_control"]["planner_limit"]
    assert plan["call_upper_bounds"] == {
        "current-workflow": 8,
        "concise-workflow": 8,
        "direct-one-shot": 2,
        "total": 18,
    }
    assert plan["authorization"]["maximum_calls_per_arm"] == {
        "current-workflow": 8,
        "concise-workflow": 8,
        "direct-one-shot": 2,
    }
    assert plan["authorization"]["maximum_usd_per_arm"] == "3.00000000"


def test_called_function_plan_compares_one_context_change_without_spending(monkeypatch, capsys):
    def unexpected(*_args, **_kwargs):
        pytest.fail("planning attempted provider construction or a workflow run")

    monkeypatch.setattr(comparison, "create_provider", unexpected)
    monkeypatch.setattr(comparison, "run_review", unexpected)
    arguments = [
        "--target",
        "anthropic-sonnet-5-5",
        "--compare-called-functions",
        "--case",
        "flask-security-1216",
        "--case",
        "flask-security-1183",
        "--reps",
        "1",
        "--analyze-concurrency",
        "1",
        "--dry-run",
    ]
    assert comparison.main(arguments) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_calls_made"] == 0
    assert plan["comparison_mode"] == "called-functions"
    assert plan["arms"] == list(comparison.CALLED_FUNCTION_ARMS)
    assert plan["instruction_control"]["current"] == plan["instruction_control"]["candidate"]
    assert plan["instruction_control"]["supplemental_context_enabled"] is True
    assert plan["called_function_control"]["enabled_by_arm"] == {
        "concise-workflow": False,
        "concise-with-helpers": True,
    }
    assert (
        plan["called_function_control"]["allocation_policy"]
        == "baseline-first-called-function-headroom-v1"
    )
    assert plan["authorization"]["total_maximum_usd"] is None
    with pytest.raises(SystemExit) as incompatible:
        comparison.main([*arguments, "--compare-instructions"])
    assert incompatible.value.code == 2


@pytest.mark.parametrize("compare_review_opening", [False, True])
def test_module_context_plan_keeps_controls_and_exposed_expanded_cases(
    monkeypatch, capsys, compare_review_opening
):
    def unexpected(*_args, **_kwargs):
        pytest.fail("free planning attempted a provider or workflow call")

    monkeypatch.setattr(comparison, "create_provider", unexpected)
    monkeypatch.setattr(comparison, "run_review", unexpected)
    arguments = [
        "--target",
        "anthropic-sonnet-5-5",
        "--compare-review-opening" if compare_review_opening else "--compare-module-context",
        "--include-path-cases",
        "--reps",
        "3",
        "--analyze-concurrency",
        "1",
        "--dry-run",
    ]
    assert comparison.main(arguments) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_calls_made"] == 0
    assert plan["comparison_mode"] == (
        "review-opening" if compare_review_opening else "module-context"
    )
    arms = (
        comparison.REVIEW_OPENING_ARMS if compare_review_opening else comparison.MODULE_CONTEXT_ARMS
    )
    assert plan["arms"] == list(arms)
    assert len(plan["cases"]) == 7 and len(plan["dataset"]["families"]) == 3
    assert plan["dataset"]["held_out"] is False
    if compare_review_opening:
        assert (
            plan["instruction_control"]["candidate"]
            == discovery_identity()["arms"]["simple-instructions"]
        )
        assert plan["instruction_control"]["current"]["prompt_version"] == CONCISE_VERSION
        assert (
            "original production planning overhead" in plan["instruction_control"]["planner_limit"]
        )
        assert "Only the clean pass-0 review opening" in plan["instruction_control"]["fixed_inputs"]
    else:
        assert plan["instruction_control"]["current"] == plan["instruction_control"]["candidate"]
    assert plan["instruction_control"]["supplemental_context_enabled"] is True
    control = plan["module_context_control"]
    assert control["enabled_by_arm"] == {
        arms[0]: compare_review_opening,
        arms[1]: True,
    }
    assert control["max_source_bytes_per_module"] == 16 * 1024
    assert control["max_rendered_supplement_bytes"] == 32 * 1024
    assert control["allocation_policy"] == "baseline-first-module-headroom-v1"
    assert "Reserve base file allocations first" in control["budget"]
    assert "cannot establish general parity" in control["release_gate"]
    assert plan["authorization"]["total_maximum_usd"] is None
    assert plan["call_upper_bounds"]["total"] == 7 * 3 * (2 * 32 + 1)
    for case in plan["cases"]:
        assert comparison.ROOT.joinpath(case["oracle_manifest"]).is_file()
        assert "development" in case["exposure"]
    subset = comparison.build_plan(
        refresh.TARGETS["anthropic-sonnet-5-5"],
        reps=1,
        compare_module_context=not compare_review_opening,
        compare_review_opening=compare_review_opening,
        include_path_cases=True,
        case_ids=("dbt-common-182",),
    )
    assert len(subset["dataset"]["families"]) == 1
    assert "broader than" not in subset["dataset"]["note"]
    incompatible_flags = ["--compare-instructions", "--compare-called-functions"]
    incompatible_flags.append(
        "--compare-module-context" if compare_review_opening else "--compare-review-opening"
    )
    if compare_review_opening:
        incompatible_flags.append("--supplemental-context")
    for flag in incompatible_flags:
        with pytest.raises(SystemExit) as incompatible:
            comparison.main([*arguments, flag])
        assert incompatible.value.code == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arm",
    ["concise-workflow", "concise-with-modules", "alternative-with-modules", "concise-with-checks"],
)
async def test_concise_workflow_changes_only_clean_pass_zero_transport(tmp_path, arm) -> None:
    fixture = EvalFixture.model_validate_json(
        (comparison.ROOT / "tests/eval/fixtures/mock_github/safe_refactor.json").read_bytes()
    )
    target = refresh.TARGETS["anthropic-sonnet-5-5"]
    original = direct_review.render_request(
        fixture, model=target.model, review_id=uuid4()
    ).model_copy(
        update={
            "system_prompt": analyze.SYSTEM_PROMPT_STABLE_PREFIX,
            "prompt_template_version": analyze.VERSION,
            "response_schema_json": ANALYZE_RESPONSE_SCHEMA_JSON,
        }
    )
    requests = [
        original,
        original.model_copy(update={"degraded_mode": True}),
        original.model_copy(
            update={
                "system_prompt": analyze.SYSTEM_PROMPT_STABLE_PREFIX
                + analyze.POST_TRACE_SYSTEM_PROMPT_SUFFIX,
            }
        ),
        original.model_copy(
            update={
                "node_id": "triage",
                "system_prompt": triage.SYSTEM_PROMPT,
                "prompt_template_version": triage.VERSION,
                "context_summary": (),
            }
        ),
    ]
    recorder = refresh.ExchangeRecorder(tmp_path, target)
    native = RecordingProvider(comparison._TeePersister(recorder, None))
    provider = comparison._ReviewProvider(
        native,
        recorder,
        refresh.Budget(Decimal("20"), len(requests)),
        {},
        arm=arm,
        slot="test",
        call_cap=len(requests),
        first_requests={},
    )
    for request in requests:
        await provider.complete(request)
    assert len(recorder.exchanges) == len(requests)
    for index, request in enumerate(requests):
        expected = request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
        if index == 0:
            expected.update(system_prompt=CONCISE_PROMPT, prompt_template_version=CONCISE_VERSION)
            if arm == "alternative-with-modules":
                expected = requests_for_discovery(
                    request.model_copy(
                        update={
                            "system_prompt": CONCISE_PROMPT,
                            "prompt_template_version": CONCISE_VERSION,
                        }
                    )
                )["simple-instructions"].model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
        assert recorder.exchanges[index]["request"] == expected


@pytest.mark.asyncio
async def test_call_cap_and_changed_initial_context_stop_before_native_generation(tmp_path) -> None:
    fixture = EvalFixture.model_validate_json(
        (comparison.ROOT / "tests/eval/fixtures/mock_github/safe_refactor.json").read_bytes()
    )
    target = refresh.TARGETS["anthropic-sonnet-5-5"]
    request = direct_review.render_request(
        fixture, model=target.model, review_id=uuid4()
    ).model_copy(update={"node_id": "triage", "context_summary": ()})
    first, native_calls = {}, []
    for arm in ("context-off", "context-on"):
        directory = tmp_path / arm
        directory.mkdir()
        recorder = refresh.ExchangeRecorder(directory, target)
        native = RecordingProvider(comparison._TeePersister(recorder, None))
        budget = refresh.Budget(Decimal("10"), 3)
        provider = comparison._ReviewProvider(
            native,
            recorder,
            budget,
            {},
            arm=arm,
            slot="test",
            call_cap=1,
            first_requests=first,
        )
        if arm == "context-off":
            await provider.complete(request)
            with pytest.raises(refresh.BudgetLimitError, match="per-review"):
                await provider.complete(request)
        else:
            with pytest.raises(ValueError, match="different first-triage"):
                await provider.complete(request.model_copy(update={"user_prompt": "different PR"}))
        native_calls.append(native.calls)
    assert native_calls == [1, 0]


@pytest.mark.asyncio
async def test_overlapping_workers_keep_distinct_receipts_and_usage(tmp_path) -> None:
    fixture = EvalFixture.model_validate_json(
        (comparison.ROOT / "tests/eval/fixtures/mock_github/safe_refactor.json").read_bytes()
    )
    target = refresh.TARGETS["anthropic-sonnet-5-5"]
    request = direct_review.render_request(fixture, model=target.model, review_id=uuid4())
    recorder = refresh.ExchangeRecorder(tmp_path, target)

    class OverlappingProvider(RecordingProvider):
        async def complete(self, request):
            await asyncio.sleep(0)
            return await super().complete(request)

    native = OverlappingProvider(comparison._TeePersister(recorder, None))
    budget = refresh.Budget(Decimal("20"), 4)
    provider = comparison._ReviewProvider(
        native, recorder, budget, {}, arm="context-on", slot="test", call_cap=4, first_requests={}
    )
    await asyncio.gather(*(provider.complete(request) for _ in range(4)))
    assert native.calls == budget.calls == len(provider.attempts) == 4
    assert budget.known_usd == Decimal("0.004")
    assert {exchange["call_index"] for exchange in recorder.exchanges} == {1, 2, 3, 4}
    assert len(list(tmp_path.glob("*.exchange.json"))) == 4
    assert all(
        call["persisted_exchange"] and call["cost_status"] == "usage-priced"
        for call in provider.attempts
    )


@pytest.mark.asyncio
async def test_workflow_setup_failure_stops_before_any_provider(tmp_path) -> None:
    plan = comparison.build_plan(
        refresh.TARGETS["anthropic-sonnet-5-5"],
        reps=1,
        calls_per_review=4,
        case_ids=("starlette-933",),
    )
    constructions = []

    def unavailable_workflow(*_args, **_kwargs):
        raise RuntimeError("injected workflow setup failure")

    def recording_factory(_target, persister):
        constructions.append(persister)
        return RecordingProvider(persister)

    report = await comparison.collect(
        plan,
        maximum_usd=Decimal("10"),
        maximum_calls=9,
        output=tmp_path / "unavailable",
        provider_factory=recording_factory,
        workflow_runner=unavailable_workflow,
    )
    assert constructions == []
    assert report["status"] == "incomplete"
    assert len(report["cases"]) == 1
    assert report["cases"][0]["status"] == "prerequisite-failed"
    assert report["cases"][0]["error_type"] == "RuntimeError"
    assert all(budget["calls"] == 0 for budget in report["budgets"].values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "fail_audit_write",
        "compare_instructions",
        "supplemental_context",
        "compare_called_functions",
        "compare_module_context",
        "compare_review_opening",
    ),
    [
        pytest.param(False, False, False, False, False, False, id="False"),
        pytest.param(True, False, False, False, False, False, id="True"),
        pytest.param(False, True, False, False, False, False, id="instructions"),
        pytest.param(False, True, True, False, False, False, id="instructions-with-context"),
        pytest.param(False, False, True, True, False, False, id="called-functions"),
        pytest.param(False, False, True, False, True, False, id="module-context"),
        pytest.param(False, False, True, False, False, True, id="review-opening"),
    ],
)
async def test_scripted_full_workflow_records_timing_pairing_and_archived_direct_replay(
    tmp_path,
    monkeypatch,
    fail_audit_write,
    compare_instructions,
    supplemental_context,
    compare_called_functions,
    compare_module_context,
    compare_review_opening,
) -> None:
    indexed = []
    if compare_called_functions or compare_module_context or compare_review_opening:
        import outrider.agent.nodes.analyze as analyze_node

        index_name = (
            "build_module_context_index"
            if compare_module_context or compare_review_opening
            else "build_called_function_index"
        )
        build_index = getattr(analyze_node, index_name)

        def tracked_index(*args, **kwargs):
            indexed.append(True)
            return build_index(*args, **kwargs)

        monkeypatch.setattr(analyze_node, index_name, tracked_index)
    if fail_audit_write:
        from outrider.audit.persister import AuditPersister

        async def failed_persistence(*_args, **_kwargs):
            raise RuntimeError("injected audit database failure")

        monkeypatch.setattr(AuditPersister, "persist", failed_persistence)
    if compare_instructions and not supplemental_context:
        import outrider.agent.nodes.analyze as analyze_node

        def unexpected_supplement(*_args, **_kwargs):
            pytest.fail("instruction comparison selected supplemental context")

        monkeypatch.setattr(analyze_node, "prepare_review_context", unexpected_supplement)
    name = "tests/eval/fixtures/mock_github/safe_refactor.json"
    fixture = EvalFixture.model_validate_json((comparison.ROOT / name).read_bytes())
    target = refresh.TARGETS["anthropic-sonnet-5-5"]
    inputs = refresh.input_hashes((refresh.Scenario(name, (), "synthetic-test"),))
    if (comparison.ROOT / "tests/__init__.py").is_file():
        inputs["tests/__init__.py"] = comparison._sha(comparison.ROOT / "tests/__init__.py")
    plan = {
        "schema": comparison.SCHEMA,
        "target": asdict(target),
        "inputs": inputs,
        "dataset": {"id": "synthetic-workflow-test", "held_out": False},
        "cases": [
            {
                "id": "scripted",
                "fixture": name,
                "fixture_sha256": comparison._sha(comparison.ROOT / name),
            }
        ],
        "schedule": [{"case_id": "scripted", "rep": 0, "order": list(comparison.ARMS)}],
        "workflow_call_cap_per_review": 4,
        "call_upper_bounds": {"direct-one-shot": 1},
    }
    if compare_review_opening:
        arms = comparison.REVIEW_OPENING_ARMS
    elif compare_module_context:
        arms = comparison.MODULE_CONTEXT_ARMS
    elif compare_called_functions:
        arms = comparison.CALLED_FUNCTION_ARMS
    elif compare_instructions:
        arms = comparison.INSTRUCTION_ARMS
    else:
        arms = comparison.ARMS
    if compare_instructions:
        plan.update(comparison_mode="instructions", arms=list(arms))
        plan["schedule"][0]["order"] = list(arms)
        plan["instruction_control"] = {"supplemental_context_enabled": supplemental_context}
    if compare_called_functions:
        plan.update(comparison_mode="called-functions", arms=list(arms))
        plan["schedule"][0]["order"] = list(arms)
    if compare_module_context:
        plan.update(comparison_mode="module-context", arms=list(arms))
        plan["schedule"][0]["order"] = list(arms)
    if compare_review_opening:
        plan.update(comparison_mode="review-opening", arms=list(arms))
        plan["schedule"][0]["order"] = list(arms)

    native_providers = []

    def scripted_provider(_target, persister):
        provider = _FixtureScriptedProvider(
            fixture.llm_responses,
            persister=persister,
            host=target.host,
            triage_paths=tuple(file.path for file in fixture.files),
        )
        native_providers.append(provider)
        return provider

    destination = tmp_path / "comparison"
    report = await comparison.collect(
        plan,
        maximum_usd=Decimal("10"),
        maximum_calls=9,
        output=destination,
        provider_factory=scripted_provider,
    )
    assert report["evidence_kind"] == "injected-provider"
    assert len(native_providers) == 3
    assert [case["status"] for case in report["cases"]] == (
        ["failed", "failed", "completed"] if fail_audit_write else ["completed"] * 3
    )
    assert report["status"] == ("incomplete" if fail_audit_write else "completed")
    assert report["cost_comparison"]["schema"] == "review-costs-v1"
    for arm in arms:
        assert (
            report["cost_comparison"]["totals"][arm]["known_usage_based_usd"]
            == report["budgets"][arm]["known_usage_based_usd"]
        )
    assert indexed == (
        [True, True]
        if compare_review_opening
        else [True]
        if compare_called_functions or compare_module_context
        else []
    )
    pair = report["first_triage_request_pairs"]["0:scripted"]
    assert pair[arms[0]] == pair[arms[1]]
    for row in report["cases"]:
        assert row["raw_exchange_files"]
        if row["arm"] != "direct-one-shot":
            assert row["diagnostics_status"] == "captured"
            assert row["finding_format"] == "native-analyze"
            assert all(event["event_type"] != "llm_call" for event in row["diagnostic_events"])
            if fail_audit_write:
                assert row["error_type"] == "RuntimeError"
                assert len(row["calls"]) == 1
                assert row["calls"][0]["status"] == "error"
                assert row["calls"][0]["persisted_exchange"]
                assert Decimal(report["budgets"][row["arm"]]["known_usage_based_usd"]) > 0
            else:
                assert row["automated_review_wall_ms"] >= 0
                assert row["database_setup_wall_ms"] >= 0
                assert row["whole_driver_wall_ms"] >= row["automated_review_wall_ms"]
                assert any(
                    event["event_type"] == "file_examination" for event in row["diagnostic_events"]
                )
        else:
            assert row["direct_assessment"]["status"] == "parsed"
            assert "admitted_findings" not in row
            exchange = json.loads((destination / row["raw_exchange_files"][0]).read_text())
            assert json.loads(exchange["request"]["user_prompt"].split("\n\n", 2)[1]) == {
                "pr_title": fixture.pr_title,
            }
    replay = comparison.replay_direct(destination)
    assert replay["reproduced_direct_reviews"] == 1 and replay["model_calls_made"] == 0
    assert json.loads((destination / "results.json").read_text()) == report
    receipts = json.loads((destination / "receipts.json").read_text())
    assert "adjudication/candidates.json" in receipts
    assert "adjudication/identity-map.json" in receipts
    assert "adjudication/review.json" not in receipts
    judgments = destination / "adjudication/review.json"
    judgments.write_text(judgments.read_text() + "\n")
    assert comparison.replay_direct(destination) == replay
    if (
        compare_instructions
        or compare_called_functions
        or compare_module_context
        or compare_review_opening
    ):
        analyze_requests = {}
        analyze_events = {}
        for row in report["cases"]:
            for name in row["raw_exchange_files"]:
                exchange = json.loads((destination / name).read_text())
                request = exchange["request"]
                if row["arm"] != "direct-one-shot" and request["node_id"] == "analyze":
                    analyze_requests[row["arm"]] = request
                    analyze_events[row["arm"]] = exchange["event"]
        current = analyze_requests[arms[0]]
        concise = analyze_requests[arms[1]]
        assert current["system_prompt"] == (
            CONCISE_PROMPT
            if compare_called_functions or compare_module_context or compare_review_opening
            else analyze.SYSTEM_PROMPT_STABLE_PREFIX
        )
        assert current["prompt_template_version"] == (
            CONCISE_VERSION
            if compare_called_functions or compare_module_context or compare_review_opening
            else analyze.VERSION
        )
        if compare_review_opening:
            assert (
                comparison.hashlib.sha256(concise["system_prompt"].encode()).hexdigest()
                == discovery_identity()["arms"]["simple-instructions"]["system_prompt_sha256"]
            )
        else:
            assert concise["system_prompt"] == CONCISE_PROMPT
        candidate_version = SIMPLE_VERSION if compare_review_opening else CONCISE_VERSION
        assert concise["prompt_template_version"] == candidate_version
        assert (
            "## Additional bounded review context" in current["user_prompt"]
        ) is supplemental_context
        assert (
            analyze_events[arms[0]]["prompt_template_version"] == current["prompt_template_version"]
        )
        assert analyze_events[arms[1]]["prompt_template_version"] == candidate_version
        assert (
            analyze_events[arms[0]]["system_prompt_hash"]
            == analyze_events[arms[1]]["system_prompt_hash"]
        ) is (compare_called_functions or compare_module_context)
        omit = {"system_prompt", "prompt_template_version", "review_id"}
        if compare_module_context:
            assert "Complete captured module" not in current["user_prompt"]
            assert "Complete captured module" in concise["user_prompt"]
            omit.update({"user_prompt", "context_summary"})
        if compare_review_opening:
            assert "Complete captured module" in current["user_prompt"]
            assert "Complete captured module" in concise["user_prompt"]
        assert {k: v for k, v in current.items() if k not in omit} == {
            k: v for k, v in concise.items() if k not in omit
        }
