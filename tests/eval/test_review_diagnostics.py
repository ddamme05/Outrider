"""Check captured-input controls and collection integrity without paid calls or a DB."""

from __future__ import annotations

import json
from dataclasses import asdict
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from outrider.agent.eval_driver import EvalFixture
from outrider.audit.events import ContextManifestEntry
from outrider.llm.base import LLMRequest
from outrider.prompts import analyze
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA_JSON

from . import direct_review
from . import review_diagnostics as diagnostics
from . import review_diagnostics_inputs as inputs
from .model_refresh import ROOT, TARGETS
from .prompt_control import CONCISE_PROMPT
from .test_model_refresh import RecordingProvider

if TYPE_CHECKING:
    from pathlib import Path


def captured_report(folder: Path) -> tuple[Path, str]:
    """Make receipt-bound local transport records using an existing checked-in source fixture."""
    folder.mkdir()
    (folder / "generation").mkdir()
    data = (ROOT / "tests/eval/fixtures/mock_github/safe_refactor.json").read_bytes()
    (folder / "generation/fixture.json").write_bytes(data)
    fixture = EvalFixture.model_validate_json(data)
    file = fixture.files[0]
    native = LLMRequest(
        system_prompt=analyze.SYSTEM_PROMPT_STABLE_PREFIX,
        user_prompt=f"File: {file.path}\n{inputs.SOURCE_CLAUSE}\n{file.content_head}",
        model="claude-sonnet-5-5",
        max_tokens=8192,
        temperature=0,
        review_id=uuid4(),
        node_id="analyze",
        is_eval=True,
        prompt_template_version=analyze.VERSION,
        degraded_mode=False,
        response_schema_json=ANALYZE_RESPONSE_SCHEMA_JSON,
        context_summary=(
            ContextManifestEntry(
                file_path=file.path,
                scope_unit_name="captured-source",
                line_start=1,
                line_end=len(file.content_head.splitlines()),
                inclusion_reason="changed_scope",
            ),
        ),
    )
    direct = direct_review.render_request(fixture, model=native.model, review_id=uuid4())
    target = TARGETS["anthropic-sonnet-5-5"]
    plan = {
        "target": asdict(target),
        "dataset": {"id": "local-protocol-fixture", "exposure": "development", "held_out": False},
        "inputs": {"fixture.json": inputs.sha(folder / "generation/fixture.json")},
        "cases": [
            {
                "id": "protocol-case",
                "fixture": "fixture.json",
                "fixture_sha256": inputs.sha(folder / "generation/fixture.json"),
            }
        ],
    }
    rows = []
    for arm, request in (("context-off", native), ("direct-one-shot", direct)):
        path = folder / f"{arm}.exchange.json"
        path.write_text(json.dumps({"request": inputs.fields(request)}))
        rows.append({"case_id": "protocol-case", "arm": arm, "raw_exchange_files": [path.name]})
    (folder / "plan.json").write_text(json.dumps(plan))
    (folder / "results.json").write_text(
        json.dumps(
            {"schema": "review-comparison-v1", "status": "completed", "plan": plan, "cases": rows}
        )
    )
    refresh_receipts(folder)
    return folder, file.path


def refresh_receipts(folder: Path) -> None:
    receipt = {
        path.name: inputs.sha(path)
        for path in folder.glob("*.json")
        if path.name != "receipts.json"
    }
    (folder / "receipts.json").write_text(json.dumps(receipt))


def test_free_plan_preserves_each_control_and_original_capture(tmp_path, monkeypatch, capsys):
    report, file_path = captured_report(tmp_path / "source")

    def unexpected(*_args, **_kwargs):
        pytest.fail("free planning constructed a provider")

    monkeypatch.setattr(diagnostics, "create_provider", unexpected)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    assert diagnostics.main(["--report", str(report), "--file", file_path, "--dry-run"]) == 0
    assert "No database. No calls made" in capsys.readouterr().out
    source = inputs.load_source(report, file_path)
    case = source.cases[0]
    requests = inputs.requests_for_case(case)
    assert inputs.fields(requests["current-scoped"]) == inputs.fields(case.native)
    assert inputs.fields(requests["direct-control"]) == inputs.fields(case.direct)
    for name, request in requests.items():
        if name == "direct-control":
            continue
        assert request.response_schema_json == case.native.response_schema_json
        assert request.model == case.native.model
        assert request.max_tokens == case.native.max_tokens
        if name.endswith("full-reference"):
            assert request.user_prompt.startswith(case.native.user_prompt)
            assert file_path in request.user_prompt
        if name.startswith("concise"):
            assert request.system_prompt == CONCISE_PROMPT
    clarified = requests["clarified-scoped"]
    assert clarified.system_prompt == case.native.system_prompt
    assert clarified.user_prompt == case.native.user_prompt.replace(
        inputs.SOURCE_CLAUSE, inputs.CLARIFIED_CLAUSE, 1
    )
    assert clarified.context_summary == case.native.context_summary


def test_discovery_plan_is_free_and_rejects_provider_profile_drift(tmp_path, monkeypatch, capsys):
    from .test_discovery_control import captured_discovery_report

    report, file_path = captured_discovery_report(tmp_path / "source")

    def unexpected(*_args, **_kwargs):
        pytest.fail("free discovery planning constructed a provider")

    monkeypatch.setattr(diagnostics, "create_provider", unexpected)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    assert (
        diagnostics.main(
            [
                "--report",
                str(report),
                "--file",
                file_path,
                "--compare-discovery",
                "--reps",
                "1",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_calls_made"] == 0 and plan["planned_calls"] == 3
    assert plan["arms"] == ["captured-control", "simple-instructions", "open-categories"]
    assert plan["comparison_mode"] == "discovery"
    assert "wire schema" in plan["restriction_scope"]
    assert "no triage" in plan["scope"]
    monkeypatch.setattr(
        diagnostics, "resolve_host_identity", lambda *_args, **_kwargs: ("changed", False, "0" * 64)
    )
    with pytest.raises(ValueError, match="provider profile differs"):
        diagnostics.build_plan(report, file_path, reps=1, compare_discovery=True)


@pytest.mark.parametrize("damage", ["receipt", "generation", "direct-render", "clause"])
def test_capture_drift_stops_before_provider_or_output(tmp_path, damage):
    report, file_path = captured_report(tmp_path / "source")
    if damage == "generation":
        (report / "generation/fixture.json").write_text("{}")
    else:
        name = (
            "direct-one-shot.exchange.json"
            if damage == "direct-render"
            else "context-off.exchange.json"
        )
        path = report / name
        exchange = json.loads(path.read_text())
        exchange["request"]["user_prompt"] += " changed"
        if damage == "clause":
            exchange["request"]["user_prompt"] = exchange["request"]["user_prompt"].replace(
                inputs.SOURCE_CLAUSE, ""
            )
        path.write_text(json.dumps(exchange))
        if damage != "receipt":
            refresh_receipts(report)
    with pytest.raises(ValueError):
        diagnostics.build_plan(report, file_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("compare_discovery", [False, True])
async def test_scripted_collector_archives_exact_requests_without_native_admission(
    tmp_path, compare_discovery
):
    if compare_discovery:
        from .test_discovery_control import captured_discovery_report

        report, file_path = captured_discovery_report(tmp_path / "source")
    else:
        report, file_path = captured_report(tmp_path / "source")
    plan = diagnostics.build_plan(report, file_path, reps=1, compare_discovery=compare_discovery)
    output = tmp_path / "collected"
    providers = []

    def factory(target, recorder):
        provider = RecordingProvider(recorder)
        providers.append(provider)
        return provider

    result = await diagnostics.collect(
        plan, maximum_usd=Decimal("9"), maximum_calls=6, output=output, provider_factory=factory
    )
    assert result["status"] == "completed"
    assert result["evidence_kind"] == "injected-provider"
    expected_calls = 3 if compare_discovery else 6
    assert len(providers) == 1 and providers[0].calls == expected_calls
    assert set(result["arm_call_counts"].values()) == {1}
    assert result["budget"]["calls"] == expected_calls
    assert Decimal(result["budget"]["known_usage_based_usd"]) == Decimal("0.001") * expected_calls
    assert not result["unattempted_calls"]
    receipts = json.loads((output / "receipts.json").read_text())
    for name, digest in receipts.items():
        assert inputs.sha(output / name) == digest
    for row in result["cases"]:
        assert row["admission_status"] == "not-evaluated"
        exchange = json.loads((output / row["raw_exchange_files"][0]).read_text())
        request = LLMRequest.model_validate(exchange["request"])
        assert (
            inputs.request_identity(request) == plan["cases"][0]["request_identities"][row["arm"]]
        )
        assert ("analysis_file" in row) == (row["arm"] != "direct-control")
    fixture = output / "generation" / plan["cases"][0]["fixture"]
    assert inputs.sha(fixture) == plan["cases"][0]["fixture_sha256"]
    assert json.loads((output / "adjudication/review.json").read_text())["status"] == "pending"


@pytest.mark.asyncio
async def test_rejected_reservation_reports_zero_model_calls_per_arm(tmp_path):
    report, file_path = captured_report(tmp_path / "source")
    plan = diagnostics.build_plan(report, file_path, reps=1)
    providers = []

    def factory(target, recorder):
        provider = RecordingProvider(recorder)
        providers.append(provider)
        return provider

    result = await diagnostics.collect(
        plan,
        maximum_usd=Decimal("0.01"),
        maximum_calls=6,
        output=tmp_path / "collected",
        provider_factory=factory,
    )
    assert result["status"] == "incomplete"
    assert providers[0].calls == result["budget"]["calls"] == 0
    assert result["cases"][0]["status"] == "budget-stopped-before-call"
    assert sum(result["arm_call_counts"].values()) == 0
    assert len(result["unattempted_calls"]) == 5


@pytest.mark.asyncio
async def test_missing_persistence_cannot_complete_and_retains_returned_response(tmp_path):
    report, file_path = captured_report(tmp_path / "source")
    plan = diagnostics.build_plan(report, file_path, reps=1)
    output = tmp_path / "collected"
    providers = []

    def factory(target, recorder):
        provider = RecordingProvider(recorder, persist=False)
        providers.append(provider)
        return provider

    result = await diagnostics.collect(
        plan,
        maximum_usd=Decimal("100"),
        maximum_calls=6,
        output=output,
        provider_factory=factory,
    )
    assert result["status"] == "incomplete"
    assert providers[0].calls == result["budget"]["calls"] == 1
    row = result["cases"][0]
    assert row["status"] == "failed"
    assert not row["raw_exchange_files"]
    returned = json.loads((output / row["unrecorded_response"]).read_text())
    assert returned["text"] == '{"findings":[]}'
    assert Decimal(result["budget"]["uncertain_budget_upper_usd"]) > 0


def test_evidence_diagnostics_keep_literal_newlines_and_claimed_locations_separate(tmp_path):
    report, file_path = captured_report(tmp_path / "source")
    case = inputs.load_source(report, file_path).cases[0]
    head = case.fixture.files[0].content_head
    quote = head.splitlines()[0]
    proposals = [
        {"evidence": quote, "line_start": 1, "line_end": 1},
        {"evidence": quote, "line_start": 2, "line_end": 2},
        {"evidence": "", "line_start": 1, "line_end": 1},
        {"evidence": r"line one\nline two", "line_start": 1, "line_end": 2},
        {"evidence": "line one\nline two", "line_start": 1, "line_end": 2},
    ]
    result = diagnostics._proposal_diagnostics(
        json.dumps({"findings": proposals}), case.fixture, case.native, file_path
    )
    rows = result["proposals"]
    assert rows[0]["exact_quote_in_claimed_span"] is True
    assert rows[1]["exact_quote_in_captured_head"] is True
    assert rows[1]["exact_quote_in_claimed_span"] is False
    assert rows[2]["exact_quote_in_captured_head"] is False
    assert rows[2]["exact_quote_in_request_text"] is False
    assert rows[3]["evidence_has_literal_backslash_n"] is True
    assert rows[3]["evidence_has_newline"] is False
    assert rows[4]["evidence_has_literal_backslash_n"] is False
    assert rows[4]["evidence_has_newline"] is True
    assert all(row["semantic_verdict"] == "pending-independent-adjudication" for row in rows)
