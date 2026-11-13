"""Check source-bound, independent discovery controls without paid calls."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from outrider.llm.host_profiles import resolve_host_identity

from . import discovery_control as control
from . import review_diagnostics_inputs as inputs
from .prompt_control import (
    CONCISE_PROMPT,
    CONCISE_VERSION,
    LEGACY_CONCISE_PROMPT,
    LEGACY_CONCISE_VERSION,
    LEGACY_V2_CONCISE_PROMPT,
    LEGACY_V2_CONCISE_VERSION,
)
from .test_review_diagnostics import captured_report, refresh_receipts

if TYPE_CHECKING:
    from pathlib import Path


def captured_discovery_report(folder: Path) -> tuple[Path, str]:
    """Add the consumed module/profile metadata to a receipt-bound source facade."""
    report, file_path = captured_report(folder)
    plan = json.loads((report / "plan.json").read_text())
    profile, reasoning, digest = resolve_host_identity(
        "anthropic", reasoning=False, models=(plan["target"]["model"],)
    )
    identity = {
        "profile_id": profile,
        "reasoning_enabled": reasoning,
        "profile_contract_digest": digest,
    }
    plan.update(
        comparison_mode="module-context",
        module_context_control={"enabled_by_arm": {"concise-with-modules": True}},
        **identity,
    )
    result = json.loads((report / "results.json").read_text())
    result["plan"] = plan
    for row in result["cases"]:
        path = report / row["raw_exchange_files"][0]
        exchange = json.loads(path.read_text())
        if row["arm"] == "context-off":
            row["arm"] = "concise-with-modules"
            exchange["request"].update(
                system_prompt=CONCISE_PROMPT,
                prompt_template_version=CONCISE_VERSION,
            )
            reference = {
                **exchange["request"]["context_summary"][0],
                "scope_unit_name": "complete captured source reference",
                "inclusion_reason": "captured_reference",
            }
            exchange["request"]["context_summary"].append(reference)
        exchange["event"] = {"model": plan["target"]["model"], **identity}
        path.write_text(json.dumps(exchange))
    (report / "plan.json").write_text(json.dumps(plan))
    (report / "results.json").write_text(json.dumps(result))
    refresh_receipts(report)
    return report, file_path


def test_discovery_controls_change_independent_prompt_sections_only(tmp_path):
    report, file_path = captured_discovery_report(tmp_path / "source")
    source = inputs.load_source(report, file_path, compare_discovery=True)
    case = source.cases[0]
    requests = inputs.requests_for_case(case, compare_discovery=True)

    assert tuple(requests) == control.DISCOVERY_ARMS
    assert inputs.fields(requests["captured-control"]) == inputs.fields(case.native)
    for request in requests.values():
        unchanged = inputs.fields(request)
        original = inputs.fields(case.native)
        for field in ("system_prompt", "prompt_template_version"):
            unchanged.pop(field)
            original.pop(field)
        assert unchanged == original
    opening, _, remainder = CONCISE_PROMPT.partition("\n\n")
    simple = requests["simple-instructions"].system_prompt
    assert simple == control.SIMPLE_OPENING + "\n\n" + remainder
    assert requests["open-categories"].system_prompt.partition("\n\n")[0] == opening
    category_line = next(
        line for line in CONCISE_PROMPT.splitlines() if line.startswith("Allowed finding_type")
    )
    assert requests["open-categories"].system_prompt == CONCISE_PROMPT.replace(
        category_line, control.OPEN_CATEGORY_GUIDANCE, 1
    )
    schema = json.loads(case.native.response_schema_json)
    assert schema["properties"]["findings"]["items"]["properties"]["finding_type"] == {
        "type": "string"
    }
    assert "not JSON decoding or admission" in control.discovery_identity()["category_control"]
    assert inputs.source_identity(source)["source_profile_identity"] == source.profile_identity
    plan = inputs.case_plan(case, compare_discovery=True)
    assert len(plan["primary_context_summary"]) == 1
    assert len(case.native.context_summary) == 2
    assert plan["primary_context_summary"][0]["inclusion_reason"] == "changed_scope"
    assert plan["request_identities"] == {
        name: inputs.request_identity(request) for name, request in requests.items()
    }


@pytest.mark.parametrize(
    "version,prompt",
    [
        (CONCISE_VERSION, CONCISE_PROMPT),
        (LEGACY_CONCISE_VERSION, LEGACY_CONCISE_PROMPT),
        (LEGACY_V2_CONCISE_VERSION, LEGACY_V2_CONCISE_PROMPT),
    ],
)
def test_captured_prompt_version_and_primary_regions_survive_diagnostics(tmp_path, version, prompt):
    report, file_path = captured_discovery_report(tmp_path / "source")
    result = json.loads((report / "results.json").read_text())
    path = report / result["cases"][0]["raw_exchange_files"][0]
    exchange = json.loads(path.read_text())
    request = exchange["request"]
    request["context_summary"][0]["inclusion_reason"] = "changed_region"
    request.update(system_prompt=prompt, prompt_template_version=version)
    path.write_text(json.dumps(exchange))
    refresh_receipts(report)

    source = inputs.load_source(report, file_path, compare_discovery=True)
    case = source.cases[0]
    controls = inputs.requests_for_case(case, compare_discovery=True)
    assert inputs.fields(controls["captured-control"]) == request
    identity = control.discovery_identity(case.native.prompt_template_version)
    for arm, transformed in controls.items():
        assert transformed.prompt_template_version == identity["arms"][arm]["prompt_version"]
    assert (
        inputs.case_plan(case, compare_discovery=True)["primary_context_summary"][0][
            "inclusion_reason"
        ]
        == "changed_region"
    )


@pytest.mark.parametrize("damage", ["mode", "prompt", "version", "schema", "profile", "duplicate"])
def test_discovery_capture_rejects_wrong_route_identity_or_ambiguous_source(tmp_path, damage):
    report, file_path = captured_discovery_report(tmp_path / "source")
    plan = json.loads((report / "plan.json").read_text())
    result = json.loads((report / "results.json").read_text())
    path = report / result["cases"][0]["raw_exchange_files"][0]
    exchange = json.loads(path.read_text())
    if damage == "mode":
        plan["comparison_mode"] = "called-functions"
        result["plan"] = plan
    elif damage == "prompt":
        exchange["request"]["system_prompt"] += " changed"
    elif damage == "version":
        exchange["request"]["prompt_template_version"] = "other-version"
    elif damage == "schema":
        exchange["request"]["response_schema_json"] = "{}"
    elif damage == "profile":
        exchange["event"]["profile_contract_digest"] = "0" * 64
    else:
        result["cases"].append(result["cases"][0])
    path.write_text(json.dumps(exchange))
    (report / "plan.json").write_text(json.dumps(plan))
    (report / "results.json").write_text(json.dumps(result))
    refresh_receipts(report)

    with pytest.raises(ValueError):
        inputs.load_source(report, file_path, compare_discovery=True)
