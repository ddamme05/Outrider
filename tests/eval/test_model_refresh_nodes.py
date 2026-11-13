"""Prove the refresh checks reach each node and reject wrong model outputs."""

from __future__ import annotations

import json

import pytest

from outrider.prompts.triage_schema import file_slot_ids

from .model_refresh_nodes import assess_saved_node_case, node_case_names, run_node_case
from .review_quality import MEASUREMENT_CONTRACT_V2
from .test_model_comparison import _ScriptedProvider
from .test_patch_admission import _EXPECTED_REPLACEMENT, _TARGET_LINE, _batch_json
from .test_scorecard import _triage_admission_specs
from .test_trace_admission import _DISTRACTOR_ID, _REAL_ID, _ranking_json


def _triage_response(*, correct: bool) -> str:
    spec = _triage_admission_specs()[0]
    expected = spec.expected
    changed_files = spec.state.pr_context.changed_files
    return json.dumps(
        {
            "file_tiers": {
                slot: expected.expected_file_tiers[file.path].value if correct else "skim"
                for slot, file in zip(file_slot_ids(len(changed_files)), changed_files, strict=True)
            },
            "overall_risk": expected.overall_risk.value if correct else "low",
            "relevant_dimensions": [dimension.value for dimension in expected.relevant_dimensions],
            "reasoning": "Review the changed query.",
        }
    )


@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("operation", ["triage", "summary", "trace", "patch"])
@pytest.mark.parametrize("contract", [None, MEASUREMENT_CONTRACT_V2], ids=["v1", "v2"])
async def test_node_refresh_detects_bad_outputs(
    operation: str, correct: bool, contract: str | None
) -> None:
    if operation == "triage":
        name = "triage/pygoat_sql_injection"
        response = _triage_response(correct=correct)
        expected_node = "triage"
    elif operation == "summary":
        name = "summary/single critical SQLi"
        response = (
            "Critical SQL injection in orders.py allows untrusted input into the query."
            if correct
            else "The changes look clean."
        )
        expected_node = "synthesize"
    elif operation == "trace":
        name = "trace/ranking"
        ids = (_REAL_ID, _DISTRACTOR_ID) if correct else (_DISTRACTOR_ID, _REAL_ID)
        response = _ranking_json(*ids)
        expected_node = "trace"
    else:
        name = "patch/safe-yaml-loader"
        replacement = _EXPECTED_REPLACEMENT if correct else "    return yaml.dump(stream)"
        response = _batch_json(_TARGET_LINE, replacement)
        expected_node = "synthesize"
    provider = _ScriptedProvider(response)
    result = await run_node_case(
        name, provider=provider, model="claude-haiku-4-5", measurement_contract=contract
    )
    assert result["n_calls"] == 1
    assert bool(result["issues"]) is not correct
    assert len(provider.calls) == 1
    assert provider.calls[0].node_id == expected_node
    assert provider.calls[0].is_eval is True
    if contract is not None:
        assessment = result["measurement_v2"]
        assert bool(assessment["issues"]) is not correct
        if operation == "summary":
            assert assessment["qualitative_review"] == "required"
            assert assessment["unscored"]


async def test_unknown_case_stops_before_provider_call() -> None:
    assert len(node_case_names()) == 13
    provider = _ScriptedProvider("unused")
    with pytest.raises(ValueError, match="unknown node scenario"):
        await run_node_case("missing", provider=provider, model="claude-haiku-4-5")
    assert provider.calls == []


@pytest.mark.parametrize(
    "tier,dimension,accepted",
    [
        ("standard", "code_quality", True),
        ("deep", "best_practices", True),
        ("skim", "code_quality", False),
        ("standard", "test_coverage", False),
    ],
)
async def test_v2_triage_accepts_prompt_overlap_but_preserves_attention(
    tier: str, dimension: str, accepted: bool
) -> None:
    spec = _triage_admission_specs()[2]
    changed_files = spec.state.pr_context.changed_files
    provider = _ScriptedProvider(
        json.dumps(
            {
                "file_tiers": dict.fromkeys(file_slot_ids(len(changed_files)), tier),
                "overall_risk": "low",
                "relevant_dimensions": [dimension],
                "reasoning": "Review the small network helper and its failure behavior.",
            }
        )
    )
    result = await run_node_case(
        "triage/missing_error_handling",
        provider=provider,
        model="claude-haiku-4-5",
        measurement_contract=MEASUREMENT_CONTRACT_V2,
    )
    assert bool(result["measurement_v2"]["issues"]) is not accepted
    assert result["issues"]  # The old exact-risk grade is retained, not rewritten.


@pytest.mark.parametrize(
    "text,finish",
    [
        ("# Review\nCritical SQL injection in orders.py.", "end_turn"),
        ("Critical SQL injection in orders.py. " * 100, "end_turn"),
        ("Critical SQL injection in orders.py.", "max_tokens"),
    ],
)
def test_v2_summary_keeps_format_length_and_output_limit_failures(text: str, finish: str) -> None:
    result = assess_saved_node_case(
        "summary/single critical SQLi",
        request={"node_id": "synthesize"},
        response={"text": text, "finish_reason": finish},
        prior_result={"n_calls": 1},
    )
    assert result["issues"]
    assert result["qualitative_review"] == "required"
