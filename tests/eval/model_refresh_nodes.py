"""Compare the other review steps using their existing scenarios and graders.

Triage, trace and patch cases use production nodes; summary cases use the
production prompt. Trace reads come from an in-memory repository.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import patch
from uuid import uuid4

from outrider.agent.nodes.patch_generation import generate_patches
from outrider.agent.nodes.trace import trace
from outrider.llm.base import INCLUDE_TEXT_OPT_IN, LLMRequest
from outrider.prompts import synthesize as summary_prompt

from .test_openai_node_admission import _invented_references, _material_omissions
from .test_patch_admission import _EXPECTED_REPLACEMENT, _admission_finding
from .test_patch_admission import _state as patch_state
from .test_scorecard import _triage_admission_specs
from .test_synthesize_summary_comparison import _scenarios
from .test_trace_admission import (
    _EchoTraceSink,
    _grade_ranking,
    _NoOpPhaseSink,
    _stub_github_factory,
)
from .test_trace_admission import _state as trace_state
from .triage_grading import grade_triage, run_triage_under_model

if TYPE_CHECKING:
    from outrider.llm.base import LLMProvider, LLMResponse


@dataclass
class _RecordedCall:
    provider: LLMProvider
    requests: list[LLMRequest] = field(default_factory=list)
    responses: list[LLMResponse] = field(default_factory=list)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        response = await self.provider.complete(request)
        self.responses.append(response)
        return response

    async def aclose(self) -> None:
        await self.provider.aclose()


def node_case_names() -> tuple[str, ...]:
    """List the fixed scenarios before constructing a paid provider."""
    triage_names = tuple(f"triage/{spec.scenario}" for spec in _triage_admission_specs())
    summary_names = tuple(f"summary/{name}" for name, _, _ in _scenarios())
    return (*triage_names, *summary_names, "trace/ranking", "patch/safe-yaml-loader")


async def _run_triage(name: str, provider: LLMProvider, model: str) -> dict[str, object]:
    spec = next(spec for spec in _triage_admission_specs() if f"triage/{spec.scenario}" == name)
    result = await run_triage_under_model(spec.state, provider=provider, model=model)
    grade = grade_triage(result, spec.expected)
    issues = []
    if grade.n_dropped_from_analysis:
        issues.append("expected files were excluded from analysis")
    if grade.under_risked:
        issues.append("risk was understated")
    if grade.dimension_recall < 1:
        issues.append("an expected review dimension was missed")
    return {"grade": asdict(grade), "issues": issues, "output": result.model_dump(mode="json")}


async def _run_summary(name: str, provider: LLMProvider, model: str) -> dict[str, object]:
    _, risk, findings = next(row for row in _scenarios() if f"summary/{row[0]}" == name)
    parts = summary_prompt.render(overall_risk=risk, findings=findings)
    response = await provider.complete(
        LLMRequest(
            system_prompt=parts.system_prompt,
            user_prompt=parts.user_prompt,
            model=model,
            max_tokens=summary_prompt.MAX_TOKENS,
            temperature=summary_prompt.TEMPERATURE,
            review_id=uuid4(),
            node_id="synthesize",
            is_eval=True,
            prompt_template_version=summary_prompt.VERSION,
            degraded_mode=False,
        )
    )
    omissions = _material_omissions(response.text, findings)
    inventions = _invented_references(response.text, findings)
    issues = []
    if not response.text.strip():
        issues.append("summary was empty")
    if omissions:
        issues.append("summary omitted important findings")
    if inventions:
        issues.append("summary invented a file or proof reference")
    return {
        "grade": {"omitted_findings": omissions, "invented_references": inventions},
        "issues": issues,
        "output": response.text,
        "qualitative_review": "pending",
    }


async def _run_trace(provider: _RecordedCall, model: str) -> dict[str, object]:
    async def read_repository(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        if path == "app/db.py":
            return b"def run_query(sql):\n    ...\n"
        return None

    with patch("outrider.agent.nodes.trace.fetch_file_content_at", read_repository):
        result = await trace(
            trace_state(),
            provider=provider,
            trace_model=model,
            phase_event_sink=_NoOpPhaseSink(),
            trace_sink=_EchoTraceSink(),
            github_factory=_stub_github_factory,
        )
    text = provider.responses[-1].text if provider.responses else ""
    verdict, detail = _grade_ranking(text)
    resolved = any(
        decision.resolution_status == "resolved" and decision.target_file == "app/db.py"
        for decision in result["trace_decisions"]
    )
    issues = []
    if verdict != "pass":
        issues.append("the relevant trace candidate was not ranked first")
    if not resolved:
        issues.append("the trace target was not resolved")
    return {"grade": {"ranking": verdict, "detail": detail, "resolved": resolved}, "issues": issues}


async def _run_patch(provider: LLMProvider, model: str) -> dict[str, object]:
    result = await generate_patches(
        (_admission_finding(),),
        patch_state(),
        provider=provider,
        patch_model=model,
        patches_enabled=True,
        max_suggestions=4,
    )
    replacement = result[0].suggested_fix
    correct = replacement == _EXPECTED_REPLACEMENT
    return {
        "grade": {"correct_replacement": correct},
        "issues": [] if correct else ["the patch did not make the reviewed safe replacement"],
        "output": replacement,
    }


def assess_saved_node_case(
    name: str,
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    prior_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assess a recorded node result without changing its original grade.

    Routing requirements are distinct from subjective exact labels. Summary
    meaning still needs review; trace and patch cover one fixed case each.
    """
    from .review_quality import MEASUREMENT_CONTRACT_V2

    if name not in node_case_names():
        raise ValueError(f"unknown node scenario: {name}")
    operation = name.split("/", 1)[0]
    issues: list[str] = []
    unscored: list[str] = []
    diagnostics: dict[str, Any] = {}
    assessment = {
        "measurement_contract": MEASUREMENT_CONTRACT_V2,
        "operation": operation,
        "issues": issues,
        "unscored": unscored,
        "diagnostics": diagnostics,
        "qualitative_review": "required" if operation == "summary" else None,
    }
    expected_node = {"patch": "synthesize", "summary": "synthesize"}.get(operation, operation)
    if request.get("node_id") != expected_node:
        issues.append("request belongs to a different review step")
    if response.get("finish_reason") not in {"stop", "end_turn"}:
        issues.append("response did not complete normally")
    if prior_result is None or prior_result.get("n_calls") != 1:
        issues.append("one replayed node result is required")
        return assessment
    if operation == "triage":
        from pydantic import ValidationError

        from outrider.schemas.triage_result import TriageResult

        try:
            actual = TriageResult.model_validate(prior_result.get("output"))
        except ValidationError:
            issues.append("triage result is invalid")
            return assessment
        spec = next(s for s in _triage_admission_specs() if f"triage/{s.scenario}" == name)
        expected = spec.expected
        if set(actual.file_tiers) != set(expected.expected_file_tiers):
            issues.append("triage must account for exactly the supplied files")
        for path, tier in expected.expected_file_tiers.items():
            admitted = actual.file_tiers.get(path)
            allowed = {"deep"} if tier.value == "deep" else {"deep", "standard"}
            if admitted is None or admitted.value not in allowed:
                issues.append(f"required review attention was lost: {path}")
        # The prompt explicitly overlaps small isolated work with typical feature
        # work. Accept that interval consistently for both non-security features.
        if spec.scenario in {"missing_error_handling", "n_plus_one_query"}:
            risks = {"low", "medium"}
        elif spec.scenario == "safe_refactor":
            risks = {"low"}
        else:
            risks = {"high", "critical"}
        if actual.overall_risk.value not in risks:
            issues.append("risk falls outside the declared scenario interval")
        dimensions = {dimension.value for dimension in actual.relevant_dimensions}
        required_groups = (
            [{"code_quality", "best_practices"}]
            if spec.scenario == "missing_error_handling"
            else [{dimension.value} for dimension in expected.relevant_dimensions]
        )
        if any(not dimensions.intersection(group) for group in required_groups):
            issues.append("required review attention was omitted")
        relevant = set().union(*required_groups)
        diagnostics.update(
            accepted_risk_levels=sorted(risks),
            accepted_dimension_groups=[sorted(group) for group in required_groups],
            additional_dimensions=sorted(dimensions - relevant),
            original_grade=prior_result.get("grade"),
        )
    elif operation == "summary":
        text = response.get("text", "")
        _, _, findings = next(row for row in _scenarios() if f"summary/{row[0]}" == name)
        omissions = _material_omissions(text, findings)
        inventions = _invented_references(text, findings)
        if not text.strip():
            issues.append("summary was empty")
        if len(text) > 2000:
            issues.append("summary exceeds the output character limit")
        if re.search(r"(?m)^\s*(?:#{1,6}\s|[-*]\s|\d+[.)]\s|```)", text):
            issues.append("summary uses a prohibited presentation format")
        if omissions:
            issues.append("summary omitted important findings")
        if inventions:
            issues.append("summary invented a file or proof reference")
        unscored.append("grounding and workflow claims require separate prose review")
        diagnostics.update(omitted_findings=omissions, invented_references=inventions)
    elif operation == "trace":
        verdict, detail = _grade_ranking(response.get("text", ""))
        resolved = prior_result.get("grade", {}).get("resolved") is True
        if verdict != "pass" or not resolved:
            issues.append("the relevant trace candidate was not ranked and resolved")
        diagnostics.update(ranking=verdict, detail=detail, resolved=resolved)
    else:
        correct = prior_result.get("output") == _EXPECTED_REPLACEMENT
        if not correct:
            issues.append("patch differs from the reviewed safe replacement")
        diagnostics["correct_replacement"] = correct
    if operation in {"trace", "patch"}:
        diagnostics["coverage"] = "one fixed scenario; not general role qualification"
    return assessment


async def run_node_case(
    name: str,
    *,
    provider: LLMProvider,
    model: str,
    measurement_contract: str | None = None,
) -> dict[str, object]:
    """Run one named case and retain failures separately from model quality."""
    if name not in node_case_names():
        raise ValueError(f"unknown node scenario: {name}")
    if measurement_contract is not None:
        from .review_quality import MEASUREMENT_CONTRACT_V2

        if measurement_contract != MEASUREMENT_CONTRACT_V2:
            raise ValueError(f"unsupported node measurement contract: {measurement_contract}")
    recorded = _RecordedCall(provider)
    if name.startswith("triage/"):
        result = await _run_triage(name, recorded, model)
    elif name.startswith("summary/"):
        result = await _run_summary(name, recorded, model)
    elif name == "trace/ranking":
        result = await _run_trace(recorded, model)
    else:
        result = await _run_patch(recorded, model)
    if len(recorded.requests) != 1 or len(recorded.responses) != 1:
        result["issues"] = [
            *result["issues"],
            "the case did not return exactly one usable response",
        ]
    result["operation"] = name.split("/", 1)[0]
    result["n_calls"] = len(recorded.requests)
    if measurement_contract is not None:
        result["measurement_v2"] = assess_saved_node_case(
            name,
            request=recorded.requests[0].model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
            if recorded.requests
            else {},
            response=recorded.responses[0].model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
            if recorded.responses
            else {},
            prior_result=result,
        )
    return result
