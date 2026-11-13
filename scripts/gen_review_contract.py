#!/usr/bin/env python
"""Export the implemented review shapes and policy to docs/review-contract.json.

Run with --check for a read-only freshness check. Imports load definitions only;
the exporter does not create providers, connect to a database, or run a review.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from outrider.agent import candidate_checks
from outrider.agent.nodes.patch_generation import PatchSuggestionBatch
from outrider.agent.nodes.trace_parser import TraceRankingResponseRaw
from outrider.policy.dimensions import FINDING_TYPE_TO_DIMENSION
from outrider.policy.severity import ACTIVE_POLICY_VERSION, SEVERITY_POLICY, FindingType
from outrider.prompts import analyze, patch, synthesize, trace, triage
from outrider.prompts.triage_schema import response_schema_json as triage_response_schema_json
from outrider.schemas.hitl import HITLDecision, HITLRequest
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA_JSON, AnalyzeResponseRaw
from outrider.schemas.native_candidate import (
    CandidateAssessmentRef,
    CandidateCheckStarted,
    ChangeAssessment,
    HeadAssessment,
    ImpactAssessment,
    NativeCandidateRef,
    RepairAssessment,
    combine_candidate_assessment,
)
from outrider.schemas.pr_context import PRContext
from outrider.schemas.review_finding import _CONFIDENCE_BY_TIER, ReviewDimension, ReviewFinding
from outrider.schemas.review_report import EMPTY_FINDINGS_SUMMARY_SOURCE, ReviewReport
from outrider.schemas.trace_decision import TraceDecision
from outrider.schemas.triage_result import TriageResult

if TYPE_CHECKING:
    from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT = _ROOT / "docs" / "review-contract.json"


def _source(value: Any) -> str:
    source_file = inspect.getsourcefile(value)
    if source_file is None:
        raise ValueError(f"No source file for {value!r}")
    path = Path(source_file).resolve().relative_to(_ROOT).as_posix()
    return f"{path}:{value.__qualname__}"


def _model_component(
    model: type[BaseModel],
    *,
    kind: str,
    version: str | None = None,
    serialization: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    mode: Literal["serialization", "validation"] = (
        "serialization" if serialization else "validation"
    )
    component: dict[str, Any] = {
        "source": _source(model),
        "kind": kind,
        "schema_mode": mode,
        "schema": model.model_json_schema(mode=mode),
    }
    if version is not None:
        component["version"] = version
    if notes is not None:
        component["notes"] = notes
    return component


def build_catalog() -> dict[str, Any]:
    components = {
        "PRContext": _model_component(PRContext, kind="review-input"),
        "TriageResult": _model_component(
            TriageResult,
            kind="validated-state-record",
            version=triage.VERSION,
            notes="Live responses use numbered file IDs, decoded to the supplied paths before "
            "constructing this state record. Policy rejects SKIP, missing/unknown paths and "
            "inactive policy versions; reasoning is clamped. Historical records keep path keys.",
        ),
        "TriageResponseWire": {
            "source": "src/outrider/prompts/triage_schema.py:response_schema_json",
            "kind": "constrained-decoding-schema",
            "version": triage.VERSION,
            "example_file_count": 2,
            "schema": json.loads(triage_response_schema_json(2)),
            "notes": "Example for two files. The actual schema requires file_1 through file_N "
            "for the supplied file count; it contains no repository paths. The same count "
            "shares a schema. Native validation still checks the returned IDs and policy.",
        },
        "AnalyzeResponseRaw": _model_component(
            AnalyzeResponseRaw,
            kind="raw-model-response",
            version=analyze.VERSION,
            notes="Bounded raw strings retain unknown categories and tiers for recorded rejection. "
            "Schema acceptance does not establish finding admission or evidence validity.",
        ),
        "AnalyzeResponseWire": {
            "source": "src/outrider/schemas/llm/analyze.py:ANALYZE_RESPONSE_SCHEMA_JSON",
            "kind": "constrained-decoding-schema",
            "version": analyze.VERSION,
            "schema": json.loads(ANALYZE_RESPONSE_SCHEMA_JSON),
            "notes": "The actual provider schema omits some raw-model bounds and keeps category "
            "and tier values as strings. Admission validates canonical values, supplied query IDs, "
            "source ranges and evidence separately.",
        },
        "TraceRankingResponseRaw": _model_component(
            TraceRankingResponseRaw,
            kind="validated-model-response",
            version=trace.VERSION,
            notes="No response_schema_json is sent. Parser additionally requires exactly the "
            "supplied IDs once each; invalid rankings retain the supplied order.",
        ),
        "TraceDecision": _model_component(TraceDecision, kind="deterministic-state-record"),
        "CandidateCheckResponseWire": {
            "source": _source(candidate_checks._response_schema),
            "kind": "constrained-decoding-schema",
            "version": candidate_checks.VERSION,
            "schema": json.loads(candidate_checks._response_schema()),
            "notes": "Opt-in investigation output. Citation identity, exact quotes, ranges and "
            "source-side requirements are validated after generation. The combined status and "
            "suggested severity are derived by code, not returned by the model.",
        },
        "NativeCandidateRef": _model_component(
            NativeCandidateRef, kind="retained-candidate-metadata", serialization=True
        ),
        "CandidateCheckStarted": _model_component(
            CandidateCheckStarted, kind="investigation-attempt-metadata", serialization=True
        ),
        "CandidateAssessmentRef": _model_component(
            CandidateAssessmentRef,
            kind="investigation-result-metadata",
            serialization=True,
            notes="Contains references and hashes rather than raw reason or source text. Legacy "
            "records can omit repair, impact or all assessment axes; current v5 responses "
            "require local HEAD, broader impact, PR relationship and repair-advice judgments.",
        ),
        "ReviewFinding": _model_component(
            ReviewFinding,
            kind="admitted-finding",
            serialization=True,
            notes="Serialization includes computed confidence. Policy, proof-boundary, coordinate "
            "and identity validators enforce rules beyond the exported JSON Schema.",
        ),
        "SummaryText": {
            "source": "src/outrider/agent/nodes/synthesize.py:synthesize",
            "kind": "assembled-summary",
            "version": synthesize.VERSION,
            "empty_findings_source": EMPTY_FINDINGS_SUMMARY_SOURCE,
            "schema": ReviewReport.model_json_schema()["properties"]["summary"],
            "notes": "Nonempty reports use plain model prose, not JSON, with no response schema. "
            "Empty reports use the versioned fixed summary without a model call. The completion "
            "event identifies the source and hashes raw model or exact template bytes. The final "
            "report bounds text. Risk and metrics come from upstream state and code.",
        },
        "PatchSuggestionBatch": _model_component(
            PatchSuggestionBatch,
            kind="validated-model-response",
            version=patch.VERSION,
            notes="Separate optional patch call; no response_schema_json is sent. Replacement "
            "eligibility, target-line equality and output safety are checked by code.",
        ),
        "ReviewReport": _model_component(
            ReviewReport, kind="assembled-review-output", serialization=True
        ),
        "HITLRequest": _model_component(HITLRequest, kind="human-approval-request"),
        "HITLDecision": _model_component(HITLDecision, kind="human-approval-decision"),
    }
    return {
        "format": "outrider-review-contract-v1",
        "human_reference": "REVIEW_CONTRACT.md",
        "purpose": "Generated implementation reference. Components are separate runtime shapes, "
        "not a new API endpoint, combined response envelope, or detection-quality guarantee.",
        "regenerate": ".venv/bin/python scripts/gen_review_contract.py",
        "check": ".venv/bin/python scripts/gen_review_contract.py --check",
        "schema_scope": "Each component schema is self-contained. Pydantic JSON Schemas describe "
        "shape and declared constraints; custom validators and pipeline admission add rules. "
        "Only components marked constrained-decoding-schema are sent as response_schema_json.",
        "http_api_schema": "dashboard/openapi.json",
        "policy_version": ACTIVE_POLICY_VERSION,
        "category_sources": [
            "src/outrider/policy/severity.py:FindingType",
            "src/outrider/policy/severity.py:SEVERITY_POLICY",
            "src/outrider/policy/dimensions.py:FINDING_TYPE_TO_DIMENSION",
        ],
        "dimensions": [dimension.value for dimension in ReviewDimension],
        "categories": [
            {
                "finding_type": finding_type.value,
                "severity": SEVERITY_POLICY[finding_type].value,
                "dimension": FINDING_TYPE_TO_DIMENSION[finding_type].value,
            }
            for finding_type in FindingType
        ],
        "confidence": {
            "source": "src/outrider/schemas/review_finding.py:_CONFIDENCE_BY_TIER",
            "meaning": "Fixed evidence-tier score, not a calibrated probability of correctness.",
            "by_evidence_tier": {tier.value: score for tier, score in _CONFIDENCE_BY_TIER.items()},
        },
        "candidate_assessment_combination": {
            "source": _source(combine_candidate_assessment),
            "meaning": "Derived provisional status. A refuted HEAD claim or demonstrated unrelated "
            "preexisting issue yields refutes. Unknown or refuted broader impact or repair "
            "advice leaves a supported local bug uncertain for review; no status removes the "
            "original finding. Historical cases retain their original combinations; repair "
            "cases show the additional v5 condition for an otherwise supported claim.",
            "cases": [
                {
                    "head_assessment": head,
                    "change_assessment": change,
                    "impact_assessment": impact,
                    "status": combine_candidate_assessment(head, change, impact),
                }
                for head in get_args(HeadAssessment)
                for change in get_args(ChangeAssessment)
                for impact in get_args(ImpactAssessment)
            ],
            "repair_cases": [
                {
                    "head_assessment": "supports",
                    "change_assessment": "introduced_or_worsened",
                    "impact_assessment": "not_claimed",
                    "repair_assessment": repair,
                    "status": combine_candidate_assessment(
                        "supports", "introduced_or_worsened", "not_claimed", repair
                    ),
                }
                for repair in get_args(RepairAssessment)
            ],
            "legacy_cases": [
                {
                    "head_assessment": head,
                    "change_assessment": change,
                    "status": combine_candidate_assessment(head, change),
                }
                for head in get_args(HeadAssessment)
                for change in get_args(ChangeAssessment)
            ],
        },
        "components": components,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without writing files")
    args = parser.parse_args()
    text = json.dumps(build_catalog(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    rendered = text.encode("utf-8")
    if args.check:
        if not _OUTPUT.exists() or _OUTPUT.read_bytes() != rendered:
            print(
                "docs/review-contract.json is stale; "
                "run .venv/bin/python scripts/gen_review_contract.py"
            )
            return 1
        print("docs/review-contract.json is current")
        return 0
    _OUTPUT.parent.mkdir(exist_ok=True)
    _OUTPUT.write_bytes(rendered)
    print("Wrote docs/review-contract.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
