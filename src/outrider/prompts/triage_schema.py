"""Constrain triage to numbered files without putting repository paths in schemas."""

import json

from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.review_finding import ReviewDimension
from outrider.schemas.triage_result import ReviewTier, RiskLevel


def file_slot_ids(file_count: int) -> tuple[str, ...]:
    return tuple(f"file_{index + 1}" for index in range(file_count))


def response_schema_json(file_count: int) -> str:
    """The same file count produces the same grammar, regardless of filenames."""
    slots = file_slot_ids(file_count)
    tier = {"type": "string", "enum": [t.value for t in ReviewTier if t is not ReviewTier.SKIP]}
    schema = {
        "type": "object",
        "properties": {
            "file_tiers": {
                "type": "object",
                "properties": dict.fromkeys(slots, tier),
                "required": list(slots),
                "additionalProperties": False,
            },
            "overall_risk": {"type": "string", "enum": [risk.value for risk in RiskLevel]},
            "relevant_dimensions": {
                "type": "array",
                "items": {"type": "string", "enum": [d.value for d in ReviewDimension]},
            },
            "reasoning": {"type": "string"},
            "policy_version": {"type": "string", "enum": [ACTIVE_POLICY_VERSION]},
        },
        "required": ["file_tiers", "overall_risk", "relevant_dimensions", "reasoning"],
        "additionalProperties": False,
    }
    return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
