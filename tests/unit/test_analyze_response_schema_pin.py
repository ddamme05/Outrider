# Per specs/2026-06-12-constrained-decoding.md — the pinned analyze response schema.
"""The hand-trimmed constrained-decoding schema (FUP-096): derivability of
the digest from the canonical JSON string, structural agreement with the
raw proposal models it mirrors, and absence of JSON Schema keywords the
Anthropic structured-outputs grammar rejects with a 400.

Structural agreement is deliberately NOT byte equality with
`model_json_schema()`: Pydantic emits `maxLength` / `maxItems` /
`minimum` constraints that the API forbids, so the pin is hand-trimmed.
What must never drift is the property-name sets and required sets — a
field added, removed, renamed, or flipped required on the raw layer
without a matching schema edit fails here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from outrider.policy.canonical import canonicalize_for_hash, compute_identity_hash
from outrider.schemas.llm.analyze import (
    ANALYZE_RESPONSE_FORMAT_DIGEST,
    ANALYZE_RESPONSE_SCHEMA,
    ANALYZE_RESPONSE_SCHEMA_JSON,
    AnalyzeFindingProposalRaw,
    AnalyzeResponseRaw,
    TraceCandidateProposalRaw,
)

# Keywords the structured-outputs grammar documents as unsupported —
# their presence anywhere in the schema turns every analyze call into
# an API 400, so the pin must stay free of them.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "maxLength",
        "minLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "patternProperties",
    }
)

_FINDING_SCHEMA = ANALYZE_RESPONSE_SCHEMA["properties"]["findings"]["items"]
_CANDIDATE_SCHEMA = _FINDING_SCHEMA["properties"]["trace_candidates"]["items"]


def _walk(node: Any) -> list[dict[str, Any]]:
    """Every dict node in the schema tree, depth-first."""
    if isinstance(node, dict):
        found = [node]
        for value in node.values():
            found.extend(_walk(value))
        return found
    if isinstance(node, list):
        found = []
        for value in node:
            found.extend(_walk(value))
        return found
    return []


def _required_fields(model: type[Any]) -> set[str]:
    return {name for name, f in model.model_fields.items() if f.is_required()}


# ---------------------------------------------------------------------------
# Digest derivability — the invariant that lets LLMRequest recompute the
# digest from its own string field without ever drifting from the constant.
# ---------------------------------------------------------------------------


def test_json_string_is_the_canonical_serialization() -> None:
    assert (
        canonicalize_for_hash(ANALYZE_RESPONSE_SCHEMA).decode("utf-8")
        == ANALYZE_RESPONSE_SCHEMA_JSON
    )


def test_digest_derives_from_both_representations() -> None:
    """digest == identity-hash(dict) == sha256(json-string): the chain
    `LLMRequest.response_format_digest` (recomputed from the string) and
    the cache key (passed the constant) both depend on."""
    assert compute_identity_hash(ANALYZE_RESPONSE_SCHEMA) == ANALYZE_RESPONSE_FORMAT_DIGEST
    assert (
        hashlib.sha256(ANALYZE_RESPONSE_SCHEMA_JSON.encode("utf-8")).hexdigest()
        == ANALYZE_RESPONSE_FORMAT_DIGEST
    )


def test_json_string_round_trips_to_the_dict() -> None:
    """What the provider sends (`json.loads` of the string) is exactly
    the pinned dict — no canonicalization loss."""
    assert json.loads(ANALYZE_RESPONSE_SCHEMA_JSON) == ANALYZE_RESPONSE_SCHEMA


# ---------------------------------------------------------------------------
# API-compatibility: the grammar's hard requirements.
# ---------------------------------------------------------------------------


def test_every_object_level_forbids_additional_properties() -> None:
    for node in _walk(ANALYZE_RESPONSE_SCHEMA):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, node


def test_no_unsupported_constraint_keywords_anywhere() -> None:
    for node in _walk(ANALYZE_RESPONSE_SCHEMA):
        present = _UNSUPPORTED_KEYWORDS.intersection(node)
        assert not present, f"unsupported keywords {present} in {node}"


# ---------------------------------------------------------------------------
# Drift against the raw proposal layer the schema mirrors.
# ---------------------------------------------------------------------------


def test_top_level_mirrors_analyze_response_raw() -> None:
    assert set(ANALYZE_RESPONSE_SCHEMA["properties"]) == set(AnalyzeResponseRaw.model_fields)
    assert set(ANALYZE_RESPONSE_SCHEMA["required"]) == _required_fields(AnalyzeResponseRaw)


def test_finding_level_mirrors_finding_proposal_raw() -> None:
    assert set(_FINDING_SCHEMA["properties"]) == set(AnalyzeFindingProposalRaw.model_fields)
    assert set(_FINDING_SCHEMA["required"]) == _required_fields(AnalyzeFindingProposalRaw)


def test_candidate_level_mirrors_trace_candidate_raw() -> None:
    assert set(_CANDIDATE_SCHEMA["properties"]) == set(TraceCandidateProposalRaw.model_fields)
    assert set(_CANDIDATE_SCHEMA["required"]) == _required_fields(TraceCandidateProposalRaw)


def test_optional_fields_admit_null() -> None:
    """The raw layer's `X | None = None` fields must accept JSON null
    under the constrained grammar, or the model is forced to emit a
    value Pydantic then rejects."""
    for field_name in ("query_match_id", "trace_path"):
        variants = _FINDING_SCHEMA["properties"][field_name]["anyOf"]
        assert {"type": "null"} in variants, field_name


def test_prose_fields_carry_nonempty_guidance_annotations() -> None:
    """FUP-169: two consecutive constrained live runs returned
    `description`/`evidence` as empty strings while the same prompt
    filled them free-form — under the grammar, the schema is where
    field guidance must live. The `description` annotations are
    API-enforced (sent on the wire) and must demand non-empty content;
    this tripwire fails if a future schema edit strips them."""
    for field_name in ("description", "evidence"):
        annotation = _FINDING_SCHEMA["properties"][field_name].get("description", "")
        assert "Non-empty" in annotation, field_name
