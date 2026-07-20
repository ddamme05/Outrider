"""Arc 2 — the derived strict schema encodes the tier/proof contract at the wire.

The arc's central claim is that a strict `json_schema` makes invalid tier/proof
shapes UNREPRESENTABLE rather than merely parser-rejected. These tests validate
that claim with the reference JSON Schema implementation.

**What a pass here does and does not mean.** `jsonschema` agreeing is evidence
the encoding is well-formed and discriminating; it is NOT evidence that OpenAI's
strict-subset validator agrees, and it is not evidence about authenticity (that
a `query_match_id` names a query that actually fired — a parser concern, see
`test_arc2_strict_parser_contract.py`). Wire acceptance is exactly what the one
paid probe row exists to answer.

See `specs/2026-07-20-arc2-strict-schema-feasibility.md`.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from spikes.openai.arc2.strict_schema import (
    TIER_PROOF_SHAPES,
    derive_strict_analyze_schema,
    schema_digest,
    strict_schema_json,
)

from outrider.policy import EvidenceTier
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA

STRICT = derive_strict_analyze_schema()
VALIDATOR = Draft202012Validator(STRICT)

_CANONICAL_FINDING = ANALYZE_RESPONSE_SCHEMA["properties"]["findings"]["items"]
_CANONICAL_PROPS: dict[str, Any] = _CANONICAL_FINDING["properties"]


def _branches() -> list[dict[str, Any]]:
    return STRICT["properties"]["findings"]["items"]["anyOf"]


def _branch_for(tier: EvidenceTier) -> dict[str, Any]:
    for branch in _branches():
        if branch["properties"]["evidence_tier"]["enum"] == [tier.value]:
            return branch
    raise AssertionError(f"no branch pins evidence_tier to {tier.value!r}")


def _finding(**overrides: Any) -> dict[str, Any]:
    """A JUDGED finding with every property present — the strict baseline."""
    base: dict[str, Any] = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "t",
        "description": "d",
        "evidence": "e",
        "line_start": 1,
        "line_end": 2,
        "trace_candidates": [],
    }
    base.update(overrides)
    return base


def _validate(*findings: dict[str, Any]) -> None:
    VALIDATOR.validate({"findings": list(findings)})


def _assert_rejected(*findings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _validate(*findings)


# --------------------------------------------------------------------------
# Derivation fidelity — the schema must stay a NARROWING of canonical.
# --------------------------------------------------------------------------


def test_derived_base_matches_canonical_properties() -> None:
    """Every branch carries exactly the canonical property set, in canonical ORDER.

    Order is asserted, not just membership: the Structured Outputs guide states
    outputs are produced in schema key order, and FUP-169 is the local scar — a
    sorted schema forced `description`/`evidence` to generate before
    `finding_type`/`title` existed in context, and three live runs returned them
    empty. A reordering here would silently probe a worse request than production.
    """
    canonical_order = list(_CANONICAL_PROPS)
    for branch in _branches():
        assert list(branch["properties"]) == canonical_order
        assert branch["required"] == canonical_order


def test_derived_types_match_canonical_for_tier_invariant_properties() -> None:
    """Properties the tier does not discriminate are carried through unchanged —
    including their `description` annotations, which are API-enforced guidance
    the constrained decoder reads (FUP-169), not decoration."""
    discriminated = {"evidence_tier", *TIER_PROOF_SHAPES[EvidenceTier.JUDGED]}
    for branch in _branches():
        for name, canonical_sub in _CANONICAL_PROPS.items():
            if name in discriminated:
                continue
            derived = branch["properties"][name]
            assert derived.get("type") == canonical_sub.get("type"), name
            assert derived.get("description") == canonical_sub.get("description"), name


def test_discriminated_properties_still_preserve_canonical_annotations() -> None:
    """The gap the previous test's skip-set left open.

    `evidence_tier`, `query_match_id`, and `trace_path` have their TYPES narrowed
    per branch — but their `description` annotations must still survive, for the
    same FUP-169 reason. This is asserted against a canonical schema mutated to
    CARRY descriptions, because canonical happens to carry none on these three
    today: without the mutation the assertion would be vacuous and a regression
    would land silently the moment someone added one.
    """
    canonical = copy.deepcopy(ANALYZE_RESPONSE_SCHEMA)
    props = canonical["properties"]["findings"]["items"]["properties"]
    annotations = {
        "evidence_tier": "How this finding's claim is justified.",
        "query_match_id": "The registry id of the tree-sitter query that fired.",
        "trace_path": "Scope units walked, in order.",
    }
    for name, text in annotations.items():
        props[name]["description"] = text

    derived = derive_strict_analyze_schema(canonical)
    for branch in derived["properties"]["findings"]["items"]["anyOf"]:
        for name, text in annotations.items():
            assert branch["properties"][name].get("description") == text, name


def test_tier_mapping_covers_every_evidence_tier() -> None:
    """Exhaustive over `EvidenceTier` — a new tier fails here rather than
    silently falling through into an undiscriminated schema."""
    assert set(TIER_PROOF_SHAPES) == set(EvidenceTier)
    assert len(_branches()) == len(EvidenceTier)


def test_strict_subset_shape() -> None:
    """Every object sets `additionalProperties: false` and lists every property
    in `required` — the two hard requirements of OpenAI strict mode."""

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(STRICT)


def test_root_is_an_object_and_not_anyof() -> None:
    """OpenAI requires the ROOT to be an object and forbids `anyOf` at the root.
    The tier union therefore lives at `findings.items`, not at the top level."""
    assert STRICT["type"] == "object"
    assert "anyOf" not in STRICT
    assert "anyOf" in STRICT["properties"]["findings"]["items"]


def test_digest_is_stable_and_order_sensitive() -> None:
    """The digest is the schema's identity in the manifest. Recomputing must be
    stable, and a property reorder must ROTATE it — emission order is part of the
    format's generative identity, so a silent reorder must not read as 'same'."""
    assert schema_digest() == schema_digest()

    reordered = derive_strict_analyze_schema()
    branch = reordered["properties"]["findings"]["items"]["anyOf"][0]
    props = branch["properties"]
    branch["properties"] = dict(reversed(list(props.items())))
    assert schema_digest(reordered) != schema_digest()
    assert strict_schema_json(reordered) != strict_schema_json()


# --------------------------------------------------------------------------
# The discriminator — without the pinned enum the design is a no-op.
# --------------------------------------------------------------------------


def test_evidence_tier_enum_pins_each_branch() -> None:
    """Each branch pins `evidence_tier` to a single-member LOWERCASE enum,
    matching the canonical `EvidenceTier` wire values the parser reads."""
    for tier in EvidenceTier:
        branch = _branch_for(tier)
        assert branch["properties"]["evidence_tier"]["enum"] == [tier.value]
        assert tier.value == tier.value.lower()


def test_each_tier_branch_accepts_its_valid_shape() -> None:
    """Positive control: the three legitimate shapes all validate."""
    _validate(_finding(evidence_tier="observed", query_match_id="q1", trace_path=None))
    _validate(_finding(evidence_tier="inferred", query_match_id=None, trace_path=["a", "b"]))
    _validate(_finding(evidence_tier="judged", query_match_id=None, trace_path=None))


def test_tier_and_proof_fields_cannot_mismatch() -> None:
    """The illusory-discrimination case, and the reason the enum pin exists.

    Without pinning `evidence_tier` per branch, the union would discriminate on
    proof-field SHAPE alone — so a payload DECLARING `judged` while carrying a
    `query_match_id` would legally match the OBSERVED branch and the whole
    discrimination would be theatre.
    """
    _assert_rejected(_finding(evidence_tier="judged", query_match_id="q1"))
    _assert_rejected(_finding(evidence_tier="observed", trace_path=["a"], query_match_id="q1"))
    _assert_rejected(_finding(evidence_tier="inferred", query_match_id="q1", trace_path=["a"]))


def test_wrong_tier_proof_fields_are_unrepresentable() -> None:
    """The core claim: wrong-tier proof cannot be expressed at all."""
    _assert_rejected(_finding(evidence_tier="judged", query_match_id="q1"))
    _assert_rejected(_finding(evidence_tier="observed", query_match_id=None))
    _assert_rejected(_finding(evidence_tier="inferred", trace_path=[]))
    _assert_rejected(_finding(evidence_tier="observed", trace_path=["a"]))


def test_unknown_tier_value_is_rejected() -> None:
    """A tier outside `EvidenceTier` matches no branch. Canonically
    `evidence_tier` is a free `string`, so this is a real narrowing."""
    _assert_rejected(_finding(evidence_tier="fabricated"))
    _assert_rejected(_finding(evidence_tier="OBSERVED", query_match_id="q1"))


# --------------------------------------------------------------------------
# trace_path — must match the PRODUCTION proof rule, not merely "non-empty".
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [[], [""], [42], [None], ["ok", ""]])
def test_trace_path_rejects_empty_and_non_string_elements(bad: list[Any]) -> None:
    """Mirrors `policy/findings.py::_trace_path_is_valid`, which rejects `[]`,
    `[""]`, `[42]`, `[None]`. Encoding it here makes those shapes
    unrepresentable at the wire instead of rejected after generation."""
    _assert_rejected(_finding(evidence_tier="inferred", query_match_id=None, trace_path=bad))


def test_trace_path_accepts_the_shapes_the_production_rule_accepts() -> None:
    from outrider.policy.findings import _trace_path_is_valid

    for good in (["a"], ["a", "b"], ["pkg.mod:fn"]):
        assert _trace_path_is_valid(good)
        _validate(_finding(evidence_tier="inferred", query_match_id=None, trace_path=good))


def test_trace_candidates_required_array_allows_empty() -> None:
    """`[]` is the production-CORRECT answer for a finding standing on this
    file's evidence alone, so requiring the key must not force invention."""
    for tier, proof in (
        ("observed", {"query_match_id": "q1"}),
        ("inferred", {"trace_path": ["a"]}),
        ("judged", {}),
    ):
        _validate(_finding(evidence_tier=tier, trace_candidates=[], **proof))


def test_every_property_is_required_so_omission_is_rejected() -> None:
    """Strict mode requires all fields present; optionality is expressed by a
    null variant, not by omission."""
    incomplete = _finding()
    del incomplete["trace_candidates"]
    _assert_rejected(incomplete)


def test_additional_properties_are_rejected() -> None:
    """A model-invented field (e.g. a self-reported `severity` or `confidence`)
    cannot ride along — `severity-set-by-policy` /
    `confidence-is-computed-not-assigned` stay true at the wire."""
    _assert_rejected(_finding(severity="critical"))
    _assert_rejected(_finding(confidence=0.9))


def test_root_narrowing_carries_canonical_additions_through() -> None:
    """The root is a narrowed COPY of canonical, not a rebuilt literal.

    This is a mutation test: it feeds a canonical shaped like a plausible FUTURE
    version of `ANALYZE_RESPONSE_SCHEMA` — a `description` annotation on `findings`
    plus a new root-level sibling property — and requires both to survive the
    derivation. Against the previous hand-authored root literal both were dropped
    silently, so the probe would have measured a schema production never sends.
    Annotations are API-enforced generative guidance (FUP-169), so dropping one at
    the root is the same class of defect as reordering properties.
    """
    canonical = copy.deepcopy(ANALYZE_RESPONSE_SCHEMA)
    canonical["properties"]["findings"]["description"] = "Findings, most severe first."
    canonical["properties"]["analysis_notes"] = {"type": "string", "description": "Scratch."}

    derived = derive_strict_analyze_schema(canonical)

    assert derived["properties"]["findings"]["description"] == "Findings, most severe first."
    assert derived["properties"]["analysis_notes"] == {
        "type": "string",
        "description": "Scratch.",
    }
    # Order preserved, and strict mode still satisfied for the new sibling.
    assert list(derived["properties"]) == ["findings", "analysis_notes"]
    assert derived["required"] == ["findings", "analysis_notes"]
    assert derived["additionalProperties"] is False
    # The substitution still happened: items is the tier-discriminated union.
    assert list(derived["properties"]["findings"]["items"]) == ["anyOf"]


def test_derivation_does_not_mutate_the_canonical_schema() -> None:
    """Narrowing a copy means canonical is untouched — it is a module-level Final
    dict shared with production, so an in-place edit here would corrupt the real
    analyze request for the rest of the process."""
    before = json.dumps(ANALYZE_RESPONSE_SCHEMA, sort_keys=True)
    derive_strict_analyze_schema()
    assert json.dumps(ANALYZE_RESPONSE_SCHEMA, sort_keys=True) == before
