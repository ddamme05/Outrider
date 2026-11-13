"""Build the experiment's strict schema from the production analyze schema.

Keep the original fields, descriptions and property order (FUP-169), then add
branches that require the proof fields allowed for each evidence tier. A response
can fit the schema and still cite false proof; production admission must check
those claims separately.

Null-only branches are part of this experiment. The probe, rather than local
schema construction, establishes whether the host accepts them (DECISIONS.md#056).
tests/unit/test_arc2_strict_schema.py checks the local mapping and production
field agreement. This module is not a production schema export.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Final

from outrider.policy import EvidenceTier
from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA

__all__ = [
    "TIER_PROOF_SHAPES",
    "derive_strict_analyze_schema",
    "schema_digest",
    "strict_schema_json",
]

# The two proof properties whose shape the tier discriminates. Everything else in
# the finding object is tier-invariant and carried through from canonical.
_PROOF_PROPERTIES: Final[tuple[str, ...]] = ("query_match_id", "trace_path")

_NULL_ONLY: Final[dict[str, Any]] = {"type": "null"}

# Probe-local semantic knowledge — the ONLY hand-authored part of the schema.
#
# `trace_path` encodes `policy/findings.py::_trace_path_is_valid`, which requires a
# non-empty sequence of NON-EMPTY strings (it rejects `[]`, `[""]`, `[42]`, `[None]`).
# `minItems: 1` + an item `pattern` of `.+` makes those shapes unrepresentable at the
# wire rather than leaving them to be rejected after generation.
#
# Precisely: this is AT LEAST AS STRICT as the production rule, not identical to it.
# `.` does not match a newline, so a newline-only element (`["\n"]`) is accepted by
# `_trace_path_is_valid` and rejected here. The divergence is in the safe direction
# (no real scope-unit identifier is a bare newline) and is left in place rather than
# widened to `[\s\S]` because the spec's per-tier matrix pins `pattern: ".+"`.
#
# `query_match_id` is a plain non-null `string` on OBSERVED (per the spec's per-tier
# matrix). Emptiness is deliberately NOT constrained here: an empty id is a
# registry-membership question, and registry membership is the parser's job, not the
# schema's. Constraining it would blur the shape/authenticity split this arc rests on.
TIER_PROOF_SHAPES: Final[dict[EvidenceTier, dict[str, dict[str, Any]]]] = {
    EvidenceTier.OBSERVED: {
        "query_match_id": {"type": "string"},
        "trace_path": dict(_NULL_ONLY),
    },
    EvidenceTier.INFERRED: {
        "query_match_id": dict(_NULL_ONLY),
        "trace_path": {
            "type": "array",
            "items": {"type": "string", "pattern": ".+"},
            "minItems": 1,
        },
    },
    EvidenceTier.JUDGED: {
        "query_match_id": dict(_NULL_ONLY),
        "trace_path": dict(_NULL_ONLY),
    },
}


def _canonical_finding_object(canonical: dict[str, Any]) -> dict[str, Any]:
    """The canonical per-finding object schema (`findings.items`)."""
    try:
        items = canonical["properties"]["findings"]["items"]
    except (KeyError, TypeError) as exc:  # pragma: no cover - canonical drift guard
        msg = (
            "ANALYZE_RESPONSE_SCHEMA no longer exposes properties.findings.items; "
            "the strict derivation cannot proceed against an unknown canonical shape."
        )
        raise ValueError(msg) from exc
    if not isinstance(items, dict) or "properties" not in items:
        msg = "canonical findings.items is not an object schema with properties"
        raise ValueError(msg)
    return items


def _strictify_object(node: dict[str, Any]) -> dict[str, Any]:
    """Recursively require every property and forbid additional ones.

    Applied to the tier-invariant sub-objects carried over from canonical (today
    only `trace_candidates.items`, which canonical already ships all-required).

    Descends `properties`, dict-form `items`, and the composition keywords, so a
    future canonical nested object does not silently arrive non-strict. It is NOT
    a fully general JSON Schema walk (no `$defs`/`$ref`, no tuple-validation
    list-form `items` — neither appears in canonical). `test_strict_subset_shape`
    is the backstop: it walks every node of the derived schema, so anything this
    misses fails loudly at test time rather than silently at the wire.
    """
    out = copy.deepcopy(node)
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        props: dict[str, Any] = out["properties"]
        out["properties"] = {name: _strictify_object(sub) for name, sub in props.items()}
        # Order-preserving: `required` follows the property insertion order.
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = _strictify_object(out["items"])
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = out.get(keyword)
        if isinstance(branches, list):
            out[keyword] = [_strictify_object(b) if isinstance(b, dict) else b for b in branches]
    return out


def _tier_branch(tier: EvidenceTier, canonical_properties: dict[str, Any]) -> dict[str, Any]:
    """One `anyOf` branch: the canonical finding object with its tier pinned.

    `evidence_tier` becomes a single-member enum. Without that pin the branches
    would discriminate on proof-field SHAPE alone, so a payload declaring
    `evidence_tier="judged"` while carrying a `query_match_id` would legally match
    the OBSERVED branch and the discrimination would be illusory.
    """
    proof = TIER_PROOF_SHAPES[tier]
    properties: dict[str, Any] = {}
    for name, subschema in canonical_properties.items():
        if name == "evidence_tier":
            # Preserve any canonical annotation (e.g. `description`) while pinning
            # the value, so the branch stays a faithful narrowing of canonical.
            pinned = copy.deepcopy(subschema)
            pinned["type"] = "string"
            pinned["enum"] = [tier.value]
            properties[name] = pinned
        elif name in proof:
            # MERGE onto the canonical subschema rather than replacing it, so a
            # canonical `description` survives the narrowing. Replacing silently
            # dropped annotations — and annotations are API-enforced generative
            # guidance (FUP-169), not decoration, so losing one on `trace_path`
            # is the same "probe a worse request than production" failure the
            # order rule guards against.
            narrowed = copy.deepcopy(subschema)
            narrowed.pop("anyOf", None)  # canonical's [T, null] union is being replaced
            narrowed.pop("type", None)
            narrowed.pop("items", None)
            narrowed.update(copy.deepcopy(proof[name]))
            properties[name] = narrowed
        else:
            properties[name] = _strictify_object(subschema)
    return {
        "type": "object",
        "properties": properties,
        # Every property required, in canonical order (strict-mode requirement).
        "required": list(properties),
        "additionalProperties": False,
    }


def derive_strict_analyze_schema(
    canonical: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the tier-discriminated strict schema from the canonical analyze schema.

    The returned schema is a NARROWING of `ANALYZE_RESPONSE_SCHEMA`: same property
    names, same types, same annotations, same order — with `evidence_tier` pinned
    per branch, the two proof properties shaped per tier, and every property
    required (which canonical is not, since canonical leaves `query_match_id`,
    `trace_path`, and `trace_candidates` optional).

    "Narrowing" is meant structurally, not aspirationally: the root is a strictified
    DEEP COPY of `canonical`, with only `findings.items` substituted. Anything
    canonical carries at the root — a new sibling property, a `description` on
    `findings` — flows through automatically.
    """
    source = ANALYZE_RESPONSE_SCHEMA if canonical is None else canonical
    finding_object = _canonical_finding_object(source)
    canonical_properties: dict[str, Any] = finding_object["properties"]

    missing = [p for p in _PROOF_PROPERTIES if p not in canonical_properties]
    if missing:
        msg = (
            f"canonical finding object is missing proof properties {missing!r}; "
            "the tier->proof mapping is stale relative to ANALYZE_RESPONSE_SCHEMA"
        )
        raise ValueError(msg)
    if "evidence_tier" not in canonical_properties:
        msg = "canonical finding object has no evidence_tier; branches cannot be discriminated"
        raise ValueError(msg)

    # Branch order follows EvidenceTier declaration order so the derivation is
    # deterministic and the digest is stable across runs.
    branches = [_tier_branch(tier, canonical_properties) for tier in EvidenceTier]

    # NARROW A COPY of the canonical root — do not rebuild one. An earlier version
    # returned a hand-authored root literal that happened to equal canonical's, so
    # the docstring's "same property names, same annotations, same order" claim held
    # by coincidence: the moment canonical grew a root property or an annotation on
    # `findings`, the probe would have silently tested a schema production does not
    # send, which is the root-level form of the FUP-169 failure this module exists to
    # avoid. Strictify FIRST, then substitute the already-strict branches, so the
    # branches are not re-walked.
    root = _strictify_object(source)
    root["properties"]["findings"]["items"] = {"anyOf": branches}
    return root


def strict_schema_json(schema: dict[str, Any] | None = None) -> str:
    """Compact ORDER-PRESERVING serialization — same recipe as the canonical
    `ANALYZE_RESPONSE_SCHEMA_JSON` (deliberately not key-sorted, because emission
    order is part of the format's generative identity; see the module docstring)."""
    target = derive_strict_analyze_schema() if schema is None else schema
    return json.dumps(target, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def schema_digest(schema: dict[str, Any] | None = None) -> str:
    """sha256 over exactly the `strict_schema_json` bytes — the schema's identity
    in the manifest. A property-order change rotates it, which is correct."""
    return hashlib.sha256(strict_schema_json(schema).encode("utf-8")).hexdigest()
