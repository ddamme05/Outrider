# See specs/2026-05-19-analyze-foundation.md §7.
"""Raw + admitted analyze proposal schemas.

Two-layer shape so the sister analyze-implementation spec's parser can
record rejection events for proposals with invalid enum values. The
raw layer admits bounded invalid strings (e.g., a model-returned
`finding_type` outside the `FindingType` enum) long enough for the
parser to emit `FindingProposalRejectedEvent`; the admitted layer is
what passes to `ReviewFinding` construction.

Without the split, a model returning a `finding_type` outside the
enum would fail Pydantic at `AnalyzeResponseRaw.model_validate(...)`
BEFORE the parser could emit the rejection event — losing the audit
signal that the model produced an invalid type.

**Line-preservation invariant** (per `DECISIONS.md#022`, span-key amended
2026-06-01 / FUP-126). Both layers are LINE-based: `line_start`/`line_end`
are 1-indexed file lines — the frame the model is actually shown
(scope-unit header + diff `@@` markers). The admitted layer's
`line_start`/`line_end` MUST equal the raw layer's byte-for-byte (no
normalization between layers), the same identity-preservation the byte
`span` field carried before this amendment: `proposal_hash` is
canonicalized from the RAW line values, so a normalized admitted line
range would describe different lines from the same hash. Byte `Span` is
NOT a proposal-schema field anymore — `ReviewFinding` is line-based, and
the only byte translation (`coordinates.line_range_to_span`) is a
parser-internal, un-clipped step for the degraded file-bounds check.
`proposal_hash` is also excluded from `finding_content_hash` per
`DECISIONS.md#025` point 3, so the recipe change is replay-safe. Tests
pin the line-preservation invariant.

`import_string` is the other cross-layer field and it IS deliberately
normalized (raw `import_string_raw` → admitted `import_string` post-
`coordinates.is_valid_trace_import_string`): line numbers are
identity-preserved because the hash folds them; `import_string` is
normalized because downstream consumers (trace node resolving the import)
need the validated canonical form. Per `DECISIONS.md#024` (Amended
2026-07-03) trace candidates carry one of two syntactic forms — dotted
Python import string or JS/TS relative specifier — validated by the
shared two-form dispatcher; the prior `candidate_path` framing was
renamed in lockstep.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outrider.coordinates import is_valid_trace_import_string
from outrider.policy import (  # noqa: TC001 — Pydantic field types
    EvidenceTier,
    FindingType,
)

# ---------------------------------------------------------------------------
# Raw layer — what the LLM provider returns. Bounded but non-canonical.
# ---------------------------------------------------------------------------


class TraceCandidateProposalRaw(BaseModel):
    """Model-proposed trace candidate as it arrives in the raw response.

    Parser admits/rejects these alongside their parent finding proposals.

    Raw and admitted layers must be materially distinct (not just
    default markers), so a downstream variable typed as the raw layer
    cannot be silently passed where the admitted layer is expected.
    The distinction is in the import field: raw layer carries
    `import_string_raw` (the model's claimed import string, unvalidated
    bounded string); admitted layer carries `import_string`
    (already passed through `coordinates.is_valid_trace_import_string` —
    NFC-normalized, per-form shape-checked: dotted module form or JS/TS
    relative-specifier form per `DECISIONS.md#024`, Amended 2026-07-03).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    import_string_raw: Annotated[str, Field(max_length=1024)]
    reason: Annotated[str, Field(max_length=500)]


class AnalyzeFindingProposalRaw(BaseModel):
    """Model-proposed finding as it arrives in the raw response.

    `finding_type` and `evidence_tier` are bounded `str` (NOT the
    canonical enums) so a model that returns an off-list value
    survives Pydantic construction long enough for the parser to emit
    a `FindingProposalRejectedEvent`. Admitted-layer construction is
    what enforces enum membership.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_type: Annotated[str, Field(max_length=128)]
    evidence_tier: Annotated[str, Field(max_length=32)]
    query_match_id: Annotated[str, Field(max_length=256)] | None = None
    trace_path: (
        Annotated[
            tuple[Annotated[str, Field(max_length=256, min_length=1)], ...],
            Field(max_length=32),
        ]
        | None
    ) = None
    title: Annotated[str, Field(max_length=120)]
    description: Annotated[str, Field(max_length=1000)]
    evidence: Annotated[str, Field(max_length=2000)]
    # 1-indexed, inclusive source line range — the frame the model is shown
    # (scope-unit header + diff `@@` markers). This IS the proposal's location:
    # the admitted layer is line-based too, with no byte `Span` field. The only
    # byte translation (`coordinates.line_range_to_span`) is parser-internal and
    # used solely for the degraded file-bounds check (FUP-126, `DECISIONS.md#022`
    # span-key amendment). Strictness matches the byte `Span` this replaced: a
    # malformed range fails here → `raw_response_unparseable`.
    line_start: Annotated[int, Field(ge=1)]
    line_end: Annotated[int, Field(ge=1)]
    trace_candidates: tuple[TraceCandidateProposalRaw, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def _line_end_not_before_start(self) -> AnalyzeFindingProposalRaw:
        if self.line_end < self.line_start:
            msg = f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            raise ValueError(msg)
        return self


class AnalyzeResponseRaw(BaseModel):
    """Top-level wrapper around the raw findings array.

    Per-call output ceiling at 50 findings — defends against a runaway
    model emission that would otherwise saturate the audit stream.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    findings: tuple[AnalyzeFindingProposalRaw, ...] = Field(max_length=50)


# ---------------------------------------------------------------------------
# Admitted layer — post-parser admission.
# ---------------------------------------------------------------------------


class TraceCandidateProposal(BaseModel):
    """Admitted trace candidate.

    Constructed by the sister analyze-implementation spec's parser only
    AFTER `coordinates.is_valid_trace_import_string(raw.import_string_raw)`
    succeeds (the two-form dispatcher per `DECISIONS.md#024`, Amended
    2026-07-03).

    Distinct field name (`import_string` vs raw's `import_string_raw`)
    means a `TraceCandidateProposal(**raw.model_dump())` swap fails
    Pydantic construction under `extra="forbid"` — the raw layer's
    `import_string_raw` is not a valid admitted field. Structural
    distinction is the pit-of-success fix; provenance markers
    (Literal["admitted"]) are belt only, validation-derived field
    shape is the load-bearing prevention.

    The `import_string` field validator below enforces the documented
    "already passed is_valid_trace_import_string" invariant at the
    schema layer — without it, the admitted-vs-raw distinction rested
    on parser flow alone; with it, the schema is the durable floor and
    any future producer / replay reconstruction validates against the
    same rule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    import_string: Annotated[str, Field(max_length=1024)]
    reason: Annotated[str, Field(max_length=500)]

    @field_validator("import_string")
    @classmethod
    def _enforce_canonical_import_string(cls, value: str) -> str:
        """Re-run `is_valid_trace_import_string` so the admitted layer
        enforces the two-form canonical invariant the layer's name
        promises (`DECISIONS.md#024`, Amended 2026-07-03).
        The raw layer (`TraceCandidateProposalRaw.import_string_raw`)
        stays loose — its whole purpose is to admit unvalidated model
        output long enough for the parser to emit a rejection event.
        """
        return is_valid_trace_import_string(value)


class AnalyzeFindingProposal(BaseModel):
    """Admitted analyze finding proposal — enum-constrained.

    V1 does not downgrade evidence tiers; failed admission produces
    a rejection event, not a downgraded admitted finding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_type: FindingType
    evidence_tier: EvidenceTier
    query_match_id: Annotated[str, Field(max_length=256)] | None = None
    trace_path: (
        Annotated[
            tuple[Annotated[str, Field(max_length=256, min_length=1)], ...],
            Field(max_length=32),
        ]
        | None
    ) = None
    title: Annotated[str, Field(max_length=120)]
    description: Annotated[str, Field(max_length=1000)]
    evidence: Annotated[str, Field(max_length=2000)]
    # Line-based, identity-preserved from the raw layer (no normalization) — see
    # the module-docstring line-preservation invariant. The byte span is a
    # parser-internal coordinate translation, not a proposal-schema field.
    line_start: Annotated[int, Field(ge=1)]
    line_end: Annotated[int, Field(ge=1)]
    trace_candidates: tuple[TraceCandidateProposal, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def _line_end_not_before_start(self) -> AnalyzeFindingProposal:
        if self.line_end < self.line_start:
            msg = f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Constrained-decoding schema (specs/2026-06-12-constrained-decoding.md,
# FUP-096). The PINNED, hand-trimmed JSON Schema the provider sends as
# `output_config.format` — NOT `AnalyzeResponseRaw.model_json_schema()`,
# because the API's supported subset excludes constructs Pydantic emits
# (`maxLength`, numeric bounds, `maxItems`>1) and requires
# `additionalProperties: false` on every object. The stripped constraints
# still enforce at parse time via the Pydantic models above — constrained
# decoding guarantees SYNTAX and shape; the raw/admitted layers stay the
# semantic gate. A drift test asserts structural agreement (property names,
# required sets, types) with the generated model schema, so a model change
# forces this constant to be revisited; byte equality is deliberately NOT
# the assertion.
# ---------------------------------------------------------------------------

ANALYZE_RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding_type": {"type": "string"},
                    "evidence_tier": {"type": "string"},
                    "query_match_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "trace_path": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ]
                    },
                    "title": {"type": "string"},
                    # Schema-level `description` annotations are API-enforced
                    # guidance the constrained decoder reads (FUP-169: two
                    # consecutive live runs returned these fields EMPTY under
                    # the grammar while the same prompt's instructions were
                    # obeyed free-form — under constrained decoding the
                    # schema, not the prompt, is where field guidance must
                    # live). The grammar cannot express minLength, so "never
                    # empty" is instruction, not enforcement; the admitted
                    # layer deliberately does not reject empty prose.
                    "description": {
                        "type": "string",
                        "description": (
                            "Non-empty explanation of why this is a problem and "
                            "what to change, specific to the code under review. "
                            "Never empty. At most 1000 characters."
                        ),
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "Non-empty verbatim quote of the offending line(s) "
                            "from the code under review. Never empty. At most "
                            "2000 characters."
                        ),
                    },
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"},
                    "trace_candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "import_string_raw": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["import_string_raw", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "finding_type",
                    "evidence_tier",
                    "title",
                    "description",
                    "evidence",
                    "line_start",
                    "line_end",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}

# Compact ORDER-PRESERVING serialization — deliberately NOT
# `canonicalize_for_hash`, whose key sorting is generatively load-bearing
# under constrained decoding: the API emits object properties in the
# schema's defined order (structured-outputs doc, "Property ordering" —
# required first, in schema order), so a sorted schema forced
# `description`/`evidence` to be generated BEFORE `finding_type`/`title`
# existed in the model's own context, and three consecutive live runs
# returned them empty (FUP-169). The dict literal above is written in
# reasoning order — classify, name, explain, quote, locate — and these
# bytes preserve it on the wire. The digest is sha256 over exactly these
# bytes, so `LLMRequest.response_format_digest` (recomputed from the
# string) and this constant can never disagree; a property-order change
# rotates the digest, which is correct — emission order is part of the
# format's identity.
ANALYZE_RESPONSE_SCHEMA_JSON: Final[str] = json.dumps(
    ANALYZE_RESPONSE_SCHEMA, separators=(",", ":"), ensure_ascii=False, allow_nan=False
)
ANALYZE_RESPONSE_FORMAT_DIGEST: Final[str] = hashlib.sha256(
    ANALYZE_RESPONSE_SCHEMA_JSON.encode("utf-8")
).hexdigest()
