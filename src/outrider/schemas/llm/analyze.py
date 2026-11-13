"""Define proposed analysis findings and their accepted response shape.

Raw proposals retain values needed to explain parser rejections. Accepted
models constrain fields; the parser validates evidence and source locations.
Line ranges are 1-indexed and unchanged between raw and accepted proposals.
proposal_hash includes those raw locations; coordinate translation stays in
coordinates/. Import paths follow the shared import rules.

See DECISIONS.md#022, #024, #025 and FUP-126.
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
    succeeds (the syntax dispatcher per `DECISIONS.md#024`, Amended
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
        enforces the syntax-dispatched canonical invariant the layer's name
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
# FUP-096). Hand-trim the provider's supported subset: omit unsupported length,
# numeric and array bounds, and forbid extra properties on every object.
# Pydantic still enforces the omitted constraints at parse time; constrained
# decoding guarantees shape, not semantic admission. A drift test checks names,
# required sets and types against AnalyzeResponseRaw.model_json_schema().
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

# Preserve property order: constrained decoding follows the schema's generation
# order (FUP-169). Classification and title must precede description/evidence,
# so canonicalize_for_hash's key sorting would change model behavior.
# Hash these exact wire bytes; order changes must rotate the format digest.
ANALYZE_RESPONSE_SCHEMA_JSON: Final[str] = json.dumps(
    ANALYZE_RESPONSE_SCHEMA, separators=(",", ":"), ensure_ascii=False, allow_nan=False
)
ANALYZE_RESPONSE_FORMAT_DIGEST: Final[str] = hashlib.sha256(
    ANALYZE_RESPONSE_SCHEMA_JSON.encode("utf-8")
).hexdigest()
