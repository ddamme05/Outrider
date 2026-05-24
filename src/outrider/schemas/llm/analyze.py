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

**Span byte-for-byte invariant.**
`AnalyzeFindingProposal.span` (admitted) MUST equal
`AnalyzeFindingProposalRaw.span` (raw) byte-for-byte. The parser MAY
reject a proposal whose span fails containment, but MUST NOT normalize/
clip/snap the span between raw and admitted layers. `proposal_hash`
(on `FindingProposalRejectedEvent`) is canonicalized from the RAW span
values; if the admitted span were normalized, downstream consumers of
the admitted `ReviewFinding` would describe different bytes from the
same hash, breaking replay reconstruction. Tests pin this invariant.

The byte-for-byte rule does NOT apply to `import_string` — that field
IS deliberately normalized between layers (raw has `import_string_raw`,
admitted has `import_string` post-`coordinates.is_valid_import_string`).
`span` is identity-preserved because rejection-event hashes depend on
it; `import_string` is normalized because downstream consumers
(trace node resolving the import) need the validated canonical form.
Per `DECISIONS.md#024` trace candidates are dotted Python import
strings; the prior `candidate_path` framing was renamed in lockstep.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from outrider.ast_facts.models import Span  # noqa: TC001 — Pydantic field type, runtime import
from outrider.coordinates import is_valid_import_string
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
    `import_string_raw` (the model's claimed dotted import string,
    unvalidated bounded string); admitted layer carries `import_string`
    (already passed through `coordinates.is_valid_import_string` —
    NFC-normalized, identifier-validity-checked, no path separators,
    no shell metacharacters, no Python keywords). Per `DECISIONS.md#024`
    trace candidates are dotted Python import strings, not file paths.
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
    span: Span
    trace_candidates: tuple[TraceCandidateProposalRaw, ...] = Field(default=(), max_length=20)


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
    AFTER `coordinates.is_valid_import_string(raw.import_string_raw)`
    succeeds.

    Distinct field name (`import_string` vs raw's `import_string_raw`)
    means a `TraceCandidateProposal(**raw.model_dump())` swap fails
    Pydantic construction under `extra="forbid"` — the raw layer's
    `import_string_raw` is not a valid admitted field. Structural
    distinction is the pit-of-success fix; provenance markers
    (Literal["admitted"]) are belt only, validation-derived field
    shape is the load-bearing prevention.

    The `import_string` field validator below enforces the documented
    "already passed is_valid_import_string" invariant at the schema
    layer — without it, the admitted-vs-raw distinction rested on
    parser flow alone; with it, the schema is the durable floor and
    any future producer / replay reconstruction validates against the
    same rule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    import_string: Annotated[str, Field(max_length=1024)]
    reason: Annotated[str, Field(max_length=500)]

    @field_validator("import_string")
    @classmethod
    def _enforce_canonical_import_string(cls, value: str) -> str:
        """Re-run `is_valid_import_string` so the admitted layer enforces
        the canonical-import-string invariant the layer's name promises.
        The raw layer (`TraceCandidateProposalRaw.import_string_raw`)
        stays loose — its whole purpose is to admit unvalidated model
        output long enough for the parser to emit a rejection event.
        """
        return is_valid_import_string(value)


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
    span: Span
    trace_candidates: tuple[TraceCandidateProposal, ...] = Field(default=(), max_length=20)
