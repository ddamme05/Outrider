# Triage-output cross-boundary models per docs/spec.md §7.2 (RiskLevel: Amended 2026-05-08)
"""Triage envelope: ReviewTier / RiskLevel / TriageResult.

These models are the typed output contract of the triage node (separate spec).
Triage runs a fast Haiku pass over PRContext, classifies each changed file
into a ReviewTier, classifies the overall PR into a RiskLevel, and identifies
which ReviewDimensions the deeper-reviewed files should be analyzed under.
The deterministic floor — that triage suggestions don't bypass review for
high-risk PRs — is enforced upstream of the model output, not by this schema.

ReviewTier values are lowercase serialized per project convention (matches
PerFindingOutcome, EvidenceTier, FindingSeverity, ReviewDimension). DEEP /
STANDARD / SKIM / SKIP are the four tiers per spec §4.1.2 + §6.10; SKIP
exists in the enum because §6.10 size-cap classification (>1000 lines OR
>30 files) maps directly to the same tier vocabulary, even though the LLM
itself is not the producer of SKIP under §4.1.2.

RiskLevel was added 2026-05-08 to close the canonical-shape gap (referenced
by TriageResult.overall_risk and ReviewReport.overall_risk in spec §7.2 but
never defined as a class). Same ladder shape as FindingSeverity minus INFO,
since "informational PR-level risk" has no operational meaning. See
specs/2026-05-08-schema-foundation.md and the spec.md §7.2 amendment for the
rationale; RiskLevel measures the PR as a whole, FindingSeverity measures
one finding within it.

TriageResult is frozen=True: it round-trips through LangGraph state JSON on
every checkpoint. The model's output is final at construction; downstream
nodes (analyze, trace, synthesize, publish) consume the value, never mutate
it. `relevant_dimensions` is `tuple[ReviewDimension, ...]` not
`list[ReviewDimension]` — same hitl/audit precedent as PRContext.changed_files
(frozen=True is faux-immutable over in-place container mutation; tuple
delivers true immutability). `file_tiers` is `Mapping[str, ReviewTier]` with
a `field_validator` that wraps the dict input in `MappingProxyType` so post-
construction mutation (`triage.file_tiers["x"] = ...`) raises `TypeError`
— closes FUP-018 in Round 24. JSON round-trip is preserved by a paired
`field_serializer` that dumps the MappingProxyType as a regular dict (via
`dict(value)`); StrEnum values serialize via Pydantic's default StrEnum
handling. Spec.md §7.2 was amended same-day (2026-05-08) to widen the field
type from `dict` to `Mapping` (the abstract supertype admits both `dict`
input and `MappingProxyType` runtime — no contradiction with #020 or §7.2's
intent; just a precision fix that lets the runtime type carry the
immutability the carrier-level annotation can't enforce alone).
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from outrider.schemas.review_finding import ReviewDimension


class ReviewTier(StrEnum):
    """Per-file review tier produced by the triage node, per spec §4.1.2.

    DEEP / STANDARD / SKIM are the LLM-produced classifications; SKIP is
    populated by the §6.10 size-cap policy gate (separate spec) when a PR
    exceeds the agentic-review thresholds (>1000 changed lines OR >30
    files — both PR-level metrics; §6.10 has no per-file size policy).
    Lowercase serialized values match project enum convention.
    """

    DEEP = "deep"
    STANDARD = "standard"
    SKIM = "skim"
    SKIP = "skip"


class RiskLevel(StrEnum):
    """PR-level risk classification produced by the triage node, per spec §7.2.

    Same ladder shape as FindingSeverity minus INFO (no operational meaning
    at PR-level). Distinct from FindingSeverity in scope: RiskLevel measures
    the PR as a whole; FindingSeverity measures one finding within it.

    Added 2026-05-08 to close the canonical-shape gap. See spec.md §7.2
    amendment + specs/2026-05-08-schema-foundation.md for rationale.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriageResult(BaseModel):
    """Output of the triage node, per spec §7.2.

    Frozen: see module docstring. All four fields are required; the LLM's
    structured output produces the values, the schema enforces the typed
    shape. relevant_dimensions enumerates which review dimensions apply to
    this PR — pure CSS changes don't get a security review, migrations
    don't get a style review (per spec §4.1.2 cost-control rationale).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_tiers: Mapping[str, ReviewTier]
    overall_risk: RiskLevel
    relevant_dimensions: tuple[ReviewDimension, ...]
    reasoning: str = Field(max_length=500)

    @field_validator("file_tiers", mode="after")
    @classmethod
    def _freeze_file_tiers(cls, value: Mapping[str, ReviewTier]) -> Mapping[str, ReviewTier]:
        """Wrap file_tiers in MappingProxyType so post-construction mutation
        (`triage.file_tiers["x"] = ...`) raises TypeError. Closes FUP-018.

        `dict(value)` first to copy the input — without the copy, the proxy
        would alias the caller's dict and mutations there would leak through
        the proxy (defeats the immutability gate)."""
        return MappingProxyType(dict(value))

    @field_serializer("file_tiers")
    def _serialize_file_tiers(self, value: Mapping[str, ReviewTier]) -> dict[str, ReviewTier]:
        """Convert MappingProxyType to dict for JSON serialization. Pydantic's
        default StrEnum handler dumps each ReviewTier as its string value
        (e.g., "deep"); the dict shape round-trips cleanly back through
        `model_validate_json` which re-applies `_freeze_file_tiers` to the
        rehydrated dict."""
        return dict(value)


__all__ = [
    "ReviewTier",
    "RiskLevel",
    "TriageResult",
]
