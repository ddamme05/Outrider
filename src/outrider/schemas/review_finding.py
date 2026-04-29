# Cross-boundary ReviewFinding per docs/spec.md §7.3 + docs/trust-boundaries.md §1
"""ReviewFinding + ReviewDimension + PublishDestination cross-boundary models.

ReviewFinding is the Pydantic carrier for every finding the agent produces.
It registers `enforce_proof_boundary` from `policy/findings.py` as a
model_validator so OBSERVED-without-query_match_id and INFERRED-without-
trace_path raise at construction time per `evidence-tier-schema-enforced`.
The `confidence` field is a `@computed_field` deriving deterministically
from `evidence_tier` per `confidence-is-computed-not-assigned` (OBSERVED=0.9,
INFERRED=0.75, JUDGED=0.5 per spec §7.3).

ReviewFinding is **NOT frozen** because it has a multi-stage lifecycle:
the analyze node constructs it with later-set fields at None;
`coordinates.tree_sitter_to_github` (separate spec) sets `publish_destination`
after computing the GitHub-comment-location translation; the HITL flow
may later set the override fields when a reviewer issues a SEVERITY_OVERRIDE
decision. Mutability lets each stage write its fields directly without
requiring `model_copy(update=...)` boilerplate. The audit-replay guarantee
for findings rides on `FindingEvent` (which IS frozen and append-only via
the audit_events trigger), not on `ReviewFinding`'s in-memory mutability.

PublishDestination uses uppercase Python member names with lowercase
serialized string values per spec §4.1.7, same convention as
EvidenceTier / FindingType / FindingSeverity.
"""

from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from outrider.policy import (
    EvidenceTier,
    FindingSeverity,
    FindingType,
    enforce_proof_boundary,
)


class ReviewDimension(StrEnum):
    """The five review-dimension axes per spec §7.3."""

    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TEST_COVERAGE = "test_coverage"
    BEST_PRACTICES = "best_practices"


class PublishDestination(StrEnum):
    """Where a finding lands when published, per spec §4.1.7.

    Set by `coordinates.tree_sitter_to_github` after computing the
    GitHub-comment-location translation; the analyze node leaves the
    field None at construction time. Backs `publish-routes-through-coordinates`.
    """

    INLINE_COMMENT = "inline_comment"
    REVIEW_BODY = "review_body"
    DASHBOARD_ONLY = "dashboard_only"


_CONFIDENCE_BY_TIER: dict[EvidenceTier, float] = {
    EvidenceTier.OBSERVED: 0.9,
    EvidenceTier.INFERRED: 0.75,
    EvidenceTier.JUDGED: 0.5,
}


class ReviewFinding(BaseModel):
    """One finding produced by the agent's review of a PR.

    Construction-time validators enforce the proof boundary
    (`enforce_proof_boundary` from policy/findings) and the line
    constraint (`line_end >= line_start`). Confidence is a computed
    field, not a settable attribute.
    """

    # Not frozen: multi-stage lifecycle. See module docstring.
    # validate_assignment=True: lifecycle writes (publish_destination,
    # override fields) re-run model_validators + Field constraints + enum
    # typing on every assignment, so post-construction mutations cannot
    # bypass the proof boundary, the line constraint, or the typed-enum
    # gates. Without this, `finding.severity = "garbage"` would silently
    # admit; with it, the assignment raises.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    finding_id: UUID = Field(default_factory=uuid4)
    review_id: UUID
    installation_id: int
    policy_version: str
    finding_type: FindingType
    dimension: ReviewDimension
    severity: FindingSeverity
    evidence_tier: EvidenceTier
    file_path: str
    line_start: int = Field(ge=1)
    line_end: int
    title: str = Field(max_length=120)
    description: str = Field(max_length=1000)
    evidence: str
    suggested_fix: str | None = None
    query_match_id: str | None = None
    trace_path: list[str] | None = None
    # Lifecycle / HITL-set fields (None at analyze-time):
    original_severity: FindingSeverity | None = None
    override_reason: str | None = None
    overrider_id: UUID | None = None
    publish_destination: PublishDestination | None = None
    # Dedup (analyze sets at construction time per spec §8.5):
    content_hash: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Deterministic mapping from evidence_tier per spec §7.3.

        OBSERVED=0.9, INFERRED=0.75, JUDGED=0.5. Read-only at the
        descriptor level — assigning to `.confidence` raises
        AttributeError regardless of model frozen-ness.
        """
        return _CONFIDENCE_BY_TIER[self.evidence_tier]

    @model_validator(mode="after")
    def _enforce_proof_boundary(self) -> Self:
        """Wire policy/findings.enforce_proof_boundary into Pydantic validation."""
        enforce_proof_boundary(
            evidence_tier=self.evidence_tier,
            query_match_id=self.query_match_id,
            trace_path=self.trace_path,
        )
        return self

    @model_validator(mode="after")
    def _enforce_line_constraint(self) -> Self:
        """line_end must be >= line_start (1-indexed per coordinates/)."""
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start "
                f"({self.line_start}); lines are 1-indexed per coordinates/"
            )
        return self


__all__ = [
    "PublishDestination",
    "ReviewDimension",
    "ReviewFinding",
]
