"""Record proposed file priorities and the review dimensions to inspect.

The triage node applies live admission checks. Results are immutable and retain
their policy version; replay can read older versions without applying today's
policy retroactively. See DECISIONS.md#020.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from outrider.policy.severity import ACTIVE_POLICY_VERSION, BARE_SEMVER_PATTERN
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
    """Triage classification plus a captured policy-version anchor.

    The four classification fields are required; policy_version defaults to the
    active version. The live triage gate rejects a different model-supplied
    version. Frozen fields, a copied read-only file map, and canonical dimension
    ordering keep checkpoint representations stable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_tiers: Mapping[str, ReviewTier]
    overall_risk: RiskLevel
    relevant_dimensions: tuple[ReviewDimension, ...]
    reasoning: str = Field(max_length=500)
    # See DECISIONS.md#028-per-review-policy-version-snapshot-anchor-on-triageresult.
    # Capture the active version at construction so findings and summary share
    # one snapshot across deploys. Synthesize checks findings against this anchor
    # and records it on SynthesizeCompletedEvent. The pattern checks shape only;
    # _enforce_triage_policy rejects a different model-supplied version on the
    # live-review path. Replay may supply an explicit historical semver.
    policy_version: str = Field(
        default_factory=lambda: ACTIVE_POLICY_VERSION,
        pattern=BARE_SEMVER_PATTERN,
    )

    @field_validator("file_tiers", mode="after")
    @classmethod
    def _freeze_file_tiers(cls, value: Mapping[str, ReviewTier]) -> Mapping[str, ReviewTier]:
        """Wrap file_tiers in MappingProxyType so post-construction mutation
        (`triage.file_tiers["x"] = ...`) raises TypeError. Closes FUP-018.

        `dict(value)` first to copy the input — without the copy, the proxy
        would alias the caller's dict and mutations there would leak through
        the proxy (defeats the immutability gate)."""
        return MappingProxyType(dict(value))

    @field_validator("relevant_dimensions", mode="after")
    @classmethod
    def _canonicalize_relevant_dimensions(
        cls, value: tuple[ReviewDimension, ...]
    ) -> tuple[ReviewDimension, ...]:
        """Reject duplicates AND canonical-sort the dimension tuple so
        set-semantic equality becomes structural equality on the wire.

        Two TriageResults whose dimensions agree as sets but differ in
        order would otherwise serialize to distinct JSON payloads,
        breaking checkpoint-comparison and audit content-hashing. A
        duplicate dimension (`["security", "security"]`) is a producer
        bug — the LLM output should never repeat a dimension — and
        silently dedup'ing would mask the bug. Fail loud matches the
        `HITLRequest._enforce_finding_partition` sibling precedent
        (rejects duplicate `finding_id`s rather than dedup'ing).
        """
        if len(value) != len(set(value)):
            raise ValueError(
                f"TriageResult.relevant_dimensions contains duplicate "
                f"dimensions: {sorted(d.value for d in value)!r}; the "
                f"field is set-semantic and duplicates indicate a "
                f"producer bug (LLM repeating an output element)."
            )
        return tuple(sorted(value))

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
