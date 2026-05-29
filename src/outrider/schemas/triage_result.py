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
(closes FUP-018). JSON round-trip is preserved by a paired
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
    # See DECISIONS.md#028-per-review-policy-version-snapshot-anchor-on-triageresult.
    # Snapshot anchor for synthesize's H-1 forge defense. Triage runs
    # FIRST in the canonical graph; the live `ACTIVE_POLICY_VERSION`
    # is captured here so a single review's findings + summary share
    # one anchor regardless of mid-deploy bumps. The schema-level
    # `pattern=BARE_SEMVER_PATTERN` rejects shape garbage, but admits
    # ANY valid semver — an LLM emitting an explicit
    # `{"policy_version": "0.0.0", ...}` in its triage JSON would
    # survive Pydantic. The triage-node gate `_enforce_triage_policy`
    # closes that producer path: Rule (d) raises
    # `TriagePolicyViolationError` if the post-validation value does
    # not equal the live `ACTIVE_POLICY_VERSION`. After the gate runs
    # the anchor is non-LLM-reachable. Synthesize compares each
    # finding's `policy_version` against this captured snapshot and
    # mirrors it onto `SynthesizeCompletedEvent.policy_version` so the
    # audit row records the snapshot under which findings were
    # classified. `default_factory` captures live at TriageResult
    # construction time; producers MAY pass an explicit value (e.g.,
    # replay paths that rehydrate from a historical audit row), in
    # which case the field's pattern is the only floor — the
    # triage-node gate runs only on the live-review path.
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
