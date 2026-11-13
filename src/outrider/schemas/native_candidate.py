"""Reference native review proposals and assessment attempts without storing their text.

Original claims and investigator prose remain in expiring model exchanges. These
models can enter checkpoints and audit events; neither contains raw model content.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID  # noqa: TC003 (Pydantic field type)

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outrider.coordinates import validate_diff_path
from outrider.policy import EvidenceTier, FindingSeverity, FindingType  # noqa: TC001
from outrider.policy.canonical import SHA256_HEX_PATTERN, compute_identity_hash
from outrider.policy.severity import ACTIVE_POLICY_VERSION, BARE_SEMVER_PATTERN, SEVERITY_POLICY

CandidateRejectionReason = Literal[
    "raw_proposal_invalid",
    "query_match_id_not_in_registry",
    "trace_path_not_admissible",
    "finding_type_not_in_enum",
    "evidence_tier_not_in_enum",
    "span_outside_scope_unit",
    "span_outside_file",
    "span_outside_degraded_context",
    "sql_injection_on_parameterized_call",
    "schema_construction_failed",
    "evidence_quote_invalid",
    "model_observed_not_supported",
]

HeadAssessment = Literal["supports", "refutes", "uncertain"]
ImpactAssessment = Literal["supports", "refutes", "uncertain", "not_claimed"]
RepairAssessment = Literal["supports", "refutes", "uncertain", "not_proposed"]
ChangeAssessment = Literal[
    "introduced_or_worsened", "fix_attempt", "unrelated_preexisting", "uncertain"
]


CandidateCheckContract = Literal[
    "head-and-change-v1", "head-change-and-impact-v1", "head-change-impact-and-repair-v1"
]


def combine_candidate_assessment(
    head: HeadAssessment,
    change: ChangeAssessment,
    impact: ImpactAssessment | None = None,
    repair: RepairAssessment | None = None,
) -> HeadAssessment:
    """Combine local truth, PR relevance, broader impact and proposed repair."""
    if head == "refutes" or change == "unrelated_preexisting":
        return "refutes"
    if (
        head == "supports"
        and change in {"introduced_or_worsened", "fix_attempt"}
        and impact in {None, "supports", "not_claimed"}
        and repair in {None, "supports", "not_proposed"}
    ):
        return "supports"
    return "uncertain"


def candidate_check_contract_for_version(version: str) -> CandidateCheckContract | None:
    """Resolve recorded formats explicitly; unknown versions cannot become legacy."""
    if version in {"native-candidate-check-v1", "native-candidate-check-v2"}:
        return None
    if version == "native-candidate-check-v3":
        return "head-and-change-v1"
    if version == "native-candidate-check-v4":
        return "head-change-and-impact-v1"
    if version == "native-candidate-check-v5":
        return "head-change-impact-and-repair-v1"
    raise ValueError("unknown native candidate check version")


def native_candidate_id(
    *,
    review_id: UUID,
    source_exchange_event_id: UUID,
    response_hash: str,
    proposal_index: int,
    file_path: str,
    pass_index: int,
    source_side: Literal["head", "base"] = "head",
) -> str:
    """Identify one occurrence; only BASE adds a side key to the legacy payload."""
    if source_side not in {"head", "base"}:
        raise ValueError("candidate source_side must be head or base")
    payload = {
        "recipe": "native-candidate-occurrence-v1",
        "review_id": str(review_id),
        "source_exchange_event_id": str(source_exchange_event_id),
        "response_hash": response_hash,
        "proposal_index": proposal_index,
        "file_path": validate_diff_path(file_path),
        "pass_index": pass_index,
    }
    if source_side == "base":
        payload["source_side"] = "base"
    return compute_identity_hash(payload)


class NativeCandidateRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(pattern=SHA256_HEX_PATTERN)
    review_id: UUID
    source_exchange_event_id: UUID
    response_hash: str = Field(pattern=SHA256_HEX_PATTERN)
    proposal_index: int = Field(ge=0, le=49, strict=True)
    file_path: str = Field(max_length=1024)
    source_side: Literal["head", "base"] = Field(
        default="head", exclude_if=lambda value: value == "head"
    )
    pass_index: int = Field(ge=0, strict=True)
    initial_disposition: Literal["admitted", "rejected", "malformed"]
    rejection_reason: CandidateRejectionReason | None = None
    proposal_hash: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)
    finding_id: UUID | None = None
    content_hash: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)
    finding_type: FindingType | None = None
    severity: FindingSeverity | None = None
    evidence_tier: EvidenceTier | None = None
    policy_version: str | None = Field(default=None, pattern=BARE_SEMVER_PATTERN)

    @field_validator("file_path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        return validate_diff_path(value)

    @model_validator(mode="after")
    def check_identity_and_admission(self) -> NativeCandidateRef:
        expected = native_candidate_id(
            review_id=self.review_id,
            source_exchange_event_id=self.source_exchange_event_id,
            response_hash=self.response_hash,
            proposal_index=self.proposal_index,
            file_path=self.file_path,
            pass_index=self.pass_index,
            source_side=self.source_side,
        )
        if self.candidate_id != expected:
            raise ValueError("candidate identity differs from its source occurrence")
        finding_fields = (
            self.finding_id,
            self.content_hash,
            self.finding_type,
            self.severity,
            self.evidence_tier,
            self.policy_version,
        )
        if self.initial_disposition == "admitted":
            if self.source_side == "base" and self.evidence_tier != "judged":
                raise ValueError("BASE candidates require judged evidence")
            if (
                self.rejection_reason is not None
                or self.proposal_hash is None
                or any(field is None for field in finding_fields)
            ):
                raise ValueError("admitted candidates require original finding metadata")
            if (
                self.finding_type is not None
                and self.policy_version == ACTIVE_POLICY_VERSION
                and (SEVERITY_POLICY[self.finding_type] != self.severity)
            ):
                raise ValueError("candidate priority differs from its active policy")
        elif any(field is not None for field in finding_fields):
            raise ValueError("unadmitted candidates cannot claim finding metadata")
        elif self.initial_disposition == "malformed":
            if self.rejection_reason != "raw_proposal_invalid" or self.proposal_hash is not None:
                raise ValueError("malformed candidates require the malformed disposition")
        elif self.rejection_reason in {None, "raw_proposal_invalid"} or self.proposal_hash is None:
            raise ValueError("rejected candidates require their validated proposal hash and reason")
        return self


class CandidateSnippetRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snippet_id: str = Field(pattern=SHA256_HEX_PATTERN)
    file_path: str = Field(max_length=1024)
    source_side: Literal["head", "base"] = Field(
        default="head", exclude_if=lambda value: value == "head"
    )
    line_start: int = Field(ge=1, strict=True)
    line_end: int = Field(ge=1, strict=True)
    source_hash: str = Field(pattern=SHA256_HEX_PATTERN)
    text_hash: str = Field(pattern=SHA256_HEX_PATTERN)

    @field_validator("file_path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        return validate_diff_path(value)

    @model_validator(mode="after")
    def ordered_range(self) -> CandidateSnippetRef:
        if self.line_end < self.line_start:
            raise ValueError("snippet source range is reversed")
        return self


class CandidateCitationRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snippet_id: str = Field(pattern=SHA256_HEX_PATTERN)
    file_path: str = Field(max_length=1024)
    source_side: Literal["head", "base"] = Field(
        default="head", exclude_if=lambda value: value == "head"
    )
    line_start: int = Field(ge=1, strict=True)
    line_end: int = Field(ge=1, strict=True)
    quote_hash: str = Field(pattern=SHA256_HEX_PATTERN)

    @field_validator("file_path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        return validate_diff_path(value)

    @model_validator(mode="after")
    def ordered_range(self) -> CandidateCitationRef:
        if self.line_end < self.line_start:
            raise ValueError("citation source range is reversed")
        return self


class CandidateCheckStarted(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(pattern=SHA256_HEX_PATTERN)
    attempt_id: UUID
    source_exchange_event_id: UUID
    prompt_hash: str = Field(pattern=SHA256_HEX_PATTERN)
    question_hash: str = Field(pattern=SHA256_HEX_PATTERN)
    model: str = Field(min_length=1, max_length=256)
    snippets: tuple[CandidateSnippetRef, ...] = Field(max_length=8)
    response_contract: CandidateCheckContract | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class CandidateAssessmentRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(pattern=SHA256_HEX_PATTERN)
    attempt_id: UUID
    assessment_exchange_event_id: UUID | None = None
    status: Literal["supports", "refutes", "uncertain"]
    head_assessment: HeadAssessment | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    change_assessment: ChangeAssessment | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    impact_assessment: ImpactAssessment | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    repair_assessment: RepairAssessment | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    outcome: Literal[
        "completed",
        "interrupted",
        "content_unavailable",
        "invalid_response",
        "provider_failed",
        "budget_exhausted",
    ]
    suggested_finding_type: FindingType | None = None
    suggested_severity: FindingSeverity | None = None
    policy_version: str = Field(pattern=BARE_SEMVER_PATTERN)
    citations: tuple[CandidateCitationRef, ...] = Field(default=(), max_length=8)
    reason_hash: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)
    reason_length: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def completed_assessment_shape(self) -> CandidateAssessmentRef:
        if (self.head_assessment is None) != (self.change_assessment is None):
            raise ValueError("HEAD and change assessments must be present together")
        if self.impact_assessment is not None and self.head_assessment is None:
            raise ValueError("impact assessments require HEAD and change judgments")
        if self.repair_assessment is not None and self.impact_assessment is None:
            raise ValueError("repair assessments require HEAD, change and impact judgments")
        if self.head_assessment is not None and self.change_assessment is not None:
            if self.outcome != "completed":
                raise ValueError("unfinished assessments cannot carry HEAD or change judgments")
            if self.status != combine_candidate_assessment(
                self.head_assessment,
                self.change_assessment,
                self.impact_assessment,
                self.repair_assessment,
            ):
                raise ValueError("assessment status differs from its structured judgments")
            sides = {citation.source_side for citation in self.citations}
            if self.head_assessment != "uncertain" and "head" not in sides:
                raise ValueError("a definite HEAD assessment requires a HEAD citation")
            if self.impact_assessment in {"supports", "refutes"} and "head" not in sides:
                raise ValueError("a definite impact assessment requires a HEAD citation")
            if self.repair_assessment in {"supports", "refutes"} and "head" not in sides:
                raise ValueError("a definite repair assessment requires a HEAD citation")
            if self.change_assessment == "unrelated_preexisting" and not {"base", "head"} <= sides:
                raise ValueError("unrelated pre-existing code requires BASE and HEAD citations")
        if self.outcome != "completed":
            if (
                self.status != "uncertain"
                or self.citations
                or self.suggested_finding_type is not None
            ):
                raise ValueError("unfinished assessments must remain uncertain and unclassified")
        elif self.assessment_exchange_event_id is None or self.reason_hash is None:
            raise ValueError(
                "completed assessments require their retained exchange and reason hash"
            )
        if self.status in {"supports", "refutes"} and not self.citations:
            raise ValueError("supports and refutes require supplied-source citations")
        if (self.suggested_finding_type is None) != (self.suggested_severity is None):
            raise ValueError("category suggestions and policy priority must be present together")
        if (
            self.suggested_finding_type is not None
            and self.policy_version == ACTIVE_POLICY_VERSION
            and SEVERITY_POLICY[self.suggested_finding_type] != self.suggested_severity
        ):
            raise ValueError("suggested priority differs from its active policy")
        return self
