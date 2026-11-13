"""Keep review candidates and append source-backed assessment attempts.

Assessments record what an investigator says. Checking a citation against captured
lines does not establish that a finding is true or qualify it for native delivery.
Every candidate remains visible, including invalid and model-refuted proposals.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from outrider.coordinates import validate_diff_path
from outrider.llm.parsing import strip_outer_json_fence
from outrider.policy import FindingType  # noqa: TC001 (Pydantic field type)
from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw

from .direct_review import DirectFinding

if TYPE_CHECKING:
    from collections.abc import Mapping


class CandidateLedgerError(ValueError):
    """A candidate or assessment cannot be recorded without losing its identity."""


class SourceCitation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str = Field(min_length=1)
    line_start: int = Field(ge=1, strict=True)
    line_end: int = Field(ge=1, strict=True)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_lines(self) -> SourceCitation:
        if self.line_end < self.line_start:
            raise ValueError("citation end precedes start")
        return self


class CandidateAssessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["supports", "refutes", "uncertain"]
    reason: str = Field(min_length=1)
    citations: tuple[SourceCitation, ...] = ()
    origin: Literal["model", "scripted"]
    suggested_finding_type: FindingType | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def require_reason_and_citations(self) -> CandidateAssessment:
        if not self.reason.strip():
            raise ValueError("assessment reason must contain text")
        if self.status in {"supports", "refutes"} and not self.citations:
            raise ValueError("supports and refutes require captured-source citations")
        return self


class DirectCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    index: int = Field(ge=0, strict=True)
    raw_proposal_json: str
    proposal: DirectFinding | None
    errors: tuple[str, ...]
    assessments: tuple[CandidateAssessment, ...] = ()


class CandidateLedger(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    raw_response: str
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finish_reason: str
    diagnostics: tuple[str, ...]
    candidates: tuple[DirectCandidate, ...]
    source_file_path: str | None = Field(default=None, exclude_if=lambda value: value is None)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object field")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError("non-finite JSON number")


def decode_candidate_json(text: str) -> object:
    """Reject duplicate fields and non-finite numbers before interpreting a response."""
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_nonfinite)


def build_ledger(
    raw_response: str,
    *,
    source_id: str,
    finish_reason: str,
    source_file_path: str | None = None,
) -> CandidateLedger:
    """Retain the exact response and each decoded array entry, without admission.

    A source path selects native proposals and must come from a verified request.
    Their readable common view never replaces the original native fields.
    """
    if not isinstance(raw_response, str) or not isinstance(finish_reason, str):
        raise CandidateLedgerError("response and finish reason must be strings")
    if not isinstance(source_id, str) or not source_id.strip():
        raise CandidateLedgerError("source identity must contain text")
    if source_file_path is not None:
        source_file_path = validate_diff_path(source_file_path)
    response_sha256 = hashlib.sha256(raw_response.encode()).hexdigest()
    diagnostics = []
    if finish_reason not in {"end_turn", "stop"}:
        diagnostics.append("discovery-did-not-finish-normally")
    candidates = []
    try:
        text = raw_response if source_file_path is None else strip_outer_json_fence(raw_response)
        raw = decode_candidate_json(text)
    except (ValueError, RecursionError):
        diagnostics.append("invalid-json-response")
    else:
        if not isinstance(raw, dict) or not isinstance(raw.get("findings"), list):
            diagnostics.append("invalid-response-envelope")
        else:
            if set(raw) != {"findings"}:
                diagnostics.append("unexpected-response-fields")
            limit = 100 if source_file_path is None else 50
            if len(raw["findings"]) > limit:
                diagnostics.append(
                    "direct-finding-limit-exceeded"
                    if source_file_path is None
                    else "native-finding-limit-exceeded"
                )
            for index, item in enumerate(raw["findings"]):
                identity = json.dumps([source_id, response_sha256, index], separators=(",", ":"))
                errors: tuple[str, ...] = ()
                view = item
                if source_file_path is not None and isinstance(item, dict):
                    try:
                        AnalyzeFindingProposalRaw.model_validate_json(json.dumps(item), strict=True)
                    except ValidationError as exc:
                        errors = tuple(
                            "native." + ".".join(map(str, error["loc"])) + ":" + error["type"]
                            for error in exc.errors(include_input=False)
                        )
                    view = {
                        key: item[key]
                        for key in ("title", "description", "evidence", "line_start", "line_end")
                        if key in item
                    }
                    view.update(file_path=source_file_path, category=item.get("finding_type"))
                try:
                    proposal = DirectFinding.model_validate(view, strict=True)
                except ValidationError as exc:
                    proposal = None
                    errors += tuple(
                        ".".join(map(str, error["loc"])) + ":" + error["type"]
                        for error in exc.errors(include_input=False)
                    )
                candidates.append(
                    DirectCandidate(
                        candidate_id=hashlib.sha256(identity.encode()).hexdigest(),
                        index=index,
                        raw_proposal_json=json.dumps(
                            item, ensure_ascii=True, separators=(",", ":")
                        ),
                        proposal=proposal,
                        errors=errors,
                    )
                )
    return CandidateLedger(
        source_id=source_id,
        raw_response=raw_response,
        response_sha256=response_sha256,
        finish_reason=finish_reason,
        diagnostics=tuple(diagnostics),
        candidates=tuple(candidates),
        source_file_path=source_file_path,
    )


def record_assessment(
    ledger: CandidateLedger,
    candidate_id: str,
    assessment: CandidateAssessment,
    *,
    contents: Mapping[str, str],
) -> CandidateLedger:
    """Append an attempt after exact captured-line checks; never remove a candidate."""
    original = build_ledger(
        ledger.raw_response,
        source_id=ledger.source_id,
        finish_reason=ledger.finish_reason,
        source_file_path=ledger.source_file_path,
    )
    original_fields = {"assessments"}
    if (
        ledger.response_sha256 != original.response_sha256
        or ledger.diagnostics != original.diagnostics
        or len(ledger.candidates) != len(original.candidates)
        or any(
            candidate.model_dump(exclude=original_fields)
            != before.model_dump(exclude=original_fields)
            for candidate, before in zip(ledger.candidates, original.candidates, strict=True)
        )
    ):
        raise CandidateLedgerError("candidate inventory differs from its original response")
    matches = [
        candidate for candidate in ledger.candidates if candidate.candidate_id == candidate_id
    ]
    if len(matches) != 1 or assessment.candidate_id != candidate_id:
        raise CandidateLedgerError("assessment must name exactly one existing candidate")
    candidate = matches[0]
    if assessment in candidate.assessments:
        raise CandidateLedgerError("assessment is already recorded")
    for citation in assessment.citations:
        content = contents.get(citation.file_path)
        if not isinstance(content, str):
            raise CandidateLedgerError("citation file is absent from captured source")
        lines = content.splitlines()
        if citation.line_end > len(lines):
            raise CandidateLedgerError("citation range exceeds captured source")
        quoted_lines = "\n".join(lines[citation.line_start - 1 : citation.line_end])
        if citation.quote != quoted_lines:
            raise CandidateLedgerError("citation quote differs from its captured line range")
    updated = DirectCandidate(
        **candidate.model_dump(exclude={"assessments"}),
        assessments=(*candidate.assessments, assessment),
    )
    return CandidateLedger(
        **ledger.model_dump(exclude={"candidates"}),
        candidates=tuple(
            updated if item.candidate_id == candidate_id else item for item in ledger.candidates
        ),
    )
