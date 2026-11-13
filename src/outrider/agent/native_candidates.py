"""Keep metadata for every native proposal before deduplication and finding caps.

Content remains in its original retained exchange. These references preserve
unknown categories and malformed entries without inventing accepted findings.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid5

from pydantic import ValidationError
from pydantic_core import from_json

from outrider.llm.parsing import strip_outer_json_fence
from outrider.policy.canonical import compute_proposal_hash, compute_response_hash
from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw
from outrider.schemas.native_candidate import NativeCandidateRef, native_candidate_id

if TYPE_CHECKING:
    from outrider.agent.nodes.analyze_parser import ParserResult, ProposalRejection
    from outrider.llm.base import LLMResponse
    from outrider.schemas.review_finding import ReviewFinding

_CANDIDATE_EVENT_NAMESPACE = UUID("ab06c881-2fcf-45b9-aedc-73bdad44283b")
_CHECK_EVENT_NAMESPACE = UUID("0526de11-1a82-4a15-a293-fde447158b2b")


def candidate_event_id(candidate_id: str) -> UUID:
    return uuid5(_CANDIDATE_EVENT_NAMESPACE, candidate_id)


def candidate_check_event_id(attempt_id: UUID, stage: str) -> UUID:
    if stage not in {"started", "assessment"}:
        raise ValueError("unknown candidate check event stage")
    return uuid5(_CHECK_EVENT_NAMESPACE, f"{attempt_id}:{stage}")


def collect_native_candidates(
    response: LLMResponse,
    parser_result: ParserResult,
    *,
    review_id: UUID,
    file_path: str,
    pass_index: int,
    source_side: Literal["head", "base"] = "head",
) -> tuple[NativeCandidateRef, ...]:
    """Capture the parser's original dispositions before producer merge or caps.

    Older scripted providers can lack exchange references. Real providers supply
    one after durable persistence. A rejected envelope has no decoded inventory.
    """
    if response.exchange_event_id is None or parser_result.response_rejection is not None:
        return ()
    envelope = from_json(
        strip_outer_json_fence(response.text), allow_inf_nan=False, allow_partial=False
    )
    findings: defaultdict[str, deque[ReviewFinding]] = defaultdict(deque)
    rejections: defaultdict[str, deque[ProposalRejection]] = defaultdict(deque)
    malformed: deque[ProposalRejection] = deque()
    for finding in parser_result.admitted_findings:
        findings[finding.proposal_hash].append(finding)
    for rejection in parser_result.proposal_rejections:
        if rejection.rejection_reason == "raw_proposal_invalid":
            malformed.append(rejection)
        else:
            rejections[rejection.proposal_hash].append(rejection)
    response_hash = compute_response_hash(response.text)
    records = []
    for index, item in enumerate(envelope["findings"]):
        base: dict[str, Any] = {
            "candidate_id": native_candidate_id(
                review_id=review_id,
                source_exchange_event_id=response.exchange_event_id,
                response_hash=response_hash,
                proposal_index=index,
                file_path=file_path,
                pass_index=pass_index,
                source_side=source_side,
            ),
            "review_id": review_id,
            "source_exchange_event_id": response.exchange_event_id,
            "response_hash": response_hash,
            "proposal_index": index,
            "file_path": file_path,
            "pass_index": pass_index,
            "source_side": source_side,
        }
        try:
            proposal = AnalyzeFindingProposalRaw.model_validate(item)
        except ValidationError:
            if not malformed:
                raise ValueError(
                    "candidate inventory differs from parser malformed entries"
                ) from None
            malformed.popleft()
            records.append(
                NativeCandidateRef(
                    **base, initial_disposition="malformed", rejection_reason="raw_proposal_invalid"
                )
            )
            continue
        proposal_hash = compute_proposal_hash(
            source_file_path=file_path,
            source_side=source_side,
            **proposal.model_dump(exclude={"trace_candidates"}),
        )
        if findings[proposal_hash]:
            finding = findings[proposal_hash].popleft()
            record = NativeCandidateRef(
                **base,
                initial_disposition="admitted",
                proposal_hash=proposal_hash,
                finding_id=finding.finding_id,
                content_hash=finding.content_hash,
                finding_type=finding.finding_type,
                severity=finding.severity,
                evidence_tier=finding.evidence_tier,
                policy_version=finding.policy_version,
            )
        elif rejections[proposal_hash]:
            rejection = rejections[proposal_hash].popleft()
            record = NativeCandidateRef(
                **base,
                initial_disposition="rejected",
                proposal_hash=proposal_hash,
                rejection_reason=rejection.rejection_reason,
            )
        else:
            raise ValueError("candidate inventory differs from its parser dispositions")
        records.append(record)
    if malformed or any(findings.values()) or any(rejections.values()):
        raise ValueError("candidate inventory did not consume all parser dispositions")
    if len(records) != parser_result.counters.n_proposals_seen:
        raise ValueError("candidate inventory differs from parser proposal count")
    return tuple(records)
