"""Keep removed-source identities distinct without changing legacy HEAD records."""

import hashlib
import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from outrider.api.dashboard.agent_view import AgentFindingView
from outrider.api.dashboard.findings import FindingView
from outrider.audit.events import (
    AnalyzeResponseRejectedEvent,
    ContextManifestEntry,
    FindingProposalRejectedEvent,
    compute_finding_content_hash,
)
from outrider.audit.persister import _lift_finding_event
from outrider.audit.replay import (
    FindingContent,
    ReconstructedFinding,
    ReplayEquivalenceError,
    _verify_full_finding,
)
from outrider.policy.canonical import compute_proposal_hash
from outrider.policy.severity import ACTIVE_POLICY_VERSION, FindingType
from outrider.schemas.native_candidate import NativeCandidateRef, native_candidate_id
from outrider.schemas.review_finding import ReviewFinding


def _finding(side="head", **changes):
    values = {
        "review_id": UUID(int=1),
        "installation_id": 1,
        "policy_version": ACTIVE_POLICY_VERSION,
        "finding_type": "correctness_bug",
        "dimension": "code_quality",
        "severity": "medium",
        "evidence_tier": "judged",
        "file_path": "removed.py",
        "source_side": side,
        "line_start": 2,
        "line_end": 2,
        "title": "Removing the registration prevents startup",
        "description": "The unchanged entry point still relies on this registration.",
        "evidence": "register_handler()",
        "proposal_hash": "a" * 64,
        "content_hash": compute_finding_content_hash(
            "removed.py",
            line_start=2,
            line_end=2,
            finding_type=FindingType.CORRECTNESS_BUG,
            source_side=side,
        ),
    }
    return ReviewFinding.model_validate({**values, **changes})


def test_legacy_head_hashes_stay_identical_and_base_identities_are_distinct():
    args = {"line_start": 2, "line_end": 2, "finding_type": FindingType.CORRECTNESS_BUG}
    expected = hashlib.sha256(
        json.dumps(["removed.py", 2, 2, "correctness_bug"], separators=(",", ":")).encode()
    ).hexdigest()
    assert compute_finding_content_hash("removed.py", **args) == expected
    assert compute_finding_content_hash("removed.py", source_side="head", **args) == expected
    assert compute_finding_content_hash("removed.py", source_side="base", **args) != expected
    proposal = {
        "source_file_path": "removed.py",
        "finding_type": "correctness_bug",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "t",
        "description": "d",
        "evidence": "register_handler()",
        "line_start": 2,
        "line_end": 2,
    }
    assert compute_proposal_hash(**proposal) == compute_proposal_hash(
        source_side="head", **proposal
    )
    assert compute_proposal_hash(**proposal) != compute_proposal_hash(
        source_side="base", **proposal
    )


def test_source_side_survives_finding_event_and_candidate_identity():
    head, base = _finding(), _finding("base")
    assert "source_side" not in head.model_dump(mode="json")
    assert base.model_dump(mode="json")["source_side"] == "base"
    event = _lift_finding_event(base, is_eval=True)
    assert event.source_side == "base"
    assert event.finding_content_hash == base.content_hash
    assert "source_side" not in _lift_finding_event(head, is_eval=True).model_dump(mode="json")
    args = {
        "review_id": UUID(int=1),
        "source_exchange_event_id": UUID(int=2),
        "response_hash": "b" * 64,
        "proposal_index": 0,
        "file_path": "removed.py",
        "pass_index": 0,
    }
    assert native_candidate_id(**args) == native_candidate_id(source_side="head", **args)
    candidate = NativeCandidateRef(
        candidate_id=native_candidate_id(source_side="base", **args),
        source_side="base",
        initial_disposition="rejected",
        rejection_reason="finding_type_not_in_enum",
        proposal_hash="a" * 64,
        **args,
    )
    assert candidate.candidate_id != native_candidate_id(**args)
    assert candidate.model_dump(mode="json")["source_side"] == "base"
    from outrider.api.dashboard.candidates import NativeCandidateView

    assert (
        NativeCandidateView.model_validate(candidate.model_dump()).model_dump()["source_side"]
        == "base"
    )
    head_candidate = NativeCandidateRef.model_validate(
        {
            **candidate.model_dump(),
            "source_side": "head",
            "candidate_id": native_candidate_id(**args),
        }
    )
    assert "source_side" not in head_candidate.model_dump()
    assert (
        NativeCandidateView.model_validate(head_candidate.model_dump()).model_dump()["source_side"]
        == "head"
    )
    manifest = ContextManifestEntry(
        file_path="removed.py",
        source_side="base",
        scope_unit_name="removed file",
        line_start=1,
        line_end=2,
        inclusion_reason="changed_region",
    )
    assert manifest.source_side == "base"


@pytest.mark.parametrize("side", ["head", "base"])
def test_http_finding_views_emit_the_source_side_explicitly(side):
    finding = _finding(side)
    fields = finding.model_dump(include=set(FindingView.model_fields))
    view = FindingView.model_validate(
        {
            **fields,
            "source_side": side,
            "content_redacted": False,
            "eligibility": None,
            "eligibility_reason": None,
            "hitl_decision": None,
            "redaction_sweep_at": None,
        }
    )
    assert view.model_dump(mode="json")["source_side"] == side
    agent = AgentFindingView.model_validate(
        {
            **view.model_dump(include=set(AgentFindingView.model_fields)),
            "hitl_gated": True,
            "reviewer_decision": None,
        }
    )
    assert agent.model_dump(mode="json")["source_side"] == side
    assert ("source_side" in finding.model_dump()) == (side == "base")


def test_replay_rejects_content_bound_to_the_wrong_source_side():
    base = _finding("base")
    content = FindingContent.model_validate(
        base.model_dump(include=set(FindingContent.model_fields))
    )
    forged = content.model_copy(update={"source_side": "head"})
    with pytest.raises(ReplayEquivalenceError, match="source_side"):
        _verify_full_finding(
            ReconstructedFinding(event=_lift_finding_event(base, is_eval=True), content=forged), {}
        )


def test_rejected_base_provenance_survives_without_changing_head_payloads():
    events = (
        AnalyzeResponseRejectedEvent(
            review_id=UUID(int=1),
            file_path="removed.py",
            source_side="base",
            response_hash="b" * 64,
            rejection_reason="raw_response_unparseable",
            rejection_detail="invalid JSON",
        ),
        FindingProposalRejectedEvent(
            review_id=UUID(int=1),
            file_path="removed.py",
            source_side="base",
            proposal_hash="a" * 64,
            claimed_finding_type_hash=None,
            claimed_finding_type_len=None,
            rejection_reason="raw_proposal_invalid",
            rejection_detail="invalid proposal",
        ),
    )
    for event in events:
        assert event.model_dump(mode="json")["source_side"] == "base"
        head = type(event).model_validate({**event.model_dump(), "source_side": "head"})
        assert "source_side" not in head.model_dump(mode="json")


@pytest.mark.parametrize(
    "changes",
    [{"evidence_tier": "inferred", "trace_path": ("scope",)}, {"suggested_fix": "replacement()"}],
)
def test_base_findings_cannot_claim_structural_proof_or_offer_a_patch(changes):
    with pytest.raises(ValidationError, match="BASE"):
        _finding("base", **changes)
