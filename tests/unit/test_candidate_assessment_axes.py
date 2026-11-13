"""Keep local bug truth, broader impact, repair advice and PR relevance separate."""

from uuid import uuid4

import pytest

from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.native_candidate import (
    CandidateAssessmentRef,
    CandidateCheckStarted,
    CandidateCitationRef,
    candidate_check_contract_for_version,
    combine_candidate_assessment,
)


def _citation(side="head"):
    return CandidateCitationRef(
        snippet_id=("1" if side == "head" else "2") * 64,
        file_path="app.py",
        source_side=side,
        line_start=1,
        line_end=1,
        quote_hash="3" * 64,
    )


def _assessment(**changes):
    return CandidateAssessmentRef(
        **{
            "candidate_id": "a" * 64,
            "attempt_id": uuid4(),
            "assessment_exchange_event_id": uuid4(),
            "status": "uncertain",
            "outcome": "completed",
            "policy_version": ACTIVE_POLICY_VERSION,
            "reason_hash": "b" * 64,
            "reason_length": 20,
            **changes,
        }
    )


@pytest.mark.parametrize(
    ("head", "change", "expected"),
    [
        ("supports", "fix_attempt", "supports"),
        ("supports", "introduced_or_worsened", "supports"),
        ("uncertain", "fix_attempt", "uncertain"),
        ("supports", "uncertain", "uncertain"),
        ("refutes", "uncertain", "refutes"),
        ("uncertain", "unrelated_preexisting", "refutes"),
    ],
)
def test_combination_preserves_fix_attemptes_and_missing_premises(head, change, expected):
    assert combine_candidate_assessment(head, change) == expected


def test_historical_metadata_roundtrips_without_new_fields():
    legacy = _assessment()
    payload = legacy.model_dump(mode="json")
    assert "head_assessment" not in payload and "change_assessment" not in payload
    assert "impact_assessment" not in payload
    assert "repair_assessment" not in payload
    assert CandidateAssessmentRef.model_validate(payload).model_dump(mode="json") == payload
    check = CandidateCheckStarted(
        candidate_id="a" * 64,
        attempt_id=uuid4(),
        source_exchange_event_id=uuid4(),
        prompt_hash="b" * 64,
        question_hash="c" * 64,
        model="captured-model",
        snippets=(),
    )
    historical = check.model_dump(mode="json")
    assert "response_contract" not in historical
    assert CandidateCheckStarted.model_validate(historical).model_dump(mode="json") == historical
    tagged = check.model_copy(update={"response_contract": "head-and-change-v1"})
    assert tagged.model_dump(mode="json")["response_contract"] == "head-and-change-v1"


def test_contract_dispatch_keeps_known_legacy_versions_and_rejects_unknown_versions():
    assert candidate_check_contract_for_version("native-candidate-check-v1") is None
    assert candidate_check_contract_for_version("native-candidate-check-v2") is None
    assert candidate_check_contract_for_version("native-candidate-check-v3") == "head-and-change-v1"
    assert (
        candidate_check_contract_for_version("native-candidate-check-v4")
        == "head-change-and-impact-v1"
    )
    assert (
        candidate_check_contract_for_version("native-candidate-check-v5")
        == "head-change-impact-and-repair-v1"
    )
    with pytest.raises(ValueError, match="unknown native candidate check version"):
        candidate_check_contract_for_version("native-candidate-check-v6")


@pytest.mark.parametrize(
    ("head", "impact", "expected"),
    [
        ("supports", "supports", "supports"),
        ("supports", "not_claimed", "supports"),
        ("supports", "uncertain", "uncertain"),
        ("supports", "refutes", "uncertain"),
        ("refutes", "uncertain", "refutes"),
    ],
)
def test_impact_does_not_refute_a_supported_local_bug(head, impact, expected):
    assert combine_candidate_assessment(head, "introduced_or_worsened", impact) == expected
    assessment = _assessment(
        status=expected,
        head_assessment=head,
        change_assessment="introduced_or_worsened",
        impact_assessment=impact,
        citations=(_citation(),),
    )
    assert assessment.head_assessment == head
    assert assessment.impact_assessment == impact
    assert assessment.status == expected


def test_v3_axes_roundtrip_without_impact_and_keep_their_original_status():
    assessment = _assessment(
        status="supports",
        head_assessment="supports",
        change_assessment="fix_attempt",
        citations=(_citation(),),
    )
    payload = assessment.model_dump(mode="json")
    assert "impact_assessment" not in payload
    assert CandidateAssessmentRef.model_validate(payload).model_dump(mode="json") == payload


def test_impact_requires_completed_axes_and_a_consistent_status():
    with pytest.raises(ValueError):
        _assessment(impact_assessment="uncertain")
    with pytest.raises(ValueError):
        _assessment(outcome="invalid_response", impact_assessment="uncertain")
    with pytest.raises(ValueError, match="status differs"):
        _assessment(
            status="supports",
            head_assessment="supports",
            change_assessment="introduced_or_worsened",
            impact_assessment="uncertain",
            citations=(_citation(),),
        )


def test_repair_requires_complete_axes_consistent_status_and_head_citation():
    with pytest.raises(ValueError, match="repair assessments require"):
        _assessment(repair_assessment="uncertain")
    with pytest.raises(ValueError, match="status differs"):
        _assessment(
            status="supports",
            head_assessment="supports",
            change_assessment="introduced_or_worsened",
            impact_assessment="not_claimed",
            repair_assessment="refutes",
            citations=(_citation(),),
        )
    with pytest.raises(ValueError, match="definite repair assessment requires a HEAD citation"):
        _assessment(
            head_assessment="uncertain",
            change_assessment="uncertain",
            impact_assessment="not_claimed",
            repair_assessment="refutes",
            citations=(_citation("base"),),
        )


def test_axes_cannot_disappear_partway_or_contradict_the_status():
    with pytest.raises(ValueError, match="present together"):
        _assessment(head_assessment="uncertain")
    with pytest.raises(ValueError, match="status differs"):
        _assessment(status="refutes", head_assessment="uncertain", change_assessment="fix_attempt")
    with pytest.raises(ValueError, match="unfinished assessments cannot carry"):
        _assessment(
            outcome="invalid_response",
            head_assessment="uncertain",
            change_assessment="uncertain",
        )


def test_definite_head_judgment_requires_head_evidence_even_when_base_has_a_quote():
    with pytest.raises(ValueError, match="HEAD citation"):
        _assessment(
            status="supports",
            head_assessment="supports",
            change_assessment="fix_attempt",
            citations=(_citation("base"),),
        )


def test_preexisting_pr_refutation_preserves_unknown_intrinsic_truth_and_both_sides():
    with pytest.raises(ValueError, match="BASE and HEAD citations"):
        _assessment(
            status="refutes",
            head_assessment="uncertain",
            change_assessment="unrelated_preexisting",
            citations=(_citation(),),
        )
    assessment = _assessment(
        status="refutes",
        head_assessment="uncertain",
        change_assessment="unrelated_preexisting",
        citations=(_citation(), _citation("base")),
    )
    assert assessment.status == "refutes"
    assert assessment.head_assessment == "uncertain"
    assert assessment.change_assessment == "unrelated_preexisting"
