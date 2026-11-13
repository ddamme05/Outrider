"""Keep already disputed findings through round and report limits."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID, uuid5

import pytest
from test_analyze_node import _RecordingAnalyzeEventSink
from test_native_candidate_checks import (
    _case as _native_candidate_case,
)
from test_native_candidate_checks import (
    _prepare as _prepare_native_check,
)
from test_native_candidate_checks import (
    _proposal as _native_proposal,
)
from test_native_candidate_checks import (
    _response as _native_response,
)
from test_synthesize_patch_pass import (
    _DispatchingProvider,
    _StubAnomalySink,
    _StubPhaseSink,
    _StubSynthesizeEventSink,
)

from outrider.agent.candidate_checks import assess_candidate_response
from outrider.agent.nodes.analyze_aggregate import analyze_aggregate
from outrider.agent.nodes.finding_cap import FindingCapOverflowError
from outrider.agent.nodes.synthesize import synthesize
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingType
from outrider.policy.canonical import compute_round_id
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.publish_eligibility import (
    is_eligible_for_v1_publish,
    verification_hold_finding_ids,
)
from outrider.policy.severity import ACTIVE_POLICY_VERSION, lookup_severity
from outrider.schemas import ChangedFile, PRContext, ReviewFinding, ReviewState
from outrider.schemas.analysis_round import AnalysisRound
from outrider.schemas.analyze_worker import AnalyzeWorkerOutcome
from outrider.schemas.native_candidate import (
    CandidateAssessmentRef,
    CandidateCitationRef,
    NativeCandidateRef,
    combine_candidate_assessment,
    native_candidate_id,
)
from outrider.schemas.triage_result import ReviewTier, RiskLevel, TriageResult

_REVIEW_ID = UUID("00000000-0000-4000-8000-000000000001")
_NOW = datetime(2026, 11, 2, tzinfo=UTC)
_LOWER_TYPES = (
    FindingType.MISSING_INPUT_VALIDATION,
    FindingType.MISSING_ERROR_HANDLING,
    FindingType.UNUSED_IMPORT,
)


def _case(high_count: int = 200) -> tuple[ReviewState, tuple[ReviewFinding, ...]]:
    findings = []
    refs = []
    for index, finding_type in enumerate(
        [FindingType.HARDCODED_SECRET] * high_count + list(_LOWER_TYPES)
    ):
        path = f"src/file_{index // 40}.py"
        line = index % 40 + 1
        proposal_hash = hashlib.sha256(str(index).encode()).hexdigest()
        finding = ReviewFinding(
            finding_id=uuid5(_REVIEW_ID, str(index)),
            review_id=_REVIEW_ID,
            installation_id=7,
            policy_version=ACTIVE_POLICY_VERSION,
            finding_type=finding_type,
            dimension=lookup_dimension(finding_type),
            severity=lookup_severity(finding_type),
            evidence_tier=EvidenceTier.JUDGED,
            file_path=path,
            line_start=line,
            line_end=line,
            title="Retained warning",
            description="Previously admitted claim pending human verification.",
            evidence="value = candidate",
            content_hash=compute_finding_content_hash(
                path, line_start=line, line_end=line, finding_type=finding_type
            ),
            proposal_hash=proposal_hash,
        )
        findings.append(finding)
        if finding_type not in _LOWER_TYPES:
            continue
        exchange_id = uuid5(_REVIEW_ID, f"exchange:{index}")
        refs.append(
            NativeCandidateRef(
                candidate_id=native_candidate_id(
                    review_id=_REVIEW_ID,
                    source_exchange_event_id=exchange_id,
                    response_hash=proposal_hash,
                    proposal_index=0,
                    file_path=path,
                    pass_index=0,
                ),
                review_id=_REVIEW_ID,
                source_exchange_event_id=exchange_id,
                response_hash=proposal_hash,
                proposal_index=0,
                file_path=path,
                pass_index=0,
                initial_disposition="admitted",
                proposal_hash=proposal_hash,
                finding_id=finding.finding_id,
                content_hash=finding.content_hash,
                finding_type=finding_type,
                severity=finding.severity,
                evidence_tier=finding.evidence_tier,
                policy_version=ACTIVE_POLICY_VERSION,
            )
        )
    outcomes = []
    for path in sorted({finding.file_path for finding in findings}):
        group = tuple(finding for finding in findings if finding.file_path == path)
        outcomes.append(
            AnalyzeWorkerOutcome(
                path=path,
                pass_index=0,
                source="parser",
                parse_status="clean",
                review_tier=ReviewTier.DEEP,
                n_proposals_seen=len(group),
                admitted_findings=group,
            )
        )
    state = ReviewState(
        review_id=_REVIEW_ID,
        received_at=_NOW,
        is_eval=True,
        pr_context=PRContext(
            installation_id=7,
            owner="owner",
            repo="repo",
            pr_number=1,
            pr_title="Retained review replay",
            head_sha="a" * 40,
            base_sha="b" * 40,
            author="developer",
            total_additions=len(findings),
            total_deletions=0,
            changed_files=tuple(
                ChangedFile(
                    path=outcome.path,
                    status="added",
                    additions=len(outcome.admitted_findings),
                    deletions=0,
                    content_head="value = candidate\n" * len(outcome.admitted_findings),
                )
                for outcome in outcomes
            ),
        ),
        triage_result=TriageResult(
            file_tiers={outcome.path: ReviewTier.DEEP for outcome in outcomes},
            overall_risk=RiskLevel.HIGH,
            relevant_dimensions=(),
            reasoning="Retained review replay.",
        ),
        analyze_worker_outcomes=outcomes,
        analyze_pass_started_at=_NOW,
        native_candidates=refs,
        candidate_assessments=[
            CandidateAssessmentRef(
                candidate_id=ref.candidate_id,
                attempt_id=uuid5(_REVIEW_ID, ref.candidate_id),
                status="uncertain",
                outcome="budget_exhausted",
                policy_version=ACTIVE_POLICY_VERSION,
            )
            for ref in refs
        ],
    )
    return state, tuple(findings)


async def _aggregate(state: ReviewState, sink: _RecordingAnalyzeEventSink):
    return await analyze_aggregate(
        state,
        analyze_event_sink=sink,
        phase_event_sink=_StubPhaseSink(),
        anomaly_sink=_StubAnomalySink(),
        analyze_model="claude-sonnet-5",
        standard_analyze_model="claude-haiku-4-5",
        total_review_budget_tokens=200_000,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["aggregate", "synthesize"])
async def test_node_cap_preserves_each_lower_severity_verification_hold(stage):
    state, findings = _case()
    expected_holds = {finding.finding_id for finding in findings[-3:]}
    assert (
        set(
            verification_hold_finding_ids(
                findings,
                state.native_candidates,
                state.candidate_assessments,
                policy_version=ACTIVE_POLICY_VERSION,
            )
        )
        == expected_holds
    )
    if stage == "aggregate":
        result = await _aggregate(state, _RecordingAnalyzeEventSink())
        kept = result["analysis_rounds"][0].findings
    else:
        state.analysis_rounds = [
            AnalysisRound(
                round_id=compute_round_id(
                    pass_index=0,
                    files_examined=tuple(state.triage_result.file_tiers),
                    files_skipped=(),
                    finding_content_hashes=tuple(f.content_hash for f in findings),
                ),
                pass_index=0,
                files_examined=tuple(state.triage_result.file_tiers),
                files_skipped=(),
                findings=findings,
                started_at=_NOW,
                ended_at=_NOW,
            )
        ]
        result = await synthesize(
            state,
            provider=_DispatchingProvider("{}"),
            synthesize_model="claude-haiku-4-5",
            patch_model="claude-haiku-5-5",
            patches_enabled=False,
            max_suggestions=5,
            phase_event_sink=_StubPhaseSink(),
            synthesize_event_sink=_StubSynthesizeEventSink(),
            anomaly_sink=_StubAnomalySink(),
        )
        kept = result["review_report"].findings
    assert expected_holds <= {finding.finding_id for finding in kept}
    assert len(kept) == len(findings)


@pytest.mark.asyncio
async def test_protected_round_overflow_fails_before_finding_emission():
    state, _ = _case(high_count=255)
    sink = _RecordingAnalyzeEventSink()
    with pytest.raises(FindingCapOverflowError):
        await _aggregate(state, sink)
    assert sink.findings == []
    assert sink.completed == []


@pytest.mark.parametrize(
    ("impact", "expected_hold"),
    [("uncertain", True), ("refutes", True), ("not_claimed", False), ("supports", False)],
)
def test_impact_review_holds_the_warning_without_removing_its_supported_bug(impact, expected_hold):
    state, findings = _case(high_count=0)
    finding = next(
        item for item in findings if item.finding_type == FindingType.MISSING_ERROR_HANDLING
    )
    original = finding.model_dump(mode="json")
    ref = next(item for item in state.native_candidates if item.finding_id == finding.finding_id)
    assessment = CandidateAssessmentRef(
        candidate_id=ref.candidate_id,
        attempt_id=uuid5(_REVIEW_ID, "impact-check"),
        assessment_exchange_event_id=uuid5(_REVIEW_ID, "impact-answer"),
        status=combine_candidate_assessment("supports", "introduced_or_worsened", impact),
        head_assessment="supports",
        change_assessment="introduced_or_worsened",
        impact_assessment=impact,
        outcome="completed",
        policy_version=ACTIVE_POLICY_VERSION,
        citations=(
            CandidateCitationRef(
                snippet_id="a" * 64,
                file_path=finding.file_path,
                line_start=finding.line_start,
                line_end=finding.line_end,
                quote_hash=hashlib.sha256(finding.evidence.encode()).hexdigest(),
            ),
        ),
        reason_hash="b" * 64,
        reason_length=30,
    )
    holds = verification_hold_finding_ids(
        [finding], [ref], [assessment], policy_version=ACTIVE_POLICY_VERSION
    )
    eligibility, _ = is_eligible_for_v1_publish(
        finding,
        hitl_request=None,
        hitl_decision=None,
        verification_hold=finding.finding_id in holds,
    )
    assert (finding.finding_id in holds) is expected_hold
    assert eligibility.value == ("withheld" if expected_hold else "eligible")
    assert assessment.head_assessment == "supports"
    assert finding.model_dump(mode="json") == original


def test_checker_qualification_creates_a_hold_instead_of_only_a_prose_caveat():
    state, original_response, refs = _native_candidate_case(
        [
            _native_proposal(
                finding_type="missing_error_handling",
                title="Local error can escape to the caller",
                description="A local error escapes and an unseen caller returns HTTP 500.",
            )
        ]
    )
    finding = state.analysis_rounds[0].findings[0]
    before = finding.model_dump(mode="json")
    ref = refs[0]
    prepared = _prepare_native_check(state, original_response, ref)
    supplied = json.loads(prepared.request.user_prompt)
    primary = next(
        item for item in supplied["reference_snippets"] if item["file_path"] == finding.file_path
    )
    answer = {
        "candidate_id": ref.candidate_id,
        "head_assessment": "supports",
        "change_assessment": "introduced_or_worsened",
        "reason": "The local defect is supported; HTTP 500 depends on an unshown caller.",
        "suggested_finding_type": finding.finding_type.value,
        "citations": [
            {
                "snippet_id": primary["snippet_id"],
                "line_start": 2,
                "line_end": 2,
                "quote": "    return checked_name(user)",
            }
        ],
    }
    schema = json.loads(prepared.request.response_schema_json)
    if "impact_assessment" in schema["properties"]:
        answer["impact_assessment"] = "uncertain"
    if "repair_assessment" in schema["properties"]:
        answer["repair_assessment"] = "not_proposed"
    response = _native_response(json.dumps(answer))
    assessment = assess_candidate_response(prepared, response, policy_version=ACTIVE_POLICY_VERSION)
    holds = verification_hold_finding_ids(
        [finding], [ref], [assessment], policy_version=ACTIVE_POLICY_VERSION
    )
    assert holds == (finding.finding_id,)
    eligibility, _ = is_eligible_for_v1_publish(
        finding, hitl_request=None, hitl_decision=None, verification_hold=bool(holds)
    )
    assert eligibility.value == "withheld"
    assert assessment.head_assessment == "supports"
    assert finding.model_dump(mode="json") == before
