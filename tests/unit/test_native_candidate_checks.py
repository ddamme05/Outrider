"""Preserve native occurrences and investigate them without accepted parents.

The provider and retained-content store are scripted. These checks establish
source binding, bounded calls and replay behavior, not semantic model quality.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from difflib import unified_diff
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from outrider.agent.candidate_checks import (
    assess_candidate_response,
    investigate_native_candidates,
    prepare_candidate_check,
)
from outrider.agent.candidate_store import (
    CandidateAttemptReceipt,
    CandidateIntegrityError,
    CandidateStore,
    RetainedCandidateExchange,
)
from outrider.agent.checkpoint_serde import build_checkpoint_serde
from outrider.agent.graph import _analyze_router
from outrider.agent.native_candidates import (
    candidate_check_event_id,
    candidate_event_id,
    collect_native_candidates,
)
from outrider.agent.nodes.analyze import AnalyzeWorkerPayload, analyze_file
from outrider.agent.nodes.analyze_aggregate import analyze_aggregate
from outrider.agent.nodes.analyze_parser import parse_analyze_response
from outrider.agent.nodes.trace import trace
from outrider.audit.events import (
    CandidateAssessmentEvent,
    CandidateCheckStartedEvent,
    LLMCallEvent,
    NativeCandidateEvent,
)
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER, source_line_count
from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.llm_call_content import LLMCallContent
from outrider.llm.base import LLMResponse
from outrider.policy import FindingType
from outrider.policy.canonical import compute_round_id
from outrider.policy.publish_eligibility import verification_hold_finding_ids
from outrider.policy.severity import ACTIVE_POLICY_VERSION, SEVERITY_POLICY
from outrider.schemas.analysis_round import AnalysisRound
from outrider.schemas.native_candidate import CandidateAssessmentRef
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState
from outrider.schemas.triage_result import ReviewTier

_SOURCE = "def load(user):\n    return checked_name(user)\n"
_HELPER = "def checked_name(user):\n    return user.name if user is not None else ''\n"
_TEST = "def test_load():\n    assert load(None) == ''\n"
_MODEL = "claude-haiku-5-5"


def _response(text, *, event_id=None):
    return LLMResponse(
        text=text,
        model=_MODEL,
        input_tokens=100,
        output_tokens=40,
        cache_read_tokens=0,
        cache_write_tokens=0,
        latency_ms=1,
        finish_reason="end_turn",
        exchange_event_id=event_id or uuid4(),
        exchange_recorded_at=datetime.now(UTC),
        profile_id="anthropic",
        reasoning_enabled=False,
        profile_contract_digest="a" * 64,
    )


def _proposal(**changes):
    return {
        "finding_type": "correctness_not_in_enum",
        "evidence_tier": "judged",
        "title": "Missing null guard",
        "description": "A null user could fail at this lookup.",
        "evidence": "    return checked_name(user)",
        "line_start": 2,
        "line_end": 2,
        **changes,
    }


def _case(items=None):
    review_id = uuid4()
    response = _response(json.dumps({"findings": items or [_proposal()]}))
    parsed = parse_analyze_response(
        response.text,
        review_id=review_id,
        installation_id=7,
        file_path="app.py",
        file_content=_SOURCE,
        file_byte_length=len(_SOURCE.encode()),
        included_scope_units=(),
        query_match_id_set=frozenset(),
        degraded_mode=False,
        active_policy_version=ACTIVE_POLICY_VERSION,
        primary_line_ranges=((1, source_line_count(_SOURCE)),),
    )
    refs = collect_native_candidates(
        response, parsed, review_id=review_id, file_path="app.py", pass_index=0
    )
    now = datetime.now(UTC)
    state = ReviewState(
        review_id=review_id,
        received_at=now,
        is_eval=True,
        pr_context=PRContext(
            installation_id=7,
            owner="owner",
            repo="repo",
            pr_number=1,
            pr_title="Change load",
            head_sha="a" * 40,
            base_sha="b" * 40,
            author="developer",
            total_additions=6,
            total_deletions=0,
            changed_files=tuple(
                ChangedFile(
                    path=path,
                    status="added",
                    additions=2,
                    deletions=0,
                    content_head=text,
                    language="python",
                )
                for path, text in (
                    ("app.py", _SOURCE),
                    ("helpers.py", _HELPER),
                    ("tests/test_app.py", _TEST),
                )
            ),
        ),
        analysis_rounds=[
            AnalysisRound(
                round_id=compute_round_id(
                    pass_index=0,
                    files_examined=("app.py",),
                    files_skipped=(),
                    finding_content_hashes=tuple(
                        finding.content_hash for finding in parsed.admitted_findings
                    ),
                ),
                pass_index=0,
                files_examined=("app.py",),
                files_skipped=(),
                findings=parsed.admitted_findings,
                started_at=now,
                ended_at=now,
            )
        ],
        native_candidates=list(refs),
    )
    return state, response, refs


def _prepare(state, response, ref):
    prepared = prepare_candidate_check(
        state,
        ref,
        response.text,
        model=_MODEL,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    assert prepared is not None
    return prepared


def _answer(request, **changes):
    supplied = json.loads(request.user_prompt)
    helper = next(
        item for item in supplied["reference_snippets"] if item["file_path"] == "helpers.py"
    )
    decision = (
        {"head_assessment": "refutes", "change_assessment": "uncertain"}
        if "head_assessment" in json.loads(request.response_schema_json)["properties"]
        else {"status": "refutes"}
    )
    if "impact_assessment" in json.loads(request.response_schema_json)["properties"]:
        decision["impact_assessment"] = "not_claimed"
    if "repair_assessment" in json.loads(request.response_schema_json)["properties"]:
        decision["repair_assessment"] = "not_proposed"
    return _response(
        json.dumps(
            {
                "candidate_id": supplied["candidate_id"],
                **decision,
                "reason": "The captured helper handles a null user.",
                "suggested_finding_type": None,
                "citations": [
                    {
                        "snippet_id": helper["snippet_id"],
                        "line_start": 2,
                        "line_end": 2,
                        "quote": "    return user.name if user is not None else ''",
                    }
                ],
                **changes,
            }
        )
    )


class _MemoryStore:
    def __init__(self, state, source, refs):
        self.state = state
        self.source = RetainedCandidateExchange(None, source)
        self.refs = refs
        self.checks = {}
        self.assessments = {}
        self.exchanges = {}

    async def load_records(self, **_scope):
        return self.refs, tuple(self.checks.values()), tuple(self.assessments.values())

    async def load_source(self, _ref, **_scope):
        return self.source

    async def load_attempt(self, *, attempt_id, **_scope):
        return CandidateAttemptReceipt(
            self.checks.get(attempt_id),
            self.assessments.get(attempt_id),
            self.exchanges.get(attempt_id),
        )

    async def emit_candidate_check_started(self, event):
        self.checks[event.check.attempt_id] = event.check

    async def emit_native_candidate(self, event):
        self.refs = (*self.refs, event.candidate)

    async def emit_candidate_assessment(self, event):
        self.assessments[event.assessment.attempt_id] = event.assessment


class _Provider:
    def __init__(self, native_response=None):
        self.requests = []
        self.native_response = native_response

    async def complete(self, request):
        self.requests.append(request)
        if request.node_id == "analyze" and self.native_response is not None:
            return self.native_response.model_copy(update={"model": request.model})
        return _answer(request)

    async def aclose(self):
        return None


def test_native_inventory_retains_each_unknown_and_malformed_occurrence_without_raw_text():
    state, response, refs = _case(
        [
            _proposal(),
            42,
            _proposal(finding_type=FindingType.MISSING_INPUT_VALIDATION.value),
            _proposal(),
        ]
    )
    assert [ref.initial_disposition for ref in refs] == [
        "rejected",
        "malformed",
        "admitted",
        "rejected",
    ]
    assert len({ref.candidate_id for ref in refs}) == 4
    assert refs[0].proposal_hash == refs[3].proposal_hash
    assert refs[2].severity == SEVERITY_POLICY[FindingType.MISSING_INPUT_VALIDATION]
    encoded = json.dumps([ref.model_dump(mode="json") for ref in refs])
    assert "correctness_not_in_enum" not in encoded
    assert "Missing null guard" not in encoded
    assert "return checked_name" not in encoded
    assert _prepare(state, response, refs[0]).request.candidate_attempt_id is not None


def test_assessment_accepts_exact_served_lines_but_not_foreign_quotes_or_labels():
    state, response, refs = _case()
    prepared = _prepare(state, response, refs[0])
    result = assess_candidate_response(
        prepared, _answer(prepared.request), policy_version=ACTIVE_POLICY_VERSION
    )
    assert result.status == "refutes"
    assert result.suggested_finding_type is None
    assert result.citations[0].file_path == "helpers.py"
    snippets = json.loads(prepared.request.user_prompt)["reference_snippets"]
    assert {item["file_path"] for item in snippets} == {"app.py", "helpers.py", "tests/test_app.py"}
    bad_citation = {
        "snippet_id": "f" * 64,
        "line_start": 2,
        "line_end": 2,
        "quote": "    return user.name if user is not None else ''",
    }
    with pytest.raises(CandidateIntegrityError, match="supplied snippet"):
        assess_candidate_response(
            prepared,
            _answer(prepared.request, citations=[bad_citation]),
            policy_version=ACTIVE_POLICY_VERSION,
        )

    bad_citation["snippet_id"] = result.citations[0].snippet_id
    bad_citation["quote"] = "invented code"
    with pytest.raises(CandidateIntegrityError, match="exact captured"):
        assess_candidate_response(
            prepared,
            _answer(prepared.request, citations=[bad_citation]),
            policy_version=ACTIVE_POLICY_VERSION,
        )
    with pytest.raises(ValueError):
        assess_candidate_response(
            prepared,
            _answer(prepared.request, suggested_finding_type="nullable bug"),
            policy_version=ACTIVE_POLICY_VERSION,
        )


def test_historical_assessment_requires_its_policy_and_cannot_override_live_mapping():
    state, response, refs = _case()
    prepared = _prepare(state, response, refs[0])
    answer = _answer(prepared.request, suggested_finding_type="auth_bypass")
    historical = {FindingType.AUTH_BYPASS: SEVERITY_POLICY[FindingType.AUTH_BYPASS]}
    restored = assess_candidate_response(
        prepared, answer, policy_version="1.2.0", historical_policy=historical
    )
    assert restored.policy_version == "1.2.0"
    assert restored.suggested_severity == "critical"
    with pytest.raises(CandidateIntegrityError, match="validated policy mapping"):
        assess_candidate_response(prepared, answer, policy_version="1.2.0")
    with pytest.raises(CandidateIntegrityError, match="override the active policy"):
        assess_candidate_response(
            prepared, answer, policy_version=ACTIVE_POLICY_VERSION, historical_policy=historical
        )
    with pytest.raises(CandidateIntegrityError, match="validated policy mapping"):
        assess_candidate_response(
            prepared,
            _answer(prepared.request, suggested_finding_type="correctness_bug"),
            policy_version="1.2.0",
            historical_policy=historical,
        )


@pytest.mark.parametrize("previous_path", [None, "old_app.py"])
def test_check_binds_shifted_base_evidence_to_its_side_and_preserves_original(previous_path):
    state, response, refs = _case()
    original = refs[0].model_dump(mode="json")
    base = "# previous header\n" + _SOURCE
    base_path = previous_path or "app.py"
    patch = "".join(
        unified_diff(
            base.splitlines(keepends=True),
            _SOURCE.splitlines(keepends=True),
            fromfile=f"a/{base_path}",
            tofile="b/app.py",
        )
    )
    changed = ChangedFile(
        path="app.py",
        previous_path=previous_path,
        status="renamed" if previous_path else "modified",
        content_base=base,
        content_head=_SOURCE,
        additions=0,
        deletions=1,
        patch=patch,
        language="python",
    )
    state = state.model_copy(
        update={
            "pr_context": state.pr_context.model_copy(
                update={"changed_files": (changed, *state.pr_context.changed_files[1:])}
            )
        }
    )
    prepared = _prepare(state, response, refs[0])
    user = json.loads(prepared.request.user_prompt)
    assert user["change_context"]["availability"] == "available"
    assert user["change_context"]["base_sha"] == state.pr_context.base_sha
    assert "-# previous header" in user["change_context"]["diff"]
    base_snippet = next(item for item in prepared.check.snippets if item.source_side == "base")
    assert base_snippet.file_path == base_path
    assert (base_snippet.line_start, base_snippet.line_end) == (1, 3)
    answer = _answer(
        prepared.request,
        head_assessment="uncertain",
        change_assessment="uncertain",
        citations=[
            {
                "snippet_id": base_snippet.snippet_id,
                "line_start": 3,
                "line_end": 3,
                "quote": "    return checked_name(user)",
            }
        ],
    )
    assessment = assess_candidate_response(prepared, answer, policy_version=ACTIVE_POLICY_VERSION)
    assert assessment.citations[0].source_side == "base"
    assert assessment.citations[0].model_dump(mode="json")["source_side"] == "base"
    assert refs[0].model_dump(mode="json") == original
    head = next(item for item in prepared.check.snippets if item.source_side == "head")
    assert "source_side" not in head.model_dump(mode="json")
    encoded = json.dumps(
        {key: user[key] for key in ("reference_snippets", "change_context", "limits")},
        ensure_ascii=False,
    ).encode()
    assert len(encoded) <= 12_000


def test_check_does_not_send_helpers_when_primary_claim_source_exceeds_cap():
    state, response, refs = _case()
    source = "def load(user):\n    return checked_name(user)  # " + "x" * 2100 + "\n"
    state = state.model_copy(
        update={
            "pr_context": state.pr_context.model_copy(
                update={
                    "changed_files": (
                        state.pr_context.changed_files[0].model_copy(
                            update={"content_head": source}
                        ),
                        *state.pr_context.changed_files[1:],
                    )
                }
            )
        }
    )
    assert (
        prepare_candidate_check(
            state,
            refs[0],
            response.text,
            model=_MODEL,
            import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
        )
        is None
    )


@pytest.mark.asyncio
async def test_flat_refutation_cannot_bypass_current_contract_and_original_finding_is_held():
    state, original, refs = _case([_proposal(finding_type="missing_error_handling")])
    store = _MemoryStore(state, original, refs)
    provider = _Provider()

    async def legacy_refutation(request):
        provider.requests.append(request)
        body = json.loads(_answer(request).text)
        body.pop("head_assessment")
        body.pop("change_assessment")
        body.pop("impact_assessment", None)
        body.pop("repair_assessment", None)
        body.update(status="refutes", reason="The PR makes the old code stricter.")
        return _response(json.dumps(body))

    provider.complete = legacy_refutation
    assessed = await investigate_native_candidates(
        state,
        provider=provider,
        model=_MODEL,
        store=store,
        sink=store,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    assert len(provider.requests) == 1
    assert [(item.status, item.outcome) for item in assessed] == [("uncertain", "invalid_response")]
    assert tuple(store.refs) == refs
    assert verification_hold_finding_ids(
        state.analysis_rounds[0].findings,
        refs,
        assessed,
        policy_version=ACTIVE_POLICY_VERSION,
    ) == (refs[0].finding_id,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repair", "status", "held"),
    [
        ("refutes", "uncertain", True),
        ("uncertain", "uncertain", True),
        ("supports", "supports", False),
        ("not_proposed", "supports", False),
    ],
)
async def test_repair_assessment_preserves_supported_bug_and_original_finding(repair, status, held):
    state, original, refs = _case([_proposal(finding_type="missing_error_handling")])
    findings = state.analysis_rounds[0].findings
    original_json = tuple(finding.model_dump_json() for finding in findings)
    store = _MemoryStore(state, original, refs)
    provider = _Provider()

    async def check_repair(request):
        provider.requests.append(request)
        return _answer(
            request,
            head_assessment="supports",
            impact_assessment="not_claimed",
            change_assessment="introduced_or_worsened",
            repair_assessment=repair,
            reason="The local failure and the proposed repair are separate judgments.",
        )

    provider.complete = check_repair
    assessed = await investigate_native_candidates(
        state,
        provider=provider,
        model=_MODEL,
        store=store,
        sink=store,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    assert len(assessed) == 1
    assessment = assessed[0]
    assert assessment.outcome == "completed"
    assert assessment.head_assessment == "supports"
    assert assessment.repair_assessment == repair
    assert assessment.status == status
    assert len(provider.requests) == 1
    assert provider.requests[0].prompt_template_version == "native-candidate-check-v5"
    assert next(iter(store.checks.values())).response_contract == "head-change-impact-and-repair-v1"
    assert tuple(store.refs) == refs
    assert tuple(state.native_candidates) == refs
    assert (
        tuple(finding.model_dump_json() for finding in state.analysis_rounds[0].findings)
        == original_json
    )
    assert verification_hold_finding_ids(
        findings, refs, assessed, policy_version=ACTIVE_POLICY_VERSION
    ) == ((refs[0].finding_id,) if held else ())


@pytest.mark.asyncio
async def test_pending_only_review_reaches_actual_trace_and_survives_checkpoint_resume():
    state, response, _ = _case()
    state.analysis_rounds = []
    state.native_candidates = []
    state.analyze_pass_started_at = state.received_at
    store = _MemoryStore(state, response, ())
    provider = _Provider(response)
    phase_sink = SimpleNamespace(emit_phase=AsyncMock())
    analyze_sink = SimpleNamespace(
        **{
            method: AsyncMock()
            for method in (
                "emit_finding",
                "emit_finding_proposal_rejected",
                "emit_analyze_response_rejected",
                "emit_analyze_completed",
                "emit_scope_exclusion",
                "emit_cache_lookup",
                "emit_cache_serve",
                "emit_observed_skip_shadow",
            )
        }
    )
    primary = state.pr_context.changed_files[0].model_copy(
        update={"patch": "@@ -0,0 +1,2 @@\n+def load(user):\n+    return checked_name(user)\n"}
    )
    worker = await analyze_file(
        AnalyzeWorkerPayload(
            review_id=state.review_id,
            installation_id=7,
            is_eval=True,
            pass_index=0,
            review_tier=ReviewTier.DEEP,
            allocation_tokens=40_000,
            changed_file=primary,
        ),
        provider=provider,
        analyze_model=_MODEL,
        standard_analyze_model=_MODEL,
        phase_event_sink=phase_sink,
        file_examination_sink=SimpleNamespace(emit_file_examination=AsyncMock()),
        analyze_event_sink=analyze_sink,
        candidate_event_sink=store,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    state.analyze_worker_outcomes = worker["analyze_worker_outcomes"]
    aggregated = await analyze_aggregate(
        state,
        analyze_event_sink=analyze_sink,
        phase_event_sink=phase_sink,
        anomaly_sink=SimpleNamespace(emit_anomaly=AsyncMock()),
        analyze_model=_MODEL,
        standard_analyze_model=_MODEL,
        total_review_budget_tokens=200_000,
    )
    for field, value in aggregated.items():
        setattr(state, field, value)
    refs = tuple(state.native_candidates)
    assert len(refs) == 1
    assert refs[0].initial_disposition == "rejected"
    assert refs[0].rejection_reason == "finding_type_not_in_enum"
    assert analyze_sink.emit_finding_proposal_rejected.await_count == 1
    assert state.analysis_rounds[0].findings == ()
    trace_sink = SimpleNamespace(emit_trace_decision=AsyncMock(), get_trace_decisions=AsyncMock())
    github_factory = AsyncMock(side_effect=AssertionError("no candidate fetch permitted"))

    async def trace_node(current):
        return await trace(
            current,
            provider=provider,
            trace_model=_MODEL,
            phase_event_sink=phase_sink,
            trace_sink=trace_sink,
            github_factory=github_factory,
            candidate_checks_enabled=True,
            candidate_store=store,
            candidate_event_sink=store,
            import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
        )

    assert _analyze_router(state) == "synthesize"
    assert _analyze_router(state, candidate_checks_enabled=True) == "trace"
    builder = StateGraph(ReviewState)
    builder.add_node("trace", trace_node)
    builder.add_edge(START, "trace")
    builder.add_edge("trace", END)
    graph = builder.compile(checkpointer=InMemorySaver(serde=build_checkpoint_serde()))
    config = {"configurable": {"thread_id": str(state.review_id)}}
    first = await graph.ainvoke(state, config)
    assert first["analysis_rounds"][0].findings == ()
    assert first["candidate_assessments"][0].status == "refutes"
    await graph.ainvoke({}, config)
    assert len(provider.requests) == 2
    assert [request.node_id for request in provider.requests] == ["analyze", "trace"]
    github_factory.assert_not_awaited()
    saved = (await graph.aget_state(config)).values
    assert saved["native_candidates"] == list(refs)
    assert len(saved["candidate_assessments"]) == 1
    metadata = json.dumps([item.model_dump(mode="json") for item in saved["candidate_assessments"]])
    assert "The captured helper" not in metadata
    assert "return user.name" not in metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("source_retained", [True, False])
@pytest.mark.parametrize("instructions_changed", [False, True])
async def test_completed_receipt_recovers_without_rebilling_and_started_only_stays_uncertain(
    source_retained,
    instructions_changed,
    monkeypatch,
):
    state, response, refs = _case([_proposal(), _proposal(title="Second occurrence")])
    store = _MemoryStore(state, response, refs)
    prepared = [_prepare(state, response, ref) for ref in refs]
    for item in prepared:
        store.checks[item.check.attempt_id] = item.check
    store.exchanges[prepared[0].check.attempt_id] = RetainedCandidateExchange(
        None, _answer(prepared[0].request)
    )
    if not source_retained:
        store.source = RetainedCandidateExchange(None, None)
    if instructions_changed:
        monkeypatch.setattr("outrider.agent.candidate_checks.VERSION", "future-instructions")
        monkeypatch.setattr(
            "outrider.agent.candidate_checks._SYSTEM", "A different review question."
        )
    provider = _Provider()
    results = await investigate_native_candidates(
        state,
        provider=provider,
        model=_MODEL,
        store=store,
        sink=store,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    assert [(item.status, item.outcome) for item in results] == (
        [("refutes", "completed"), ("uncertain", "interrupted")]
        if source_retained
        else [("uncertain", "content_unavailable")] * 2
    )
    assert provider.requests == []
    assert len(store.checks) == 2


@pytest.mark.asyncio
async def test_trace_restores_durable_ref_before_holding_an_expired_started_check():
    state, response, refs = _case(
        [_proposal(finding_type=FindingType.MISSING_ERROR_HANDLING.value)]
    )
    store = _MemoryStore(state, response, refs)
    check = _prepare(state, response, refs[0]).check
    store.checks[check.attempt_id] = check
    store.source = RetainedCandidateExchange(None, None)
    state.native_candidates = []
    provider = _Provider()
    github_factory = AsyncMock(side_effect=AssertionError("no candidate fetch permitted"))
    delta = await trace(
        state,
        provider=provider,
        trace_model=_MODEL,
        phase_event_sink=SimpleNamespace(emit_phase=AsyncMock()),
        trace_sink=SimpleNamespace(
            emit_trace_decision=AsyncMock(), get_trace_decisions=AsyncMock()
        ),
        github_factory=github_factory,
        candidate_checks_enabled=True,
        candidate_store=store,
        candidate_event_sink=store,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    recovered = state.model_copy(update=delta)
    assert recovered.native_candidates == list(refs)
    assert recovered.candidate_assessments[0].outcome == "content_unavailable"
    assert verification_hold_finding_ids(
        recovered.analysis_rounds[0].findings,
        recovered.native_candidates,
        recovered.candidate_assessments,
        policy_version=ACTIVE_POLICY_VERSION,
    ) == (state.analysis_rounds[0].findings[0].finding_id,)
    assert provider.requests == []
    assert len(store.checks) == 1
    github_factory.assert_not_awaited()


@pytest.mark.asyncio
async def test_calls_remain_bounded_and_invalid_response_is_visible_without_reclassification():
    state, response, refs = _case([_proposal(title=f"Occurrence {i}") for i in range(4)])
    store = _MemoryStore(state, response, refs)
    provider = _Provider()

    async def invalid(request):
        provider.requests.append(request)
        return _answer(request, suggested_finding_type="made up category")

    provider.complete = invalid
    results = await investigate_native_candidates(
        state,
        provider=provider,
        model=_MODEL,
        store=store,
        sink=store,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    assert len(provider.requests) == len(results) == 2
    assert all(
        item.outcome == "invalid_response" and item.status == "uncertain" for item in results
    )
    assert all(item.suggested_finding_type is None for item in results)
    assert len(store.refs) == 4
    assert all(ref.initial_disposition == "rejected" for ref in store.refs)


@pytest.mark.asyncio
async def test_retained_source_reader_enforces_owner_hash_and_actual_expiry():
    state, response, refs = _case()
    event = LLMCallEvent(
        event_id=response.exchange_event_id,
        review_id=state.review_id,
        is_eval=True,
        model=_MODEL,
        node_id="analyze",
        input_tokens=100,
        output_tokens=40,
        cached_tokens=0,
        cost_usd=0.001,
        pricing_version="v8",
        latency_ms=1,
        prompt_hash="c" * 64,
        cache_hit=False,
        context_summary=(),
        prompt_template_version="analyze-v12",
        system_prompt_hash="d" * 64,
        degraded_mode=False,
        finish_reason="end_turn",
    )
    row = AuditEvent(
        **{
            name: getattr(event, name)
            for name in ("event_id", "review_id", "event_type", "timestamp", "is_eval", "phase_key")
        },
        sequence_number=1,
        payload=event.model_dump(mode="json"),
    )
    content = LLMCallContent(
        event_id=event.event_id,
        installation_id=7,
        is_eval=True,
        prompt="retained prompt",
        completion=response.text,
        retention_expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, _query):
            return SimpleNamespace(
                one_or_none=lambda: SimpleNamespace(installation_id=7, is_eval=True)
            )

        async def get(self, model, _identity):
            return row if model is AuditEvent else content

    store = CandidateStore(Session)
    loaded = await store.load_source(refs[0], installation_id=7, is_eval=True)
    assert loaded.response.text == response.text
    with pytest.raises(CandidateIntegrityError, match="owner"):
        await store.load_source(refs[0], installation_id=8, is_eval=True)
    with pytest.raises(CandidateIntegrityError, match="owner"):
        await store.load_source(refs[0], installation_id=7, is_eval=False)
    content.completion = "tampered"
    with pytest.raises(CandidateIntegrityError, match="response hash"):
        await store.load_source(refs[0], installation_id=7, is_eval=True)
    content.retention_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert (await store.load_source(refs[0], installation_id=7, is_eval=True)).response is None


def test_current_assessment_fields_are_required_without_reinterpreting_legacy_answers():
    state, response, refs = _case()
    prepared = _prepare(state, response, refs[0])
    body = json.loads(_answer(prepared.request).text)
    for required in ("impact_assessment", "repair_assessment"):
        missing = {key: value for key, value in body.items() if key != required}
        with pytest.raises(ValueError, match=required):
            assess_candidate_response(
                prepared, _response(json.dumps(missing)), policy_version="1.3.0"
            )

    body.pop("repair_assessment")
    v4 = replace(
        prepared,
        check=prepared.check.model_copy(update={"response_contract": "head-change-and-impact-v1"}),
    )
    historical = assess_candidate_response(v4, _response(json.dumps(body)), policy_version="1.3.0")
    assert historical.status == "refutes"
    assert historical.impact_assessment == "not_claimed"
    assert "repair_assessment" not in historical.model_dump(mode="json")

    body.pop("impact_assessment")
    v3 = replace(
        prepared,
        check=prepared.check.model_copy(update={"response_contract": "head-and-change-v1"}),
    )
    historical = assess_candidate_response(v3, _response(json.dumps(body)), policy_version="1.3.0")
    assert historical.status == "refutes"
    assert "impact_assessment" not in historical.model_dump(mode="json")
    assert "repair_assessment" not in historical.model_dump(mode="json")

    legacy_body = {
        key: value
        for key, value in body.items()
        if key not in {"head_assessment", "change_assessment"}
    }
    legacy_body["status"] = "refutes"
    legacy = replace(prepared, check=prepared.check.model_copy(update={"response_contract": None}))
    flat = assess_candidate_response(
        legacy, _response(json.dumps(legacy_body)), policy_version="1.3.0"
    )
    assert flat.status == "refutes"
    assert "impact_assessment" not in flat.model_dump(mode="json")
    assert "repair_assessment" not in flat.model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("contract", "impact", "repair", "valid"),
    [
        ("head-change-impact-and-repair-v1", "not_claimed", "not_proposed", True),
        ("head-change-impact-and-repair-v1", "not_claimed", None, False),
        ("head-change-and-impact-v1", "not_claimed", None, True),
        ("head-change-and-impact-v1", "not_claimed", "not_proposed", False),
        ("head-change-and-impact-v1", None, None, False),
        ("head-and-change-v1", None, None, True),
        ("head-and-change-v1", "not_claimed", None, False),
    ],
)
async def test_stored_completed_assessment_keeps_its_started_response_contract(
    contract, impact, repair, valid
):
    state, response, refs = _case()
    prepared = _prepare(state, response, refs[0])
    assessment = assess_candidate_response(
        prepared, _answer(prepared.request), policy_version="1.3.0"
    )
    assessment = CandidateAssessmentRef.model_validate(
        assessment.model_copy(
            update={"impact_assessment": impact, "repair_assessment": repair}
        ).model_dump(mode="json")
    )
    started = prepared.check.model_copy(update={"response_contract": contract})
    common = {"review_id": state.review_id, "is_eval": True}
    events = [
        NativeCandidateEvent(
            **common, event_id=candidate_event_id(refs[0].candidate_id), candidate=refs[0]
        ),
        CandidateCheckStartedEvent(
            **common,
            event_id=candidate_check_event_id(started.attempt_id, "started"),
            check=started,
        ),
        CandidateAssessmentEvent(
            **common,
            event_id=candidate_check_event_id(started.attempt_id, "assessment"),
            assessment=assessment,
        ),
    ]
    rows = [
        AuditEvent(
            **{
                name: getattr(event, name)
                for name in (
                    "event_id",
                    "review_id",
                    "event_type",
                    "timestamp",
                    "is_eval",
                )
            },
            phase_key=getattr(event, "phase_key", None),
            sequence_number=number,
            payload=event.model_dump(mode="json"),
        )
        for number, event in enumerate(events, 1)
    ]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, _query):
            return SimpleNamespace(
                one_or_none=lambda: SimpleNamespace(installation_id=7, is_eval=True)
            )

        async def scalars(self, _query):
            return SimpleNamespace(all=lambda: rows)

    store = CandidateStore(Session)
    if valid:
        _, _, loaded = await store.load_records(
            review_id=state.review_id, installation_id=7, is_eval=True
        )
        assert loaded == (assessment,)
    else:
        with pytest.raises(CandidateIntegrityError, match="response contract"):
            await store.load_records(review_id=state.review_id, installation_id=7, is_eval=True)


def test_policy_suggestions_cannot_assign_arbitrary_priority():
    state, response, refs = _case()
    prepared = _prepare(state, response, refs[0])
    result = assess_candidate_response(
        prepared,
        _answer(prepared.request, suggested_finding_type="missing_input_validation"),
        policy_version=ACTIVE_POLICY_VERSION,
    )
    assert result.suggested_severity == SEVERITY_POLICY[FindingType.MISSING_INPUT_VALIDATION]
    payload = result.model_dump(mode="json")
    payload["suggested_severity"] = "critical"
    with pytest.raises(ValueError, match="active policy"):
        CandidateAssessmentRef.model_validate(payload)
