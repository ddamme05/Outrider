"""Analyze node body tests — spec §7 first landing.

Pins four scenarios per the user's commit-7 scope:

1. **Clean file** — tier=DEEP, parses cleanly, model returns admittable
   proposal → 1 admitted finding in the returned AnalysisRound;
   FileExaminationEvent(parse_status="clean") fires; FindingEvent
   fires; AnalyzeCompletedEvent shows n_findings_emitted=1.
2. **Triage-skipped** — tier=SKIM (or absent from tier map) → file
   NOT in iteration scope; no events for that file; appears NOWHERE
   in AnalysisRound.files_examined OR files_skipped (only kept files
   appear there).
3. **Parser rejection** — tier=DEEP, model returns proposal that
   fails enum admission → 1 ProposalRejection lifted to
   FindingProposalRejectedEvent; admitted_findings empty;
   AnalyzeCompletedEvent shows n_proposals_rejected=1.
4. **Budget skip** — tier=DEEP, estimated cost exceeds per-file cap
   → FileExaminationEvent(skip_reason=COST_BUDGET_EXHAUSTED) fires;
   no LLM call; file appears in AnalysisRound.files_skipped.

Test infrastructure: inline recorder sinks + mock LLM provider + mock
ImportPathResolver. Per the user direction: "without special mocks
beyond provider/context inputs" — all four scenarios share the same
recorder/provider/resolver scaffolding; only the file content + tier
map + provider response differ per test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from outrider.agent.nodes.analyze import (
    DEFAULT_REVIEW_BUDGET_TOKENS,
    PER_FILE_CAP_FRACTION,
    analyze,
)
from outrider.ast_facts.models import SkipReason
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewState
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel, TriageResult

if TYPE_CHECKING:
    from outrider.audit.events import (
        AnalyzeCompletedEvent,
        AnalyzeResponseRejectedEvent,
        FileExaminationEvent,
        FindingEvent,
        FindingProposalRejectedEvent,
        ReviewPhaseEvent,
    )

# ---------------------------------------------------------------------------
# Recorder sinks (inline; commit-7 keeps test infra minimal per user
# direction "without special mocks beyond provider/context inputs").
# ---------------------------------------------------------------------------


class _RecordingFileExaminationSink:
    """Captures `FileExaminationEvent` emissions for assertion."""

    def __init__(self) -> None:
        self.events: list[FileExaminationEvent] = []

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        self.events.append(event)


class _RecordingPhaseEventSink:
    """Inline copy of the conftest recorder; kept local for test
    self-containment."""

    def __init__(self) -> None:
        self.events: list[ReviewPhaseEvent] = []

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        self.events.append(event)


class _RecordingAnalyzeEventSink:
    """Captures every emission of the four analyze-specific event
    types into per-type lists for assertion. The aggregate `events`
    list preserves emission order across types so tests can pin
    event-ordering invariants."""

    def __init__(self) -> None:
        self.findings: list[FindingEvent] = []
        self.proposal_rejections: list[FindingProposalRejectedEvent] = []
        self.response_rejections: list[AnalyzeResponseRejectedEvent] = []
        self.completed: list[AnalyzeCompletedEvent] = []
        self.events: list[Any] = []

    async def emit_finding(self, event: FindingEvent) -> None:
        self.findings.append(event)
        self.events.append(event)

    async def emit_finding_proposal_rejected(self, event: FindingProposalRejectedEvent) -> None:
        self.proposal_rejections.append(event)
        self.events.append(event)

    async def emit_analyze_response_rejected(self, event: AnalyzeResponseRejectedEvent) -> None:
        self.response_rejections.append(event)
        self.events.append(event)

    async def emit_analyze_completed(self, event: AnalyzeCompletedEvent) -> None:
        self.completed.append(event)
        self.events.append(event)


class _StubLLMProvider:
    """Returns a canned JSON response per call. Tests configure the
    response text; the provider doesn't enforce LLMRequest's full
    contract beyond what's needed for the wrapper's internal validation
    of return type."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self.response_text,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=250,
        )


# ---------------------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------------------


_SIMPLE_PY = b"""\
def my_function():
    return 42

def another_function(x):
    y = x + 1
    return y
"""

_REVIEW_ID = UUID("12345678-1234-5678-1234-567812345678")
_INSTALLATION_ID = 99999


def _build_changed_file(
    *,
    path: str = "src/example.py",
    content: bytes = _SIMPLE_PY,
    patch: str | None = "@@ -1,1 +1,2 @@\n+def my_function():\n+    return 42\n",
) -> ChangedFile:
    return ChangedFile(
        path=path,
        status="modified",
        additions=2,
        deletions=0,
        patch=patch,
        content_base="def my_function():\n    return 0\n",
        content_head=content.decode("utf-8"),
        previous_path=None,
        language="python",
    )


def _build_pr_context(*, changed_files: tuple[ChangedFile, ...] | None = None) -> PRContext:
    if changed_files is None:
        changed_files = (_build_changed_file(),)
    return PRContext(
        installation_id=_INSTALLATION_ID,
        owner="acme",
        repo="widget",
        pr_number=42,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title="Test PR",
        pr_body=None,
        author="someone",
        total_additions=2,
        total_deletions=0,
        changed_files=changed_files,
    )


def _build_triage_result(
    *,
    file_tiers: dict[str, ReviewTier] | None = None,
    overall_risk: RiskLevel = RiskLevel.MEDIUM,
) -> TriageResult:
    if file_tiers is None:
        file_tiers = {"src/example.py": ReviewTier.DEEP}
    return TriageResult(
        file_tiers=file_tiers,
        overall_risk=overall_risk,
        relevant_dimensions=(ReviewDimension.SECURITY,),
        reasoning="test",
    )


def _build_review_state(
    *,
    pr_context: PRContext | None = None,
    triage_result: TriageResult | None = None,
    is_eval: bool = True,
) -> ReviewState:
    return ReviewState(
        review_id=_REVIEW_ID,
        received_at=datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC),
        pr_context=pr_context or _build_pr_context(),
        triage_result=triage_result or _build_triage_result(),
        is_eval=is_eval,
    )


def _build_finding_proposal_json(
    *,
    finding_type: str = "sql_injection",
    evidence_tier: str = "judged",
    span_byte_start: int = 0,
    span_byte_end: int = 20,
) -> str:
    """Build a JSON `AnalyzeResponseRaw` payload with one proposal."""
    return json.dumps(
        {
            "findings": [
                {
                    "finding_type": finding_type,
                    "evidence_tier": evidence_tier,
                    "query_match_id": None,
                    "trace_path": None,
                    "title": "Test finding",
                    "description": "A test finding for the analyze node body unit tests.",
                    "evidence": "def my_function():\n    return 42",
                    "span": {
                        "byte_start": span_byte_start,
                        "byte_end": span_byte_end,
                    },
                    "trace_candidates": [],
                }
            ]
        }
    )


@pytest.fixture
def deps() -> dict[str, Any]:
    """Default per-scenario dependency bundle. Tests override the
    `provider` and `state` entries; the recorder sinks + resolver are
    shared scaffolding."""
    return {
        "provider": _StubLLMProvider(_build_finding_proposal_json()),
        "analyze_model": "claude-sonnet-4-6",
        "phase_event_sink": _RecordingPhaseEventSink(),
        "file_examination_sink": _RecordingFileExaminationSink(),
        "analyze_event_sink": _RecordingAnalyzeEventSink(),
        "import_path_resolver": MagicMock(),
        "active_policy_version": ACTIVE_POLICY_VERSION,
        "total_review_budget_tokens": DEFAULT_REVIEW_BUDGET_TOKENS,
    }


# ---------------------------------------------------------------------------
# Scenario 1 — clean file admits one finding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_file_admits_one_finding(deps: dict[str, Any]) -> None:
    """Tier=DEEP file with a JUDGED proposal whose span lands in a
    scope unit → 1 admitted finding in AnalysisRound + 1 FindingEvent
    emitted + AnalyzeCompletedEvent shows n_findings_emitted=1."""
    state = _build_review_state()
    result = await analyze(state, **deps)

    # State delta shape
    assert "analysis_rounds" in result
    assert "trace_candidates" in result
    assert len(result["analysis_rounds"]) == 1
    round_ = result["analysis_rounds"][0]
    assert len(round_.findings) == 1
    assert round_.files_examined == ("src/example.py",)
    assert round_.files_skipped == ()

    # Audit events
    phase_events = deps["phase_event_sink"].events
    assert len(phase_events) == 2
    assert phase_events[0].marker == "start"
    assert phase_events[1].marker == "end"
    assert phase_events[0].phase_id == phase_events[1].phase_id

    fe_events = deps["file_examination_sink"].events
    assert len(fe_events) == 1
    assert fe_events[0].parse_status == "clean"
    assert fe_events[0].skip_reason is None

    finding_events = deps["analyze_event_sink"].findings
    assert len(finding_events) == 1

    completed = deps["analyze_event_sink"].completed
    assert len(completed) == 1
    assert completed[0].n_findings_emitted == 1
    assert completed[0].n_proposals_seen == 1
    assert completed[0].n_proposals_rejected == 0
    assert completed[0].n_responses_rejected == 0
    assert completed[0].n_llm_calls == 1
    assert completed[0].n_files_analyzed == 1
    assert completed[0].n_files_skipped == 0

    # Provider was called exactly once
    assert len(deps["provider"].calls) == 1


# ---------------------------------------------------------------------------
# Scenario 2 — triage-skipped file (tier=SKIM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_skim_file_excluded_from_iteration(deps: dict[str, Any]) -> None:
    """Tier=SKIM file → NOT in iteration scope. No FileExaminationEvent
    fires for that file; the file does not appear in files_examined or
    files_skipped. Provider is NEVER called. AnalyzeCompletedEvent
    shows zero counters."""
    state = _build_review_state(
        triage_result=_build_triage_result(file_tiers={"src/example.py": ReviewTier.SKIM}),
    )
    result = await analyze(state, **deps)

    round_ = result["analysis_rounds"][0]
    assert round_.files_examined == ()
    assert round_.files_skipped == ()
    assert len(round_.findings) == 0

    # NO FileExaminationEvent fires for SKIM files
    fe_events = deps["file_examination_sink"].events
    assert len(fe_events) == 0

    # NO LLM call
    assert len(deps["provider"].calls) == 0

    # Completed event still fires with zero counters
    completed = deps["analyze_event_sink"].completed
    assert len(completed) == 1
    assert completed[0].n_files_analyzed == 0
    assert completed[0].n_files_skipped == 0
    assert completed[0].n_llm_calls == 0


# ---------------------------------------------------------------------------
# Scenario 3 — parser rejection (bogus finding_type)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parser_rejection_lifts_to_audit_event(deps: dict[str, Any]) -> None:
    """Tier=DEEP file, model returns a proposal with a finding_type
    NOT in the FindingType enum → parser rejects at finding_type_not_in_enum
    → ProposalRejection lifted to FindingProposalRejectedEvent. No
    admitted findings; AnalyzeCompletedEvent counters reflect."""
    deps["provider"] = _StubLLMProvider(
        _build_finding_proposal_json(finding_type="unknown_bogus_type")
    )
    state = _build_review_state()

    result = await analyze(state, **deps)

    round_ = result["analysis_rounds"][0]
    assert len(round_.findings) == 0

    pr_events = deps["analyze_event_sink"].proposal_rejections
    assert len(pr_events) == 1
    assert pr_events[0].rejection_reason == "finding_type_not_in_enum"

    # No admitted findings
    assert len(deps["analyze_event_sink"].findings) == 0

    completed = deps["analyze_event_sink"].completed
    assert completed[0].n_findings_emitted == 0
    assert completed[0].n_proposals_seen == 1
    assert completed[0].n_proposals_rejected == 1
    assert completed[0].n_responses_rejected == 0


# ---------------------------------------------------------------------------
# Scenario 4 — budget skip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exhausted_skips_file_without_llm_call(deps: dict[str, Any]) -> None:
    """Per-file budget cap = total_review_budget * 0.25. Set the budget
    so low that the prompt's estimated cost exceeds the cap → file
    rejected at cost gate with skip_reason=COST_BUDGET_EXHAUSTED → no
    LLM call → file in files_skipped."""
    # Tiny budget: per-file cap = budget * 0.25 = 25 tokens; the prompt
    # easily exceeds this.
    deps["total_review_budget_tokens"] = 100
    state = _build_review_state()

    result = await analyze(state, **deps)

    round_ = result["analysis_rounds"][0]
    assert round_.files_skipped == ("src/example.py",)
    assert round_.files_examined == ()
    assert len(round_.findings) == 0

    fe_events = deps["file_examination_sink"].events
    assert len(fe_events) == 1
    assert fe_events[0].parse_status == "skipped"
    assert fe_events[0].skip_reason == SkipReason.COST_BUDGET_EXHAUSTED

    # Provider was NEVER called
    assert len(deps["provider"].calls) == 0

    # Completed event counters
    completed = deps["analyze_event_sink"].completed
    assert completed[0].n_llm_calls == 0
    assert completed[0].n_files_analyzed == 0
    assert completed[0].n_files_skipped == 1


# ---------------------------------------------------------------------------
# Counter-source-of-truth + event-ordering pins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_event_counters_match_local_bookkeeping(deps: dict[str, Any]) -> None:
    """Spec §7 step 5 + commit-7 design: AnalyzeCompletedEvent counters
    come from per-file accumulators populated from ParserResult.counters
    — NEVER from re-reading the audit stream. Mixed scenario: 1 admitted
    + 1 rejected ensures the accounting equation holds at the producer
    side."""
    # Two proposals in one response: one admitted, one rejected.
    response_json = json.dumps(
        {
            "findings": [
                {
                    "finding_type": "sql_injection",
                    "evidence_tier": "judged",
                    "query_match_id": None,
                    "trace_path": None,
                    "title": "Admitted finding",
                    "description": "An admitted finding.",
                    "evidence": "def my_function():\n    return 42",
                    "span": {"byte_start": 0, "byte_end": 20},
                    "trace_candidates": [],
                },
                {
                    "finding_type": "definitely_not_a_real_enum_value",
                    "evidence_tier": "judged",
                    "query_match_id": None,
                    "trace_path": None,
                    "title": "Rejected finding",
                    "description": "A rejected finding.",
                    "evidence": "irrelevant",
                    "span": {"byte_start": 30, "byte_end": 50},
                    "trace_candidates": [],
                },
            ]
        }
    )
    deps["provider"] = _StubLLMProvider(response_json)
    state = _build_review_state()

    await analyze(state, **deps)

    completed = deps["analyze_event_sink"].completed[0]
    # Accounting equation
    assert completed.n_proposals_seen == (
        completed.n_findings_emitted + completed.n_proposals_rejected
    )
    assert completed.n_proposals_seen == 2
    assert completed.n_findings_emitted == 1
    assert completed.n_proposals_rejected == 1


@pytest.mark.asyncio
async def test_phase_events_bracket_the_pass(deps: dict[str, Any]) -> None:
    """`phase-events-bound-work`: start fires before any per-file work;
    end fires after all per-file work + completed event."""
    state = _build_review_state()

    await analyze(state, **deps)

    phase_events = deps["phase_event_sink"].events
    assert len(phase_events) == 2
    assert phase_events[0].marker == "start"
    assert phase_events[1].marker == "end"
    assert phase_events[0].node_id == "analyze"
    assert phase_events[1].node_id == "analyze"
    # Same phase_id
    assert phase_events[0].phase_id == phase_events[1].phase_id


@pytest.mark.asyncio
async def test_is_eval_propagates_through_emitted_events(deps: dict[str, Any]) -> None:
    """`is_eval` from `ReviewState` flows to every emitted event so
    eval runs produce eval-tagged audit rows."""
    state = _build_review_state(is_eval=True)

    await analyze(state, **deps)

    assert all(e.is_eval for e in deps["phase_event_sink"].events)
    assert all(e.is_eval for e in deps["file_examination_sink"].events)
    assert all(e.is_eval for e in deps["analyze_event_sink"].events)


# ---------------------------------------------------------------------------
# Configuration knob pins
# ---------------------------------------------------------------------------


def test_per_file_cap_fraction_is_quarter() -> None:
    """V1 fairness guard per FUP-044: one file can starve at most four
    others. Drift here changes the audit signal for cost-budget skips."""
    assert PER_FILE_CAP_FRACTION == 0.25


def test_default_review_budget_is_pinned() -> None:
    """Default lives in the module so node-test scenarios can reason
    about cost-gate behavior. Production wires a tighter value from
    settings; the default is the unit-test baseline."""
    assert DEFAULT_REVIEW_BUDGET_TOKENS == 200_000


def test_no_review_id_kwarg_in_signature() -> None:
    """`review_id` flows from `state.review_id` — never as a kwarg.
    Pin so a future refactor that drifts the contract surfaces here."""
    import inspect

    sig = inspect.signature(analyze)
    assert "review_id" not in sig.parameters
