"""Analyze node body tests — spec §7.

Covers file-outcome coverage, changed-region intersection, and
registry-query firing:

1. **Clean file** — tier=DEEP, parses cleanly, model returns admittable
   proposal → 1 admitted finding; FileExaminationEvent(parse_status="clean").
2. **Triage-skipped** — tier=SKIM → file NOT in iteration scope.
3. **Parser rejection** — tier=DEEP, model returns proposal that
   fails enum admission → ProposalRejection lifted.
4. **Budget skip** — tier=DEEP, estimated cost exceeds per-file cap.
5. **NO_REVIEWABLE_CONTEXT** — content_head and content_base both None.
6. **NO_CHANGED_SCOPE_UNITS** — clean parse, patch doesn't intersect
   any scope unit.
7. **Degraded mode (has_error in changed region)** — clean parse but
   the patched scope unit has `has_error=True`; degraded LLM call
   with `degradation_reason="tree_has_error_in_changed_regions"`.
8. **Changed-region intersection trims `included_scope_units`** — a
   file with two functions, patch only touches one → only that
   scope unit reaches the prompt + parser admission.
9. **Registry-query firing** — `query_match_id_set` constructed from
   `queries.registry.REGISTERED_QUERY_IDS` against the file content;
   non-empty for a typical Python file.

Test infrastructure: inline recorder sinks + mock LLM provider + mock
ImportPathResolver. Per the user direction: "without special mocks
beyond provider/context inputs" — all scenarios share the same
scaffolding; only the file content + tier map + provider response differ.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from outrider.agent.nodes.analyze import (
    DEFAULT_REVIEW_BUDGET_TOKENS,
    MAX_PER_FILE_TOKENS_ABSOLUTE,
    PER_FILE_CAP_FRACTION,
    _compute_per_file_cap,
    _estimate_tokens,
    _round_ended_at,
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
        CacheLookupEvent,
        FileExaminationEvent,
        FindingEvent,
        FindingProposalRejectedEvent,
        ReviewPhaseEvent,
        ScopeExclusionEvent,
    )
    from outrider.schemas.review_finding import ReviewFinding

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


def _lift_finding_event(finding: ReviewFinding, *, is_eval: bool) -> FindingEvent:
    """Lift an admitted ``ReviewFinding`` to its metadata-only ``FindingEvent``,
    mirroring ``AuditPersister._lift_finding_event`` so the recorder captures the
    same event the production sink would emit (keeps FindingEvent-field assertions
    such as ``finding_content_hash`` / ``is_eval`` valid under the new sink
    signature)."""
    from outrider.audit.events import FindingEvent

    return FindingEvent(
        review_id=finding.review_id,
        is_eval=is_eval,
        finding_id=finding.finding_id,
        finding_type=finding.finding_type,
        severity=finding.severity,
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        dimension=finding.dimension,
        finding_content_hash=finding.content_hash,
        evidence_tier=finding.evidence_tier,
        query_match_id=finding.query_match_id,
        trace_path=finding.trace_path,
        policy_version=finding.policy_version,
        proposal_hash=finding.proposal_hash,
    )


class _RecordingAnalyzeEventSink:
    """Captures every emission of the seven analyze-specific event
    types into per-type lists for assertion. The aggregate `events`
    list preserves emission order across types so tests can pin
    event-ordering invariants."""

    def __init__(self) -> None:
        self.findings: list[FindingEvent] = []
        self.proposal_rejections: list[FindingProposalRejectedEvent] = []
        self.response_rejections: list[AnalyzeResponseRejectedEvent] = []
        self.completed: list[AnalyzeCompletedEvent] = []
        self.scope_exclusions: list[ScopeExclusionEvent] = []
        self.cache_lookups: list[CacheLookupEvent] = []
        self.cache_serves: list[object] = []
        self.observed_skip_shadows: list[object] = []
        self.events: list[Any] = []

    async def emit_finding(self, finding: ReviewFinding, *, is_eval: bool) -> None:
        event = _lift_finding_event(finding, is_eval=is_eval)
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

    async def emit_scope_exclusion(self, event: ScopeExclusionEvent) -> None:
        self.scope_exclusions.append(event)
        self.events.append(event)

    async def emit_cache_lookup(self, event: CacheLookupEvent) -> None:
        self.cache_lookups.append(event)
        self.events.append(event)

    async def emit_cache_serve(self, event: object) -> None:
        self.cache_serves.append(event)
        self.events.append(event)

    async def emit_observed_skip_shadow(self, event: object) -> None:
        self.observed_skip_shadows.append(event)
        self.events.append(event)


class _StubLLMProvider:
    """Returns a canned JSON response per call. Tests configure the
    response text; the provider doesn't enforce LLMRequest's full
    contract beyond what's needed for the wrapper's internal validation
    of return type."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[LLMRequest] = []

    async def aclose(self) -> None:
        return None

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

# Default unidiff-format patch for `_SIMPLE_PY`. The hunk modifies
# `my_function` (1 context line + 1 added line = source count 1,
# target count 2). The added line lands at target line 2 which is
# inside `my_function` (lines 1-2); `another_function` (lines 4-6)
# does not intersect, so changed-region intersection includes only
# `my_function`.
_DEFAULT_PATCH_TEMPLATE = (
    "--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,2 @@\n def my_function():\n+    return 42\n"
)


def _build_changed_file(
    *,
    path: str = "src/example.py",
    content: bytes = _SIMPLE_PY,
    patch: str | None = None,
    content_head: str | None = "__default__",
    content_base: str | None = "def my_function():\n    return 0\n",
) -> ChangedFile:
    """Construct a `ChangedFile` for analyze-node tests.

    `patch=None` defaults to a valid unidiff-format patch keyed off
    the given `path`; pass a string to override (e.g., for a
    NO_CHANGED_SCOPE_UNITS test that targets non-scope-unit lines).
    Pass `patch=""` to test the no-patch case.

    `content_head` defaults to `content.decode("utf-8")` via the
    sentinel string `"__default__"`; pass `None` explicitly to
    suppress content_head (e.g., to construct a binary-file case
    paired with `content_base=None`).
    """
    if patch is None:
        patch = _DEFAULT_PATCH_TEMPLATE.format(path=path)
    head: str | None = content.decode("utf-8") if content_head == "__default__" else content_head
    return ChangedFile(
        path=path,
        status="modified",
        additions=2,
        deletions=0,
        patch=patch,
        content_base=content_base,
        content_head=head,
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
    line_start: int = 1,
    line_end: int = 2,
) -> str:
    """Build a JSON `AnalyzeResponseRaw` payload with one proposal. Default line
    range 1-2 lands in `my_function` (the changed scope unit in `_SIMPLE_PY`)."""
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
                    "line_start": line_start,
                    "line_end": line_end,
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
        "standard_analyze_model": "claude-sonnet-4-6",
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
# Scenario 4b — bounded high-risk reserve (specs/2026-06-17-analyze-cost-fairness.md
# Stage 1). With `_estimate_tokens` monkeypatched to 0, every file's estimate is
# exactly `analyze_prompt.MAX_TOKENS` (8192 = E), so budgets are deterministic.
# ---------------------------------------------------------------------------

_E_TOKENS = 8192  # analyze_prompt.MAX_TOKENS — the per-file estimate when _estimate_tokens→0

# A high-risk file: its added patch line carries `os.system(...)`, which
# policy.recall flags → eligible for the reserve. content_head matches so the
# changed scope (`my_function`) parses and triggers an LLM call like the default.
_HIGH_RISK_CONTENT = b"def my_function():\n    os.system(cmd)\n"
_HIGH_RISK_PATCH_TEMPLATE = (
    "--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,2 @@\n def my_function():\n+    os.system(cmd)\n"
)


def _build_high_risk_file(path: str) -> ChangedFile:
    return _build_changed_file(
        path=path,
        content=_HIGH_RISK_CONTENT,
        patch=_HIGH_RISK_PATCH_TEMPLATE.format(path=path),
    )


# Budget regime for the reserve tests: budget = 4E.
#   per_file_cap = min(int(4E * 0.25), 60000) = E  → never the binding constraint
#   general      = 4E - reserved = 3E              → fits exactly the first 3 files
#   reserved     = int(4E * 0.25) = E              → covers exactly ONE high-risk file
# Crucially, a SINGLE pool (the pre-Stage-1 baseline) of 4E covers exactly 4 files,
# so a 5th file starves UNLESS the reserve admits it. This makes the rescue/bound
# tests load-bearing under the revert-the-fold check: delete the reserve and the
# high-risk file at position 5 skips, flipping the asserted outcome.
_RESERVE_TEST_BUDGET = 4 * _E_TOKENS


@pytest.mark.asyncio
async def test_reserve_rescues_high_risk_file_under_budget_pressure(
    deps: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A high-risk file (blatant signature) LATE in iteration is analyzed via the
    reserve, jumping ahead of a benign file that already drained the general pool —
    the deterministic fix for the verified PR #8 starvation.

    Load-bearing: 4 benign files exhaust a single 4E pool, so under the pre-Stage-1
    single pool the high-risk 5th file would skip. It survives ONLY because of the
    reserve — and a benign file at position 4 (app/d.py) starves in the SAME run,
    which is the re-prioritization the reserve exists to produce."""
    monkeypatch.setattr("outrider.agent.nodes.analyze._estimate_tokens", lambda _text: 0)
    deps["total_review_budget_tokens"] = _RESERVE_TEST_BUDGET
    paths = ["app/a.py", "app/b.py", "app/c.py", "app/d.py", "app/danger.py"]
    changed = (
        _build_changed_file(path=paths[0]),
        _build_changed_file(path=paths[1]),
        _build_changed_file(path=paths[2]),
        _build_changed_file(path=paths[3]),
        _build_high_risk_file(paths[4]),
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=changed),
        triage_result=_build_triage_result(file_tiers=dict.fromkeys(paths, ReviewTier.DEEP)),
    )

    result = await analyze(state, **deps)
    round_ = result["analysis_rounds"][0]

    # General drained by the first 3 benign files; app/d.py (benign, position 4)
    # starves, yet the high-risk file at position 5 is rescued by the reserve.
    assert "app/danger.py" in round_.files_examined
    assert "app/d.py" in round_.files_skipped
    assert len(deps["provider"].calls) == 4


@pytest.mark.asyncio
async def test_benign_file_starves_where_high_risk_would_be_rescued(
    deps: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A/B control for the rescue test: SAME budget and SAME 5th position, but the
    5th file is BENIGN (no signature) → it starves COST_BUDGET_EXHAUSTED. The ONLY
    difference from the rescue case is the signature in the patch, so the flipped
    outcome (examined → skipped) isolates the reserve's effect to policy.recall."""
    monkeypatch.setattr("outrider.agent.nodes.analyze._estimate_tokens", lambda _text: 0)
    deps["total_review_budget_tokens"] = _RESERVE_TEST_BUDGET
    paths = ["app/a.py", "app/b.py", "app/c.py", "app/d.py", "app/benign.py"]
    changed = tuple(_build_changed_file(path=p) for p in paths)
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=changed),
        triage_result=_build_triage_result(file_tiers=dict.fromkeys(paths, ReviewTier.DEEP)),
    )

    result = await analyze(state, **deps)
    round_ = result["analysis_rounds"][0]

    # Benign 5th file gets no reserve → starves where the high-risk file was rescued.
    assert "app/benign.py" in round_.files_skipped
    assert len(deps["provider"].calls) == 3  # only the first 3 (general pool) fired


@pytest.mark.asyncio
async def test_reserve_is_bounded_second_high_risk_overflows(
    deps: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reserve is BOUNDED. With general drained by 4 benign files, the reserve
    (= E) covers exactly ONE high-risk file; a second high-risk file overflows it,
    falls back to the exhausted general pool, and skips — proving it is not an
    unlimited bypass.

    Load-bearing: under the pre-Stage-1 single 4E pool the four benign files
    exhaust the budget, so BOTH high-risk files would skip. The reserve is what
    admits the first one (`h1`); the second (`h2`) skipping is the bound."""
    monkeypatch.setattr("outrider.agent.nodes.analyze._estimate_tokens", lambda _text: 0)
    deps["total_review_budget_tokens"] = _RESERVE_TEST_BUDGET
    paths = ["app/a.py", "app/b.py", "app/c.py", "app/d.py", "app/h1.py", "app/h2.py"]
    changed = (
        _build_changed_file(path=paths[0]),
        _build_changed_file(path=paths[1]),
        _build_changed_file(path=paths[2]),
        _build_changed_file(path=paths[3]),
        _build_high_risk_file(paths[4]),
        _build_high_risk_file(paths[5]),
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=changed),
        triage_result=_build_triage_result(file_tiers=dict.fromkeys(paths, ReviewTier.DEEP)),
    )

    result = await analyze(state, **deps)
    round_ = result["analysis_rounds"][0]

    assert "app/h1.py" in round_.files_examined  # reserve admits the first high-risk
    assert "app/h2.py" in round_.files_skipped  # bounded: the second overflows


@pytest.mark.asyncio
async def test_reserve_not_wasted_when_high_risk_iterates_early(
    deps: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reserved-then-general contract: a high-risk file EARLY in iteration spends
    its dedicated reserve first, leaving the general pool intact for benign files.

    This is the guard against a general-FIRST rule (which the first cut used): under
    general-first the early high-risk file would burn general, leave the reserve
    unused, and starve a benign file the reserve could have covered — lower
    throughput. Order [high-risk, benign x4] @ 4E (general=3E, reserved=E):
    reserved-then-general examines danger + 3 benign (the reserve frees general),
    skipping only app/d.py (throughput 4); general-first would examine only danger
    + 2 benign, skipping app/c.py too (throughput 3, reserve wasted). The
    `app/c.py examined` + `calls == 4` assertions fail under general-first."""
    monkeypatch.setattr("outrider.agent.nodes.analyze._estimate_tokens", lambda _text: 0)
    deps["total_review_budget_tokens"] = _RESERVE_TEST_BUDGET
    paths = ["app/danger.py", "app/a.py", "app/b.py", "app/c.py", "app/d.py"]
    changed = (
        _build_high_risk_file(paths[0]),
        _build_changed_file(path=paths[1]),
        _build_changed_file(path=paths[2]),
        _build_changed_file(path=paths[3]),
        _build_changed_file(path=paths[4]),
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=changed),
        triage_result=_build_triage_result(file_tiers=dict.fromkeys(paths, ReviewTier.DEEP)),
    )

    result = await analyze(state, **deps)
    round_ = result["analysis_rounds"][0]

    assert "app/danger.py" in round_.files_examined  # high-risk drew the reserve
    assert "app/c.py" in round_.files_examined  # general intact → 3rd benign survives
    assert "app/d.py" in round_.files_skipped  # only the 4th benign starves
    assert len(deps["provider"].calls) == 4


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
                    "line_start": 1,
                    "line_end": 2,
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
                    "line_start": 1,
                    "line_end": 2,
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
    eval runs produce eval-tagged audit rows. Non-empty length asserts
    guard `all(...)` from passing vacuously when an event list is empty."""
    state = _build_review_state(is_eval=True)

    await analyze(state, **deps)

    phase_events = deps["phase_event_sink"].events
    fe_events = deps["file_examination_sink"].events
    analyze_events = deps["analyze_event_sink"].events
    assert len(phase_events) > 0
    assert len(fe_events) > 0
    assert len(analyze_events) > 0
    assert all(e.is_eval for e in phase_events)
    assert all(e.is_eval for e in fe_events)
    assert all(e.is_eval for e in analyze_events)


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


def test_max_per_file_tokens_absolute_is_pinned() -> None:
    """Absolute ceiling on per-file pre-flight token estimate. Decouples
    the per-file cap from caller-configurable budget — a 20× budget
    inflation can't drag the per-file cap into Sonnet-call-overflow
    territory. Drift here changes the audit signal for cost gates."""
    assert MAX_PER_FILE_TOKENS_ABSOLUTE == 60_000


def test_compute_per_file_cap_fraction_drives_at_default_budget() -> None:
    """At the default budget (200K tokens), the fraction (0.25) is the
    binding constraint: 200K * 0.25 = 50K < 60K absolute ceiling. The
    helper picks the tighter of the two."""
    assert _compute_per_file_cap(DEFAULT_REVIEW_BUDGET_TOKENS) == 50_000


def test_compute_per_file_cap_absolute_ceiling_clamps_inflated_budget() -> None:
    """At 1M budget, the fraction would yield 250K but the absolute
    ceiling clamps to 60K. The min() picks the tighter constraint.
    Pins the clamp value directly so a future drift in either ceiling
    surfaces here without needing the full cost-gate flow."""
    assert _compute_per_file_cap(1_000_000) == MAX_PER_FILE_TOKENS_ABSOLUTE


def test_compute_per_file_cap_tiny_budget_yields_tiny_cap() -> None:
    """Tiny budgets: the fraction still drives. 100 tokens * 0.25 = 25.
    The fraction-truncating `int(...)` is the runtime; pinning the
    behavior catches accidental round-vs-truncate changes."""
    assert _compute_per_file_cap(100) == 25


def test_compute_per_file_cap_tie_point_picks_absolute() -> None:
    """At the exact tie (budget × 0.25 == MAX_PER_FILE_TOKENS_ABSOLUTE,
    i.e., budget=240_000 → both ceilings yield 60_000), `min(a, b)`
    returns either value since they're equal — the observable result is
    60_000 either way. Pin so a future refactor to a branched
    `if fraction > ABSOLUTE: ...` doesn't accidentally pick the wrong
    branch at the tie point and drift to off-by-one."""
    tie_budget = MAX_PER_FILE_TOKENS_ABSOLUTE * 4  # budget where 0.25*budget == ABSOLUTE
    assert tie_budget == 240_000  # sanity: pins the arithmetic relationship
    assert _compute_per_file_cap(tie_budget) == MAX_PER_FILE_TOKENS_ABSOLUTE


def test_compute_per_file_cap_zero_budget_is_kill_switch() -> None:
    """Budget = 0 → cap = 0. Every prompt's `estimated_tokens > 0`
    (since `analyze_prompt.MAX_TOKENS` is positive), so every file
    skips with `COST_BUDGET_EXHAUSTED`. This is a kill-switch by
    construction — not a bug. Pin so the kill-switch semantics are
    documented as part of the contract, not just emergent."""
    assert _compute_per_file_cap(0) == 0


def test_compute_per_file_cap_negative_budget_is_kill_switch() -> None:
    """Negative budget produces a negative cap (mathematically valid
    but operationally a stricter kill-switch — `estimated_tokens > -N`
    is always True). Pin the documented contract: invalid/sentinel
    budget values produce the safe outcome (skip everything), not a
    permissive cap that admits LLM calls under bogus configuration."""
    assert _compute_per_file_cap(-1) == 0  # int(-1 * 0.25) = 0 (Python int truncates toward zero)
    # At very-negative budgets the fraction goes negative AND min picks it.
    assert _compute_per_file_cap(-100) == -25


def test_estimate_tokens_counts_bytes_not_codepoints() -> None:
    """Anthropic's BPE tokenizer operates on UTF-8 bytes. Counting
    Python codepoints (`len(str)`) under-counts multi-byte sequences:
    `len("中") == 1` codepoint but `"中".encode("utf-8") == 3` bytes.

    The fix: `_estimate_tokens` encodes to UTF-8 first, then counts.
    Verified against fixtures the prior implementation would have
    silently under-counted.
    """
    # ASCII baseline: 1 byte → ceiling(1/3) = 1 token.
    assert _estimate_tokens("a") == 1
    # CJK character: 3 bytes → ceiling(3/3) = 1 token.
    # Prior (codepoint-counting) impl would have returned `1 // 3 == 0`.
    assert _estimate_tokens("中") == 1
    # Emoji: 4 bytes → ceiling(4/3) = 2 tokens.
    # Prior impl would have returned `1 // 3 == 0`.
    assert _estimate_tokens("🎉") == 2
    # Pure-ASCII over-cap: 6 bytes → ceiling(6/3) = 2 tokens.
    assert _estimate_tokens("abcdef") == 2


def test_estimate_tokens_rounds_up_not_down() -> None:
    """The cost gate is a pre-flight safety guard; ceiling-division
    is conservative-up (over-estimates rather than under-estimates).
    Catch any future refactor that flips back to floor-division."""
    # 4 bytes → ceiling(4/3) = 2 tokens. Floor-div would yield 1.
    assert _estimate_tokens("abcd") == 2
    # 2 bytes → ceiling(2/3) = 1 token. Floor-div would yield 0.
    assert _estimate_tokens("ab") == 1
    # Edge: 0 bytes → 0 tokens (no overshoot on empty).
    assert _estimate_tokens("") == 0


@pytest.mark.asyncio
async def test_inflated_budget_does_not_lift_per_file_cap_past_absolute(
    deps: dict[str, Any],
) -> None:
    """Caller passes a 20× budget. `PER_FILE_CAP_FRACTION` would lift the
    per-file cap to 1M tokens; the absolute ceiling clamps it at 60K. A
    prompt below 60K passes; one above would fail. We can't engineer a
    >60K prompt in unit test, but we can verify the cap is computed via
    `min(fraction*budget, ABSOLUTE)` by setting a budget where the
    fraction would exceed ABSOLUTE and asserting the LLM call still
    fires (prompt is well under 60K)."""
    # 1M budget → fraction*budget = 250K. min(250K, 60K) = 60K. The
    # default prompt is well under 60K, so the LLM call fires (the
    # absolute-ceiling clamp prevents over-permissive cap).
    deps["total_review_budget_tokens"] = 1_000_000
    state = _build_review_state()

    await analyze(state, **deps)

    # Provider was called — confirms cost gate did not block.
    assert len(deps["provider"].calls) == 1


def test_no_review_id_kwarg_in_signature() -> None:
    """`review_id` flows from `state.review_id` — never as a kwarg.
    Pin so a future refactor that drifts the contract surfaces here."""
    import inspect

    sig = inspect.signature(analyze)
    assert "review_id" not in sig.parameters


# ---------------------------------------------------------------------------
# Second-landing outcomes: NO_REVIEWABLE_CONTEXT, NO_CHANGED_SCOPE_UNITS,
# degraded mode, changed-region intersection, registry-query firing.
# ---------------------------------------------------------------------------


# NO_REVIEWABLE_CONTEXT in V1 is unreachable from a valid `ChangedFile`:
# the schema's `enforce_status_invariants` validator rejects every status
# with missing content, and `parse_python` only returns
# `parser_outcome="failed"` on UTF-8 decode failure — which cannot fire
# from `str.encode("utf-8")`. The analyze branch is retained as
# defensive code for spec compliance + future schema relaxation; a unit
# test would require mocking `parse_python` (brittle) or a custom
# `ChangedFile` validator bypass (worse). Integration coverage lands
# alongside binary-file handling whenever it migrates into analyze.
# FUP-053 tracks the raw-bytes intake path that would make
# `failed+degraded_llm` reachable.


def test_str_to_utf8_roundtrip_cannot_produce_invalid_utf8() -> None:
    """Pin the upstream gate that makes analyze's `failed+degraded_llm`
    outcome V1-unreachable: any Python `str`, re-encoded with UTF-8,
    decodes back via strict UTF-8 — which is exactly the gate
    `parse_python` step 2 runs.

    If this property ever breaks (e.g., the language adds a string
    flavor that doesn't round-trip), the analyze module docstring's
    'V1 unreachable' note needs revisiting and FUP-053 may already
    have a reachable trigger.
    """
    samples = [
        "",
        "ascii",
        "café",  # multibyte
        "𝕳𝖊𝖑𝖑𝖔",  # 4-byte UTF-8 chars
        "\x00 inline null",  # NUL in str is fine; bytes form is also valid UTF-8
        "mixed \n\t\r whitespace",
    ]
    for s in samples:
        # Round-trip must succeed (no UnicodeDecodeError).
        roundtripped = s.encode("utf-8").decode("utf-8")
        assert roundtripped == s


@pytest.mark.asyncio
async def test_non_python_file_routed_to_skip_without_calling_provider(
    deps: dict[str, Any],
) -> None:
    """V1 language gate: a non-`.py` file classified DEEP/STANDARD by
    triage must NOT reach `parse_python` or the Python query registry.
    Routes through `SkipReason.UNSUPPORTED_LANGUAGE` per
    `DECISIONS.md#018` Amended 2026-05-21. Pin: no LLM call, no
    FindingEvent, and a single skip-shaped FileExaminationEvent.
    """
    js_file = _build_changed_file(
        path="src/example.js",
        content=b"export function hello() {\n  return 42;\n}\n",
        patch=(
            "--- a/src/example.js\n+++ b/src/example.js\n@@ -1,1 +1,2 @@\n"
            " export function hello() {\n+  return 42;\n"
        ),
        content_base="export function hello() {\n  return 0;\n}\n",
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(js_file,)),
        triage_result=TriageResult(
            file_tiers={"src/example.js": ReviewTier.DEEP},
            overall_risk=RiskLevel.MEDIUM,
            relevant_dimensions=(ReviewDimension.CODE_QUALITY,),
            reasoning="js file forced through analyze for the language-gate test",
        ),
    )

    await analyze(state, **deps)

    # No LLM call: the gate fires before content selection / parse.
    assert deps["provider"].calls == []
    # No findings (the file never reached the parser).
    assert deps["analyze_event_sink"].findings == []
    # One skip-shaped FileExaminationEvent for the non-Python file.
    skip_events = [e for e in deps["file_examination_sink"].events if e.parse_status == "skipped"]
    assert len(skip_events) == 1
    assert skip_events[0].file_path == "src/example.js"
    assert skip_events[0].skip_reason == SkipReason.UNSUPPORTED_LANGUAGE


@pytest.mark.asyncio
async def test_no_changed_scope_units_when_patch_targets_outside_scopes(
    deps: dict[str, Any],
) -> None:
    """Clean parse + patch whose target lines fall outside every scope
    unit's line range → `skipped+NO_CHANGED_SCOPE_UNITS`. Example:
    `_SIMPLE_PY` has functions at lines 1-2 and 4-6; a patch targeting
    line 8 (past the file end, just a comment append) intersects nothing.
    """
    # Patch adds a trailing comment line at target line 8. `_SIMPLE_PY`
    # is 6 lines; the comment lands after both functions, outside any
    # scope unit's line range.
    extended_content = _SIMPLE_PY + b"# trailing comment\n"
    out_of_scope_patch = (
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -6,1 +6,2 @@\n"
        "     return y\n"
        "+# trailing comment\n"
    )
    cf = _build_changed_file(content=extended_content, patch=out_of_scope_patch)
    state = _build_review_state(pr_context=_build_pr_context(changed_files=(cf,)))

    await analyze(state, **deps)

    fe_events = deps["file_examination_sink"].events
    assert len(fe_events) == 1
    assert fe_events[0].skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS
    assert len(deps["provider"].calls) == 0


@pytest.mark.asyncio
async def test_no_patch_clean_parse_skips_as_no_changed_scope_units(
    deps: dict[str, Any],
) -> None:
    """Clean parse + `patch=None` (binary content GitHub didn't ship
    a diff for, or an oversized response) → no patched_file → no
    intersection → `NO_CHANGED_SCOPE_UNITS`."""
    cf = _build_changed_file(patch="")
    state = _build_review_state(pr_context=_build_pr_context(changed_files=(cf,)))

    await analyze(state, **deps)

    fe_events = deps["file_examination_sink"].events
    assert len(fe_events) == 1
    assert fe_events[0].skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS
    assert len(deps["provider"].calls) == 0


@pytest.mark.asyncio
async def test_parser_skipped_file_skips_before_patch_lookup(
    deps: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parser-skipped file (VENDORED `vendor/*` path) is skipped BEFORE
    `lookup_patched_file` runs.

    Regression for the degradation-decision extraction: `lookup_patched_file`
    RAISES `CoordinateError` on a malformed/duplicate patch (it returns None only
    for the absent cases), and a skipped file may carry such a patch (the skip is
    path/size-keyed, independent of patch validity). Calling it for a skipped file
    would crash the whole pass (no try/except per the module's provider-failure
    policy). Here `lookup_patched_file` is stubbed to raise unconditionally; the
    test passes only because the parser-skip return fires first — `analyze` must
    not raise, and the file skips cleanly with `VENDORED`."""
    import outrider.agent.nodes.analyze as analyze_mod
    from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind

    def _raise_malformed(*_args: object, **_kwargs: object) -> object:
        raise CoordinateError("stub: malformed patch", kind=CoordinateErrorKind.MALFORMED_PATCH)

    monkeypatch.setattr(analyze_mod, "lookup_patched_file", _raise_malformed)

    vendored = _build_changed_file(path="vendor/lib.py")
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(vendored,)),
        triage_result=_build_triage_result(file_tiers={"vendor/lib.py": ReviewTier.DEEP}),
    )

    result = await analyze(state, **deps)  # must NOT raise

    round_ = result["analysis_rounds"][0]
    assert round_.files_skipped == ("vendor/lib.py",)
    fe_events = deps["file_examination_sink"].events
    assert len(fe_events) == 1
    assert fe_events[0].parse_status == "skipped"
    assert fe_events[0].skip_reason == SkipReason.VENDORED
    assert len(deps["provider"].calls) == 0


@pytest.mark.asyncio
async def test_changed_region_intersection_includes_only_intersecting_unit(
    deps: dict[str, Any],
) -> None:
    """`_SIMPLE_PY` has `my_function` (lines 1-2) and `another_function`
    (lines 4-6). The default patch only touches lines 1-2 →
    `included_scope_units` contains only `my_function`. Verified via the
    `LLMRequest.context_summary` payload: one entry for `my_function`,
    zero for `another_function`."""
    state = _build_review_state()

    await analyze(state, **deps)

    request = deps["provider"].calls[0]
    summary_names = {entry.scope_unit_name for entry in request.context_summary}
    assert "my_function" in summary_names
    assert "another_function" not in summary_names
    # Inclusion reason for the intersected unit is `changed_scope`.
    assert all(entry.inclusion_reason == "changed_scope" for entry in request.context_summary)


@pytest.mark.asyncio
async def test_registry_query_firing_populates_query_match_id_list(
    deps: dict[str, Any],
) -> None:
    """For a Python file with function definitions, `query_match_id_set`
    contains `python.function_definition`. The registry-query block lives
    in `user_prompt` (per-file content; the system prompt is the
    cross-file stable cache prefix per the analyze-v4 repartition)."""
    state = _build_review_state()

    await analyze(state, **deps)

    request = deps["provider"].calls[0]
    assert "python.function_definition" in request.user_prompt
    assert "(no registry query matches" not in request.user_prompt


@pytest.mark.asyncio
async def test_observed_proposal_with_registered_query_id_admits(
    deps: dict[str, Any],
) -> None:
    """A model proposal claiming `evidence_tier=observed` with a
    `query_match_id` that's actually in the file's registry-fired set
    → admitted through the parser's OBSERVED admission step."""
    response_json = json.dumps(
        {
            "findings": [
                {
                    "finding_type": "sql_injection",
                    "evidence_tier": "observed",
                    "query_match_id": "python.function_definition",
                    "trace_path": None,
                    "title": "Admitted OBSERVED finding",
                    "description": "Tied to a real registry match.",
                    "evidence": "def my_function():\n    return 42",
                    "line_start": 1,
                    "line_end": 2,
                    "trace_candidates": [],
                }
            ]
        }
    )
    deps["provider"] = _StubLLMProvider(response_json)
    state = _build_review_state()

    result = await analyze(state, **deps)

    round_ = result["analysis_rounds"][0]
    assert len(round_.findings) == 1
    assert round_.findings[0].evidence_tier == "observed"
    assert round_.findings[0].query_match_id == "python.function_definition"
    assert len(deps["analyze_event_sink"].proposal_rejections) == 0


@pytest.mark.asyncio
async def test_observed_proposal_with_unregistered_query_id_rejects(
    deps: dict[str, Any],
) -> None:
    """A model claim of `evidence_tier=observed` with a `query_match_id`
    NOT in the file's registry-fired set rejects with
    `query_match_id_not_in_registry` — the proof-boundary defense
    against fabricated structural evidence."""
    response_json = json.dumps(
        {
            "findings": [
                {
                    "finding_type": "sql_injection",
                    "evidence_tier": "observed",
                    "query_match_id": "python.fabricated_pattern",
                    "trace_path": None,
                    "title": "Fabricated OBSERVED claim",
                    "description": "Cites an id that isn't in the registry.",
                    "evidence": "irrelevant",
                    "line_start": 1,
                    "line_end": 2,
                    "trace_candidates": [],
                }
            ]
        }
    )
    deps["provider"] = _StubLLMProvider(response_json)
    state = _build_review_state()

    result = await analyze(state, **deps)

    round_ = result["analysis_rounds"][0]
    assert len(round_.findings) == 0
    rejections = deps["analyze_event_sink"].proposal_rejections
    assert len(rejections) == 1
    assert rejections[0].rejection_reason == "query_match_id_not_in_registry"


@pytest.mark.asyncio
async def test_degraded_path_when_has_error_in_changed_scope_unit(
    deps: dict[str, Any],
) -> None:
    """A file whose changed scope unit carries `has_error=True` (tree-
    sitter ERROR node inside the function) routes through degraded mode
    with `degradation_reason="tree_has_error_in_changed_regions"`.
    `LLMRequest.degraded_mode=True`, `context_summary=()`, and the
    user_prompt uses the degraded template (mentions `DEGRADED`).
    """
    # Engineered content: `my_function` has an incomplete `if` statement
    # that tree-sitter parses as a top-level function with an ERROR
    # sub-node inside its body.
    broken_content = b"def my_function():\n    if\n    return 42\n"
    broken_patch = (
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1,1 +1,3 @@\n"
        " def my_function():\n"
        "+    if\n"
        "+    return 42\n"
    )
    cf = _build_changed_file(content=broken_content, patch=broken_patch)
    state = _build_review_state(pr_context=_build_pr_context(changed_files=(cf,)))

    await analyze(state, **deps)

    fe_events = deps["file_examination_sink"].events
    # FileExaminationEvent.parse_status="degraded" per spec §7 step 3e
    # for the degraded+degraded_llm outcome.
    assert len(fe_events) == 1
    assert fe_events[0].parse_status == "degraded"

    request = deps["provider"].calls[0]
    assert request.degraded_mode is True
    assert request.degradation_reason == "tree_has_error_in_changed_regions"
    # Degraded request has empty context_summary per spec §7 step 3f.
    assert request.context_summary == ()
    # The user_prompt uses the degraded template signal.
    assert "DEGRADED" in request.user_prompt


# ---------------------------------------------------------------------------
# Aggregate-accounting regression pins
# ---------------------------------------------------------------------------
#
# Two focused tests pin the producer-side aggregate accounting on
# `AnalyzeCompletedEvent`: cache-token split (reads ≠ writes in the
# event row, the 12.5× pricing differential motivated the split) AND
# Decimal cost accumulation (`total_cost_usd` is `float(sum_of_Decimals)`,
# not `sum_of_floats`).
#
# These exercise the node's per-pass aggregation directly — they
# deliberately do NOT route through the graph builder or test the
# multi-node wiring; that path is covered in `test_analyze_graph_wiring.py`.
# Aggregate-accounting drift would otherwise pass the existing tests
# because the graph-wiring tests use mocks returning zero-valued cache
# tokens and a single LLM call (so the split + multi-call sum paths
# never fire).


class _ConfigurableTokensStubProvider:
    """`LLMProvider` stub returning a configurable token-count response.

    Two configurations:
      - `tokens_per_call`: a single dict applied to every call.
      - `token_specs`: a list of dicts, one per expected call, returned
        in order. Raises `IndexError` if calls exceed the list length —
        a fixture overrun is a test bug.

    Per-call captures live on `self.calls` for assertion. Used only by
    the two aggregate-accounting regression tests; `_StubLLMProvider`
    above remains the default scaffolding for outcome tests.
    """

    def __init__(
        self,
        *,
        response_text: str,
        tokens_per_call: dict[str, int] | None = None,
        token_specs: list[dict[str, int]] | None = None,
    ) -> None:
        if (tokens_per_call is None) == (token_specs is None):
            msg = "exactly one of tokens_per_call / token_specs must be set"
            raise ValueError(msg)
        self._text = response_text
        self._tokens_per_call = tokens_per_call
        self._token_specs = list(token_specs) if token_specs is not None else None
        self.calls: list[LLMRequest] = []

    async def aclose(self) -> None:
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self._token_specs is not None:
            spec = self._token_specs.pop(0)
        else:
            assert self._tokens_per_call is not None  # narrowing for mypy
            spec = self._tokens_per_call
        return LLMResponse(
            text=self._text,
            model=request.model,
            input_tokens=spec["input_tokens"],
            output_tokens=spec["output_tokens"],
            cache_read_tokens=spec["cache_read_tokens"],
            cache_write_tokens=spec["cache_write_tokens"],
            finish_reason="end_turn",
            latency_ms=250,
        )


@pytest.mark.asyncio
async def test_aggregate_cache_tokens_keep_read_and_write_distinct(
    deps: dict[str, Any],
) -> None:
    """The aggregate event splits cache reads and cache writes into
    separate columns because the 12.5× pricing differential makes the
    lumped field uninformative for cost analysis (cache_write 1.25×
    base, cache_read 0.1× base). This regression pin would catch a
    drift that re-lumped them into a single `total_cached_tokens`
    field.

    Uses values where reads ≠ writes AND both are non-zero — the
    existing graph-wiring tests' mocks return zero for both, so the
    split path is uncovered there.
    """
    cache_read = 700
    cache_write = 100
    assert cache_read != cache_write  # fixture must use distinct values

    deps["provider"] = _ConfigurableTokensStubProvider(
        response_text=_build_finding_proposal_json(),
        tokens_per_call={
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
        },
    )
    state = _build_review_state()

    await analyze(state, **deps)

    completed = deps["analyze_event_sink"].completed
    assert len(completed) == 1
    assert completed[0].total_cache_read_tokens == cache_read
    assert completed[0].total_cache_write_tokens == cache_write


@pytest.mark.asyncio
async def test_total_cost_usd_is_decimal_sum_then_float_cast(deps: dict[str, Any]) -> None:
    """`AnalyzeCompletedEvent.total_cost_usd == float(sum_of_per_call_Decimals)`.

    Producer contract: each per-file cost comes back from
    `compute_cost_usd` as `Decimal`; the main loop accumulates Decimals
    across files and casts to `float` ONCE at AnalyzeCompletedEvent
    construction. The prior shape (per-file `float(Decimal)` cast +
    float-sum) drifted at FP noise (~5e-17 USD per 50 files), and
    `LLMCallEvent.cost_usd` sum on replay didn't match the aggregate's
    self-reported `total_cost_usd`.

    Test design: three DEEP files with DIFFERENT token-count tuples per
    call (so per-call cost Decimals are distinct non-trivial values).
    Compute the expected aggregate via the same `compute_cost_usd` +
    Decimal-sum + float-cast path; assert the event's `total_cost_usd`
    matches exactly. A regression to float-sum-per-call would surface
    as inequality at FP precision.
    """
    paths = ("src/a.py", "src/b.py", "src/c.py")
    token_specs: list[dict[str, int]] = [
        {
            "input_tokens": 1234,
            "output_tokens": 567,
            "cache_read_tokens": 89,
            "cache_write_tokens": 12,
        },
        {
            "input_tokens": 2345,
            "output_tokens": 678,
            "cache_read_tokens": 90,
            "cache_write_tokens": 23,
        },
        {
            "input_tokens": 3456,
            "output_tokens": 789,
            "cache_read_tokens": 11,
            "cache_write_tokens": 34,
        },
    ]
    assert len(paths) == len(token_specs)

    changed_files = tuple(_build_changed_file(path=p) for p in paths)
    pr_context = _build_pr_context(changed_files=changed_files)
    triage_result = _build_triage_result(
        file_tiers=dict.fromkeys(paths, ReviewTier.DEEP),
    )
    state = _build_review_state(pr_context=pr_context, triage_result=triage_result)

    provider = _ConfigurableTokensStubProvider(
        response_text=_build_finding_proposal_json(),
        token_specs=token_specs,
    )

    from outrider.llm.pricing import compute_cost_usd

    model = "claude-sonnet-4-6"  # matches the `analyze_model` in default deps
    expected_decimal_sum = sum(
        (
            compute_cost_usd(
                model,
                input_tokens=s["input_tokens"],
                cache_write_tokens=s["cache_write_tokens"],
                cache_read_tokens=s["cache_read_tokens"],
                output_tokens=s["output_tokens"],
            )
            for s in token_specs
        ),
        start=Decimal("0"),
    )
    expected_total_cost_usd = float(expected_decimal_sum)

    # Override the default provider with our varying-tokens stub; keep
    # the rest of the dep bundle.
    deps_copy: dict[str, Any] = {**deps, "provider": provider}

    await analyze(state, **deps_copy)

    completed = deps_copy["analyze_event_sink"].completed
    assert len(completed) == 1
    assert completed[0].n_llm_calls == 3  # all three files fired
    # The structural equation: aggregate equals one-float-cast of the
    # Decimal-sum. A float-sum-per-call regression would (occasionally)
    # produce a different float at FP-noise precision.
    assert completed[0].total_cost_usd == expected_total_cost_usd


# ---------------------------------------------------------------------------
# _round_ended_at — FUP-141 monotonic-derived AnalysisRound.ended_at
# ---------------------------------------------------------------------------
# The AnalysisRound ordering invariant is `ended_at >= started_at`. The old
# code took a second wall-clock read (`datetime.now(UTC)`) for ended_at, which
# a backwards wall-clock jump (NTP step / VM resume / WSL2 skew) between round
# start and end could violate. The fix derives ended_at from a monotonic delta
# instead. These tests pin `ended_at >= started_at` across every sign of the
# monotonic delta — the wall clock is never re-read, so it can't break it.


def test_round_ended_at_is_monotonic_derived(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward elapsed: ended_at = started_at + (monotonic delta), to the second.

    The helper reads `time.monotonic()` exactly once; with a fixed start mark
    the result is the wall-clock start plus the elapsed monotonic seconds — NOT
    a second `datetime.now()` read."""
    import outrider.agent.nodes.analyze as analyze_mod

    monkeypatch.setattr(analyze_mod.time, "monotonic", lambda: 1000.5)
    started_at = datetime(2026, 6, 2, 12, 0, 0, tzinfo=UTC)
    ended = _round_ended_at(started_at, started_mono=1000.0)
    assert ended == started_at + timedelta(seconds=0.5)
    assert ended >= started_at


def test_round_ended_at_zero_elapsed_equals_started_at(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero elapsed (start mark == end mark): ended_at == started_at exactly.

    A round that completes within the monotonic clock's resolution lands on the
    boundary of the invariant — equal is allowed, before is not."""
    import outrider.agent.nodes.analyze as analyze_mod

    monkeypatch.setattr(analyze_mod.time, "monotonic", lambda: 1000.0)
    started_at = datetime(2026, 6, 2, 12, 0, 0, tzinfo=UTC)
    ended = _round_ended_at(started_at, started_mono=1000.0)
    assert ended == started_at


def test_round_ended_at_clamps_backwards_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    """FUP-141 regression: even a backwards monotonic delta yields ended_at ==
    started_at, never before it.

    `time.monotonic()` is non-decreasing by contract, so this is the defensive
    `max(0.0, …)` floor — but it is exactly the condition the old wall-clock
    re-read failed under. Pinning it proves the ordering invariant holds for a
    negative delta, the worst case, without depending on the platform clock."""
    import outrider.agent.nodes.analyze as analyze_mod

    # End mark BEFORE the start mark — a contractually-impossible monotonic
    # regression standing in for the backwards wall-clock jump the old code hit.
    monkeypatch.setattr(analyze_mod.time, "monotonic", lambda: 999.0)
    started_at = datetime(2026, 6, 2, 12, 0, 0, tzinfo=UTC)
    ended = _round_ended_at(started_at, started_mono=1000.0)
    assert ended == started_at
    assert ended >= started_at


class _BackwardsWallClock:
    """`datetime` stand-in whose FIRST `now()` read is later than every
    subsequent one — a backwards wall-clock jump (NTP step / VM resume /
    WSL2 skew) between a node's start read and any later read.

    Post-fix `analyze` reads the wall clock once (`started_at`) and derives
    `ended_at` from a monotonic delta, so the jump can't trip the
    `AnalysisRound` ordering invariant. If a refactor reintroduces a second
    wall read for `ended_at` (the FUP-141 bug), that read returns the earlier
    instant and the round's validator rejects `ended_at < started_at` —
    failing this test. Guards the call site the three helper tests can't reach.
    """

    _later = datetime(2026, 6, 2, 12, 0, 1, tzinfo=UTC)
    _earlier = datetime(2026, 6, 2, 11, 59, 59, tzinfo=UTC)

    def __init__(self) -> None:
        self._first = True

    def now(self, tz: object = None) -> datetime:
        if self._first:
            self._first = False
            return self._later
        return self._earlier


@pytest.mark.asyncio
async def test_analyze_round_ordering_holds_under_backwards_wall_clock(
    deps: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node-level FUP-141 regression: a backwards wall-clock jump during the
    round still yields `ended_at >= started_at` on the constructed
    `AnalysisRound`.

    The three `_round_ended_at` tests pin the helper in isolation; this pins
    the NODE wiring. Reverting the call site (`analyze.py`) to a second
    `datetime.now(UTC)` read reintroduces the bug AND fails this test (the
    second wall read returns the earlier instant → validator raises), where
    the helper tests would stay green."""
    import outrider.agent.nodes.analyze as analyze_mod

    monkeypatch.setattr(analyze_mod, "datetime", _BackwardsWallClock())
    state = _build_review_state()
    result = await analyze(state, **deps)

    round_ = result["analysis_rounds"][0]
    # started_at is the single wall read (the later instant); ended_at is
    # monotonic-derived, so ordering holds despite the backwards jump.
    assert round_.started_at == _BackwardsWallClock._later
    assert round_.ended_at >= round_.started_at


# ---------------------------------------------------------------------------
# Tiered model routing (specs/2026-06-08-analyze-tiered-model-routing.md):
# DEEP → analyze_model, STANDARD → standard_analyze_model, recorded on the event.
# ---------------------------------------------------------------------------


async def test_analyze_routes_standard_tier_to_standard_model(deps: dict[str, Any]) -> None:
    """The cost lever: a STANDARD-tier file's LLM call uses `standard_analyze_model`; a
    DEEP-tier file's uses `analyze_model`. Both are read from config (the call site is
    literal-free), and `AnalyzeCompletedEvent` records the DEEP model on `analyze_model`
    and the STANDARD model on `standard_analyze_model`."""
    deep_file = _build_changed_file(path="src/deep.py")
    standard_file = _build_changed_file(path="src/standard.py")
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(deep_file, standard_file)),
        triage_result=_build_triage_result(
            file_tiers={
                "src/deep.py": ReviewTier.DEEP,
                "src/standard.py": ReviewTier.STANDARD,
            }
        ),
    )
    # Empty-findings response: a real LLM call per file (so routing is exercised)
    # without two identical findings tripping AnalysisRound's proposal-hash uniqueness.
    provider = _StubLLMProvider(json.dumps({"findings": []}))
    routed_deps = {
        **deps,
        "provider": provider,
        "analyze_model": "claude-sonnet-4-6",
        "standard_analyze_model": "claude-haiku-4-5",
    }

    await analyze(state, **routed_deps)

    # Two LLM calls, routed by tier — iteration order is `changed_files` order.
    assert [c.model for c in provider.calls] == ["claude-sonnet-4-6", "claude-haiku-4-5"]
    event = routed_deps["analyze_event_sink"].completed[0]
    assert event.analyze_model == "claude-sonnet-4-6"  # DEEP
    assert event.standard_analyze_model == "claude-haiku-4-5"  # STANDARD


async def test_analyze_standard_model_none_when_no_standard_file(deps: dict[str, Any]) -> None:
    """`AnalyzeCompletedEvent.standard_analyze_model` is `None` when no STANDARD-tier
    file was analyzed (here: a DEEP-only pass) — even though the config carries a
    distinct STANDARD model, it isn't 'used' this pass."""
    state = _build_review_state(
        triage_result=_build_triage_result(file_tiers={"src/example.py": ReviewTier.DEEP})
    )
    provider = _StubLLMProvider(json.dumps({"findings": []}))
    routed_deps = {
        **deps,
        "provider": provider,
        "analyze_model": "claude-sonnet-4-6",
        "standard_analyze_model": "claude-haiku-4-5",
    }

    await analyze(state, **routed_deps)

    assert provider.calls[0].model == "claude-sonnet-4-6"  # DEEP file on analyze_model
    event = routed_deps["analyze_event_sink"].completed[0]
    assert event.analyze_model == "claude-sonnet-4-6"
    assert event.standard_analyze_model is None


async def test_analyze_standard_model_recorded_even_when_equals_deep_inert(
    deps: dict[str, Any],
) -> None:
    """Under INERT config (standard_analyze_model == analyze_model == Sonnet), a
    STANDARD-tier file still records that model on the event — NOT `None`.
    `standard_analyze_model is None` means "no STANDARD call ran", not "same model as
    DEEP" (observability over distinctness). Pins the corrected contract."""
    standard_file = _build_changed_file(path="src/standard.py")
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(standard_file,)),
        triage_result=_build_triage_result(file_tiers={"src/standard.py": ReviewTier.STANDARD}),
    )
    provider = _StubLLMProvider(json.dumps({"findings": []}))
    # Inert: both tiers on the same model.
    routed_deps = {
        **deps,
        "provider": provider,
        "analyze_model": "claude-sonnet-4-6",
        "standard_analyze_model": "claude-sonnet-4-6",
    }

    await analyze(state, **routed_deps)

    assert provider.calls[0].model == "claude-sonnet-4-6"  # STANDARD routed (== DEEP)
    event = routed_deps["analyze_event_sink"].completed[0]
    # A STANDARD LLM call ran → the field is the model used, even though it equals DEEP.
    assert event.standard_analyze_model == "claude-sonnet-4-6"
