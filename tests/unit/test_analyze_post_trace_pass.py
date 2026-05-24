"""Analyze post-trace (pass-1) pass — integration test for the M8 loop output.

The trace-node arc landed all the trace-side machinery + graph wiring,
but the consumer side (analyze re-examining `state.trace_fetched_files`
+ admitting INFERRED findings with `trace_path`) wasn't wired until
the 2026-05-24 fold per Codex review findings 1 + 2.

This file pins the load-bearing pass-1 contracts:

  1. `pass_index` derives from `len(state.analysis_rounds)`. A second
     analyze invocation (post-trace re-entry) sees `pass_index=1`,
     NOT `0`. Without this, the `round_id` reducer would collide and
     silently dedup the second pass.
  2. Pass 1 iterates `state.trace_fetched_files` (NOT
     `pr_context.changed_files`) — trace-resolved files, no PR diff.
  3. Pass 1 admits INFERRED proposals with non-empty `trace_path`;
     pass 0 still rejects (no trace context yet).
  4. The returned `AnalysisRound` carries `pass_index=1`, distinct
     `files_examined` (the trace-fetched file path), and distinct
     `round_id` from pass 0.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest  # noqa: TC002 — used at runtime as parameter type

from outrider.agent.nodes.analyze import analyze
from outrider.llm.base import LLMResponse
from outrider.policy import FindingType
from outrider.policy.canonical import compute_round_id
from outrider.schemas import (
    AnalysisRound,
    ReviewState,
    TraceFetchedFile,
)
from outrider.schemas.pr_context import PRContext

if TYPE_CHECKING:
    from pathlib import Path

    from outrider.audit.events import (
        AnalyzeCompletedEvent,
        AnalyzeResponseRejectedEvent,
        FileExaminationEvent,
        FindingEvent,
        FindingProposalRejectedEvent,
        ReviewPhaseEvent,
    )
    from outrider.llm.base import LLMRequest


# ---------------------------------------------------------------------------
# Minimal test scaffolding
# ---------------------------------------------------------------------------


class _MockLLMProvider:
    """Returns a canned analyze response per call. Tests construct the
    response shape per scenario."""

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
            latency_ms=42,
        )


class _RecordingPhaseSink:
    def __init__(self) -> None:
        self.events: list[ReviewPhaseEvent] = []

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        self.events.append(event)


class _RecordingFileExaminationSink:
    def __init__(self) -> None:
        self.events: list[FileExaminationEvent] = []

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        self.events.append(event)


class _RecordingAnalyzeEventSink:
    def __init__(self) -> None:
        self.findings: list[FindingEvent] = []
        self.proposal_rejections: list[FindingProposalRejectedEvent] = []
        self.response_rejections: list[AnalyzeResponseRejectedEvent] = []
        self.completed: list[AnalyzeCompletedEvent] = []

    async def emit_finding(self, event: FindingEvent) -> None:
        self.findings.append(event)

    async def emit_finding_proposal_rejected(self, event: FindingProposalRejectedEvent) -> None:
        self.proposal_rejections.append(event)

    async def emit_analyze_response_rejected(self, event: AnalyzeResponseRejectedEvent) -> None:
        self.response_rejections.append(event)

    async def emit_analyze_completed(self, event: AnalyzeCompletedEvent) -> None:
        self.completed.append(event)


class _StubImportPathResolver:
    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:  # noqa: ARG002
        return []


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def _build_seed_state(
    *,
    analysis_rounds: list[AnalysisRound],
    trace_fetched_files: list[TraceFetchedFile],
) -> ReviewState:
    """Seed state for the analyze() call. Pass 0 → empty analysis_rounds
    + empty trace_fetched_files. Pass 1 → analysis_rounds=[round_0] +
    trace_fetched_files=[fetched_file]."""
    return ReviewState(
        review_id=uuid4(),
        pr_context=PRContext(
            installation_id=12345,
            owner="o",
            repo="r",
            pr_number=1,
            pr_title="x",
            head_sha="a" * 40,
            base_sha="b" * 40,
            author="dev",
            total_additions=5,
            total_deletions=2,
            changed_files=(),
        ),
        received_at=datetime.now(UTC),
        analysis_rounds=analysis_rounds,
        trace_fetched_files=trace_fetched_files,
    )


def _build_round_0(file_path: str = "src/foo.py") -> AnalysisRound:
    """Build a pass-0 AnalysisRound representing analyze's first
    invocation. Used as the seed state for pass-1 tests."""
    now = datetime.now(UTC)
    round_id = compute_round_id(
        pass_index=0,
        files_examined=(file_path,),
        files_skipped=(),
        finding_content_hashes=(),
    )
    return AnalysisRound(
        round_id=round_id,
        pass_index=0,
        findings=(),
        files_examined=(file_path,),
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )


async def test_pass_index_derives_from_analysis_rounds_state() -> None:
    """`pass_index` = `len(state.analysis_rounds)`. Without this
    derivation, hardcoded pass_index=0 would make the round_id
    reducer collide on the second analyze pass and silently dedup.

    Exercises analyze() with an empty-input pass-0 state (no
    changed files, no trace_fetched_files) and pass-1 state (one
    seed round, no fetched files). The AnalyzeCompletedEvent's
    pass_index field is the load-bearing observable: it must be
    derived from state, not hardcoded. Empty inputs keep the LLM
    out of the picture so the test exercises ONLY the routing /
    derivation logic.
    """
    provider = _MockLLMProvider(response_text=json.dumps({"findings": []}))
    phase_sink = _RecordingPhaseSink()
    file_examination_sink = _RecordingFileExaminationSink()
    analyze_event_sink_0 = _RecordingAnalyzeEventSink()

    # Pass 0: empty analysis_rounds + empty pr_context.changed_files
    # → analyze iterates pr-files (zero) → emits AnalyzeCompletedEvent
    # with pass_index=0.
    state_pass_0 = _build_seed_state(analysis_rounds=[], trace_fetched_files=[])
    await analyze(
        state_pass_0,
        provider=provider,  # type: ignore[arg-type]
        analyze_model="claude-sonnet-4-6-20251015",
        phase_event_sink=phase_sink,  # type: ignore[arg-type]
        file_examination_sink=file_examination_sink,  # type: ignore[arg-type]
        analyze_event_sink=analyze_event_sink_0,  # type: ignore[arg-type]
        import_path_resolver=_StubImportPathResolver(),  # type: ignore[arg-type]
    )
    assert len(analyze_event_sink_0.completed) == 1
    assert analyze_event_sink_0.completed[0].pass_index == 0

    # Pass 1: one round in state + empty trace_fetched_files →
    # analyze iterates trace-fetched-files (zero) → emits
    # AnalyzeCompletedEvent with pass_index=1.
    analyze_event_sink_1 = _RecordingAnalyzeEventSink()
    state_pass_1 = _build_seed_state(
        analysis_rounds=[_build_round_0()],
        trace_fetched_files=[],
    )
    await analyze(
        state_pass_1,
        provider=provider,  # type: ignore[arg-type]
        analyze_model="claude-sonnet-4-6-20251015",
        phase_event_sink=phase_sink,  # type: ignore[arg-type]
        file_examination_sink=file_examination_sink,  # type: ignore[arg-type]
        analyze_event_sink=analyze_event_sink_1,  # type: ignore[arg-type]
        import_path_resolver=_StubImportPathResolver(),  # type: ignore[arg-type]
    )
    assert len(analyze_event_sink_1.completed) == 1
    assert analyze_event_sink_1.completed[0].pass_index == 1

    # No LLM call fired in either pass (both file lists were empty).
    assert len(provider.calls) == 0


async def test_pass_1_emits_round_with_pass_index_1_and_distinct_round_id() -> None:
    """End-to-end pass-1 test: state has one pass-0 round + one
    trace-fetched file. Mock LLM returns an INFERRED proposal.
    Verify:
      - The returned state delta's new AnalysisRound carries pass_index=1
      - The new round's round_id differs from the seed round_0
      - INFERRED finding admitted with non-empty trace_path
      - The trace-fetched file's path lands in files_examined
    """
    # Mock LLM provider returns an INFERRED proposal with valid
    # trace_path. The parser's pass_index=1 admission gate accepts it.
    # trace_path elements MUST appear in the trace-fetched file's
    # scope-unit set (post-Codex-round-2 deterministic-proof check).
    # parse_python yields scope units named `authenticate` +
    # `validate_token` for this content (top-level function defs;
    # qualified_name == name when there's no enclosing class/module).
    inferred_response = json.dumps(
        {
            "findings": [
                {
                    "finding_type": "sql_injection",
                    "evidence_tier": "inferred",
                    "query_match_id": None,
                    "trace_path": ["authenticate", "validate_token"],
                    "title": "Auth check skipped in helper",
                    "description": "The helper bypasses the validation in the parent",
                    "evidence": "def authenticate(token: str) -> bool:\n    return True\n",
                    "span": {"byte_start": 0, "byte_end": 40},
                    "trace_candidates": [],
                }
            ]
        }
    )
    provider = _MockLLMProvider(response_text=inferred_response)

    fetched_file = TraceFetchedFile(
        path="src/middleware/auth.py",
        content_head=(
            "def authenticate(token: str) -> bool:\n"
            "    return True\n"
            "\n"
            "def validate_token(token: str) -> bool:\n"
            "    return token == 'admin'\n"
        ),
        source_finding_id=uuid4(),
    )

    state = _build_seed_state(
        analysis_rounds=[_build_round_0()],
        trace_fetched_files=[fetched_file],
    )

    phase_sink = _RecordingPhaseSink()
    file_examination_sink = _RecordingFileExaminationSink()
    analyze_event_sink = _RecordingAnalyzeEventSink()

    state_delta = await analyze(
        state,
        provider=provider,  # type: ignore[arg-type]
        analyze_model="claude-sonnet-4-6-20251015",
        phase_event_sink=phase_sink,  # type: ignore[arg-type]
        file_examination_sink=file_examination_sink,  # type: ignore[arg-type]
        analyze_event_sink=analyze_event_sink,  # type: ignore[arg-type]
        import_path_resolver=_StubImportPathResolver(),  # type: ignore[arg-type]
    )

    # The returned state delta carries one new AnalysisRound for pass 1.
    new_rounds: Any = state_delta["analysis_rounds"]
    assert len(new_rounds) == 1
    new_round = new_rounds[0]
    assert new_round.pass_index == 1

    # The pass-1 round's round_id is distinct from the seed pass-0 round_id
    # — the reducer key. If they collided, the reducer would silently drop
    # one (the bug Codex caught on round-N+1).
    seed_round_id = state.analysis_rounds[0].round_id
    assert new_round.round_id != seed_round_id

    # files_examined lists the trace-fetched file, NOT pr_context.changed_files
    # (which is empty in this state).
    assert "src/middleware/auth.py" in new_round.files_examined

    # The LLM admitted an INFERRED finding with the trace_path from the
    # mock response. Pass 0 would have rejected this — pass 1 admits.
    assert len(new_round.findings) == 1
    finding = new_round.findings[0]
    assert finding.evidence_tier.value == "inferred"
    assert finding.trace_path == ("authenticate", "validate_token")
    assert finding.finding_type == FindingType.SQL_INJECTION

    # The provider was called exactly once (one trace-fetched file).
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call.node_id == "analyze"
    # Verify the post-trace SUFFIX was appended (not just the base
    # prompt's mention of the suffix). Use phrases that exist ONLY in
    # POST_TRACE_SYSTEM_PROMPT_SUFFIX: "REPLACES the pass-0 schema" and
    # the literal pass-1 output schema fragment `<observed|inferred|judged>`.
    # Asserting the section heading alone would be vacuous — Codex
    # round-3 finding F2 caught the earlier version of this test
    # passing on a phrase that existed in the BASE prompt too.
    assert "REPLACES the pass-0 schema" in call.system_prompt
    assert "<observed|inferred|judged>" in call.system_prompt
    # The user-prompt names the source finding via render_post_trace's
    # user template (POST_TRACE_USER_TEMPLATE).
    assert str(fetched_file.source_finding_id) in call.user_prompt

    # Phase events: one start + one end.
    assert len(phase_sink.events) == 2
    assert phase_sink.events[0].marker == "start"
    assert phase_sink.events[1].marker == "end"
    assert phase_sink.events[0].phase_id == phase_sink.events[1].phase_id

    # FileExaminationEvent fired once for the trace-fetched file.
    assert len(file_examination_sink.events) == 1
    assert file_examination_sink.events[0].file_path == "src/middleware/auth.py"

    # FindingEvent emitted once for the admitted INFERRED finding.
    assert len(analyze_event_sink.findings) == 1
    assert len(analyze_event_sink.proposal_rejections) == 0

    # AnalyzeCompletedEvent carries pass_index=1.
    assert len(analyze_event_sink.completed) == 1
    completed = analyze_event_sink.completed[0]
    assert completed.pass_index == 1
    assert completed.n_findings_emitted == 1


@pytest.fixture
def pytest_mock_provider() -> _MockLLMProvider:
    """Placeholder fixture for the routing-only test above."""
    return _MockLLMProvider(response_text=json.dumps({"findings": []}))
