# Per specs/2026-06-10-trivial-scope-filter.md — analyze-node wiring tests.
"""Trivial-scope filter wiring through the analyze node.

Pins the four behavior contracts the spec fixes at the node level:
shadow mode classifies + audits without changing behavior; enforcing
mode skips all-trivial files with `ALL_SCOPES_TRIVIAL` (no LLM call)
and excludes trivial scopes from mixed files' prompts; the baseline
cost gate wins precedence (`COST_BUDGET_EXHAUSTED`, no classification);
and the span-filtered query-ID set feeds the prompt (the parser gets
the same set by construction — one variable feeds both).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS, analyze
from outrider.ast_facts.models import SkipReason, TrivialityReason
from outrider.ast_facts.triviality import TRIVIAL_FILTER_VERSION
from outrider.llm.anthropic_provider import (
    _ANTHROPIC_CONTRACT_DIGEST,
    _ANTHROPIC_PROFILE_ID,
)
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ChangedFile, PRContext, ReviewState
from outrider.schemas.triage_result import (
    ReviewDimension,
    ReviewTier,
    RiskLevel,
    TriageResult,
)

_REVIEW_ID = UUID("87654321-4321-8765-4321-876543218765")


class _StubLLMProvider:
    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    async def aclose(self) -> None:
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=json.dumps({"findings": []}),
            model=request.model,
            input_tokens=100,
            output_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=10,
            profile_id=_ANTHROPIC_PROFILE_ID,
            reasoning_enabled=False,
            profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
        )


class _RecordingPhaseEventSink:
    async def emit_phase(self, event: Any) -> None:  # noqa: ARG002
        return None


class _RecordingFileExaminationSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit_file_examination(self, event: Any) -> None:
        self.events.append(event)


class _RecordingAnalyzeEventSink:
    def __init__(self) -> None:
        self.findings: list[Any] = []
        self.proposal_rejections: list[Any] = []
        self.response_rejections: list[Any] = []
        self.completed: list[Any] = []
        self.scope_exclusions: list[Any] = []
        self.cache_lookups: list[Any] = []
        self.cache_serves: list[Any] = []
        self.observed_skip_shadows: list[Any] = []

    async def emit_finding(self, finding: Any, *, is_eval: bool) -> None:
        self.findings.append((finding, is_eval))

    async def emit_finding_proposal_rejected(self, event: Any) -> None:
        self.proposal_rejections.append(event)

    async def emit_analyze_response_rejected(self, event: Any) -> None:
        self.response_rejections.append(event)

    async def emit_analyze_completed(self, event: Any) -> None:
        self.completed.append(event)

    async def emit_scope_exclusion(self, event: Any) -> None:
        self.scope_exclusions.append(event)

    async def emit_cache_lookup(self, event: Any) -> None:
        self.cache_lookups.append(event)

    async def emit_cache_serve(self, event: Any) -> None:
        self.cache_serves.append(event)

    async def emit_observed_skip_shadow(self, event: Any) -> None:
        self.observed_skip_shadows.append(event)


# Head: module import (query-bearing, outside every scope), alpha with a
# comment-only change at line 5, beta with a code change at line 10.
_HEAD_MIXED = """\
import os


def alpha():
    # tweaked note
    return os.sep


def beta(x):
    y = len(x)
    return y
"""

_BASE_MIXED = """\
import os


def alpha():
    return os.sep


def beta(x):
    return y
"""

_PATCH_MIXED = (
    "--- a/src/mixed.py\n+++ b/src/mixed.py\n"
    "@@ -4,2 +4,3 @@\n"
    " def alpha():\n"
    "+    # tweaked note\n"
    "     return os.sep\n"
    "@@ -8,2 +9,3 @@\n"
    " def beta(x):\n"
    "+    y = len(x)\n"
    "     return y\n"
)

# All-trivial variant: only alpha exists; only the comment changes.
_HEAD_TRIVIAL = """\
import os


def alpha():
    # tweaked note
    return os.sep
"""

_BASE_TRIVIAL = """\
import os


def alpha():
    return os.sep
"""

_PATCH_TRIVIAL = (
    "--- a/src/trivial.py\n+++ b/src/trivial.py\n"
    "@@ -4,2 +4,3 @@\n"
    " def alpha():\n"
    "+    # tweaked note\n"
    "     return os.sep\n"
)


def _changed_file(*, path: str, head: str, base: str, patch: str) -> ChangedFile:
    return ChangedFile(
        path=path,
        status="modified",
        additions=1,
        deletions=0,
        patch=patch,
        content_base=base,
        content_head=head,
        previous_path=None,
        language="python",
    )


def _state(changed_file: ChangedFile) -> ReviewState:
    pr_context = PRContext(
        installation_id=1,
        owner="acme",
        repo="widget",
        pr_number=7,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title="t",
        pr_body=None,
        author="someone",
        total_additions=1,
        total_deletions=0,
        changed_files=(changed_file,),
    )
    triage = TriageResult(
        file_tiers={changed_file.path: ReviewTier.DEEP},
        overall_risk=RiskLevel.MEDIUM,
        relevant_dimensions=(ReviewDimension.SECURITY,),
        reasoning="test",
    )
    return ReviewState(
        review_id=_REVIEW_ID,
        received_at=datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC),
        pr_context=pr_context,
        triage_result=triage,
        is_eval=True,
    )


async def _run(
    state: ReviewState,
    *,
    enabled: bool,
    budget: int = DEFAULT_REVIEW_BUDGET_TOKENS,
) -> tuple[_StubLLMProvider, _RecordingFileExaminationSink, _RecordingAnalyzeEventSink]:
    provider = _StubLLMProvider()
    exam_sink = _RecordingFileExaminationSink()
    analyze_sink = _RecordingAnalyzeEventSink()
    await analyze(
        state,
        provider=provider,  # type: ignore[arg-type]
        analyze_model="claude-sonnet-4-6",
        standard_analyze_model="claude-sonnet-4-6",
        phase_event_sink=_RecordingPhaseEventSink(),
        file_examination_sink=exam_sink,
        analyze_event_sink=analyze_sink,
        anomaly_sink=AsyncMock(),
        import_path_resolver=MagicMock(),
        active_policy_version=ACTIVE_POLICY_VERSION,
        total_review_budget_tokens=budget,
        trivial_scope_filter_enabled=enabled,
    )
    return provider, exam_sink, analyze_sink


@pytest.mark.asyncio
async def test_shadow_mode_classifies_audits_and_changes_nothing() -> None:
    """Flag off (default): the classifier runs and emits applied=False
    would-exclude telemetry, but the LLM call happens exactly as
    baseline — the prompt still carries the trivial scope."""
    cf = _changed_file(
        path="src/trivial.py", head=_HEAD_TRIVIAL, base=_BASE_TRIVIAL, patch=_PATCH_TRIVIAL
    )
    provider, exam_sink, analyze_sink = await _run(_state(cf), enabled=False)

    assert len(provider.calls) == 1  # baseline behavior preserved
    assert "alpha" in provider.calls[0].user_prompt
    [event] = analyze_sink.scope_exclusions
    assert event.applied is False
    assert event.filter_version == TRIVIAL_FILTER_VERSION
    assert event.file_path == "src/trivial.py"
    [entry] = event.entries
    assert entry.trivial is True
    assert entry.reason == TrivialityReason.ALL_LINES_ORDINARY_COMMENT
    assert entry.head_added_lines == (5,)
    assert event.is_eval is True
    # Baseline FileExaminationEvent: clean, not skipped.
    [exam] = exam_sink.events
    assert exam.parse_status == "clean"


@pytest.mark.asyncio
async def test_enforcing_all_trivial_skips_without_llm_call() -> None:
    """Flag on + every admitted scope trivial: no LLM call; the single
    FileExaminationEvent carries ALL_SCOPES_TRIVIAL; the exclusion event
    records applied=True."""
    cf = _changed_file(
        path="src/trivial.py", head=_HEAD_TRIVIAL, base=_BASE_TRIVIAL, patch=_PATCH_TRIVIAL
    )
    provider, exam_sink, analyze_sink = await _run(_state(cf), enabled=True)

    assert provider.calls == []
    [exam] = exam_sink.events
    assert exam.parse_status == "skipped"
    assert exam.skip_reason == SkipReason.ALL_SCOPES_TRIVIAL
    [event] = analyze_sink.scope_exclusions
    assert event.applied is True
    assert all(entry.trivial for entry in event.entries)


@pytest.mark.asyncio
async def test_enforcing_mixed_excludes_trivial_scope_from_prompt_and_queries() -> None:
    """Flag on + one trivial scope, one code scope: the LLM call happens
    over the kept scope only — the trivial scope's body leaves the
    prompt AND context_summary, and query IDs whose matches fall outside
    kept scopes (module-level import_statement) stop advertising."""
    cf = _changed_file(path="src/mixed.py", head=_HEAD_MIXED, base=_BASE_MIXED, patch=_PATCH_MIXED)
    provider, exam_sink, analyze_sink = await _run(_state(cf), enabled=True)

    [request] = provider.calls
    assert "beta" in request.user_prompt
    assert "def alpha" not in request.user_prompt  # excluded scope body gone
    # context_summary stays "what the LLM saw": kept scope only.
    assert [e.scope_unit_name for e in request.context_summary] == ["beta"]
    # Query-ID span filtering: import_statement fires only at module level
    # (outside every kept scope) and must not advertise; function_definition
    # matches inside beta and stays.
    assert "python.function_definition" in request.user_prompt
    assert "python.import_statement" not in request.user_prompt
    [event] = analyze_sink.scope_exclusions
    assert event.applied is True
    by_name = {e.scope_qualified_name: e for e in event.entries}
    assert by_name["alpha"].trivial is True
    assert by_name["beta"].trivial is False
    assert by_name["beta"].reason == TrivialityReason.NON_COMMENT_CONTENT
    # Baseline emission still clean (file was reviewed).
    [exam] = exam_sink.events
    assert exam.parse_status == "clean"


@pytest.mark.asyncio
async def test_shadow_mode_mixed_keeps_full_prompt_and_query_ids() -> None:
    """Shadow contrast for the mixed case: full prompt, full query set."""
    cf = _changed_file(path="src/mixed.py", head=_HEAD_MIXED, base=_BASE_MIXED, patch=_PATCH_MIXED)
    provider, _, analyze_sink = await _run(_state(cf), enabled=False)

    [request] = provider.calls
    assert "def alpha" in request.user_prompt
    assert "python.import_statement" in request.user_prompt
    [event] = analyze_sink.scope_exclusions
    assert event.applied is False
    assert len(event.entries) == 2


@pytest.mark.asyncio
async def test_cost_gate_wins_precedence_over_all_scopes_trivial() -> None:
    """Pinned order: the baseline cost gate runs FIRST. A budget-exhausted
    file skips COST_BUDGET_EXHAUSTED and is never classified — no
    ScopeExclusionEvent, even with the flag on and an all-trivial file."""
    cf = _changed_file(
        path="src/trivial.py", head=_HEAD_TRIVIAL, base=_BASE_TRIVIAL, patch=_PATCH_TRIVIAL
    )
    provider, exam_sink, analyze_sink = await _run(_state(cf), enabled=True, budget=0)

    assert provider.calls == []
    [exam] = exam_sink.events
    assert exam.skip_reason == SkipReason.COST_BUDGET_EXHAUSTED
    assert analyze_sink.scope_exclusions == []
