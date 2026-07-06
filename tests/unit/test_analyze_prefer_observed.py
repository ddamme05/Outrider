# Per DECISIONS.md#054 — prefer-OBSERVED dedup accounting at the analyze node.
"""Prefer-OBSERVED merge: counter accounting + surviving tier (node-level).

The driver-backed eval collision scenarios assert the surviving finding is
OBSERVED, but they do NOT inspect the `AnalyzeCompletedEvent` counters — and an
accounting regression would only surface as the equation validator raising
mid-`run_review`. This pins the counters directly: a same-line collision evicts
the JUDGED proposal, keeps the OBSERVED, and balances the proposal-accounting
equation with `n_proposals_superseded_by_observed == 1`
(`seen=1, emitted=1, observed=1, rejected=0, superseded=1`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from test_analyze_node import run_analyze_pass_kw

from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS
from outrider.llm.anthropic_provider import (
    _ANTHROPIC_CONTRACT_DIGEST,
    _ANTHROPIC_PROFILE_ID,
)
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.policy import EvidenceTier
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ChangedFile, PRContext, ReviewState
from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel, TriageResult

_REVIEW_ID = UUID("0fedcba9-8765-4321-0fed-cba987654321")

# os.system on line 5 fires the command_injection_os_system OBSERVED query; the
# scripted model JUDGES command_injection on the SAME line -> a content_hash
# collision (file, line 5-5, command_injection).
_HEAD = 'import os\n\n\ndef run(name):\n    os.system("echo " + name)\n    return True\n'
_BASE = "import os\n"
_PATCH = (
    "--- a/app/run.py\n+++ b/app/run.py\n"
    "@@ -1 +1,6 @@\n import os\n+\n+\n+def run(name):\n"
    '+    os.system("echo " + name)\n'
    "+    return True\n"
)
_JUDGED_RESPONSE = json.dumps(
    {
        "findings": [
            {
                "finding_type": "command_injection",
                "evidence_tier": "judged",
                "query_match_id": None,
                "trace_path": None,
                "title": "Command injection via os.system in run()",
                "description": (
                    "run() passes an untrusted argument into os.system, a "
                    "command-injection sink. Use subprocess with an argument list."
                ),
                "evidence": '    os.system("echo " + name)',
                "line_start": 5,
                "line_end": 5,
                "trace_candidates": [],
            }
        ]
    }
)


class _JudgedProvider:
    async def aclose(self) -> None:
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=_JUDGED_RESPONSE,
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


class _PhaseSink:
    async def emit_phase(self, event: Any) -> None:  # noqa: ARG002
        return None


class _FileExamSink:
    async def emit_file_examination(self, event: Any) -> None:  # noqa: ARG002
        return None


class _AnalyzeSink:
    def __init__(self) -> None:
        self.findings: list[Any] = []
        self.completed: list[Any] = []

    async def emit_finding(self, finding: Any, *, is_eval: bool) -> None:  # noqa: ARG002
        self.findings.append(finding)

    async def emit_finding_proposal_rejected(self, event: Any) -> None:
        raise AssertionError(f"unexpected proposal rejection: {event!r}")

    async def emit_analyze_response_rejected(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_analyze_completed(self, event: Any) -> None:
        self.completed.append(event)

    async def emit_scope_exclusion(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_cache_lookup(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_cache_serve(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def emit_observed_skip_shadow(self, event: Any) -> None:  # noqa: ARG002
        return None


def _state() -> ReviewState:
    cf = ChangedFile(
        path="app/run.py",
        status="modified",
        additions=4,
        deletions=0,
        patch=_PATCH,
        content_base=_BASE,
        content_head=_HEAD,
        previous_path=None,
        language="python",
    )
    pr_context = PRContext(
        installation_id=1,
        owner="acme",
        repo="widget",
        pr_number=9,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title="t",
        pr_body=None,
        author="someone",
        total_additions=4,
        total_deletions=0,
        changed_files=(cf,),
    )
    triage = TriageResult(
        file_tiers={cf.path: ReviewTier.DEEP},
        overall_risk=RiskLevel.HIGH,
        relevant_dimensions=(ReviewDimension.SECURITY,),
        reasoning="test",
    )
    return ReviewState(
        review_id=_REVIEW_ID,
        received_at=datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC),
        pr_context=pr_context,
        triage_result=triage,
        is_eval=True,
    )


@pytest.mark.asyncio
async def test_prefer_observed_collision_counter_accounting() -> None:
    """A model JUDGED + producer OBSERVED collision on one line: the OBSERVED
    survives, exactly one FindingEvent fires, and the AnalyzeCompletedEvent
    balances with n_proposals_superseded_by_observed == 1."""
    analyze_sink = _AnalyzeSink()
    await run_analyze_pass_kw(
        _state(),
        provider=_JudgedProvider(),  # type: ignore[arg-type]
        analyze_model="claude-sonnet-4-6",
        standard_analyze_model="claude-sonnet-4-6",
        phase_event_sink=_PhaseSink(),
        file_examination_sink=_FileExamSink(),
        analyze_event_sink=analyze_sink,
        anomaly_sink=AsyncMock(),
        import_path_resolver=MagicMock(),
        active_policy_version=ACTIVE_POLICY_VERSION,
        total_review_budget_tokens=DEFAULT_REVIEW_BUDGET_TOKENS,
        trivial_scope_filter_enabled=False,
    )

    # Exactly one finding survived, and it is the OBSERVED one (query_match_id kept).
    assert len(analyze_sink.findings) == 1
    finding = analyze_sink.findings[0]
    assert finding.evidence_tier is EvidenceTier.OBSERVED
    assert finding.query_match_id == "python.command_injection_os_system"

    # The accounting equation balances with the superseded term ADDED.
    [completed] = analyze_sink.completed
    assert completed.n_proposals_superseded_by_observed == 1
    assert completed.n_proposals_seen == 1
    assert completed.n_findings_emitted == 1  # the one surviving FindingEvent
    assert completed.n_findings_observed == 1
    assert completed.n_proposals_rejected == 0
    # seen == (emitted - served - observed) + rejected + superseded
    assert (
        completed.n_proposals_seen
        == (
            completed.n_findings_emitted
            - completed.n_findings_served
            - completed.n_findings_observed
        )
        + completed.n_proposals_rejected
        + completed.n_proposals_superseded_by_observed
    )
