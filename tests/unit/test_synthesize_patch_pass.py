# Full-node synthesize test for the suggested-patch pass (DECISIONS.md#040).
"""End-to-end-through-the-node coverage that the patch pass runs INSIDE synthesize:
after final dedup, before the summary call + ReviewReport, with the patch LLM call
stamped to `node_id="synthesize"` (its cost rolls into synthesize's aggregate).

The pure parser/orchestration logic lives in `test_patch_generation.py`; this file
locks the WIRING — a patched finding flows out of synthesize on the canonical
`review_report.findings`, and a disabled flag makes no patch call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from outrider.agent.nodes.synthesize import synthesize
from outrider.audit.aggregates import ReviewLLMAggregates
from outrider.audit.events import compute_finding_content_hash
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import compute_round_id
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import patch as patch_prompt
from outrider.prompts import synthesize as synthesize_prompt
from outrider.schemas import ChangedFile, PRContext, ReviewFinding, ReviewState
from outrider.schemas.analysis_round import AnalysisRound
from outrider.schemas.triage_result import (
    ReviewDimension,
    ReviewTier,
    RiskLevel,
    TriageResult,
)

# ---------------------------------------------------------------------------
# Sinks (duck-typed stubs; synthesize only calls these members)
# ---------------------------------------------------------------------------


class _StubPhaseSink:
    async def emit_phase(self, event: Any) -> None:  # noqa: ARG002
        return None


class _StubSynthesizeEventSink:
    async def emit_synthesize_completed(self, event: Any) -> None:  # noqa: ARG002
        return None

    async def query_review_llm_aggregates(  # noqa: ARG002
        self, *, review_id: Any, is_eval: bool
    ) -> ReviewLLMAggregates:
        return ReviewLLMAggregates(
            llm_calls_made=0, total_input_tokens=0, total_output_tokens=0, total_cost_usd=0.0
        )


class _StubAnomalySink:
    async def emit_anomaly(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class _DispatchingProvider:
    """Returns the patch batch on the patch call (keyed by prompt_template_version,
    since the patch + summary calls share node_id='synthesize') and prose otherwise."""

    def __init__(self, patch_batch_json: str, *, summary: str = "Looks reasonable.") -> None:
        self.patch_batch_json = patch_batch_json
        self.summary = summary
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        text = (
            self.patch_batch_json
            if request.prompt_template_version == patch_prompt.VERSION
            else self.summary
        )
        return LLMResponse(
            text=text,
            model=request.model,
            input_tokens=5,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=1,
        )

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_high_finding(*, file_path: str = "src/foo.py", line: int = 2) -> ReviewFinding:
    finding_type = FindingType.HARDCODED_SECRET  # HIGH per SEVERITY_POLICY
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=finding_type,
        severity=FindingSeverity.HIGH,
        file_path=file_path,
        line_start=line,
        line_end=line,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=line,
            line_end=line,
            finding_type=finding_type,
        ),
        proposal_hash="a" * 64,
    )


def _make_state(finding: ReviewFinding, *, content_head: str) -> ReviewState:
    changed_file = ChangedFile(
        path=finding.file_path,
        status="modified",  # type: ignore[arg-type]
        additions=1,
        deletions=1,
        patch="@@ -1,3 +1,3 @@\n a\n-old\n+new\n c\n",
        content_base="a\nold\nc\n",
        content_head=content_head,
        previous_path=None,
    )
    pr_context = PRContext(
        installation_id=42,
        owner="o",
        repo="r",
        pr_number=1,
        pr_title="t",
        base_sha="1" * 40,
        head_sha="0" * 40,
        author="a",
        total_additions=1,
        total_deletions=0,
        changed_files=(changed_file,),
    )
    round_id = compute_round_id(
        pass_index=0,
        files_examined=(finding.file_path,),
        files_skipped=(),
        finding_content_hashes=(finding.content_hash,),
    )
    now = datetime.now(UTC)
    analysis_round = AnalysisRound(
        round_id=round_id,
        pass_index=0,
        findings=(finding,),
        files_examined=(finding.file_path,),
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )
    triage_result = TriageResult(
        file_tiers={finding.file_path: ReviewTier.DEEP},
        overall_risk=RiskLevel.HIGH,
        relevant_dimensions=(ReviewDimension.SECURITY,),
        reasoning="r",
        policy_version=ACTIVE_POLICY_VERSION,
    )
    return ReviewState(
        review_id=uuid4(),
        pr_context=pr_context,
        received_at=now,
        analysis_rounds=[analysis_round],
        triage_result=triage_result,
    )


def _batch_json(finding_id: Any, original_line: str, replacement_line: str) -> str:
    import json

    return json.dumps(
        {
            "items": [
                {
                    "finding_id": str(finding_id),
                    "original_line": original_line,
                    "replacement_line": replacement_line,
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_sets_suggested_fix_on_report_finding() -> None:
    """patches_enabled=True: the HIGH finding's `suggested_fix` is set from the patch
    batch and rides out on the canonical `review_report.findings`."""
    finding = _make_high_finding(line=2)
    state = _make_state(finding, content_head="a = 1\nreturn x\nc = 3\n")
    provider = _DispatchingProvider(
        _batch_json(finding.finding_id, "return x", "return sanitize(x)")
    )

    result = await synthesize(
        state,
        provider=provider,  # type: ignore[arg-type]
        synthesize_model="stub-synthesize-model",
        patch_model="stub-patch-model",
        patches_enabled=True,
        max_suggestions=5,
        phase_event_sink=_StubPhaseSink(),
        synthesize_event_sink=_StubSynthesizeEventSink(),
        anomaly_sink=_StubAnomalySink(),
    )

    report = result["review_report"]
    assert len(report.findings) == 1
    assert report.findings[0].suggested_fix == "return sanitize(x)"
    # Two provider calls: the patch call (patch-v1) + the summary call.
    versions = [r.prompt_template_version for r in provider.requests]
    assert patch_prompt.VERSION in versions
    # The patch call is stamped to synthesize so its cost rolls into the aggregate.
    patch_req = next(
        r for r in provider.requests if r.prompt_template_version == patch_prompt.VERSION
    )
    assert patch_req.node_id == "synthesize"
    assert patch_req.model == "stub-patch-model"


@pytest.mark.asyncio
async def test_synthesize_disabled_makes_no_patch_call() -> None:
    """patches_enabled=False: only the summary call fires; no finding is patched."""
    finding = _make_high_finding(line=2)
    state = _make_state(finding, content_head="a = 1\nreturn x\nc = 3\n")
    provider = _DispatchingProvider(
        _batch_json(finding.finding_id, "return x", "return sanitize(x)")
    )

    result = await synthesize(
        state,
        provider=provider,  # type: ignore[arg-type]
        synthesize_model="stub-synthesize-model",
        patch_model="stub-patch-model",
        patches_enabled=False,
        max_suggestions=5,
        phase_event_sink=_StubPhaseSink(),
        synthesize_event_sink=_StubSynthesizeEventSink(),
        anomaly_sink=_StubAnomalySink(),
    )

    report = result["review_report"]
    assert report.findings[0].suggested_fix is None
    # Exactly one call (the summary); no patch-v1 call.
    assert all(r.prompt_template_version != patch_prompt.VERSION for r in provider.requests)
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_summary_request_carries_configured_model_and_prompt_version() -> None:
    """Zero-spend pins for the DECISIONS#043 flip mechanics: the summary
    LLMRequest carries (a) the injected `synthesize_model` verbatim — the
    config→request routing the Haiku default flows through — and (b)
    `prompts/synthesize.VERSION` as `prompt_template_version`, pinned to
    "synthesize-v3" (the no-pipeline-claims prompt bump) so audit
    provenance records both halves of the flip."""
    finding = _make_high_finding(line=2)
    state = _make_state(finding, content_head="a = 1\nreturn x\nc = 3\n")
    provider = _DispatchingProvider(
        _batch_json(finding.finding_id, "return x", "return sanitize(x)")
    )

    await synthesize(
        state,
        provider=provider,  # type: ignore[arg-type]
        synthesize_model="stub-synthesize-model",
        patch_model="stub-patch-model",
        patches_enabled=False,
        max_suggestions=5,
        phase_event_sink=_StubPhaseSink(),
        synthesize_event_sink=_StubSynthesizeEventSink(),
        anomaly_sink=_StubAnomalySink(),
    )

    [summary_req] = provider.requests
    assert summary_req.model == "stub-synthesize-model"
    assert summary_req.prompt_template_version == synthesize_prompt.VERSION
    assert synthesize_prompt.VERSION == "synthesize-v3"
