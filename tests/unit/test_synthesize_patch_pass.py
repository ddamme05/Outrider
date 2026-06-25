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

from outrider.agent.nodes.finding_cap import FindingCapOverflowError
from outrider.agent.nodes.synthesize import synthesize
from outrider.anomaly.rule_names import AnomalyRuleName
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
    def __init__(self) -> None:
        self.completed: list[Any] = []

    async def emit_synthesize_completed(self, event: Any) -> None:
        self.completed.append(event)

    async def query_review_llm_aggregates(  # noqa: ARG002
        self, *, review_id: Any, is_eval: bool
    ) -> ReviewLLMAggregates:
        return ReviewLLMAggregates(
            llm_calls_made=0, total_input_tokens=0, total_output_tokens=0, total_cost_usd=0.0
        )


class _StubAnomalySink:
    def __init__(self) -> None:
        self.anomalies: list[dict[str, Any]] = []

    async def emit_anomaly(self, **kwargs: Any) -> None:
        self.anomalies.append(kwargs)


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


def _make_finding(
    *, finding_type: FindingType, severity: FindingSeverity, line: int
) -> ReviewFinding:
    file_path = "src/foo.py"
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=finding_type,
        severity=severity,
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
            file_path=file_path, line_start=line, line_end=line, finding_type=finding_type
        ),
        proposal_hash=f"{line:064x}",  # distinct per finding (round requires unique)
    )


def _make_multi_state(findings: tuple[ReviewFinding, ...]) -> ReviewState:
    file_path = findings[0].file_path
    changed_file = ChangedFile(
        path=file_path,
        status="modified",  # type: ignore[arg-type]
        additions=1,
        deletions=1,
        patch="@@ -1,3 +1,3 @@\n a\n-old\n+new\n c\n",
        content_base="a\nold\nc\n",
        content_head="a\nb\nc\n",
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
    now = datetime.now(UTC)
    round_id = compute_round_id(
        pass_index=0,
        files_examined=(file_path,),
        files_skipped=(),
        finding_content_hashes=tuple(f.content_hash for f in findings),
    )
    analysis_round = AnalysisRound(
        round_id=round_id,
        pass_index=0,
        findings=findings,
        files_examined=(file_path,),
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )
    triage_result = TriageResult(
        file_tiers={file_path: ReviewTier.DEEP},
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


@pytest.mark.asyncio
async def test_synthesize_report_cap_keeps_gated_drops_non_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FUP-180: synthesize re-caps the deduped cross-round union to the report bound,
    gated-aware. With soft_cap=2 and 2 HIGH (gated) + LOW + INFO (non-gated), the report
    keeps the 2 HIGH and drops the 2 non-gated — gated findings reach HITL, non-gated
    are degraded. (The cap runs BEFORE the summary call, so the summary describes only
    the kept set.)"""
    monkeypatch.setattr("outrider.agent.nodes.synthesize.MAX_FINDINGS_PER_REPORT", 2)
    findings = (
        _make_finding(
            finding_type=FindingType.HARDCODED_SECRET, severity=FindingSeverity.HIGH, line=2
        ),
        _make_finding(finding_type=FindingType.XSS, severity=FindingSeverity.HIGH, line=4),
        _make_finding(
            finding_type=FindingType.MISSING_ERROR_HANDLING, severity=FindingSeverity.LOW, line=6
        ),
        _make_finding(
            finding_type=FindingType.UNUSED_IMPORT, severity=FindingSeverity.INFO, line=8
        ),
    )
    state = _make_multi_state(findings)
    provider = _DispatchingProvider(_batch_json(uuid4(), "x", "y"))
    event_sink = _StubSynthesizeEventSink()
    anomaly_sink = _StubAnomalySink()

    result = await synthesize(
        state,
        provider=provider,  # type: ignore[arg-type]
        synthesize_model="m",
        patch_model="m",
        patches_enabled=False,
        max_suggestions=5,
        phase_event_sink=_StubPhaseSink(),
        synthesize_event_sink=event_sink,
        anomaly_sink=anomaly_sink,
    )

    report = result["review_report"]
    assert len(report.findings) == 2
    assert all(f.severity is FindingSeverity.HIGH for f in report.findings)
    # The degrade counter lands on SynthesizeCompletedEvent (2 non-gated dropped).
    assert event_sink.completed[0].n_findings_dropped_over_cap == 2
    # kept (2) == soft_cap (2), not a gated overflow → no anomaly.
    assert not anomaly_sink.anomalies


@pytest.mark.asyncio
async def test_synthesize_gated_overflow_emits_anomaly(monkeypatch: pytest.MonkeyPatch) -> None:
    """FUP-180: when gated (CRITICAL/HIGH) findings alone exceed the soft report cap, the
    report keeps them ALL (they reach HITL) and a loud GATED_FINDINGS_OVER_CAP anomaly
    fires. 3 HIGH with soft_cap=2 → report has 3, anomaly emitted."""
    monkeypatch.setattr("outrider.agent.nodes.synthesize.MAX_FINDINGS_PER_REPORT", 2)
    findings = (
        _make_finding(
            finding_type=FindingType.HARDCODED_SECRET, severity=FindingSeverity.HIGH, line=2
        ),
        _make_finding(finding_type=FindingType.XSS, severity=FindingSeverity.HIGH, line=4),
        _make_finding(
            finding_type=FindingType.PATH_TRAVERSAL, severity=FindingSeverity.HIGH, line=6
        ),
    )
    state = _make_multi_state(findings)
    provider = _DispatchingProvider(_batch_json(uuid4(), "x", "y"))
    anomaly_sink = _StubAnomalySink()

    result = await synthesize(
        state,
        provider=provider,  # type: ignore[arg-type]
        synthesize_model="m",
        patch_model="m",
        patches_enabled=False,
        max_suggestions=5,
        phase_event_sink=_StubPhaseSink(),
        synthesize_event_sink=_StubSynthesizeEventSink(),
        anomaly_sink=anomaly_sink,
    )

    assert len(result["review_report"].findings) == 3  # all gated kept, exceeding soft_cap
    assert any(
        a["rule_name"] is AnomalyRuleName.GATED_FINDINGS_OVER_CAP for a in anomaly_sink.anomalies
    )


@pytest.mark.asyncio
async def test_synthesize_hard_cap_fails_loud_no_strand(monkeypatch: pytest.MonkeyPatch) -> None:
    """FUP-180: gated findings exceeding the hard ceiling make synthesize FAIL LOUD
    (FindingCapOverflowError), raised after dedup but BEFORE the patch/summary LLM calls
    + ReviewReport + SynthesizeCompletedEvent — so nothing is stranded (no completion
    event, no LLM call fired). Symmetric to analyze's fail-loud test."""
    monkeypatch.setattr("outrider.agent.nodes.synthesize.MAX_FINDINGS_HARD_CAP", 2)
    findings = (
        _make_finding(
            finding_type=FindingType.HARDCODED_SECRET, severity=FindingSeverity.HIGH, line=2
        ),
        _make_finding(finding_type=FindingType.XSS, severity=FindingSeverity.HIGH, line=4),
        _make_finding(
            finding_type=FindingType.PATH_TRAVERSAL, severity=FindingSeverity.HIGH, line=6
        ),
    )
    state = _make_multi_state(findings)
    provider = _DispatchingProvider(_batch_json(uuid4(), "x", "y"))
    event_sink = _StubSynthesizeEventSink()

    with pytest.raises(FindingCapOverflowError):
        await synthesize(
            state,
            provider=provider,  # type: ignore[arg-type]
            synthesize_model="m",
            patch_model="m",
            patches_enabled=True,
            max_suggestions=5,
            phase_event_sink=_StubPhaseSink(),
            synthesize_event_sink=event_sink,
            anomaly_sink=_StubAnomalySink(),
        )

    # Clean crash: no SynthesizeCompletedEvent (no strand), and the raise fired BEFORE
    # any patch/summary LLM call.
    assert event_sink.completed == []
    assert provider.requests == []
