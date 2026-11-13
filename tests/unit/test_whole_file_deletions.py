"""Review removal regressions without treating deleted bodies as current code.

The real analysis, summary, approval and publish nodes use recorded model and
GitHub responses. These checks establish source provenance and delivery rules,
not whether a live model will discover the scripted issue.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from difflib import unified_diff
from unittest.mock import MagicMock

import pytest
from test_analyze_node import (
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _RecordingAnalyzeEventSink,
    _RecordingAnomalySink,
    _RecordingFileExaminationSink,
    _RecordingPhaseEventSink,
    _StubLLMProvider,
    run_analyze_pass,
)
from test_hitl_node import _RecordingHITLSink, _RecordingStatusSink
from test_publish_node_end_to_end import (
    _RecordingPublishEventSink,
    _RecordingReviewStatusSink,
    _stub_github_factory,
    _StubPublisher,
)
from test_synthesize_patch_pass import (
    _DispatchingProvider,
    _StubSynthesizeEventSink,
)

from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS
from outrider.agent.nodes.hitl import hitl
from outrider.agent.nodes.hitl_config import HITLConfig
from outrider.agent.nodes.publish import publish
from outrider.agent.nodes.synthesize import synthesize
from outrider.audit.events import PublishDestination, compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.pr_context import ChangedFile
from outrider.schemas.triage_result import ReviewTier

_POLICY = "def authorization_required():\n    return True\n"
_APP_BASE = (
    "from policy import authorization_required\n\n"
    "def handle():\n    return authorization_required()\n"
)
_APP_HEAD = _APP_BASE.replace(
    "return authorization_required()", "return bool(authorization_required())"
)


def _removed(path: str = "policy.py", source: str = _POLICY) -> ChangedFile:
    patch = "".join(
        unified_diff(source.splitlines(keepends=True), [], fromfile="a/" + path, tofile="/dev/null")
    )
    return ChangedFile(
        path=path,
        status="removed",
        additions=0,
        deletions=len(source.splitlines()),
        patch=patch,
        content_base=source,
        content_head=None,
        language="python",
    )


def _state(removed: ChangedFile, *, captured_importer: bool = False):
    files = [removed]
    tiers = {removed.path: ReviewTier.DEEP}
    if captured_importer:
        files.append(
            ChangedFile(
                path="app.py",
                status="modified",
                additions=1,
                deletions=1,
                patch="".join(
                    unified_diff(
                        _APP_BASE.splitlines(keepends=True),
                        _APP_HEAD.splitlines(keepends=True),
                        fromfile="a/app.py",
                        tofile="b/app.py",
                    )
                ),
                content_base=_APP_BASE,
                content_head=_APP_HEAD,
                language="python",
            )
        )
        # Captured reference material remains available without a second review call.
        tiers["app.py"] = ReviewTier.SKIM
    context = _build_pr_context(changed_files=tuple(files)).model_copy(
        update={
            "total_additions": sum(file.additions for file in files),
            "total_deletions": sum(file.deletions for file in files),
        }
    )
    state = _build_review_state(
        pr_context=context, triage_result=_build_triage_result(file_tiers=tiers)
    )
    state.received_at = datetime.now(UTC)
    return state


def _deps(response: dict):
    return {
        "provider": _StubLLMProvider(json.dumps(response)),
        "analyze_model": "claude-sonnet-5",
        "standard_analyze_model": "claude-haiku-4-5",
        "phase_event_sink": _RecordingPhaseEventSink(),
        "file_examination_sink": _RecordingFileExaminationSink(),
        "analyze_event_sink": _RecordingAnalyzeEventSink(),
        "anomaly_sink": _RecordingAnomalySink(),
        "import_path_resolver": MagicMock(),
        "active_policy_version": ACTIVE_POLICY_VERSION,
        "total_review_budget_tokens": DEFAULT_REVIEW_BUDGET_TOKENS,
    }


async def _publish(state):
    sink, publisher = _RecordingPublishEventSink(), _StubPublisher()
    await publish(
        state,
        publisher=publisher,
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_RecordingReviewStatusSink(),
        github_factory=_stub_github_factory,
    )
    return sink, publisher


@pytest.mark.asyncio
async def test_removed_module_warning_reaches_approval_and_body_publication(monkeypatch):
    monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", "removal-test-secret-0123456789012345")
    state = _state(_removed(), captured_importer=True)
    deps = _deps(
        {
            "findings": [
                {
                    "finding_type": "correctness_bug",
                    "evidence_tier": "judged",
                    "title": "Removing policy breaks the surviving importer",
                    "description": "app.py still imports from the deleted policy module.",
                    "evidence": _POLICY.rstrip("\n"),
                    "line_start": 1,
                    "line_end": 2,
                    "trace_candidates": [],
                }
            ]
        }
    )
    delta = await run_analyze_pass(state, deps)
    assert len(deps["provider"].calls) == 1
    request = deps["provider"].calls[0]
    assert request.degraded_mode is True
    assert request.degradation_reason.value == "file_removal"
    assert "from policy import authorization_required" in request.user_prompt
    assert any(
        entry.file_path == "policy.py" and getattr(entry, "source_side", None) == "base"
        for entry in request.context_summary
    )
    assert any(
        entry.file_path == "app.py" and getattr(entry, "source_side", "head") == "head"
        for entry in request.context_summary
    )
    state = state.model_copy(update=delta)
    [finding] = state.analysis_rounds[0].findings
    assert getattr(finding, "source_side", None) == "base"
    assert finding.severity is FindingSeverity.MEDIUM
    assert finding.evidence_tier is EvidenceTier.JUDGED
    assert finding.model_dump(mode="json")["source_side"] == "base"
    assert getattr(deps["analyze_event_sink"].findings[0], "source_side", None) == "base"
    assert finding.content_hash != compute_finding_content_hash(
        "policy.py", line_start=1, line_end=2, finding_type=FindingType.CORRECTNESS_BUG
    )

    summary = _DispatchingProvider('{"items": []}')
    state = state.model_copy(
        update=await synthesize(
            state,
            provider=summary,
            synthesize_model="claude-haiku-4-5",
            patch_model="claude-haiku-5-5",
            patches_enabled=True,
            max_suggestions=5,
            phase_event_sink=_RecordingPhaseEventSink(),
            synthesize_event_sink=_StubSynthesizeEventSink(),
            anomaly_sink=_RecordingAnomalySink(),
        )
    )
    assert len(summary.requests) == 1  # Summary only: no patch on absent HEAD source.
    assert state.review_report.findings[0].suggested_fix is None
    _, unapproved = await _publish(state)
    assert unapproved.create_calls == []

    def approve(payload):
        assert payload["findings_requiring_approval"] == [str(finding.finding_id)]
        assert payload["auto_post_findings"] == []
        return {
            "reviewer_id": "admin",
            "decisions": [
                {
                    "finding_id": str(finding.finding_id),
                    "outcome": "approve",
                    "reason": "Checked importer.",
                }
            ],
            "decided_at": datetime.now(UTC).isoformat(),
        }

    monkeypatch.setattr("outrider.agent.nodes.hitl.interrupt", approve)
    state = state.model_copy(
        update=await hitl(
            state,
            phase_event_sink=_RecordingPhaseEventSink(),
            hitl_event_sink=_RecordingHITLSink(),
            review_status_sink=_RecordingStatusSink(),
            hitl_config=HITLConfig(),
        )
    )
    sink, approved = await _publish(state)
    assert sink.routing[0].destination is PublishDestination.REVIEW_BODY
    [posted] = approved.create_calls
    assert not posted["comments"]
    assert "policy.py" in posted["body"]
    assert finding.title in posted["body"]
    assert "base" in posted["body"].lower()


@pytest.mark.asyncio
async def test_removed_unsafe_body_never_generates_observed_findings():
    source = "import time\n\nasync def obsolete_handler():\n    time.sleep(1)\n"
    state = _state(_removed("obsolete.py", source))
    deps = _deps({"findings": []})
    result = await run_analyze_pass(state, deps)
    assert len(deps["provider"].calls) == 1
    assert result["analysis_rounds"][0].findings == ()
    assert deps["analyze_event_sink"].findings == []
    request = deps["provider"].calls[0]
    assert "removed" in (request.system_prompt + request.user_prompt).lower()


@pytest.mark.parametrize(
    ("suffix", "source"),
    [
        ("toml", "require_auth = true\n"),
        ("yaml", "require_auth: true\n"),
        ("yml", "require_auth: true\n"),
        ("json", '{"require_auth": true}\n'),
        ("ini", "[auth]\nrequired = true\n"),
        ("cfg", "[auth]\nrequired = true\n"),
    ],
)
@pytest.mark.parametrize("has_finding", [False, True], ids=["empty", "finding"])
@pytest.mark.asyncio
async def test_removed_configuration_completes_without_trace_collection(
    suffix, source, has_finding
):
    proposal = {
        "finding_type": "correctness_bug",
        "evidence_tier": "judged",
        "title": "Removing the required authentication setting changes behavior",
        "description": "The removed configuration supplied the authentication requirement.",
        "evidence": source.rstrip("\n"),
        "line_start": 1,
        "line_end": len(source.splitlines()),
        "trace_candidates": [{"import_string_raw": "auth", "reason": "Check the consumer."}],
    }
    deps = _deps({"findings": [proposal] if has_finding else []})
    removed = _removed(f"settings.{suffix}", source).model_copy(update={"language": None})

    result = await run_analyze_pass(_state(removed), deps)

    [request] = deps["provider"].calls
    assert request.degradation_reason.value == "file_removal"
    assert request.context_summary[0].source_side == "base"
    assert not result["trace_candidates"]
    [round_] = result["analysis_rounds"]
    assert len(round_.findings) == int(has_finding)
    if has_finding:
        [finding] = round_.findings
        assert finding.source_side == "base"
        assert finding.evidence == source.rstrip("\n")
    assert len(deps["analyze_event_sink"].completed) == 1


@pytest.mark.asyncio
async def test_missing_surviving_context_is_explicit_unknown():
    deps = _deps({"findings": []})
    result = await run_analyze_pass(_state(_removed()), deps)
    assert len(deps["provider"].calls) == 1
    request = deps["provider"].calls[0]
    assert not any(
        getattr(entry, "source_side", "head") == "head" for entry in request.context_summary
    )
    text = (request.system_prompt + request.user_prompt).lower()
    assert "unknown" in text
    assert result["analysis_rounds"][0].findings == ()


@pytest.mark.asyncio
async def test_misaligned_removed_base_is_rejected_before_model_call():
    removed = _removed().model_copy(
        update={"content_base": "def different_body():\n    return False\n"}
    )
    deps = _deps({"findings": []})
    await run_analyze_pass(_state(removed), deps)
    assert deps["provider"].calls == []
    [examination] = deps["file_examination_sink"].events
    assert examination.parse_status == "skipped"
    assert examination.skip_reason.value == "PATCH_BASE_MISALIGNED"
