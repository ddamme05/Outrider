"""Review the exact Humanize #297 date-function bodies with an empty model answer.

The fixture excerpts come from python-humanize/humanize, BASE
471cd9b3492318ef639973d29ba929ca458068db and HEAD
b7aa0a2909690c39ef76502c77501c2a7845cf73. Both excerpts retain the local
imports and omit unrelated functions. The fixture carries the upstream license.
The correction and unrelated edit below are authored controls.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from outrider.agent import run_review
from outrider.policy import EvidenceTier, FindingSeverity, FindingType

if TYPE_CHECKING:
    from outrider.agent import EvalRunResult

_FIXTURES = (
    Path(__file__).resolve().parents[2] / "fixtures" / "mock_github" / "humanize_datetime_awareness"
)
_FILE_PATH = "src/humanize/time.py"
_QUERY = "python.datetime_now_without_offset_check"
_GUARD = "if isinstance(value, dt.datetime) and value.tzinfo is not None:"
_CALL = "today = dt.datetime.now(value.tzinfo).date()"


def _fixture_source(name: str) -> str:
    return (_FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


def _review(tmp_path: Path, base: str, head: str) -> EvalRunResult:
    patch = list(difflib.unified_diff(base.splitlines(), head.splitlines(), lineterm=""))[2:]
    additions = sum(line.startswith("+") for line in patch)
    deletions = sum(line.startswith("-") for line in patch)
    fixture = {
        "installation_id": 7401,
        "owner": "python-humanize",
        "repo": "humanize",
        "pr_number": 297,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "pr_title": "Handle timezone-aware dates",
        "author": "contributor",
        "total_additions": additions,
        "total_deletions": deletions,
        "files": [
            {
                "path": _FILE_PATH,
                "status": "modified",
                "additions": additions,
                "deletions": deletions,
                "content_base": base,
                "content_head": head,
                "patch": "\n".join(patch) + "\n",
            }
        ],
        "llm_responses": {
            "triage": [
                json.dumps(
                    {
                        "file_tiers": {_FILE_PATH: "deep"},
                        "overall_risk": "medium",
                        "relevant_dimensions": ["code_quality"],
                        "reasoning": "Review the changed date logic.",
                    }
                )
            ],
            "analyze": ['{"findings": []}'],
            "synthesize": ["Review completed."],
        },
    }
    fixture_path = tmp_path / "review.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    return run_review(fixture_path)


def test_datetime_check_reaches_inline_review_without_a_model_finding(tmp_path: Path) -> None:
    head = _fixture_source("head")
    expected_call_lines = {
        line_no for line_no, line in enumerate(head.splitlines(), 1) if line.strip() == _CALL
    }
    assert len(expected_call_lines) == 2
    result = _review(tmp_path, _fixture_source("base"), head)
    assert len(result.findings) == 2
    assert not result.hitl_gated
    for finding in result.findings:
        assert finding.query_match_id == _QUERY
        assert finding.evidence_tier == EvidenceTier.OBSERVED
        assert finding.finding_type == FindingType.CORRECTNESS_BUG
        assert finding.severity == FindingSeverity.MEDIUM
        assert finding.file_path == _FILE_PATH
    assert len(result.published_comments) == 2
    expected_spans = {(call_line - 1, call_line) for call_line in expected_call_lines}
    assert {(finding.line_start, finding.line_end) for finding in result.findings} == expected_spans
    assert {
        (comment.path, comment.line, comment.side) for comment in result.published_comments
    } == {(_FILE_PATH, guard_line, "RIGHT") for guard_line, _ in expected_spans}


@pytest.mark.parametrize("control", ["corrected_guard", "unrelated_edit"])
def test_datetime_check_stays_quiet_on_control(tmp_path: Path, control: str) -> None:
    head = _fixture_source("head")
    if control == "corrected_guard":
        base = _fixture_source("base")
        candidate = head.replace(
            _GUARD,
            _GUARD.replace("value.tzinfo is not None", "value.utcoffset() is not None"),
        )
    else:
        base = head
        # The changed line selects the same scope while leaving the guard and call untouched.
        candidate = head.replace('return _("today")', 'return _("today")  # relative label')
    assert candidate != base
    result = _review(tmp_path, base, candidate)
    assert not result.findings
    assert not result.published_comments
