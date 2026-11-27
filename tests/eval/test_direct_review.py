"""The direct comparator reports the same side and source that it showed the model."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from outrider.agent.eval_driver import EvalFixture
from outrider.llm.base import INCLUDE_TEXT_OPT_IN

from . import direct_review


@pytest.fixture
def mixed_changes() -> EvalFixture:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/mock_github/safe_refactor.json").read_bytes()
    )
    fixture["files"] = [
        {
            "path": "removed.py",
            "status": "removed",
            "content_base": "gone()\n",
            "content_head": None,
            "patch": "@@ -1 +0,0 @@\n-gone()",
            "additions": 0,
            "deletions": 1,
        },
        {
            "path": "current.py",
            "status": "modified",
            "content_base": "old()\n",
            "content_head": "current()\n",
            "patch": "@@ -1 +1 @@\n-old()\n+current()",
            "additions": 1,
            "deletions": 1,
        },
    ]
    fixture["repository_contents_head"] = {
        "helper.py": "helper()\n",
    }
    return EvalFixture.model_validate(fixture)


def test_request_records_the_side_that_supplied_each_excerpt(mixed_changes):
    request = direct_review.render_request(mixed_changes, model="test", review_id=uuid4())
    assert "Complete base source (lines 1–1):\n```\n1 | gone()" in request.user_prompt
    assert [(row.file_path, row.source_side) for row in request.context_summary] == [
        ("removed.py", "base"),
        ("current.py", "head"),
        ("helper.py", "head"),
    ]


def test_assessment_uses_only_the_source_rendered_for_that_path(mixed_changes):
    def proposal(path, quote, line=1):
        return dict(
            file_path=path,
            evidence=quote,
            line_start=line,
            line_end=line,
            title="Candidate",
            description="Candidate to check.",
            category=None,
        )

    rows = [
        proposal("removed.py", "gone()"),
        proposal("removed.py", "unserved()", 2),
        proposal("current.py", "current()"),
        proposal("helper.py", "helper()"),
        proposal("unknown.py", "gone()"),
    ]
    result = direct_review.assess_response(json.dumps({"findings": rows}), mixed_changes)
    gone, unserved, current, helper, unknown = result["findings"]
    assert (
        gone["source_side"],
        gone["location_in_captured_base"],
        gone["quote_in_captured_base"],
    ) == ("base", True, True)
    assert gone["quote_in_captured_head"] is gone["location_in_captured_head"] is None
    assert not unserved["location_in_captured_base"] and not unserved["quote_in_captured_base"]
    assert current["quote_in_captured_head"] and helper["quote_in_captured_head"]
    assert "source_side" not in current and "quote_in_captured_base" not in helper
    assert unknown["quote_in_captured_head"] is None


def test_missing_removed_source_is_unknown_even_if_the_fixture_contains_head_text(mixed_changes):
    # The renderer selects BASE for removals and never sends this HEAD text.
    removed = mixed_changes.files[0].model_copy(
        update={"content_base": None, "content_head": "unserved()"}
    )
    fixture = mixed_changes.model_copy(update={"files": [removed, *mixed_changes.files[1:]]})
    request = direct_review.render_request(fixture, model="test", review_id=uuid4())
    assert "unserved()" not in request.user_prompt
    assert "Captured base source is unavailable or empty." in request.user_prompt
    answer = {
        "file_path": "removed.py",
        "line_start": 1,
        "line_end": 1,
        "evidence": "unserved()",
        "title": "Candidate",
        "description": "Candidate.",
        "category": None,
    }
    (row,) = direct_review.assess_response(json.dumps({"findings": [answer]}), fixture)["findings"]
    assert row["source_side"] == "base"
    assert row["location_in_captured_base"] is row["quote_in_captured_base"] is None
    assert row["location_in_captured_head"] is row["quote_in_captured_head"] is None


def test_archived_v1_request_reproduces_its_original_manifest(mixed_changes):
    rid = uuid4()
    current = direct_review.render_request(mixed_changes, model="test", review_id=rid)
    old = direct_review.render_request(
        mixed_changes, model="test", review_id=rid, template_version="direct-pr-review-v1"
    )
    before = old.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
    after = current.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
    assert before["prompt_template_version"] == "direct-pr-review-v1"
    assert after["prompt_template_version"] == direct_review.VERSION
    assert before["context_summary"][0].get("source_side") is None
    assert after["context_summary"][0]["source_side"] == "base"
    assert before["context_summary"][1:] == after["context_summary"][1:]
    assert before["user_prompt"] == after["user_prompt"]
    with pytest.raises(ValueError, match="direct review version"):
        direct_review.render_request(
            mixed_changes, model="test", review_id=rid, template_version="unknown"
        )
