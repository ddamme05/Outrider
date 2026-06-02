"""Unit tests for the eval driver's fixture-replay doubles (no DB, no graph).

Covers the four `src`-local adapters in `outrider.agent.eval_driver` in
isolation: the scripted LLM provider's `(node, call-index)` keying, the fake
GitHub client's wire-shape fidelity (newline-wrapped base64 that survives
intake's strip-then-`validate=True` decode), the capturing publisher, and the
`EvalFixture` schema's `extra="forbid"` discipline. The full DB-backed run is
covered by `tests/eval/test_run_review_driver.py`.
"""

from __future__ import annotations

import base64
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.agent.eval_driver import (
    EvalDriverError,
    EvalFixture,
    EvalRunResult,
    _CapturingPublisher,
    _FixtureScriptedProvider,
    _github_factory_for,
)
from outrider.llm.base import LLMRequest

# asyncio_mode = "auto" (pyproject) auto-detects async tests; no marker needed.

_INSTALLATION_ID = 4242
_BASE_SHA = "a" * 40
_HEAD_SHA = "b" * 40
# >60 chars so base64.encodebytes wraps with newlines (the real GitHub shape).
_HEAD_CONTENT = (
    "def vulnerable(user_input):\n    return run('SELECT * FROM t WHERE x=' + user_input)\n"
)
_BASE_CONTENT = "def vulnerable(user_input):\n    return 1\n"


def _fixture(**overrides: object) -> EvalFixture:
    base: dict[str, object] = {
        "installation_id": _INSTALLATION_ID,
        "owner": "acme",
        "repo": "widget",
        "pr_number": 7,
        "base_sha": _BASE_SHA,
        "head_sha": _HEAD_SHA,
        "pr_title": "t",
        "author": "someone",
        "total_additions": 1,
        "total_deletions": 0,
        "files": [
            {
                "path": "app/views.py",
                "status": "modified",
                "additions": 1,
                "deletions": 0,
                "patch": (
                    "@@ -1,2 +1,2 @@\n def vulnerable(user_input):\n"
                    "-    return 1\n+    return run('x')\n"
                ),
                "content_base": _BASE_CONTENT,
                "content_head": _HEAD_CONTENT,
            }
        ],
        "llm_responses": {"triage": ["{}"], "analyze": ["{}"]},
    }
    base.update(overrides)
    return EvalFixture.model_validate(base)


def _triage_request() -> LLMRequest:
    # triage does not require context_summary (only analyze does).
    return LLMRequest(
        system_prompt="s",
        user_prompt="u",
        model="claude-haiku",
        max_tokens=100,
        temperature=0.0,
        review_id=uuid4(),
        node_id="triage",
        prompt_template_version="v1",
        degraded_mode=False,
    )


# ---------------------------------------------------------------------------
# _FixtureScriptedProvider
# ---------------------------------------------------------------------------


async def test_scripted_provider_returns_response_by_node_and_call_index() -> None:
    provider = _FixtureScriptedProvider({"triage": ["first", "second"]})
    req = _triage_request()
    r1 = await provider.complete(req)
    r2 = await provider.complete(req)
    assert r1.text == "first"
    assert r2.text == "second"
    # echoes the request model + emits a valid LLMResponse shape.
    assert r1.model == "claude-haiku"
    assert r1.finish_reason == "end_turn"


async def test_scripted_provider_raises_loud_when_exhausted() -> None:
    provider = _FixtureScriptedProvider({"triage": ["only-one"]})
    req = _triage_request()
    await provider.complete(req)
    with pytest.raises(EvalDriverError, match="no scripted LLM response"):
        await provider.complete(req)


async def test_scripted_provider_raises_for_unscripted_node() -> None:
    provider = _FixtureScriptedProvider({"triage": ["x"]})
    # node_id="analyze" is unscripted in this provider.
    req = LLMRequest(
        system_prompt="s",
        user_prompt="u",
        model="m",
        max_tokens=100,
        temperature=0.0,
        review_id=uuid4(),
        node_id="synthesize",
        prompt_template_version="v1",
        degraded_mode=False,
    )
    with pytest.raises(EvalDriverError, match="no scripted LLM response"):
        await provider.complete(req)


# ---------------------------------------------------------------------------
# _FixtureGitHubClient (via _github_factory_for) — wire-shape fidelity
# ---------------------------------------------------------------------------


async def test_github_factory_rejects_wrong_installation_id() -> None:
    factory = _github_factory_for(_fixture())
    with pytest.raises(EvalDriverError, match="unexpected installation_id"):
        factory(_INSTALLATION_ID + 1)


async def test_github_double_serves_file_list() -> None:
    client = _github_factory_for(_fixture())(_INSTALLATION_ID)
    resp = await client.rest.pulls.async_list_files("acme", "widget", 7)
    metas = resp.parsed_data
    assert len(metas) == 1
    assert metas[0].filename == "app/views.py"
    assert metas[0].status == "modified"
    assert metas[0].previous_filename is None


async def test_github_content_base64_wrapped_survives_intake_decode() -> None:
    repos = _github_factory_for(_fixture())(_INSTALLATION_ID).rest.repos
    resp = await repos.async_get_content("acme", "widget", "app/views.py", ref=_HEAD_SHA)
    data = resp.parsed_data
    assert data.encoding == "base64"
    # Real GitHub wraps base64 with newlines; assert ours does too (content >60 chars).
    assert "\n" in data.content, "fixture base64 must be newline-wrapped to match GitHub wire shape"
    # Mirror intake's decode (github/fetch.py): strip \n/\r then b64decode(validate=True).
    normalized = data.content.replace("\n", "").replace("\r", "")
    decoded = base64.b64decode(normalized, validate=True).decode()
    assert decoded == _HEAD_CONTENT


async def test_github_double_serves_base_vs_head_by_ref() -> None:
    repos = _github_factory_for(_fixture())(_INSTALLATION_ID).rest.repos
    base = await repos.async_get_content("acme", "widget", "app/views.py", ref=_BASE_SHA)
    head = await repos.async_get_content("acme", "widget", "app/views.py", ref=_HEAD_SHA)
    assert base64.b64decode(base.parsed_data.content.replace("\n", "")).decode() == _BASE_CONTENT
    assert base64.b64decode(head.parsed_data.content.replace("\n", "")).decode() == _HEAD_CONTENT


async def test_github_double_raises_on_missing_content() -> None:
    repos = _github_factory_for(_fixture())(_INSTALLATION_ID).rest.repos
    with pytest.raises(EvalDriverError, match="no content for path"):
        await repos.async_get_content("acme", "widget", "app/views.py", ref="c" * 40)


async def test_github_double_renamed_file_keys_base_content_at_previous_path() -> None:
    fixture = _fixture(
        files=[
            {
                "path": "app/new_name.py",
                "status": "renamed",
                "additions": 1,
                "deletions": 1,
                "previous_path": "app/old_name.py",
                "content_base": _BASE_CONTENT,
                "content_head": _HEAD_CONTENT,
            }
        ]
    )
    repos = _github_factory_for(fixture)(_INSTALLATION_ID).rest.repos
    # base content is read at the OLD path; head at the NEW path (intake's rename shape).
    base = await repos.async_get_content("acme", "widget", "app/old_name.py", ref=_BASE_SHA)
    head = await repos.async_get_content("acme", "widget", "app/new_name.py", ref=_HEAD_SHA)
    assert base64.b64decode(base.parsed_data.content.replace("\n", "")).decode() == _BASE_CONTENT
    assert base64.b64decode(head.parsed_data.content.replace("\n", "")).decode() == _HEAD_CONTENT


# ---------------------------------------------------------------------------
# _CapturingPublisher
# ---------------------------------------------------------------------------


async def test_capturing_publisher_records_create_review_and_reports_no_prior() -> None:
    publisher = _CapturingPublisher()
    assert (
        await publisher.find_existing_review_on_head_sha(
            gh=None,  # type: ignore[arg-type]
            owner="acme",
            repo="widget",
            pull_number=7,
            head_sha=_HEAD_SHA,
            body_marker="m",
        )
        is None
    )
    created = await publisher.create_review(
        gh=None,  # type: ignore[arg-type]
        owner="acme",
        repo="widget",
        pull_number=7,
        head_sha=_HEAD_SHA,
        review_status="COMMENT",
        body_marker="m",
        comments=(),
    )
    assert created.comments_posted == 0
    assert len(publisher.create_review_calls) == 1
    assert publisher.create_review_calls[0]["review_status"] == "COMMENT"


# ---------------------------------------------------------------------------
# EvalFixture schema + EvalRunResult contract
# ---------------------------------------------------------------------------


def test_eval_fixture_forbids_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        _fixture(language="python")  # not a fixture field — intake derives it


def test_eval_run_result_iterates_and_lens_over_findings_and_exposes_state() -> None:
    r = EvalRunResult(
        review_id=uuid4(),
        findings=(),
        trace_decisions=(),
        published_comments=(),
        hitl_gated=True,
    )
    assert list(r) == []
    assert len(r) == 0
    assert r.hitl_gated is True
    assert r.hitl_request is None
