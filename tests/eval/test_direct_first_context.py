"""Check captured-only retrieval, source accounting and investigation limits."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from outrider.agent.eval_driver import EvalFixture, EvalFixtureFile

from .direct_first_candidates import build_ledger
from .direct_first_context import (
    MAX_CONTEXT_CHARS,
    MAX_PARSE_BYTES,
    MAX_SNIPPET_CHARS,
    MAX_SNIPPETS,
    VERSION,
    captured_heads,
    prepare_context,
    prepare_followup,
)


def _fixture(contents: dict[str, str], **changes) -> EvalFixture:
    return EvalFixture(
        installation_id=1,
        owner="example",
        repo="reviewed",
        pr_number=1,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title="A source change",
        pr_body="PRIVATE_PR_BODY",
        author="PRIVATE_AUTHOR",
        total_additions=1,
        total_deletions=0,
        files=tuple(
            EvalFixtureFile(
                path=path, status="added", additions=1, deletions=0, content_head=source
            )
            for path, source in contents.items()
        ),
        llm_responses={},
        **changes,
    )


def _candidate(path="app.py", line=3, evidence="result = normalize(value)"):
    ledger = build_ledger(
        json.dumps(
            {
                "findings": [
                    {
                        "file_path": path,
                        "title": "Check this call",
                        "description": "Investigate the call.",
                        "evidence": evidence,
                        "line_start": line,
                        "line_end": line,
                        "category": "logic",
                    }
                ]
            }
        ),
        source_id="test-discovery",
        finish_reason="end_turn",
    )
    return ledger.candidates[0]


def _assert_exact_snippets(request, contents):
    payload = json.loads(request.user_prompt)
    snippets = payload["reference_excerpts"]
    assert len(snippets) == len(request.context_summary) <= MAX_SNIPPETS
    total_chars = 0
    for snippet, manifest in zip(snippets, request.context_summary, strict=True):
        assert snippet["file_path"] == manifest.file_path
        assert snippet["line_start"] == manifest.line_start
        assert snippet["line_end"] == manifest.line_end
        source_lines = contents[manifest.file_path].splitlines()
        expected = "\n".join(
            f"{number} | {source_lines[number - 1]}"
            for number in range(manifest.line_start, manifest.line_end + 1)
        )
        assert snippet["numbered_source"] == expected
        rendered_chars = len(json.dumps(snippet, ensure_ascii=False))
        assert rendered_chars <= MAX_SNIPPET_CHARS
        total_chars += rendered_chars
    assert total_chars <= MAX_CONTEXT_CHARS
    assert len(json.dumps(snippets, ensure_ascii=False)) <= MAX_CONTEXT_CHARS
    return payload


@pytest.mark.parametrize("candidate_path", ["app.py", "./app.py"])
def test_followup_includes_primary_callee_and_possible_test_caller_without_private_metadata(
    tmp_path, monkeypatch, candidate_path
):
    from . import direct_first_context

    (tmp_path / "unavailable.py").write_text("OUTSIDE_CAPTURE_MUST_NOT_APPEAR = True")
    monkeypatch.chdir(tmp_path)
    contents = {
        "app.py": "from helpers import normalize\n"
        "def handle(value):\n"
        "    result = normalize(value)\n"
        "    return result\n",
        "tests/test_app.py": "from app import handle\n"
        "def test_handle():\n"
        "    assert handle(2) == 2\n",
    }
    additional = {"helpers.py": "def normalize(value):\n    return value\n"}
    fixture = _fixture(contents, repository_contents_head=additional)
    candidate = _candidate(candidate_path)
    review_id = uuid4()
    request = prepare_followup(fixture, candidate, model="test-model", review_id=review_id)
    assert request is not None
    payload = _assert_exact_snippets(request, {**contents, **additional})
    assert payload["candidate"]["candidate_id"] == candidate.candidate_id
    assert {entry.file_path for entry in request.context_summary} == {
        "app.py",
        "tests/test_app.py",
        "helpers.py",
    }
    assert "possible caller by name" in request.user_prompt
    assert "OUTSIDE_CAPTURE_MUST_NOT_APPEAR" not in request.user_prompt
    assert "PRIVATE_PR_BODY" not in request.user_prompt
    assert "PRIVATE_AUTHOR" not in request.user_prompt
    assert "not resolved calls or test coverage" in payload["retrieval_limits"]["matching"]
    assert request.model == "test-model" and request.review_id == review_id
    assert request.is_eval and request.node_id == "analyze"
    assert request.max_tokens == 2048 and request.temperature == 0.0
    assert request.cache_control and request.prompt_template_version == VERSION
    schema = json.loads(request.response_schema_json)
    assert set(schema["properties"]) == {"candidate_id", "status", "reason", "citations"}
    assert "origin" not in schema["properties"]

    context = prepare_context(fixture)

    def unexpected_parse(*_args, **_kwargs):
        raise AssertionError("prepared context must reuse parser facts")

    monkeypatch.setattr(direct_first_context, "parse_source", unexpected_parse)
    reused = prepare_followup(
        fixture, candidate, model="test-model", review_id=review_id, context=context
    )
    assert reused.model_dump_json() == request.model_dump_json()
    changed = fixture.model_copy(update={"repository_contents_head": {"helpers.py": "changed"}})
    with pytest.raises(ValueError, match="different captured heads"):
        prepare_followup(
            changed, candidate, model="test-model", review_id=review_id, context=context
        )


def test_followup_caps_show_omissions_and_unsupported_primary_keeps_exact_complete_lines(
    monkeypatch,
):
    from . import direct_first_context

    contents = {
        "notes.txt": "call(value)\n" + "x" * 2200 + "\n",
        "oversized.py": "#" * (MAX_PARSE_BYTES + 1),
        **{
            f"related_{number}.py": "def call(value):\n    # " + "x" * 1400 + "\n    return value\n"
            for number in range(10)
        },
    }
    original_parse = direct_first_context.parse_source
    parsed_paths = []

    def observed_parse(source, file_path, resolver):
        assert file_path not in {"notes.txt", "oversized.py"}
        parsed_paths.append(file_path)
        return original_parse(source, file_path, resolver)

    monkeypatch.setattr(direct_first_context, "parse_source", observed_parse)
    request = prepare_followup(
        _fixture(contents),
        _candidate("notes.txt", 1, "call(value)"),
        model="test-model",
        review_id=uuid4(),
    )
    assert request is not None and len(parsed_paths) == 10
    payload = _assert_exact_snippets(request, contents)
    primary = payload["reference_excerpts"][0]
    assert primary["file_path"] == "notes.txt"
    assert primary["line_start"] == primary["line_end"] == 1
    assert primary["numbered_source"] == "1 | call(value)"
    unindexed = {
        row["file_path"]: row["reason"] for row in payload["retrieval_limits"]["unindexed_files"]
    }
    assert unindexed == {
        "notes.txt": "unsupported source extension",
        "oversized.py": "source exceeds parser byte cap",
    }
    omitted = payload["retrieval_limits"]["clipped_or_omitted_snippets"]
    assert any(row["file_path"] == "notes.txt" for row in omitted)
    assert any(row["reason"] == "context count or character cap" for row in omitted)
    assert "oversized.py" in payload["retrieval_limits"]["files_without_excerpts"]


@pytest.mark.parametrize("source", [None, "", "only_one_line\n"])
def test_followup_needs_a_captured_primary_head_range(source):
    fixture = _fixture({} if source is None else {"app.py": source})
    assert prepare_followup(fixture, _candidate(), model="test-model", review_id=uuid4()) is None


@pytest.mark.parametrize(
    ("original", "duplicate"),
    [("app.py", "app.py"), ("app.py", "./app.py"), ("caf\u00e9.py", "cafe\u0301.py")],
)
def test_captured_heads_rejects_duplicate_canonical_paths(original, duplicate):
    fixture = _fixture({original: "source"})
    second = fixture.files[0].model_copy(update={"path": duplicate})
    fixture = fixture.model_copy(update={"files": (*fixture.files, second)})
    with pytest.raises(ValueError, match="duplicate captured source path"):
        captured_heads(fixture)
