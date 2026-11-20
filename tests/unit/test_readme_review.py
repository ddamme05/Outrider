"""Check README review through ordinary file planning, analysis and admission."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import test_analyze_node as analyze_tests

from outrider.agent.nodes.publish_render import _build_finding_comment_body
from outrider.agent.nodes.readme_review import implementation_sources
from outrider.prompts.analyze import README_VERSION
from outrider.schemas.triage_result import ReviewTier

if TYPE_CHECKING:
    from outrider.schemas.pr_context import ChangedFile

deps = analyze_tests.deps


def changed(path: str, before: str, after: str) -> ChangedFile:
    from difflib import unified_diff

    patch = "".join(
        unified_diff(before.splitlines(True), after.splitlines(True), f"a/{path}", f"b/{path}")
    )
    return analyze_tests._build_changed_file(
        path=path, content=after.encode(), content_base=before, patch=patch
    )


@pytest.mark.parametrize("path", ["src/storage.pyi", "src/storage.d.ts"])
def test_declarations_cannot_corroborate_runtime_behavior(path: str) -> None:
    doc = changed("README.md", "Old.\n", "Calling store always returns zero.\n")
    declaration = changed(path, "", "def store() -> int: ...\n")
    assert implementation_sources(doc, (doc, declaration)) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "path", "quote", "line", "expected"),
    [
        ("send", "src/storage.py", "return False", 3, True),
        ("withhold", "src/storage.py", "return False", 3, False),
        ("uncertain", "src/storage.py", "return False", 3, False),
        ("missing", "src/storage.py", "return False", 3, False),
        ("send", "src/storage.py", "return None", 3, False),
        ("send", "other.py", "return False", 3, False),
        ("send", "src/storage.py", "return False", 4, False),
    ],
)
async def test_changed_readme_needs_captured_support(
    deps: dict[str, Any], verdict: str, path: str, quote: str, line: int, expected: bool
) -> None:
    source = changed(
        "src/storage.py", "def store():\n    pass\n", "def store():\n    return False\n"
    )
    doc = changed(
        "README.md",
        "# Store\n\nOld contract.\nSee details below.\n",
        "# Store\n\n- `store` returns true.\nSee details below.\n",
    )
    state = analyze_tests._build_review_state(
        pr_context=analyze_tests._build_pr_context(changed_files=(doc, source)),
        triage_result=analyze_tests._build_triage_result(
            file_tiers={doc.path: ReviewTier.STANDARD, source.path: ReviewTier.SKIP}
        ),
    )
    answer = json.loads(
        analyze_tests._build_finding_proposal_json(
            finding_type="correctness_bug", line_start=line, line_end=line
        )
    )
    answer["findings"][0]["evidence"] = (
        doc.content_head.splitlines()[line - 1] if doc.content_head else ""
    )
    answer["support"] = (
        [dict(verdict=verdict, implementation_path=path, implementation_quote=quote)]
        if verdict != "missing"
        else []
    )
    deps["provider"] = analyze_tests._StubLLMProvider(json.dumps(answer))
    result = await analyze_tests.run_analyze_pass(state, deps)
    [request] = deps["provider"].calls
    assert request.prompt_template_version == README_VERSION
    assert "See details below." in request.user_prompt
    assert set(e.file_path for e in request.context_summary) == {doc.path, source.path}
    assert bool(result["analysis_rounds"][0].findings) == expected
    if expected:
        [finding] = result["analysis_rounds"][0].findings
        body = _build_finding_comment_body(finding, effective_severity=finding.severity)
        assert quote not in finding.description
        assert quote in body and path in body
        assert finding.evidence == answer["findings"][0]["evidence"]
        assert finding.model_dump()["implementation_citation"] == {"path": path, "quote": quote}
    if verdict == "missing":
        [rejection] = deps["analyze_event_sink"].response_rejections
        assert rejection.rejection_detail == "support x1"


@pytest.mark.asyncio
async def test_readme_without_implementation_spends_nothing(deps: dict[str, Any]) -> None:
    doc = changed("README.md", "Original.\n", "Reworded.\n")
    state = analyze_tests._build_review_state(
        pr_context=analyze_tests._build_pr_context(changed_files=(doc,)),
        triage_result=analyze_tests._build_triage_result(file_tiers={doc.path: ReviewTier.DEEP}),
    )
    result = await analyze_tests.run_analyze_pass(state, deps)
    assert not deps["provider"].calls
    assert not result["analysis_rounds"][0].findings


@pytest.mark.asyncio
async def test_readme_preserves_code_requests_at_a_tight_budget(deps: dict[str, Any]) -> None:
    files = tuple(
        changed(
            f"src/module_{index}.py", "def store():\n    pass\n", "def store():\n    return 1\n"
        )
        for index in range(7)
    )
    readme = changed("README.md", "Old.\n", "Calling store always returns zero.\n")
    deps["total_review_budget_tokens"] = 140_000

    async def code_requests(readme_tier: ReviewTier) -> dict[str | None, str]:
        provider = analyze_tests._StubLLMProvider('{"findings":[],"support":[]}')
        state = analyze_tests._build_review_state(
            pr_context=analyze_tests._build_pr_context(changed_files=(readme, *files)),
            triage_result=analyze_tests._build_triage_result(
                file_tiers={
                    **dict.fromkeys((f.path for f in files), ReviewTier.STANDARD),
                    readme.path: readme_tier,
                }
            ),
        )
        await analyze_tests.run_analyze_pass(state, {**deps, "provider": provider})
        return {
            request.phase_key: request.user_prompt
            for request in provider.calls
            if request.prompt_template_version != README_VERSION
        }

    baseline = await code_requests(ReviewTier.SKIP)
    assert 0 < len(baseline) < len(files)
    assert await code_requests(ReviewTier.STANDARD) == baseline
