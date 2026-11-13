"""Whole local parent classes remain bounded, reference-only native context."""

# ruff: noqa: F811 -- pytest injects the existing imported deps fixture.
from __future__ import annotations

import json
from difflib import unified_diff
from typing import Any

import pytest
from test_analyze_node import (  # noqa: F401 -- reuse existing public recorder/fixture
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _StubLLMProvider,
    deps,
    run_analyze_pass,
)

from outrider.agent.nodes import analyze as node
from outrider.agent.nodes.analyze_context import (
    add_parent_constructor_context,
    prepare_review_context,
)
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER
from outrider.schemas.pr_context import ChangedFile
from outrider.schemas.triage_result import ReviewTier

_PATH = "app/catalog.py"
_DOC = "Public parent contract documentation belongs to the complete class."
_SOURCE = f'''class Base:
    """{_DOC}"""
    def __init__(self, width):
        self.width = self._checked_width(width)

    def _checked_width(self, width):
        if width < 0:
            raise ValueError("negative width")
        return width


class Child(Base):
    def __init__(self, width):
        super().__init__(width + 1)
'''


def _input(source: str = _SOURCE) -> tuple[Any, ChangedFile]:
    before = source.replace("width + 1", "width")
    patch = "".join(
        unified_diff(
            before.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"a/{_PATH}",
            tofile=f"b/{_PATH}",
        )
    )
    changed = ChangedFile(
        path=_PATH,
        status="modified",
        additions=1,
        deletions=1,
        content_base=before,
        content_head=source,
        patch=patch,
        language="python",
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(changed,)),
        triage_result=_build_triage_result(file_tiers={_PATH: ReviewTier.DEEP}),
    )
    return state, changed


@pytest.mark.asyncio
async def test_complete_parent_docs_and_inherited_method_are_reference_only(
    deps: dict[str, Any],
) -> None:
    state, _ = _input()
    deps["provider"] = _StubLLMProvider(
        json.dumps(
            {
                "findings": [
                    {
                        "finding_type": "correctness_bug",
                        "evidence_tier": "judged",
                        "query_match_id": None,
                        "trace_path": None,
                        "title": "Reference location",
                        "description": "Unchanged parent source is not a primary finding location.",
                        "evidence": 'raise ValueError("negative width")',
                        "line_start": 8,
                        "line_end": 8,
                        "trace_candidates": [],
                    }
                ]
            }
        )
    )
    result = await run_analyze_pass(state, deps)
    (request,) = deps["provider"].calls
    assert "self.width = self._checked_width(width)" in request.user_prompt
    assert _DOC in request.user_prompt
    assert "def _checked_width(self, width):" in request.user_prompt
    entries = [
        entry
        for entry in request.context_summary
        if entry.scope_unit_name.startswith("parent class ")
    ]
    assert len(entries) == 1
    assert (entries[0].line_start, entries[0].line_end) == (1, 9)
    assert entries[0].inclusion_reason == "same_file_context"
    assert not result["analysis_rounds"][0].findings
    (rejected,) = deps["analyze_event_sink"].proposal_rejections
    assert rejected.rejection_reason == "span_outside_scope_unit"


@pytest.mark.asyncio
async def test_insufficient_headroom_preserves_existing_constructor_context(
    deps: dict[str, Any],
) -> None:
    state, changed = _input()
    prepared = node._select_file_review(
        changed_file=changed, import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER
    )
    assert isinstance(prepared, node._PreparedFileReview)
    base = prepare_review_context(
        file_path=_PATH,
        content=changed.content_head,
        patch=changed.patch,
        changed_files=(changed,),
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    context = add_parent_constructor_context(
        base,
        file_path=_PATH,
        head_sha=state.pr_context.head_sha,
        source=prepared.content,
        selected_scope_units=prepared.decision.included_scope_units,
    )
    expected = node._prepare_file_prompt(
        prepared, file_path=_PATH, pass_index=0, review_context=context.text
    )
    deps.update(
        provider=_StubLLMProvider('{"findings":[]}'),
        total_review_budget_tokens=4 * expected.estimated_tokens,
    )
    result = await run_analyze_pass(state, deps)
    (request,) = deps["provider"].calls
    assert request.user_prompt == expected.prompt_parts.user_prompt
    assert request.system_prompt == expected.prompt_parts.system_prompt
    assert "parent constructor " in " ".join(
        entry.scope_unit_name for entry in request.context_summary
    )
    assert _DOC not in request.user_prompt
    assert result["analysis_rounds"][0].files_examined == (_PATH,)


@pytest.mark.asyncio
async def test_oversized_parent_class_is_omitted_without_truncating_constructor_context(
    deps: dict[str, Any],
) -> None:
    source = _SOURCE.replace(_DOC, "Complete parent documentation. " * 700)
    state, _ = _input(source)
    deps["provider"] = _StubLLMProvider('{"findings":[]}')
    await run_analyze_pass(state, deps)
    (request,) = deps["provider"].calls
    assert "self.width = self._checked_width(width)" in request.user_prompt
    assert "Complete parent documentation." not in request.user_prompt
    assert not any(
        entry.scope_unit_name.startswith("parent class ") for entry in request.context_summary
    )
