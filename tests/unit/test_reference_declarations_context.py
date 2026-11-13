"""Declarations used by supplied parent references remain bounded source data."""

# ruff: noqa: F811 -- pytest injects the existing imported deps fixture.
from __future__ import annotations

import hashlib
import json
from difflib import unified_diff
from typing import Any

import pytest
from test_analyze_node import (  # noqa: F401 -- existing native recorder/fixture
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _StubLLMProvider,
    deps,
    run_analyze_pass,
)

from outrider.agent.nodes import analyze as node
from outrider.agent.nodes.analyze_context import (
    add_parent_class_context,
    add_parent_constructor_context,
    prepare_review_context,
)
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER
from outrider.schemas.pr_context import ChangedFile
from outrider.schemas.triage_result import ReviewTier

_PATH = "app/values.py"
_SOURCE = """from math import isfinite as accepts_value

guard_tag = "finite value required"


class ValueOptions:
    magnitude: float


class Base:
    def __init__(self, value):
        self.value = self._checked(value)

    def _checked(self, value):
        if not accepts_value(value):
            raise ValueError(guard_tag)
        return value


class Child(Base):
    def __init__(self, value, options: ValueOptions | None = None):
        super().__init__(value + 1)
"""


def _input(source: str = _SOURCE) -> tuple[Any, ChangedFile]:
    before = source.replace("value + 1", "value")
    changed = ChangedFile(
        path=_PATH,
        status="modified",
        additions=1,
        deletions=1,
        content_base=before,
        content_head=source,
        patch="".join(
            unified_diff(
                before.splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile=f"a/{_PATH}",
                tofile=f"b/{_PATH}",
            )
        ),
        language="python",
    )
    return (
        _build_review_state(
            pr_context=_build_pr_context(changed_files=(changed,)),
            triage_result=_build_triage_result(file_tiers={_PATH: ReviewTier.DEEP}),
        ),
        changed,
    )


@pytest.mark.asyncio
async def test_supplied_parent_declarations_reach_provider_without_finding_locations(
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
                        "description": "Supplied declarations remain reference-only source.",
                        "evidence": "from math import isfinite as accepts_value",
                        "line_start": 1,
                        "line_end": 1,
                        "trace_candidates": [],
                    }
                ]
            }
        )
    )
    result = await run_analyze_pass(state, deps)
    (request,) = deps["provider"].calls
    assert "from math import isfinite as accepts_value" in request.user_prompt
    assert 'guard_tag = "finite value required"' in request.user_prompt
    assert "class ValueOptions:\n" in request.user_prompt
    assert "magnitude: float" in request.user_prompt
    assert hashlib.sha256(_SOURCE.encode()).hexdigest() in request.user_prompt
    declarations = [
        entry
        for entry in request.context_summary
        if entry.scope_unit_name.startswith("reference declaration ")
    ]
    assert [(entry.line_start, entry.line_end) for entry in declarations] == [
        (1, 1),
        (3, 3),
        (6, 7),
    ]
    assert all(entry.inclusion_reason == "same_file_context" for entry in declarations)
    assert not result["analysis_rounds"][0].findings
    (rejected,) = deps["analyze_event_sink"].proposal_rejections
    assert rejected.rejection_reason == "span_outside_scope_unit"


@pytest.mark.asyncio
async def test_tight_headroom_preserves_complete_parent_context(deps: dict[str, Any]) -> None:
    state, changed = _input()
    prepared = node._select_file_review(
        changed_file=changed, import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER
    )
    assert isinstance(prepared, node._PreparedFileReview)
    context = prepare_review_context(
        file_path=_PATH,
        content=changed.content_head,
        patch=changed.patch,
        changed_files=(changed,),
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    for add_context in (add_parent_constructor_context, add_parent_class_context):
        context = add_context(
            context,
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
    assert "raise ValueError(guard_tag)" in request.user_prompt
    assert "from math import isfinite as accepts_value" not in request.user_prompt
    assert not any(
        entry.scope_unit_name.startswith("reference declaration ")
        for entry in request.context_summary
    )
    assert result["analysis_rounds"][0].files_examined == (_PATH,)


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable_parent", [True, False])
async def test_unavailable_or_competing_bindings_withhold_declarations(
    deps: dict[str, Any], unavailable_parent: bool
) -> None:
    source = """from external_types import Base
from math import isfinite as accepts_value


class Child(Base):
    def __init__(self, value):
        super().__init__(value + 1)
"""
    if not unavailable_parent:
        source = (
            _SOURCE.replace(
                "from math import isfinite as accepts_value",
                "from math import isfinite as accepts_value, isnan as is_nan",
            ).replace(
                "if not accepts_value(value):",
                "if not accepts_value(value) or is_nan(value):",
            )
            + "\n\ndef accepts_value(value):\n    return True\n"
        )
    state, _ = _input(source)
    deps["provider"] = _StubLLMProvider('{"findings":[]}')
    await run_analyze_pass(state, deps)
    (request,) = deps["provider"].calls
    assert "from math import isfinite as accepts_value" not in request.user_prompt
    if unavailable_parent:
        assert not any(
            entry.scope_unit_name.startswith("reference declaration ")
            for entry in request.context_summary
        )
    else:
        assert "if not accepts_value(value) or is_nan(value):" in request.user_prompt
        assert not any(
            entry.scope_unit_name.startswith("reference declaration ") and entry.line_start == 1
            for entry in request.context_summary
        )
