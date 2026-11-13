"""Bounded same-file constructor references through the real analysis pipeline."""

# ruff: noqa: F811 — pytest injects the imported deps fixture by parameter name.
from __future__ import annotations

from difflib import unified_diff
from math import ceil
from typing import Any

import pytest
from test_analyze_node import (  # noqa: F401 — existing public test harness/fixture
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _StubLLMProvider,
    deps,
    run_analyze_pass,
)

from outrider.agent.nodes.analyze import (
    PER_FILE_CAP_FRACTION,
    _compute_per_file_cap,
    _estimate_tokens,
    _prepare_file_prompt,
    _PreparedFileReview,
    _select_file_review,
)
from outrider.agent.nodes.analyze_context import prepare_review_context
from outrider.schemas.pr_context import ChangedFile
from outrider.schemas.triage_result import ReviewTier

_PATH = "app/records.py"
_GUARD = "if width <= 0 or width % 8:"
_PARENT = """class AlignedStore{generic}:
    def __init__(self, width):
        if width <= 0 or width % 8:
            raise ValueError("unaligned record width")
        self.width = width
        self.guard_finished = True


"""
_CHILD = """class PacketStore({bases}):
    def __init__(self, width={width}{extra_argument}):
        super().__init__(width)
"""


def _state(source: str) -> Any:
    before = source.replace("width=10", "width=16")
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
    return _build_review_state(
        pr_context=_build_pr_context(changed_files=(changed,)),
        triage_result=_build_triage_result(file_tiers={_PATH: ReviewTier.DEEP}),
    )


def _source(
    *, generic: bool = False, between: str = "", bases: str | None = None, extra_argument: str = ""
) -> str:
    base = _PARENT.format(generic="[T]" if generic else "")
    return (
        base
        + between
        + _CHILD.format(
            bases=bases or ("AlignedStore[int]" if generic else "AlignedStore"),
            width=10,
            extra_argument=extra_argument,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True], ids=["plain-base", "subscripted-base"])
async def test_changed_constructor_receives_complete_local_parent_without_new_finding_bounds(
    deps: dict[str, Any], generic: bool
) -> None:
    source = _source(generic=generic)
    # Deliberately locate a proposal in reference-only parent code. Extra context
    # must not make that unchanged location reviewable as a primary finding.
    deps["provider"] = _StubLLMProvider(
        '{"findings":[{"finding_type":"correctness_bug","evidence_tier":"judged",'
        '"query_match_id":null,"trace_path":null,"title":"Reference location",'
        '"description":"Reference source is not a primary finding location.",'
        f'"evidence":"{_GUARD}","line_start":3,"line_end":3,"trace_candidates":[]}}]}}'
    )
    result = await run_analyze_pass(_state(source), deps)

    (request,) = deps["provider"].calls
    assert _GUARD in request.user_prompt
    assert "self.guard_finished = True" in request.user_prompt
    parent_entries = [
        entry
        for entry in request.context_summary
        if entry.file_path == _PATH and entry.scope_unit_name.startswith("parent constructor ")
    ]
    assert len(parent_entries) == 1
    assert (parent_entries[0].line_start, parent_entries[0].line_end) == (2, 6)
    assert parent_entries[0].inclusion_reason == "same_file_context"
    assert not result["analysis_rounds"][0].findings
    assert deps["analyze_event_sink"].proposal_rejections


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("between", "bases", "extra_argument"),
    [
        ("AlignedStore = object\n\n", "AlignedStore", ""),
        ("AlignedStore.__init__ = lambda self, width: None\n\n", "AlignedStore", ""),
        (_PARENT.format(generic=""), "AlignedStore", ""),
        ("class AuditMixin:\n    pass\n\n", "AlignedStore, AuditMixin", ""),
        ("from external import *\n\n", "AlignedStore", ""),
        ("", "AlignedStore", ", super=None"),
    ],
    ids=[
        "base-rebound",
        "constructor-rebound",
        "duplicate-base",
        "multiple-bases",
        "star-import",
        "super-shadowed",
    ],
)
async def test_ambiguous_parent_bindings_do_not_supply_a_contract(
    deps: dict[str, Any], between: str, bases: str, extra_argument: str
) -> None:
    deps["provider"] = _StubLLMProvider('{"findings":[]}')
    await run_analyze_pass(
        _state(_source(between=between, bases=bases, extra_argument=extra_argument)), deps
    )
    (request,) = deps["provider"].calls
    assert _GUARD not in request.user_prompt
    assert not any(
        entry.scope_unit_name.startswith("parent constructor ") for entry in request.context_summary
    )


@pytest.mark.asyncio
async def test_oversized_parent_is_omitted_whole(deps: dict[str, Any]) -> None:
    source = _source().replace(
        "        self.width = width\n",
        '        """' + "Long parent contract. " * 250 + '"""\n        self.width = width\n',
    )
    deps["provider"] = _StubLLMProvider('{"findings":[]}')
    await run_analyze_pass(_state(source), deps)
    (request,) = deps["provider"].calls
    assert _GUARD not in request.user_prompt
    assert "Long parent contract." not in request.user_prompt
    assert not any(
        entry.scope_unit_name.startswith("parent constructor ") for entry in request.context_summary
    )


@pytest.mark.asyncio
async def test_parent_reference_uses_spare_headroom_without_defunding_primary(
    deps: dict[str, Any],
) -> None:
    source = _source().replace(
        "        self.width = width\n",
        '        """' + "Bounded parent contract. " * 100 + '"""\n        self.width = width\n',
    )
    state = _state(source)
    (changed_file,) = state.pr_context.changed_files
    prepared = _select_file_review(
        changed_file=changed_file, import_path_resolver=deps["import_path_resolver"]
    )
    assert isinstance(prepared, _PreparedFileReview)
    base_context = prepare_review_context(
        file_path=_PATH,
        content=source,
        patch=changed_file.patch,
        changed_files=state.pr_context.changed_files,
        import_path_resolver=deps["import_path_resolver"],
    )
    base_tokens = _prepare_file_prompt(
        prepared, file_path=_PATH, pass_index=0, review_context=base_context.text
    ).estimated_tokens
    # Fit the base supplement exactly; optional parent source must use spare headroom.
    budget = ceil(base_tokens / PER_FILE_CAP_FRACTION)
    deps.update(total_review_budget_tokens=budget, provider=_StubLLMProvider('{"findings":[]}'))
    baseline = await run_analyze_pass(state, {**deps, "supplemental_context_enabled": False})
    (before,) = deps["provider"].calls
    deps["provider"] = _StubLLMProvider('{"findings":[]}')
    candidate = await run_analyze_pass(state, deps)
    (after,) = deps["provider"].calls
    assert (
        candidate["analysis_rounds"][0].files_examined
        == baseline["analysis_rounds"][0].files_examined
        == (_PATH,)
    )
    assert _GUARD not in after.user_prompt
    assert not any(
        entry.scope_unit_name.startswith("parent constructor ") for entry in after.context_summary
    )
    assert tuple(
        e for e in before.context_summary if e.inclusion_reason == "changed_scope"
    ) == tuple(e for e in after.context_summary if e.inclusion_reason == "changed_scope")
    actual = (
        _estimate_tokens(after.system_prompt)
        + _estimate_tokens(after.user_prompt)
        + after.max_tokens
    )
    assert actual <= _compute_per_file_cap(budget)
