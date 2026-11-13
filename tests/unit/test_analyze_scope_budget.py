"""Fund the selected scope rather than all unchanged source in its file."""

# ruff: noqa: F811 — pytest injects the imported deps fixture by parameter name.
from __future__ import annotations

from difflib import unified_diff
from typing import Any

import pytest
from test_analyze_node import (  # noqa: F401 — imports the existing pytest fixture
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _StubLLMProvider,
    deps,
    run_analyze_pass,
)

from outrider.agent.nodes.analyze import _compute_per_file_cap, _estimate_tokens
from outrider.ast_facts.models import SkipReason
from outrider.schemas.pr_context import ChangedFile
from outrider.schemas.triage_result import ReviewTier

_REVIEW_BUDGET = 88_000
_PATH = "src/large_module.py"


def _changed_file(before: str, after: str) -> ChangedFile:
    patch = "".join(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{_PATH}",
            tofile=f"b/{_PATH}",
        )
    )
    return ChangedFile(
        path=_PATH,
        status="modified",
        additions=1,
        deletions=1,
        content_base=before,
        content_head=after,
        patch=patch,
        language="python",
    )


@pytest.mark.asyncio
async def test_small_changed_scope_in_large_file_reaches_model(deps: dict[str, Any]) -> None:
    """Unchanged definitions must not consume the changed function's allocation."""
    unchanged = "\n".join(
        f"def unchanged_{index:04d}():\n    return {index}\n" for index in range(1_200)
    )
    before = "def changed():\n    return 0\n\n" + unchanged
    after = "def changed():\n    return 1\n\n" + unchanged
    changed = _changed_file(before, after)
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(changed,)),
        triage_result=_build_triage_result(file_tiers={_PATH: ReviewTier.DEEP}),
    )
    deps["provider"] = _StubLLMProvider('{"findings":[]}')
    deps["total_review_budget_tokens"] = _REVIEW_BUDGET
    # Isolate the primary request; optional reference selection has its own tests.
    deps["supplemental_context_enabled"] = False

    result = await run_analyze_pass(state, deps)

    assert len(deps["provider"].calls) == 1
    (request,) = deps["provider"].calls
    assert "def changed():" in request.user_prompt
    assert "unchanged_1199" not in request.user_prompt
    actual_request_tokens = (
        _estimate_tokens(request.system_prompt)
        + _estimate_tokens(request.user_prompt)
        + request.max_tokens
    )
    assert actual_request_tokens <= _compute_per_file_cap(_REVIEW_BUDGET)
    (round_,) = result["analysis_rounds"]
    assert round_.files_examined == (_PATH,)
    assert round_.files_skipped == ()


@pytest.mark.asyncio
async def test_oversized_selected_scope_still_skips_without_model(deps: dict[str, Any]) -> None:
    """The same limit still rejects a genuinely large rendered function."""
    prefix = "def changed():\n    value = '" + "x" * 50_000 + "'\n"
    changed = _changed_file(prefix + "    return 0\n", prefix + "    return 1\n")
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(changed,)),
        triage_result=_build_triage_result(file_tiers={_PATH: ReviewTier.DEEP}),
    )
    deps["provider"] = _StubLLMProvider('{"findings":[]}')
    deps["total_review_budget_tokens"] = _REVIEW_BUDGET
    deps["supplemental_context_enabled"] = False

    result = await run_analyze_pass(state, deps)

    assert deps["provider"].calls == []
    (event,) = deps["file_examination_sink"].events
    assert event.skip_reason is SkipReason.COST_BUDGET_EXHAUSTED
    (round_,) = result["analysis_rounds"]
    assert round_.files_examined == ()
    assert round_.files_skipped == (_PATH,)
