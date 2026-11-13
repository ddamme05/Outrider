"""Optional helper references preserve funded primary reviews under tight budgets."""

from __future__ import annotations

from difflib import unified_diff
from typing import TYPE_CHECKING

import pytest
import test_analyze_node as recorders

from outrider.agent.nodes import analyze as analyze_node
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER

if TYPE_CHECKING:
    from outrider.schemas.pr_context import ChangedFile


def _changed_files(suffix: str, language: str) -> tuple[ChangedFile, ...]:
    files = []
    for number in range(3):
        helper = f"helper_{number}"
        contract = "Public helper contract. " * 55
        if suffix == ".py":
            source = (
                f'def {helper}(value):\n    """{contract}"""\n    return value\n\n'
                f"def changed(value):\n    return {helper}(value) + 1\n"
            )
        else:
            source = (
                f"function {helper}(value) {{\n    /* {contract} */\n    return value;\n}}\n\n"
                f"function changed(value) {{\n    return {helper}(value) + 1;\n}}\n"
            )
        before = source.replace(f"{helper}(value) + 1", f"{helper}(value)")
        path = f"app/file{number}{suffix}"
        patch = "".join(
            unified_diff(
                before.splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile="a/" + path,
                tofile="b/" + path,
            )
        )
        files.append(
            recorders._build_changed_file(
                path=path, content=source.encode(), content_base=before, patch=patch
            ).model_copy(update={"language": language})
        )
    return tuple(files)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "language", "budget"),
    [
        pytest.param(".py", "python", 74_640, id="python"),
        pytest.param(".js", "javascript", 74_740, id="javascript"),
        pytest.param(".ts", "typescript", 74_740, id="typescript"),
    ],
)
async def test_optional_helpers_preserve_primary_reviews(
    suffix: str, language: str, budget: int
) -> None:
    deps = recorders.deps.__wrapped__()
    files = _changed_files(suffix, language)
    state = recorders._build_review_state(
        pr_context=recorders._build_pr_context(changed_files=files),
        triage_result=recorders._build_triage_result(
            file_tiers={file.path: recorders.ReviewTier.STANDARD for file in files}
        ),
    )
    deps.update(
        provider=recorders._StubLLMProvider('{"findings":[]}'),
        total_review_budget_tokens=budget,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    baseline = await recorders.run_analyze_pass(state, deps)
    baseline_requests = {request.phase_key: request for request in deps["provider"].calls}
    assert len(baseline_requests) == 3

    deps["provider"] = recorders._StubLLMProvider('{"findings":[]}')
    expanded = await recorders.run_analyze_pass(
        state, {**deps, "called_function_context_enabled": True}
    )
    expanded_requests = {request.phase_key: request for request in deps["provider"].calls}
    assert expanded_requests.keys() == baseline_requests.keys()
    assert (
        [outcome.skip_reason for outcome in expanded["analyze_worker_outcomes"]]
        == [outcome.skip_reason for outcome in baseline["analyze_worker_outcomes"]]
        == [None, None, None]
    )
    for key, request in expanded_requests.items():
        assert request.user_prompt == baseline_requests[key].user_prompt
        assert request.context_summary == baseline_requests[key].context_summary
    assert deps["total_review_budget_tokens"] == budget
    assert (
        sum(outcome.estimated_tokens for outcome in expanded["analyze_worker_outcomes"]) <= budget
    )

    deps["provider"] = recorders._StubLLMProvider('{"findings":[]}')
    await recorders.run_analyze_pass(
        state,
        {
            **deps,
            "total_review_budget_tokens": analyze_node.DEFAULT_REVIEW_BUDGET_TOKENS,
            "called_function_context_enabled": True,
        },
    )
    assert len(deps["provider"].calls) == 3
    for number, request in enumerate(deps["provider"].calls):
        assert any(
            entry.inclusion_reason == "captured_reference"
            and entry.scope_unit_name.startswith(f"helper_{number}@")
            for entry in request.context_summary
        )
        assert tuple(
            entry for entry in request.context_summary if entry.inclusion_reason == "changed_scope"
        ) == tuple(
            entry
            for entry in baseline_requests[request.phase_key].context_summary
            if entry.inclusion_reason == "changed_scope"
        )
