"""Test runner settings reach only funded HEAD test-file reviews."""

# ruff: noqa: F811 — pytest injects the imported deps fixture by name.
from __future__ import annotations

import asyncio
from difflib import unified_diff
from typing import Any

import pytest
from test_analyze_node import (  # noqa: F401
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _StubLLMProvider,
    deps,
    run_analyze_pass,
)
from test_import_reference_context import _GitHubHTTPError, _RepositoryContents

from outrider.agent.nodes.analyze import analyze
from outrider.schemas import ChangedFile, ReviewState
from outrider.schemas.triage_result import ReviewTier

PACKAGE = """{
  "scripts": {
    "test": "ava"
  },
  "devDependencies": {
    "ava": "6.2.0"
  },
  "ava": {
    "serial": true
  }
}
"""
TEST = "import test from 'ava';\ntest('example', t => {\n  t.is(42, 42);\n});\n"


def changed(path: str, before: str, after: str) -> ChangedFile:
    patch = "".join(
        unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="a/" + path,
            tofile="b/" + path,
        )
    )
    return ChangedFile(
        path=path,
        status="modified",
        additions=1,
        deletions=1,
        patch=patch,
        content_base=before,
        content_head=after,
    )


def review(test_path: str, package_path: str | None = None) -> ReviewState:
    files = [changed(test_path, TEST.replace("42", "41"), TEST)]
    if package_path is not None:
        files.append(changed(package_path, PACKAGE.replace("true", "false"), PACKAGE))
    return _build_review_state(
        pr_context=_build_pr_context(changed_files=tuple(files)),
        triage_result=_build_triage_result(file_tiers={test_path: ReviewTier.STANDARD}),
    )


@pytest.mark.asyncio
async def test_captured_sibling_settings_reach_actual_request_with_exact_source_ranges(
    deps: dict[str, Any],
) -> None:
    github = _RepositoryContents({})
    state = review("packages/colors/test.js", "packages/colors/package.json")
    provider = _StubLLMProvider('{"findings":[]}')
    await run_analyze_pass(state, {**deps, "provider": provider, "github_factory": github.factory})
    (request,) = provider.calls
    assert '"serial": true' in request.user_prompt
    refs = [
        item for item in request.context_summary if item.file_path == "packages/colors/package.json"
    ]
    assert [(item.line_start, item.line_end) for item in refs] == [(3, 3), (8, 10)]
    assert all(
        item.source_side == "head" and item.inclusion_reason == "captured_reference"
        for item in refs
    )
    github.factory.assert_not_awaited()


@pytest.mark.asyncio
async def test_unchanged_nearest_owner_uses_pinned_lookup_once_across_tests(
    deps: dict[str, Any],
) -> None:
    github = _RepositoryContents({"packages/colors/package.json": PACKAGE})
    first = review("packages/colors/tests/one.test.js")
    second = changed("packages/colors/tests/two.test.ts", TEST.replace("42", "41"), TEST)
    files = first.pr_context.changed_files + (second,)
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=files),
        triage_result=_build_triage_result(file_tiers={f.path: ReviewTier.STANDARD for f in files}),
        is_eval=False,
    )
    command = await analyze(state, **deps, github_factory=github.factory)
    assert all('"serial": true' in send.arg.review_context for send in command.goto)
    assert github.requests == [
        (state.pr_context.owner, state.pr_context.repo, path, state.pr_context.head_sha)
        for path in ("packages/colors/tests/package.json", "packages/colors/package.json")
    ]
    github.factory.assert_awaited_once_with(state.pr_context.installation_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("nearer", [None, _GitHubHTTPError(403), b"\x80", "{}", "x" * 32769])
async def test_unresolved_nearer_package_blocks_captured_ancestor(
    deps: dict[str, Any],
    nearer: str | bytes | Exception | None,
) -> None:
    state = review("packages/colors/test.js", "package.json")
    github = _RepositoryContents({"packages/colors/package.json": nearer})
    command = await analyze(state, **deps, github_factory=github.factory)
    (requested,) = command.goto
    assert '"serial": true' not in requested.arg.review_context
    assert not requested.arg.supplemental_context_manifest
    assert [req[2] for req in github.requests] == ["packages/colors/package.json"]


@pytest.mark.asyncio
async def test_package_lookups_share_one_deadline_and_keep_captured_settings(
    deps: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib import import_module

    analyze_module = import_module("outrider.agent.nodes.analyze")
    monkeypatch.setattr(analyze_module, "TEST_LAUNCH_LOOKUP_SECONDS", 0.03)
    paths = ["a/test.js", "b/test.js", "c/test.js"]
    files = [changed(path, TEST.replace("42", "41"), TEST) for path in paths]
    files.append(changed("c/package.json", PACKAGE.replace("true", "false"), PACKAGE))
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=tuple(files)),
        triage_result=_build_triage_result(file_tiers=dict.fromkeys(paths, ReviewTier.STANDARD)),
    )
    cancelled = []

    async def delayed_client(installation_id: int) -> Any:
        try:
            await asyncio.Future()
        finally:
            cancelled.append(installation_id)

    command = await asyncio.wait_for(analyze(state, **deps, github_factory=delayed_client), 2)
    contexts = {send.arg.changed_file.path: send.arg.review_context for send in command.goto}
    assert [path for path in paths if '"serial": true' in contexts[path]] == ["c/test.js"]
    assert cancelled == [state.pr_context.installation_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "source", "budget"),
    [
        ("implementation.js", TEST, None),
        ("test.js", TEST.replace("'ava'", "'./helpers.js'"), None),
        ("test.ts", "import type {Test} from 'ava';\nexport type Run = Test<42>;\n", None),
        ("test.js", TEST, 1),
    ],
)
async def test_ineligible_or_unfunded_file_makes_no_launch_reads(
    deps: dict[str, Any], path: str, source: str, budget: int | None
) -> None:
    github = _RepositoryContents({"package.json": PACKAGE})
    state = _build_review_state(
        pr_context=_build_pr_context(
            changed_files=(changed(path, source.replace("42", "41"), source),)
        ),
        triage_result=_build_triage_result(file_tiers={path: ReviewTier.STANDARD}),
    )
    settings = dict(deps, import_reference_context_enabled=True, github_factory=github.factory)
    if budget is not None:
        settings["total_review_budget_tokens"] = budget
    await analyze(state, **settings)
    github.factory.assert_not_awaited()


@pytest.mark.asyncio
async def test_python_and_launch_lookups_share_the_existing_request_limit(
    deps: dict[str, Any],
) -> None:
    names = [f"helper{n}" for n in range(8)]
    source = "".join(f"from .{name} import {name}\n" for name in names)
    source += "def changed():\n    return 42 + " + " + ".join(f"{n}()" for n in names) + "\n"
    files = (
        changed("pkg/check.py", source.replace("42", "41"), source),
        changed("test.js", TEST.replace("42", "41"), TEST),
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=files),
        triage_result=_build_triage_result(
            file_tiers={file.path: ReviewTier.STANDARD for file in files}
        ),
    )
    github = _RepositoryContents({"package.json": PACKAGE})
    command = await analyze(
        state, **deps, import_reference_context_enabled=True, github_factory=github.factory
    )
    assert len(github.requests) == 16
    assert all(req[2].endswith(".py") for req in github.requests)
    target = next(send.arg for send in command.goto if send.arg.changed_file.path == "test.js")
    assert not target.supplemental_context_manifest


@pytest.mark.asyncio
async def test_tight_primary_allocation_omits_unfunded_launch_context(deps: dict[str, Any]) -> None:
    state = review("test.js", "package.json")
    github = _RepositoryContents({})
    baseline = await analyze(state, **deps, github_factory=github.factory)
    (generous,) = baseline.goto
    assert '"serial": true' in generous.arg.review_context
    unknown = await analyze(review("test.js", "example.json"), **deps)
    (primary,) = unknown.goto
    tight = await analyze(
        state, **{**deps, "total_review_budget_tokens": primary.arg.allocation_tokens * 4}
    )
    (kept,) = tight.goto
    assert kept.arg.allocation_tokens >= primary.arg.allocation_tokens
    assert '"serial": true' not in kept.arg.review_context
    assert not kept.arg.supplemental_context_manifest
