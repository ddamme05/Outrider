"""Keep captured sibling references inside the existing planner and cache contracts."""

# ruff: noqa: F811 — pytest injects the imported deps fixture by parameter name.
from __future__ import annotations

from datetime import UTC, datetime
from difflib import unified_diff
from typing import Any
from uuid import UUID, uuid4

import pytest
from test_analyze_node import (  # noqa: F401 — reuse the existing pytest fixture.
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _StubLLMProvider,
    deps,
    run_analyze_pass,
)

from outrider.agent.nodes.analyze import analyze
from outrider.agent.nodes.cache_config import CacheMode
from outrider.cache import CacheEntry, CacheScope
from outrider.schemas import ChangedFile, ReviewState
from outrider.schemas.triage_result import ReviewTier


class _MemoryCache:
    """Reuse only the exact keys written by earlier reviews, as serve mode does."""

    def __init__(self) -> None:
        self.scope = CacheScope(
            installation_id=99999,
            repo_id=7,
            is_eval=False,
            retention_expires_at=datetime(2099, 1, 1, tzinfo=UTC),
        )
        self.entries: dict[str, CacheEntry] = {}
        self.lookups: list[str] = []

    async def resolve_scope(self, review_id: UUID) -> CacheScope:  # noqa: ARG002
        return self.scope

    async def lookup(
        self, cache_key: str, *, is_eval: bool, exclude_source_review_id: UUID | None = None
    ) -> CacheEntry | None:
        assert is_eval is self.scope.is_eval
        self.lookups.append(cache_key)
        entry = self.entries.get(cache_key)
        if entry is not None and entry.source_review_id == exclude_source_review_id:
            return None
        return entry

    async def write(self, **values: Any) -> None:
        self.entries[values["cache_key"]] = CacheEntry(
            cache_key=values["cache_key"],
            payload=values["payload"],
            source_review_id=values["source_review_id"],
            file_path=values["file_path"],
            created_at=datetime.now(UTC),
        )


def _changed_file(path: str, before: str, after: str) -> ChangedFile:
    return ChangedFile(
        path=path,
        status="modified",
        additions=1,
        deletions=1,
        content_base=before,
        content_head=after,
        patch="".join(
            unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        ),
        language="python",
    )


def _state(
    *, sibling_type: str = "int", count: int = 1, review_declaration: bool = False
) -> ReviewState:
    files = []
    tiers = {}
    for index in range(count):
        stem = f"pkg/contracts{index}"
        implementation = _changed_file(
            stem + ".py",
            "def parse(value):\n    return value\n",
            "def parse(value):\n    return int(value)\n",
        )
        declaration = _changed_file(
            stem + ".pyi",
            "def parse(value: str) -> float: ...\n",
            f"def parse(value: str) -> {sibling_type}: ...\n",
        )
        files.extend((implementation, declaration))
        tiers[declaration.path if review_declaration else implementation.path] = ReviewTier.DEEP
    return _build_review_state(
        pr_context=_build_pr_context(changed_files=tuple(files)),
        triage_result=_build_triage_result(file_tiers=tiers),
        is_eval=False,
    ).model_copy(update={"review_id": uuid4()})


@pytest.mark.asyncio
async def test_sibling_change_invalidates_the_actual_worker_cache(deps: dict[str, Any]) -> None:
    store = _MemoryCache()
    provider = _StubLLMProvider('{"findings":[]}')
    options = {
        **deps,
        "provider": provider,
        "analyze_cache_store": store,
        "cache_mode": CacheMode.SERVE,
        "companion_context_enabled": True,
    }
    original = await run_analyze_pass(_state(), options)
    repeated = await run_analyze_pass(_state(), options)
    assert len(provider.calls) == 1
    assert original["analyze_worker_outcomes"][0].source == "parser"
    assert repeated["analyze_worker_outcomes"][0].source == "cache_serve"

    changed = await run_analyze_pass(_state(sibling_type="str"), options)

    assert len(provider.calls) == 2
    assert changed["analyze_worker_outcomes"][0].source == "parser"
    assert store.lookups[0] == store.lookups[1] != store.lookups[2]
    first, second = provider.calls
    assert "def parse(value: str) -> int: ..." in first.user_prompt
    assert "def parse(value: str) -> str: ..." in second.user_prompt
    assert first.system_prompt == second.system_prompt
    assert first.response_schema_json == second.response_schema_json
    assert tuple(
        e for e in first.context_summary if e.inclusion_reason == "changed_scope"
    ) == tuple(e for e in second.context_summary if e.inclusion_reason == "changed_scope")
    references = [e for e in first.context_summary if e.inclusion_reason == "captured_reference"]
    assert [(e.file_path, e.line_start, e.line_end) for e in references] == [
        ("pkg/contracts0.pyi", 1, 1)
    ]


@pytest.mark.asyncio
async def test_disabled_companion_context_keeps_the_original_request_and_cache(
    deps: dict[str, Any],
) -> None:
    store = _MemoryCache()
    provider = _StubLLMProvider('{"findings":[]}')
    options = {
        **deps,
        "provider": provider,
        "analyze_cache_store": store,
        "cache_mode": CacheMode.SERVE,
    }
    await run_analyze_pass(_state(), options)
    await run_analyze_pass(_state(sibling_type="str"), options)

    assert len(provider.calls) == 1
    assert store.lookups[0] == store.lookups[1]
    assert "def parse(value: str) -> int: ..." not in provider.calls[0].user_prompt
    assert not any(
        e.inclusion_reason == "captured_reference" for e in provider.calls[0].context_summary
    )


@pytest.mark.asyncio
async def test_declaration_review_receives_the_captured_implementation(
    deps: dict[str, Any],
) -> None:
    provider = _StubLLMProvider('{"findings":[]}')
    result = await run_analyze_pass(
        _state(review_declaration=True),
        {**deps, "provider": provider, "companion_context_enabled": True},
    )

    (request,) = provider.calls
    assert "return int(value)" in request.user_prompt
    references = [e for e in request.context_summary if e.inclusion_reason == "captured_reference"]
    assert [(e.file_path, e.line_start, e.line_end) for e in references] == [
        ("pkg/contracts0.py", 1, 2)
    ]
    assert result["analysis_rounds"][0].files_examined == ("pkg/contracts0.pyi",)


@pytest.mark.asyncio
async def test_companion_omission_preserves_tight_budget_coverage_and_cache(
    deps: dict[str, Any],
) -> None:
    state = _state(count=4)
    generous_plan = await analyze(state, **deps)
    largest_allocation = max(send.arg.allocation_tokens for send in generous_plan.goto)
    options = {**deps, "total_review_budget_tokens": largest_allocation * 4}
    baseline = await analyze(state, **options)
    candidate = await analyze(state, **options, companion_context_enabled=True)

    assert len(baseline.goto) == 4
    assert sum(send.arg.allocation_tokens > 0 for send in baseline.goto) == 3
    assert [send.arg for send in candidate.goto] == [send.arg for send in baseline.goto]

    store = _MemoryCache()
    provider = _StubLLMProvider('{"findings":[]}')
    options.update(provider=provider, analyze_cache_store=store, cache_mode=CacheMode.SERVE)
    before = await run_analyze_pass(state, options)
    after = await run_analyze_pass(
        state.model_copy(update={"review_id": uuid4()}),
        {**options, "companion_context_enabled": True},
    )

    assert len(provider.calls) == 3
    assert store.lookups[:3] == store.lookups[3:]
    assert after["analysis_rounds"][0].files_examined == before["analysis_rounds"][0].files_examined
    assert after["analysis_rounds"][0].files_skipped == before["analysis_rounds"][0].files_skipped
    assert (
        sum(o.estimated_tokens for o in after["analyze_worker_outcomes"])
        <= (options["total_review_budget_tokens"])
    )
