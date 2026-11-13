"""Related test references retain the decorators that supply their arguments."""

from __future__ import annotations

from difflib import unified_diff
from typing import TYPE_CHECKING

import pytest
import test_analyze_node as recorders

from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER

if TYPE_CHECKING:
    from outrider.llm.base import LLMRequest
    from outrider.schemas.pr_context import ChangedFile


_TEST = (
    '@pytest.mark.parametrize("value,expected", [\n'
    '    ("x", "x"),\n'
    "])\n"
    "def test_normalize(value, expected):\n"
    "    assert normalize(value) == expected\n"
)


def _changed_file(path: str, before: str, after: str) -> ChangedFile:
    patch = "".join(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/" + path,
            tofile="b/" + path,
        )
    )
    return recorders._build_changed_file(
        path=path, content=after.encode(), content_base=before, patch=patch
    )


async def _request_with_test(before: str, after: str) -> LLMRequest:
    source = _changed_file(
        "app/normalize.py",
        "def normalize(value):\n    return value\n",
        "def normalize(value):\n    return value.strip()\n",
    )
    test = _changed_file("tests/test_normalize.py", before, after)
    state = recorders._build_review_state(
        pr_context=recorders._build_pr_context(changed_files=(source, test)),
        triage_result=recorders._build_triage_result(
            file_tiers={source.path: recorders.ReviewTier.STANDARD}
        ),
    )
    deps = recorders.deps.__wrapped__()
    deps.update(
        provider=recorders._StubLLMProvider('{"findings":[]}'),
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    await recorders.run_analyze_pass(state, deps)
    assert len(deps["provider"].calls) == 1
    return deps["provider"].calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            _TEST,
            _TEST.replace('    ("x", "x"),\n', '    ("x", "x"),\n    (" y ", "y"),\n'),
            id="decorator-only-addition",
        ),
        pytest.param(
            _TEST, _TEST.replace('("x", "x")', '(" x ", "x")'), id="decorator-replacement"
        ),
        pytest.param(
            "@custom_cases(load_cases())\n" + _TEST,
            "@custom_cases(load_cases())\n" + _TEST.replace("== expected", "== str(expected)"),
            id="stacked-dynamic-decorators",
        ),
    ],
)
async def test_related_test_source_includes_complete_function(before: str, after: str) -> None:
    request = await _request_with_test(before, after)
    assert after in request.user_prompt
    entries = [
        entry for entry in request.context_summary if entry.file_path == "tests/test_normalize.py"
    ]
    assert len(entries) == 1
    assert (entries[0].line_start, entries[0].line_end) == (1, len(after.splitlines()))
    assert request.user_prompt.count("def test_normalize(") == 1


@pytest.mark.asyncio
async def test_oversized_decorated_test_is_omitted_whole() -> None:
    before = _TEST.replace('("x", "x")', '("' + "x" * 1300 + '", "x")')
    request = await _request_with_test(before, before.replace("== expected", "== str(expected)"))
    assert "def test_normalize(" not in request.user_prompt
    assert all(entry.file_path != "tests/test_normalize.py" for entry in request.context_summary)


@pytest.mark.asyncio
async def test_parenthesized_decorator_is_complete_or_omitted() -> None:
    before = _TEST.replace("@pytest.mark.parametrize", "@(\n    pytest.mark.parametrize").replace(
        "])\n", "])\n)\n"
    )
    after = before.replace("== expected", "== str(expected)")
    request = await _request_with_test(before, after)
    entries = [
        entry for entry in request.context_summary if entry.file_path == "tests/test_normalize.py"
    ]
    if entries:
        assert after in request.user_prompt
        assert (entries[0].line_start, entries[0].line_end) == (1, len(after.splitlines()))
    else:
        assert "def test_normalize(" not in request.user_prompt
