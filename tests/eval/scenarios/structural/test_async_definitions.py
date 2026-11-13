"""Check that async definitions and async-with blocks produce valid source scopes.

Call the parser directly without a model. Async functions use the existing
function scope type; no async-specific field is expected."""

from unittest.mock import MagicMock

from outrider.ast_facts import parse_python

SOURCE = """\
import asyncio


async def fetch_data(url: str) -> dict:
    async with asyncio.timeout(5):
        return {"url": url}


async def stream_lines(handle):
    async for line in handle:
        yield line.strip()
"""

EXPECTED_SCOPE_NAMES = ("fetch_data", "stream_lines")


def test_async_def_parses_into_function_kind_scope_unit() -> None:
    """async def fn() produces ScopeUnit with kind='function' (no async-specific field)."""
    result = parse_python(SOURCE.encode(), "test.py", MagicMock())
    names = tuple(s.name for s in result.scope_units if s.kind == "function")
    for expected in EXPECTED_SCOPE_NAMES:
        assert expected in names


def test_async_def_line_spans_include_async_keyword() -> None:
    """ScopeUnit's line_start covers the line with the `async` keyword, not the body."""
    result = parse_python(SOURCE.encode(), "test.py", MagicMock())
    fetch_data = next(s for s in result.scope_units if s.name == "fetch_data")
    # The `async def fetch_data(...)` line is line 4 (1-indexed) in SOURCE.
    assert fetch_data.line_start == 4
