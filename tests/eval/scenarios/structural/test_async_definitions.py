"""Structural eval scenario: async def + async with parse cleanly.

Per spec §11.2: `async def` and `async with` parse into the canonical
`ScopeUnit` shape (`kind="function"` for `async def`; line spans cover
the async syntax).

The canonical `ScopeUnit` does NOT distinguish async at the structural
layer (`kind: Literal["function", "method", "class"]` per spec §5.4
line 480) — async-ness is semantic, not structural. This test asserts
existing fields handle async syntax without breaking; it does NOT
assert any async-specific field.

V1: scaffolded; assertion runs at `ast_facts/` flip time.
"""

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
