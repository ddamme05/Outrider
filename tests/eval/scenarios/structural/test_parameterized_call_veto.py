"""Check when a parameterized SQL call can rule out a model proposal.

Use real parsing and line containment, including indented execute calls.
No model runs; unit tests cover the individual call-shape variations."""

from __future__ import annotations

from outrider.ast_facts.parameterized_calls import scan_parameterized_calls
from outrider.coordinates import line_range_vetoed_by_parameterized_call

_HANDLER_SOURCE = (
    "import db\n"  # 1
    "\n"  # 2
    "\n"  # 3
    "class SearchRepo:\n"  # 4
    "    def find(self, cursor, query, limit):\n"  # 5
    "        page_size = int(limit)\n"  # 6
    '        cursor.execute("SELECT * FROM r WHERE q = %s LIMIT %s", (query, page_size))\n'  # 7
    "        rows = cursor.fetchall()\n"  # 8
    '        cursor.execute(f"SELECT count(*) FROM {query}")\n'  # 9
    "        return rows\n"  # 10
)


def _veto(line_start: int, line_end: int) -> bool:
    scan = scan_parameterized_calls(_HANDLER_SOURCE.encode("utf-8"))
    return line_range_vetoed_by_parameterized_call(line_start, line_end, scan)


def test_indented_parameterized_execute_is_vetoed() -> None:
    """The spec-pinned case: nested two levels deep (class → method body),
    single line, literal SQL + separate params tuple. A whole-line byte
    span starts at column 0, before the call node's token start — line
    space is what makes this veto actually fire on real code."""
    assert _veto(7, 7) is True


def test_fstring_execute_is_never_vetoed() -> None:
    """Line 9 is a real injection vector — the model's sql_injection
    claim there must always flow through."""
    assert _veto(9, 9) is False


def test_range_spanning_safe_and_unsafe_is_not_vetoed() -> None:
    assert _veto(7, 9) is False


def test_range_wider_than_the_call_is_not_vetoed() -> None:
    assert _veto(6, 7) is False


def test_unrelated_lines_are_not_vetoed() -> None:
    assert _veto(6, 6) is False
    assert _veto(8, 8) is False


def test_syntax_error_anywhere_disables_the_veto() -> None:
    """`parse-errors-degrade-to-judged`: a tree with ANY error returns the
    empty scan, so the veto cannot fire on recovery-shaped nodes — even
    when the error is far from the call site."""
    broken = _HANDLER_SOURCE + "def broken(:\n"
    scan = scan_parameterized_calls(broken.encode("utf-8"))
    assert line_range_vetoed_by_parameterized_call(7, 7, scan) is False
