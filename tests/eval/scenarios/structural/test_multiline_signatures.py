"""Check that multiline function signatures have correct whole-function spans."""

from unittest.mock import MagicMock

from outrider.ast_facts import parse_python

SOURCE = """\
def process(
    user_id: int,
    items: list[dict[str, str]],
    *,
    flush: bool = True,
) -> tuple[int, int]:
    return (user_id, len(items))
"""

EXPECTED_LINE_START = 1
EXPECTED_LINE_END = 7


def test_multiline_signature_line_spans() -> None:
    """def process(...) -> tuple[int, int]: spans lines 1 (def) through 7 (last body line)."""
    result = parse_python(SOURCE.encode(), "test.py", MagicMock())
    process = next(s for s in result.scope_units if s.name == "process")
    assert process.line_start == EXPECTED_LINE_START
    assert process.line_end == EXPECTED_LINE_END
