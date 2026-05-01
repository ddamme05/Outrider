"""Structural eval scenario: multi-line function signatures with type hints.

Per spec §11.2: multi-line function signatures with type hints resolve
byte-spans correctly (line_start covers the `def` line; line_end covers
the last line of the function body, not just the signature).

V1: scaffolded; assertion runs at `ast_facts/` flip time.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires ast_facts")

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
    from outrider.ast_facts import extract_scopes  # type: ignore[import-not-found]

    scopes = extract_scopes(SOURCE)
    process = next(s for s in scopes if s.name == "process")
    assert process.line_start == EXPECTED_LINE_START
    assert process.line_end == EXPECTED_LINE_END
