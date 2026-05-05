"""Structural eval scenario: graceful degradation on syntax error outside diff.

Per docs/spec.md §11.2 + the `parse-errors-degrade-to-judged` invariant:
a file with a syntax error OUTSIDE the changed region degrades gracefully
— the analyze node reports degraded=False because the diff line itself
lies in parseable code, even though the file as a whole has errors.

V1: still skipped after coordinates lands. ast_facts ships the parse
result (with whatever `scope_units` tree-sitter recovers from the
parseable region); coordinates ships `diff_line_to_scope` (maps a
parseable-region diff line to its scope). The "graceful" derivation
that combines both into a degraded=False marker lives in the analyze
node — this scenario flips when the analyze-node spec lands.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from outrider.ast_facts import parse_python
from outrider.coordinates import diff_line_to_scope

pytestmark = pytest.mark.skip(reason="requires analyze-node degraded derivation")

# The syntax error is at the bottom of the file (unmatched paren); the
# diff hunk lies in the parseable region above.
SOURCE = """\
def changed_function():
    return 42  # diff hunk lives here, line 2


def broken_function(  # syntax error: unmatched paren, line 5
    return None
"""

DIFF_LINE = 2

EXPECTED_DEGRADED_MARKER = False  # graceful degradation, NOT full degraded mode
EXPECTED_RESOLVED_NAME = "changed_function"


def test_syntax_error_outside_diff_degrades_gracefully() -> None:
    """Diff line in parseable region → ScopeUnit returned; eventual analyze-node
    degraded=False because the diff line itself is parseable.
    """
    parse_result = parse_python(
        source=SOURCE.encode("utf-8"),
        file_path="test.py",
        resolver=MagicMock(),
    )

    # ast_facts contract: tree-sitter recovers from the trailing syntax
    # error and extracts the parseable region's scopes. The diff line
    # itself maps to a real ScopeUnit via diff_line_to_scope.
    scope = diff_line_to_scope(
        file_path="test.py",
        diff_line=DIFF_LINE,
        scope_units=list(parse_result.scope_units),
    )
    assert scope is not None
    assert scope.name == EXPECTED_RESOLVED_NAME

    # Pending analyze-node: combines parse_result + diff-line mapping to
    # derive degraded=False (the diff line is parseable even though the
    # whole-file parse may report errors elsewhere).
    # assert analyze_node_derive_degraded(parse_result, DIFF_LINE) is EXPECTED_DEGRADED_MARKER
