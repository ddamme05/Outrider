"""Structural eval scenario: graceful degradation on syntax error outside diff.

Per spec §11.2 + `parse-errors-degrade-to-judged`: a file with a syntax
error OUTSIDE the changed region degrades gracefully — `ast_facts/`
returns `ScopeUnit` objects for the parseable region, not an empty set.

V1: still skipped after the ast_facts/ landing. ast_facts/ ships the
per-scope `has_error` map, but the "graceful degradation" outcome
combines that map with `coordinates/`'s changed-region-to-scope mapping
in the analyze node. This scenario flips when `coordinates/` and the
analyze-node spec land.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires coordinates + analyze-node degraded derivation")

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
    """ast_facts returns ScopeUnits for the parseable region; degraded marker is False."""
    from outrider.ast_facts import resolve_line_to_scope  # type: ignore[import-not-found]

    result = resolve_line_to_scope(SOURCE, DIFF_LINE)
    assert result.scope.name == EXPECTED_RESOLVED_NAME
    assert result.degraded is EXPECTED_DEGRADED_MARKER
