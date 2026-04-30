"""Structural eval scenario: parse-degraded fallback on syntax error inside diff.

Per spec §11.2 + `parse-errors-degrade-to-judged`: a file with a syntax
error INSIDE the changed region triggers the parse-degraded fallback.
Empty ScopeUnit set + a degraded marker; downstream findings produced
under degraded mode get downgraded to JUDGED tier (per the invariant).

V1: scaffolded; assertion runs at `ast_facts/` flip time.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires ast_facts")

# Syntax error is inside the changed region; the diff line itself lies
# in unparseable code.
SOURCE = """\
def healthy_function():
    return 1


def broken_function(  # diff hunk lives here, line 5 — parser fails on this region
    return None
"""

DIFF_LINE = 5

EXPECTED_DEGRADED_MARKER = True
EXPECTED_SCOPE_UNITS = ()  # empty set — parser can't resolve anything in the broken region


def test_syntax_error_inside_diff_triggers_parse_degraded_fallback() -> None:
    """ast_facts returns empty ScopeUnit set + degraded=True for unparseable diff region."""
    from outrider.ast_facts import resolve_line_to_scope  # type: ignore[import-not-found]

    result = resolve_line_to_scope(SOURCE, DIFF_LINE)
    assert result.degraded is EXPECTED_DEGRADED_MARKER
    assert tuple(result.scopes) == EXPECTED_SCOPE_UNITS
