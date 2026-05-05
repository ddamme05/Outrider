"""Structural eval scenario: parse-degraded fallback on syntax error inside diff.

Per docs/spec.md §11.2 + the `parse-errors-degrade-to-judged` invariant:
a file with a syntax error INSIDE the changed region triggers the
parse-degraded fallback. Empty `ScopeUnit` set + a degraded marker
derived by the analyze node; downstream findings produced under degraded
mode get downgraded to JUDGED tier per the invariant.

V1: still skipped after coordinates lands. ast_facts ships the parse
result (`parser_outcome="failed"` + empty `scope_units` for an unparseable
file); coordinates ships `diff_line_to_scope` (returns None for any line
when `scope_units` is empty). The "degraded" derivation that combines
both into a single per-finding marker lives in the analyze node — this
scenario flips when the analyze-node spec lands.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from outrider.ast_facts import parse_python
from outrider.coordinates import diff_line_to_scope

pytestmark = pytest.mark.skip(reason="requires analyze-node degraded derivation")

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


def test_syntax_error_inside_diff_triggers_parse_degraded_fallback() -> None:
    """Whole-file unparseable → parser_outcome=failed, empty scope_units;
    diff_line_to_scope returns None; eventual analyze-node degraded=True.
    """
    parse_result = parse_python(
        source=SOURCE.encode("utf-8"),
        file_path="test.py",
        resolver=MagicMock(),
    )

    # ast_facts contract: failed parse → empty scope_units.
    assert parse_result.parser_outcome == "failed"
    assert parse_result.scope_units == ()

    # coordinates contract: empty scope_units → None for any diff_line.
    scope = diff_line_to_scope(
        file_path="test.py",
        diff_line=DIFF_LINE,
        scope_units=list(parse_result.scope_units),
    )
    assert scope is None

    # Pending analyze-node: combines parse_outcome="failed" with diff-line
    # mapping to derive degraded=True; this assertion lands when the
    # analyze-node spec ships.
    # assert analyze_node_derive_degraded(parse_result, DIFF_LINE) is EXPECTED_DEGRADED_MARKER
