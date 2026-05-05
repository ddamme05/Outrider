"""Structural eval scenario: changed line resolves to its owning ScopeUnit.

Per docs/spec.md §11.2: given a Python source file and a diff line, the
scope at that line is found by parsing via `ast_facts.parse_python` and
mapping via `coordinates.diff_line_to_scope` per `docs/spec.md` §5.6 /
`docs/trust-boundaries.md` §3 (translation lives in `coordinates/`).

V1: live. ast_facts ships scope extraction; coordinates ships the
line-to-scope mapper; this test exercises both together for the
nested-function innermost-scope case.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from outrider.ast_facts import parse_python
from outrider.coordinates import diff_line_to_scope

# Source file under test (synthetic). Module-top constant — no absent-module imports.
SOURCE = """\
def outer_function():
    def nested_helper():
        return 42  # changed line — line 3 of file
    return nested_helper()
"""

DIFF_LINE = 3  # line number of the change inside SOURCE


def test_changed_line_resolves_to_owning_function() -> None:
    """Line 3 of SOURCE resolves to the nested_helper ScopeUnit, not outer_function.

    Innermost-scope rule: line 3 is inside both `outer_function` (lines 1-4)
    and `nested_helper` (lines 2-3); `diff_line_to_scope` returns the
    smaller line span (`nested_helper`).
    """
    parse_result = parse_python(
        source=SOURCE.encode("utf-8"),
        file_path="test.py",
        resolver=MagicMock(),  # parse_python doesn't invoke resolver on the clean path
    )

    assert parse_result.parser_outcome == "clean"
    assert parse_result.scope_units, "fixture must produce scope_units; otherwise test is vacuous"

    scope = diff_line_to_scope(
        file_path="test.py",
        diff_line=DIFF_LINE,
        scope_units=list(parse_result.scope_units),
    )

    assert scope is not None
    assert scope.qualified_name == "outer_function.nested_helper"
    assert scope.kind == "function"
