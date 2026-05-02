"""Structural eval scenario: changed line resolves to its owning ScopeUnit.

Per spec §11.2: given a diff hunk and a Python source file, `ast_facts/`
resolves the changed line to its containing `ScopeUnit` (function or
method), including nested-function cases.

V1: still skipped after the ast_facts/ landing. Diff-line-to-owning-scope
translation is owned by `coordinates.diff_line_to_scope` per
`docs/trust-boundaries.md` §3 (no coordinate math outside `coordinates/`);
this scenario flips when the `coordinates/` module spec lands.
The expected-output fixture stays a raw dict; the `coordinates/` spec
PR reshapes it into the canonical typed instance.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires coordinates.diff_line_to_scope")

# Source file under test (synthetic). Module-top constant — no absent-module imports.
SOURCE = """\
def outer_function():
    def nested_helper():
        return 42  # changed line — line 3 of file
    return nested_helper()
"""

DIFF_LINE = 3  # line number of the change inside SOURCE

# Expected: line 3 resolves to `nested_helper` (the inner function),
# not to `outer_function`. Raw dict shape; the `coordinates/` spec PR
# reshapes this into the canonical typed `ScopeUnit` when the scenario
# flips live.
EXPECTED_SCOPE = {
    "kind": "function",
    "name": "nested_helper",
    "qualified_name": "outer_function.nested_helper",
}


def test_changed_line_resolves_to_owning_function() -> None:
    """Line 3 of SOURCE resolves to the nested_helper ScopeUnit, not outer_function."""
    from outrider.ast_facts import resolve_line_to_scope  # type: ignore[import-not-found]

    result = resolve_line_to_scope(SOURCE, DIFF_LINE)
    assert result.degraded is False
    scope = result.scope
    assert scope.qualified_name == EXPECTED_SCOPE["qualified_name"]
    assert scope.kind == EXPECTED_SCOPE["kind"]
