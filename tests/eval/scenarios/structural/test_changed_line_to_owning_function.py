"""Structural eval scenario: changed line resolves to its owning ScopeUnit.

Per spec §11.2: given a diff hunk and a Python source file, `ast_facts/`
resolves the changed line to its containing `ScopeUnit` (function or
method), including nested-function cases.

V1 (this commit): scaffolded with `pytest.mark.skip("requires ast_facts")`.
The expected-output fixture is a raw dict at module top — NOT a typed
`ScopeUnit` instance, since `outrider.ast_facts` doesn't exist yet.
The `ast_facts/` spec PR flips this skip marker, reshapes the expected
output into a typed instance, and starts asserting against real output.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires ast_facts")

# Source file under test (synthetic). Module-top constant — no absent-module imports.
SOURCE = """\
def outer_function():
    def nested_helper():
        return 42  # changed line — line 3 of file
    return nested_helper()
"""

DIFF_LINE = 3  # line number of the change inside SOURCE

# Expected: line 3 resolves to `nested_helper` (the inner function),
# not to `outer_function`. Raw dict shape; typed ScopeUnit at flip time.
EXPECTED_SCOPE = {
    "kind": "function",
    "name": "nested_helper",
    "qualified_name": "outer_function.nested_helper",
}


def test_changed_line_resolves_to_owning_function() -> None:
    """Line 3 of SOURCE resolves to the nested_helper ScopeUnit, not outer_function."""
    from outrider.ast_facts import resolve_line_to_scope  # type: ignore[import-not-found]

    scope = resolve_line_to_scope(SOURCE, DIFF_LINE)
    assert scope.qualified_name == EXPECTED_SCOPE["qualified_name"]
    assert scope.kind == EXPECTED_SCOPE["kind"]
