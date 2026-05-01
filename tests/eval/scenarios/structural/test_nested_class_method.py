"""Structural eval scenario: nested class methods resolve to qualified names.

Per spec §11.2: methods inside nested classes resolve to the right
`ScopeUnit` qualified-name path.

V1: scaffolded; assertion runs at `ast_facts/` flip time.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires ast_facts")

SOURCE = """\
class Outer:
    class Inner:
        def deep_method(self):
            return "deep"

    def shallow_method(self):
        return "shallow"
"""

EXPECTED_QUALIFIED_NAMES = (
    "Outer.Inner.deep_method",
    "Outer.shallow_method",
)


def test_nested_class_method_qualified_names() -> None:
    """deep_method qualifies under Outer.Inner; shallow_method under Outer."""
    from outrider.ast_facts import extract_scopes  # type: ignore[import-not-found]

    scopes = extract_scopes(SOURCE)
    method_names = tuple(s.qualified_name for s in scopes if s.kind == "method")
    for expected in EXPECTED_QUALIFIED_NAMES:
        assert expected in method_names
