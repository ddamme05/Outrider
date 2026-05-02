"""Structural eval scenario: nested class methods resolve to qualified names.

Per spec §11.2: methods inside nested classes resolve to the right
`ScopeUnit` qualified-name path.

V1: live (flipped on the ast_facts/ V1 spec landing). Calls `parse_python`
directly and gates current ast_facts behavior for nested-class qualified names.
"""

from unittest.mock import MagicMock

from outrider.ast_facts import parse_python

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
    result = parse_python(SOURCE.encode(), "test.py", MagicMock())
    method_names = tuple(s.qualified_name for s in result.scope_units if s.kind == "method")
    for expected in EXPECTED_QUALIFIED_NAMES:
        assert expected in method_names
