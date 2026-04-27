"""Q2 — scope-node identification when decorators are present.

When a function has one or more decorators, tree-sitter-python wraps the
``function_definition`` inside a ``decorated_definition`` node. This matters
for two reasons:

1. The natural parent pointer from a matched ``function_definition`` walks to
   ``decorated_definition``, not to the enclosing ``class_definition.body``.
   Qualified-name derivation (Q3) must handle that jump.
2. ``start_point`` of the ``function_definition`` is the ``def`` line;
   ``start_point`` of the wrapping ``decorated_definition`` is the first
   ``@decorator`` line. Which one goes into ScopeUnit.line_start (spec §5.4)
   determines whether findings on decorator lines are attributed to the
   function or treated as module-level.
"""

from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor

FIXTURES = Path(__file__).parent.parent / "fixtures"
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

FUNC_QUERY = Query(
    PY_LANGUAGE,
    """
    (function_definition name: (identifier) @name) @func
    """,
)


def main() -> None:
    source = (FIXTURES / "nested_and_decorators.py").read_bytes()
    tree = parser.parse(source)

    # captures are list[Node] per key (see NOTES.md — the python-source-analysis
    # doc's example omits this, but the 0.25.2 API returns lists).
    matches = QueryCursor(FUNC_QUERY).matches(tree.root_node)
    named: dict[str, object] = {}
    for _pat, caps in matches:
        name_node = caps["name"][0]
        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        named[name] = caps["func"][0]

    # Sanity: fixture contains a decorator-stacked method called `greet`
    # (@staticmethod @log) and an un-decorated method `say`-less helper.
    assert "greet" in named, "fixture changed — expected 'greet' method"

    greet_func = named["greet"]
    parent = greet_func.parent
    assert parent is not None, "function_definition must have a parent"

    assert parent.type == "decorated_definition", (
        f"Q2 FAIL: expected 'decorated_definition' as parent of decorated "
        f"function, got {parent.type!r}"
    )

    # decorated_definition.start_point is the first decorator line,
    # function_definition.start_point is the def line.
    assert parent.start_point[0] < greet_func.start_point[0], (
        "Q2 FAIL: decorated_definition.start_point should precede function_definition.start_point"
    )

    # Un-decorated function 'top_level' should have a plain 'module' ancestor,
    # not wrapped in decorated_definition.
    top = named["top_level"]
    assert top.parent.type == "module", (
        f"Q2 FAIL: un-decorated top-level function's parent should be "
        f"'module', got {top.parent.type!r}"
    )

    print(
        "Q2 OK: decorated functions are wrapped in 'decorated_definition'; "
        "un-decorated top-level functions have 'module' as parent. "
        "ScopeUnit.line_start MUST use decorated_definition.start_point when "
        "decorators exist, otherwise findings on decorator lines orphan."
    )


if __name__ == "__main__":
    main()
