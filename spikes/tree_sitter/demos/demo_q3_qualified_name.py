"""Q3 — qualified-name derivation.

spec §5.4 requires ``ScopeUnit.qualified_name`` in the form
``"ClassName.method_name"`` (and, by extension, deeper nestings).

Walk from a function_definition upward: skip ``decorated_definition``,
``block``, and ``module`` nodes; collect the ``name`` field of every
``function_definition`` or ``class_definition`` encountered; join with '.'.
"""

from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor

FIXTURES = Path(__file__).parent.parent / "fixtures"
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

SCOPE_QUERY = Query(
    PY_LANGUAGE,
    """
    (function_definition name: (identifier) @name) @scope
    (class_definition    name: (identifier) @name) @scope
    """,
)

NAMED_SCOPES = {"function_definition", "class_definition"}


def qualified_name(node, source: bytes) -> str:
    parts: list[str] = []
    cur = node
    while cur is not None:
        if cur.type in NAMED_SCOPES:
            name_field = cur.child_by_field_name("name")
            if name_field is not None:
                parts.append(source[name_field.start_byte : name_field.end_byte].decode("utf-8"))
        cur = cur.parent
    return ".".join(reversed(parts))


def main() -> None:
    source = (FIXTURES / "nested_and_decorators.py").read_bytes()
    tree = parser.parse(source)

    qnames: set[str] = set()
    for _pat, caps in QueryCursor(SCOPE_QUERY).matches(tree.root_node):
        # caps["scope"] is list[Node] — one element per match.
        qnames.add(qualified_name(caps["scope"][0], source))

    expected = {
        "log",
        "log.wrapper",
        "retry",
        "retry.decorator",
        "retry.decorator.wrapper",
        "Outer",
        "Outer.Inner",
        "Outer.Inner.greet",
        "Outer.Inner.greet._clean",
        "Outer.Inner.fetch",
        "top_level",
        "top_level.nested_one",
        "top_level.nested_one.nested_two",
    }
    missing = expected - qnames
    extra = qnames - expected
    assert not missing, f"Q3 FAIL: missing qualified names: {sorted(missing)}"
    assert not extra, (
        f"Q3 FAIL: unexpected qualified names: {sorted(extra)}. "
        "Update the fixture or the expected set — drift here means ast_facts "
        "will also get qualified_name wrong."
    )
    print(
        f"Q3 OK: all {len(expected)} expected qualified names derived "
        "by walking function_definition/class_definition parent chain."
    )


if __name__ == "__main__":
    main()
