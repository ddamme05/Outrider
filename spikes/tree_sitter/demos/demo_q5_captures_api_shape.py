"""Q5 — captures API shape regression guard.

Motivation: the canonical docs example in
`aegis-docs::tree-sitter/python-source-analysis.md` §"Complete Runnable Example"
writes `captures["function.def"]` and uses the result as a bare `Node`
(`defn.start_byte`). On the spike's pinned stack
(`tree-sitter==0.25.2`, `tree-sitter-python==0.25.0`), that raises
`AttributeError: 'list' object has no attribute 'start_byte'` because the
actual shape is `list[Node]`, not `Node`.

Every other demo in this spike subscripts with `caps[key][0]`, so the list
shape is assumed implicitly. This demo isolates the assumption so that a
future tree-sitter release that flips the shape (either way) fails here
first, with a clear error, before all the other demos fail mysteriously.

Not a correctness check of tree-sitter; a regression guard against
upstream API drift.
"""

import tree_sitter_python
from tree_sitter import Language, Node, Parser, Query, QueryCursor

PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

SOURCE = b"""\
def alpha():
    pass


def beta(x, y):
    return x + y


def gamma():
    pass
"""

# Un-quantified single-capture query: one `@name` per match.
SINGLE_CAPTURE_QUERY = Query(
    PY_LANGUAGE,
    """
    (function_definition name: (identifier) @name)
    """,
)


def main() -> None:
    tree = parser.parse(SOURCE)

    # --- Assertion 1: captures[key] is list[Node], not bare Node.
    matches = list(QueryCursor(SINGLE_CAPTURE_QUERY).matches(tree.root_node))
    assert len(matches) == 3, (
        f"Q5 FAIL: expected 3 function matches, got {len(matches)}"
    )

    for pat_idx, caps in matches:
        assert isinstance(caps, dict), (
            f"Q5 FAIL: caps is {type(caps).__name__}, expected dict"
        )
        name_val = caps["name"]
        assert isinstance(name_val, list), (
            f"Q5 FAIL: captures['name'] is {type(name_val).__name__}, "
            "expected list. The docs example at "
            "aegis-docs::tree-sitter/python-source-analysis.md treats this "
            "as a bare Node; on tree-sitter==0.25.2 it is list[Node]."
        )
        assert len(name_val) == 1, (
            f"Q5 FAIL: un-quantified capture returned {len(name_val)} "
            "elements, expected exactly 1 per match"
        )
        assert isinstance(name_val[0], Node), (
            f"Q5 FAIL: captures['name'][0] is {type(name_val[0]).__name__}, "
            "expected Node"
        )

    # --- Assertion 2: treating the raw capture value as a Node raises.
    # This is the docs-vs-reality drift the spike caught. If tree-sitter ever
    # flips the shape back to bare Node, this assertion starts failing and
    # we know to re-evaluate every `caps[key][0]` subscript in ast_facts/.
    _, first_caps = matches[0]
    try:
        _ = first_caps["name"].start_byte
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "Q5 FAIL: first_caps['name'].start_byte did NOT raise "
            "AttributeError. The captures API shape may have changed back "
            "to bare Node. Re-read "
            "aegis-docs::tree-sitter/python-source-analysis.md and audit "
            "every list-subscripted capture call in the spike and in "
            "ast_facts/ — the whole spike assumes list[Node]."
        )

    # --- Assertion 3: the three function names extract correctly via the
    # documented list[0]-subscript pattern. This is the pattern ast_facts/
    # will use; asserting it here ensures a future API change can't silently
    # break extraction by making the bracket-subscript succeed with wrong data.
    names = []
    for _, caps in matches:
        name_node = caps["name"][0]
        names.append(SOURCE[name_node.start_byte : name_node.end_byte].decode("utf-8"))
    assert names == ["alpha", "beta", "gamma"], (
        f"Q5 FAIL: extracted names {names!r}, expected ['alpha', 'beta', 'gamma']"
    )

    print(
        "Q5 OK: captures API is list[Node] as of tree-sitter==0.25.2. "
        "Bare-Node access raises AttributeError as expected; "
        "list[0]-subscript extracts names correctly."
    )


if __name__ == "__main__":
    main()
