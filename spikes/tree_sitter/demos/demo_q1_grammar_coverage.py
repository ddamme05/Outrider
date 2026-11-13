"""Check the pinned Python grammar against the spike's source examples.

Parse the PyGoat fixture and modern Python syntax, reporting ERROR and MISSING
nodes. The examples reveal grammar limits relevant to
parse-errors-degrade-to-judged; they do not cover every Python program.
"""

from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor

FIXTURES = Path(__file__).parent.parent / "fixtures"
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

ERROR_QUERY = Query(PY_LANGUAGE, "(ERROR) @e (MISSING) @m")


def parse(path: Path):
    source = path.read_bytes()
    tree = parser.parse(source)
    captures = QueryCursor(ERROR_QUERY).matches(tree.root_node)
    error_nodes = []
    for _pat, caps in captures:
        for key in ("e", "m"):
            # caps values are list[Node] — one element per match, not a bare Node
            # (the canonical docs example shows bare Node; the 0.25.2 API returns
            # a list. Treat as list always.)
            for node in caps.get(key, []):
                error_nodes.append((key, node))
    return tree, error_nodes


def main() -> None:
    # Q1a — eval target must parse clean.
    pygoat = FIXTURES / "pygoat_introduction_views.py"
    tree, errors = parse(pygoat)
    assert not tree.root_node.has_error, (
        f"Q1a FAIL: {pygoat.name} has parse errors ({len(errors)} ERROR/MISSING nodes)"
    )
    print(f"Q1a OK: {pygoat.name} parsed clean, no ERROR/MISSING.")

    # Q1b — modern Python features.
    modern = FIXTURES / "modern_python.py"
    tree, errors = parse(modern)
    if tree.root_node.has_error or errors:
        # Non-fatal: document which constructs the grammar can't handle.
        locations = sorted({n.start_point[0] + 1 for _k, n in errors})
        print(
            f"Q1b PARTIAL: {modern.name} has {len(errors)} error nodes at "
            f"lines {locations}. Document these as grammar limitations."
        )
        raise AssertionError(
            "Q1b: tree-sitter-python 0.25.0 cannot cleanly parse "
            f"modern_python.py (lines: {locations}). Inspect and add to NOTES."
        )
    print(f"Q1b OK: {modern.name} parsed clean — 3.13 features supported.")


if __name__ == "__main__":
    main()
