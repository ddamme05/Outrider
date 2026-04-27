"""Q4 — byte/point duality on multi-byte UTF-8.

python-source-analysis.md uses ``source[start_byte:end_byte].decode("utf-8")``
to extract text from a node. For that to be correct on files with non-ASCII
identifiers or emoji:

1. start_byte/end_byte must always land on UTF-8 character boundaries.
2. start_point/end_point (row, column-in-bytes) must stay consistent with
   start_byte/end_byte — decoding slices taken either way must produce the
   same text.

Coordinate translation (spec §5.6) uses bytes; if points and bytes ever
disagree, coordinates.tree_sitter_to_github() has no single correct answer.
"""

from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor

FIXTURES = Path(__file__).parent.parent / "fixtures"
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

FUNC_QUERY = Query(
    PY_LANGUAGE,
    "(function_definition name: (identifier) @name) @func",
)


def slice_by_points(source: bytes, node) -> bytes:
    """Reconstruct a node's bytes via start_point/end_point (row, col-in-bytes).

    tree-sitter columns are byte offsets within the row, not codepoint offsets.
    """
    lines = source.split(b"\n")
    start_row, start_col = node.start_point
    end_row, end_col = node.end_point
    if start_row == end_row:
        return lines[start_row][start_col:end_col]
    chunks = [lines[start_row][start_col:]]
    chunks.extend(lines[start_row + 1 : end_row])
    chunks.append(lines[end_row][:end_col])
    return b"\n".join(chunks)


def main() -> None:
    source = (FIXTURES / "non_ascii.py").read_bytes()
    tree = parser.parse(source)
    assert not tree.root_node.has_error, (
        "Q4 pre-req FAIL: non_ascii.py has parse errors; grammar does not "
        "support non-ASCII identifiers cleanly."
    )

    matches = list(QueryCursor(FUNC_QUERY).matches(tree.root_node))
    names_found: list[str] = []
    for _pat, caps in matches:
        name_node = caps["name"][0]
        func_node = caps["func"][0]

        # Check 1: name bytes decode to a non-ASCII identifier where expected.
        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        names_found.append(name)

        # Check 2: bytes-sliced and point-sliced views of the function body
        # are identical.
        by_bytes = source[func_node.start_byte : func_node.end_byte]
        by_points = slice_by_points(source, func_node)
        assert by_bytes == by_points, (
            f"Q4 FAIL: byte-slice and point-slice disagree for {name!r}. "
            f"Coordinate translation cannot trust both."
        )

        # Check 3: decoding doesn't raise — byte offsets landed on char
        # boundaries.
        _ = by_bytes.decode("utf-8")

    # Check 4: the Greek-letter identifier α survived parsing as a function name.
    assert "α" in names_found, (
        f"Q4 FAIL: expected 'α' in function names, got {names_found}. "
        "tree-sitter-python should accept PEP 3131 non-ASCII identifiers."
    )

    # Check 5: the class Привет should also parse; query classes separately.
    class_q = Query(PY_LANGUAGE, "(class_definition name: (identifier) @n)")
    class_names: list[str] = []
    for _pat, caps in QueryCursor(class_q).matches(tree.root_node):
        n = caps["n"][0]
        class_names.append(source[n.start_byte : n.end_byte].decode("utf-8"))
    assert "Привет" in class_names, f"Q4 FAIL: expected Cyrillic class 'Привет' in {class_names}"

    print(
        f"Q4 OK: byte-slice and point-slice agree across {len(matches)} "
        "functions; non-ASCII identifiers parse; byte offsets land on "
        "UTF-8 character boundaries. Using bytes for coordinate math is safe."
    )


if __name__ == "__main__":
    main()
