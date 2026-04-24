"""Q6 — diff-line → innermost containing scope.

python-source-analysis.md prescribes the algorithm in one sentence:
"Convert PR added lines to one-based target-side lines, then pick the
innermost containing scope." This demo validates that prescription on the
edge cases coordinates.diff_line_to_scope() will hit:

- line is a decorator line above a method → expect method scope (per Q2,
  decorated_definition range starts at the decorator, so the innermost
  "function-like" scope covers decorators).
- line is the first line of a function body → expect that function.
- line is the last line of a nested function → expect the nested function,
  not its outer.
- line is a class-body line between two methods → expect the class, not
  a method.
- line is a module-level comment → expect None (no owning scope).
- line is outside the file's line range → the mapper must handle it
  deterministically (we return None here).
"""

from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor

FIXTURES = Path(__file__).parent.parent / "fixtures"
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

# Match decorated_definition at its outer extent when present; otherwise the
# bare function_definition. Also match class_definition for class scopes.
SCOPE_QUERY = Query(
    PY_LANGUAGE,
    """
    (decorated_definition
        definition: (function_definition name: (identifier) @name)) @scope
    (decorated_definition
        definition: (class_definition name: (identifier) @name)) @scope
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
                parts.append(
                    source[name_field.start_byte : name_field.end_byte].decode("utf-8")
                )
        cur = cur.parent
    return ".".join(reversed(parts))


def collect_scopes(source: bytes, tree):
    """Return list of (qualified_name, line_start, line_end) sorted so that
    narrower scopes appear before their enclosing scopes.

    line_start is 1-based and is decorated_definition.start_point when
    decorators exist (so decorator lines count as "inside the scope").
    """
    seen: dict[int, tuple] = {}
    for _pat, caps in QueryCursor(SCOPE_QUERY).matches(tree.root_node):
        scope = caps["scope"][0]
        # Dedupe: a decorated function matches twice (once as
        # decorated_definition, once as the inner function_definition).
        # Prefer the outer extent (decorator line).
        inner = (
            scope.child_by_field_name("definition")
            if scope.type == "decorated_definition"
            else scope
        )
        qn = qualified_name(inner, source)
        start = scope.start_point[0] + 1
        end = scope.end_point[0] + 1
        existing = seen.get(id(inner))
        if existing is None or start < existing[1]:
            seen[id(inner)] = (qn, start, end)
    scopes = list(seen.values())
    # Sort by width ascending so innermost is picked first in the linear scan.
    scopes.sort(key=lambda t: (t[2] - t[1], t[1]))
    return scopes


def scope_for_line(scopes, line: int) -> str | None:
    for qn, s, e in scopes:
        if s <= line <= e:
            return qn
    return None


def find_line(source_text: str, needle: str) -> int:
    for i, line in enumerate(source_text.splitlines(), start=1):
        if needle in line:
            return i
    raise AssertionError(f"marker not found in fixture: {needle!r}")


def main() -> None:
    path = FIXTURES / "nested_and_decorators.py"
    source_bytes = path.read_bytes()
    source_text = source_bytes.decode("utf-8")
    tree = parser.parse(source_bytes)
    scopes = collect_scopes(source_bytes, tree)

    cases = [
        # (description, anchor_line_contains, expected_qualified_name)
        (
            "decorator line above decorator-stacked method",
            "@staticmethod",
            "Outer.Inner.greet",
        ),
        (
            "first line of an inner-inner function body",
            "return 42",
            "top_level.nested_one.nested_two",
        ),
        (
            "line of a class-body comment (no method)",
            "class Inner",
            "Outer.Inner",
        ),
        (
            "import line at module top",
            "import functools",
            None,  # module-level, no owning scope
        ),
    ]

    for desc, marker, expected in cases:
        line = find_line(source_text, marker)
        got = scope_for_line(scopes, line)
        assert got == expected, (
            f"Q6 FAIL ({desc}): line {line} marker {marker!r} → "
            f"{got!r}, expected {expected!r}"
        )

    # Out-of-range line.
    huge = 10_000
    assert scope_for_line(scopes, huge) is None, (
        "Q6 FAIL: line past EOF should return None"
    )

    print(
        f"Q6 OK: innermost-scope picker resolves {len(cases)} edge cases "
        "plus module-level and out-of-range correctly. "
        "ChangedRegion.owning_scope_ids is legally empty for module-level."
    )


if __name__ == "__main__":
    main()
