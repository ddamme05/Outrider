"""Q7 — parse-error localization (inside vs outside a scope).

spec §5.5 distinguishes:
- "Parse with errors outside changed regions" → clean-parse semantics for
  findings inside the changed scope.
- "Parse with errors inside changed regions" → file degrades to JUDGED-only.

For that policy to be implementable, ast_facts must be able to answer
'is the error inside this scope?' deterministically. Tree-sitter exposes
``node.has_error`` per-node, so a scope with has_error=False is provably
clean even when root_node.has_error=True. This demo proves that holds.
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


def functions_with_error_status(source: bytes) -> dict[str, bool]:
    tree = parser.parse(source)
    out: dict[str, bool] = {}
    for _pat, caps in QueryCursor(FUNC_QUERY).matches(tree.root_node):
        name_node = caps["name"][0]
        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        out[name] = caps["func"][0].has_error
    return out, tree


def main() -> None:
    # Case 1: syntax error INSIDE one function.
    inside = (FIXTURES / "syntax_error_inside_scope.py").read_bytes()
    funcs, tree = functions_with_error_status(inside)
    assert tree.root_node.has_error, (
        "Q7 pre-req FAIL: root_node.has_error should be True for "
        "syntax_error_inside_scope.py"
    )
    assert funcs == {
        "good_function": False,
        "bad_function": True,
        "also_good": False,
    }, f"Q7 FAIL (case 1): per-function has_error wrong, got {funcs}"
    print(
        "Q7 case 1 OK: errors inside bad_function are localized — "
        "good_function and also_good report has_error=False even though "
        "root_node.has_error=True."
    )

    # Case 2: syntax error OUTSIDE any function (module-level garbage).
    outside = (FIXTURES / "syntax_error_outside_scope.py").read_bytes()
    funcs, tree = functions_with_error_status(outside)
    assert tree.root_node.has_error, (
        "Q7 pre-req FAIL: root_node.has_error should be True for "
        "syntax_error_outside_scope.py"
    )
    assert all(not v for v in funcs.values()), (
        f"Q7 FAIL (case 2): module-level error leaked into a function scope. "
        f"Per-function has_error: {funcs}. "
        "If this fires, parse-errors-degrade-to-judged needs a different "
        "localization strategy than per-node has_error."
    )
    print(
        f"Q7 case 2 OK: module-level syntax error did NOT set has_error=True "
        f"on any of {sorted(funcs)}. "
        "Conclusion: per-scope has_error is a valid signal for the §5.5 "
        "'errors outside changed regions' policy."
    )


if __name__ == "__main__":
    main()
