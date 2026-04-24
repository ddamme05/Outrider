# Fixtures

Fixtures used by the tree-sitter spike. Kept local to the spike — not shared
with `tests/fixtures/` — so nothing outside the spike starts depending on them.

| File | Purpose |
|---|---|
| `modern_python.py` | PEP 695 generics, type aliases, match/case, walrus, async. Grammar-coverage check for 3.13 features. |
| `nested_and_decorators.py` | Nested classes, inner functions, decorator stacks. Qualified-name derivation. |
| `non_ascii.py` | Non-ASCII identifiers and emoji. Byte/point duality on multi-byte UTF-8. |
| `syntax_error_inside_scope.py` | Syntax error inside one function; the other functions must still parse. |
| `syntax_error_outside_scope.py` | Syntax error at module level; function bodies must still parse cleanly. |
| `pygoat_introduction_views.py` | First 200 lines of `adeyosemanputra/pygoat@master:introduction/views.py` — real Django views, real decorators, real SQLi target. Proves the grammar works on eval-target code. |
