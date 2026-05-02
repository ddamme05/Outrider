# Tree-sitter query registry per
# specs/2026-04-30-ast-facts-module.md Internal contracts.
"""Query registry — module marker.

`queries/` is a sibling of `ast_facts/` under trust-boundary #4
(`ast-facts-is-the-only-tree-sitter-consumer`): the two modules share
the privilege of importing `tree_sitter`. No tree-sitter import here at
the package level — `registry.py` does the loading.
"""
