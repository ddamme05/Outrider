"""Structural source checks and their registry.

Import queries.registry to load the checks. Raw tree-sitter objects are allowed
only here and in ast_facts/; callers receive domain spans instead.
See docs/trust-boundaries.md §4.
"""
