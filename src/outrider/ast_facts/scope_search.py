# Shared scope/node geometry helpers per
# specs/2026-07-02-js-ts-tree-sitter-adapters.md.
"""Language-agnostic scope-lookup and error-line helpers.

Extracted from `python_adapter.py` so the JS/TS adapters share the
byte-range geometry instead of duplicating it: `innermost_scope_containing`
is pure `ScopeUnit` math, `find_node_by_span` is the same containment
lookup parameterized by each language's scope-defining node types, and
`error_lines_from_tree` walks ERROR/MISSING nodes identically for every
grammar. Raw `tree_sitter` objects are consumed here but never returned
beyond `ast_facts/` (AST firewall): callers receive `ScopeUnit | None`,
a `Node` that stays adapter-internal, or `frozenset[int]`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

    from outrider.ast_facts.models import ScopeUnit


def innermost_scope_containing(
    sorted_scopes: list[ScopeUnit], byte_start: int, byte_end: int
) -> ScopeUnit | None:
    """Return the smallest scope whose byte range contains the given range,
    or None if no scope contains it (module-level).

    "Smallest" = smallest enclosing span. Single-pass over sorted scopes,
    tracking the best so far. We compare by `(byte_end - byte_start)`
    rather than by `byte_start` alone because two scopes can share
    `byte_start` after a decorator-inclusive span adjustment (Python's
    `_scope_byte_range` widens a decorated function's span to its
    parent's); a strict `byte_start >` tiebreaker would let the outer
    (wider) scope win in that case, mis-attributing nested call sites
    and assignments.
    """
    best: ScopeUnit | None = None
    best_span: int | None = None
    for scope in sorted_scopes:
        if scope.byte_start <= byte_start and byte_end <= scope.byte_end:
            span = scope.byte_end - scope.byte_start
            if best_span is None or span < best_span:
                best = scope
                best_span = span
    return best


def find_node_by_span(
    root: Node,
    byte_start: int,
    byte_end: int,
    target_types: frozenset[str],
) -> Node | None:
    """Find the OUTERMOST node of a scope-defining type whose span lies
    within [byte_start, byte_end]. Used by the Python adapter's
    `compute_parser_outcome` to locate the tree-sitter node for a
    `ScopeUnit` (the JS/TS adapters use `error_byte_spans_from_tree`
    intersection instead — their decorator span-widening has no wrapper
    node for a containment lookup to land on).

    `target_types` is the per-language set of scope-defining node types
    — including any wrapper types whose `has_error` must propagate to
    the scope (Python passes `decorated_definition` so a decorator-region
    syntax error reaches the wrapped scope's `has_error`; see the
    call-site comment in `python_adapter.py`).

    "Outermost" = smallest start_byte; on tie, largest end_byte. Each
    ScopeUnit corresponds to ONE scope-defining node — the outer one.
    Picking the deepest contained match would incorrectly attribute a
    nested scope's `has_error` to the outer scope. Nested scopes are
    separate ScopeUnits with their own unit_ids.
    """
    best: Node | None = None
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        contains = (
            node.type in target_types
            and byte_start <= node.start_byte
            and node.end_byte <= byte_end
        )
        if contains and (
            best is None
            or node.start_byte < best.start_byte
            or (node.start_byte == best.start_byte and node.end_byte > best.end_byte)
        ):
            best = node
        for child in node.children:
            stack.append(child)
    return best


def error_byte_spans_from_tree(tree: Tree) -> tuple[tuple[int, int], ...]:
    """Byte spans `[start, end)` of every ERROR or MISSING node.

    The per-scope `has_error` primitive for adapters whose ScopeUnit
    spans don't correspond to a single tree node (JS/TS decorator
    span-widening): a scope has an error iff any error span intersects
    its byte range. Zero-width MISSING nodes yield `start == end`;
    callers treat them as points.
    """
    spans: list[tuple[int, int]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            spans.append((node.start_byte, node.end_byte))
        stack.extend(node.children)
    return tuple(spans)


def error_lines_from_tree(tree: Tree) -> frozenset[int]:
    """1-indexed source lines covered by tree-sitter ERROR or MISSING nodes.

    Scope-INDEPENDENT, unlike per-scope `has_error`: a syntax error that
    breaks a scope's header yields no scope node, so it is invisible to
    `has_error` but IS an ERROR node here — the signal
    degrade-don't-skip uses for the no-scope case (see DECISIONS.md#033).
    A multi-line ERROR node contributes every line in its row span; a
    MISSING / zero-width node (`start_point == end_point`) contributes
    its single line.
    """
    lines: set[int] = set()
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            # tree-sitter rows are 0-indexed; source lines are 1-indexed.
            for row in range(node.start_point[0], node.end_point[0] + 1):
                lines.add(row + 1)
        stack.extend(node.children)
    return frozenset(lines)
