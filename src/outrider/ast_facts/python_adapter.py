# Python adapter implementing LanguageAdapter Protocol per
# specs/2026-04-30-ast-facts-module.md.
"""Python adapter for tree-sitter-python 0.25.0.

Wraps the Month 0 spike's validated primitives:
  * `decorated_definition` wraps `function_definition` with the
    decorator span; `decorated_definition.start_point` precedes the
    inner `function_definition.start_point`.
  * Per-scope `node.has_error` is reliable — a module-level syntax
    error does NOT taint sibling function scopes.
  * Byte/point duality on multi-byte UTF-8: `start_byte` / `end_byte`
    always land on character boundaries.

Defines the canonical `parse_python(...)` orchestration function per
Internal contracts: runs the size→pattern→decode→parse pipeline,
discards extracted tuples on a post-parse failed outcome, returns
`ParseResult` with the empty-tuples shape on skip/fail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import tree_sitter_python
from tree_sitter import Language, Node, Parser, Tree

from outrider.ast_facts.models import (
    AssignmentSite,
    CallSite,
    ComputedParserOutcome,
    ImportRef,
    ImportResolution,
    LexicalBinding,
    ParseResult,
    ScopeUnit,
    SkipReason,
    compute_unit_id,
)
from outrider.ast_facts.parser_outcome import should_skip
from outrider.ast_facts.scope_search import (
    error_lines_from_tree,
    find_node_by_span,
    innermost_scope_containing,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from outrider.ast_facts.base import ImportPathResolver

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_PY_LANGUAGE: Final = Language(tree_sitter_python.language())
_PARSER: Final = Parser(_PY_LANGUAGE)


# ---------------------------------------------------------------------------
# PythonAdapter
# ---------------------------------------------------------------------------


class PythonAdapter:
    """Implements `LanguageAdapter` for Python via tree-sitter-python."""

    def __init__(self, resolver: ImportPathResolver) -> None:
        self._resolver = resolver

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, source: bytes) -> Tree:
        return _PARSER.parse(source)

    @staticmethod
    def _node_text(node: Node) -> str:
        # Node.text returns bytes; decode UTF-8. Source has already
        # passed UTF-8 validation per the parse_python pipeline.
        return node.text.decode("utf-8") if node.text else ""

    @staticmethod
    def _scope_byte_range(node: Node) -> tuple[int, int]:
        """For function/class definitions, return the byte span that
        includes any wrapping `decorated_definition` per Month 0 spike."""
        if node.parent is not None and node.parent.type == "decorated_definition":
            return node.parent.start_byte, node.parent.end_byte
        return node.start_byte, node.end_byte

    @staticmethod
    def _scope_line_range(node: Node) -> tuple[int, int]:
        """Same span semantics as `_scope_byte_range` but in 1-indexed lines."""
        outer = (
            node.parent
            if node.parent is not None and node.parent.type == "decorated_definition"
            else node
        )
        return outer.start_point[0] + 1, outer.end_point[0] + 1

    @staticmethod
    def _decorator_strings(node: Node) -> tuple[str, ...]:
        """Decorator text from a wrapping `decorated_definition`, in source order.

        The `@` prefix is stripped — `@app.route("/x")` becomes
        `app.route("/x")` per the eval-harness scaffold convention.
        """
        if node.parent is None or node.parent.type != "decorated_definition":
            return ()
        decorators: list[str] = []
        for child in node.parent.children:
            if child.type == "decorator":
                text = PythonAdapter._node_text(child)
                if text.startswith("@"):
                    text = text[1:]
                decorators.append(text)
        return tuple(decorators)

    # ------------------------------------------------------------------
    # extract_scopes
    # ------------------------------------------------------------------

    def extract_scopes(self, source: bytes, file_path: str) -> tuple[ScopeUnit, ...]:
        return self._extract_scopes_from_tree(self._parse(source), file_path)

    def _extract_scopes_from_tree(self, tree: Tree, file_path: str) -> tuple[ScopeUnit, ...]:
        scopes: list[ScopeUnit] = []
        self._walk_for_scopes(tree.root_node, file_path=file_path, out=scopes)
        return tuple(scopes)

    def _emit_scope(
        self,
        out: list[ScopeUnit],
        node: Node,
        *,
        kind: str,
        name: str,
        qual_path: tuple[str, ...],
        file_path: str,
        parent_unit_id: str | None,
    ) -> str:
        """Append one `ScopeUnit` and return its `unit_id`.

        The byte and line ranges come from `_scope_byte_range` /
        `_scope_line_range`, which widen a decorated definition to cover its
        decorators — which is why the `decorated_definition` wrapper node itself
        never emits.
        """
        qualified_name = ".".join((*qual_path, name))
        byte_start, byte_end = self._scope_byte_range(node)
        line_start, line_end = self._scope_line_range(node)
        unit_id = compute_unit_id(file_path, kind=kind, qualified_name=qualified_name)
        out.append(
            ScopeUnit(
                unit_id=unit_id,
                kind=kind,
                name=name,
                qualified_name=qualified_name,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                byte_start=byte_start,
                byte_end=byte_end,
                decorators=self._decorator_strings(node),
                parent_scope_id=parent_unit_id,
            )
        )
        return unit_id

    def _walk_for_scopes(self, node: Node, *, file_path: str, out: list[ScopeUnit]) -> None:
        """Iterative pre-order scope walk with state carried per frame.

        Each stack frame is a `(node, qual_path, in_class, parent_unit_id)`
        tuple — the four pieces of state the recursive form carried.
        Iterative (not recursive) so adversarially deep parse trees
        can't exhaust Python's recursion limit; tree depth is
        independent of the `MAX_PARSE_BYTES` size cap, so a small file
        with thousands of nested expressions could otherwise blow the
        stack here.

        For function/class nodes: emit the `ScopeUnit`, then push only
        the body frame with updated `qual_path` / `in_class` /
        `parent_unit_id` (siblings of the body are not scope-bearing for
        these node types).

        For other nodes (including the `decorated_definition` wrapper):
        push every child with the inherited frame state. Children are
        pushed in reverse so leftmost is popped first, preserving the
        pre-order traversal of the recursive form. `decorated_definition`
        wrappers don't emit a `ScopeUnit` themselves — the inner
        function/class node handles decorator-span detection via
        `_scope_byte_range` / `_decorator_strings`.
        """
        # The root frame is invariant: module scope has no qualified path, is not
        # inside a class, and has no parent unit.
        stack: list[tuple[Node, tuple[str, ...], bool, str | None]] = [(node, (), False, None)]
        while stack:
            cur_node, cur_qual_path, cur_in_class, cur_parent_unit_id = stack.pop()
            node_type = cur_node.type

            if node_type == "function_definition":
                name_node = cur_node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = self._node_text(name_node)
                unit_id = self._emit_scope(
                    out,
                    cur_node,
                    kind="method" if cur_in_class else "function",
                    name=name,
                    qual_path=cur_qual_path,
                    file_path=file_path,
                    parent_unit_id=cur_parent_unit_id,
                )
                body = cur_node.child_by_field_name("body")
                if body is not None:
                    stack.append((body, (*cur_qual_path, name), False, unit_id))
                continue

            if node_type == "class_definition":
                name_node = cur_node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = self._node_text(name_node)
                unit_id = self._emit_scope(
                    out,
                    cur_node,
                    kind="class",
                    name=name,
                    qual_path=cur_qual_path,
                    file_path=file_path,
                    parent_unit_id=cur_parent_unit_id,
                )
                body = cur_node.child_by_field_name("body")
                if body is not None:
                    stack.append((body, (*cur_qual_path, name), True, unit_id))
                continue

            # Other node (including `decorated_definition` wrapper): push
            # all children carrying the inherited frame state. Reversed
            # so leftmost is popped first → preserves pre-order.
            for child in reversed(cur_node.children):
                stack.append((child, cur_qual_path, cur_in_class, cur_parent_unit_id))

    # ------------------------------------------------------------------
    # extract_imports
    # ------------------------------------------------------------------

    def extract_imports(self, source: bytes, file_path: str) -> tuple[ImportRef, ...]:
        return self._extract_imports_from_tree(self._parse(source), file_path)

    def _extract_imports_from_tree(self, tree: Tree, file_path: str) -> tuple[ImportRef, ...]:
        imports: list[ImportRef] = []
        for node in self._walk(tree.root_node):
            if node.type == "import_statement":
                imports.extend(self._build_direct_imports(node, file_path))
            elif node.type == "import_from_statement":
                imp = self._build_from_import(node, file_path)
                if imp is not None:
                    imports.append(imp)
        return tuple(imports)

    def _build_direct_imports(self, node: Node, file_path: str) -> list[ImportRef]:
        """`import x` / `import x.y` / `import x as y` — possibly multiple
        per statement (`import a, b`). is_simple_direct=False per spec
        (only `from x import y` with non-relative module is True)."""
        line = node.start_point[0] + 1
        results: list[ImportRef] = []
        for name_node in node.children_by_field_name("name"):
            if name_node.type == "aliased_import":
                inner = name_node.child_by_field_name("name")
                module = self._node_text(inner) if inner is not None else ""
                alias = name_node.child_by_field_name("alias")
                names = (self._node_text(alias),) if alias is not None else ()
            else:
                module = self._node_text(name_node)
                names = ()
            results.append(
                ImportRef(
                    file_path=file_path,
                    line=line,
                    import_kind="direct",
                    module=module,
                    names=names,
                    is_simple_direct=False,
                )
            )
        return results

    def _build_from_import(self, node: Node, file_path: str) -> ImportRef | None:
        line = node.start_point[0] + 1
        module_node = node.child_by_field_name("module_name")
        if module_node is None:
            return None
        # Star: a wildcard_import child is present.
        is_star = any(c.type == "wildcard_import" for c in node.children)
        # Relative: module_name is a relative_import node.
        is_relative = module_node.type == "relative_import"

        if is_star:
            module = self._node_text(module_node)
            return ImportRef(
                file_path=file_path,
                line=line,
                import_kind="star",
                module=module,
                names=(),
                is_simple_direct=False,
            )
        if is_relative:
            module = self._node_text(module_node)
            # Collect imported name(s)
            names = self._collect_from_imported_names(node)
            return ImportRef(
                file_path=file_path,
                line=line,
                import_kind="relative",
                module=module,
                names=names,
                is_simple_direct=False,
            )
        # Plain `from x import y`
        module = self._node_text(module_node)
        names = self._collect_from_imported_names(node)
        # is_simple_direct only on this exact shape: non-relative module,
        # at least one name, no star
        is_simple_direct = bool(names) and not is_relative and not is_star
        return ImportRef(
            file_path=file_path,
            line=line,
            import_kind="from",
            module=module,
            names=names,
            is_simple_direct=is_simple_direct,
        )

    def _collect_from_imported_names(self, node: Node) -> tuple[str, ...]:
        """Imported names from `import_from_statement.name` field(s).

        Each `name` field child is a `dotted_name` or `aliased_import`.
        For `aliased_import`, return the alias text; otherwise, keep
        the full `dotted_name` text. This matches the V1 trace-node
        behavior, which preserves import identity rather than reducing
        names to their final segment.
        """
        names: list[str] = []
        for child in node.children_by_field_name("name"):
            if child.type == "aliased_import":
                alias = child.child_by_field_name("alias")
                if alias is not None:
                    names.append(self._node_text(alias))
            else:
                names.append(self._node_text(child))
        return tuple(names)

    # ------------------------------------------------------------------
    # extract_call_sites
    # ------------------------------------------------------------------

    def extract_call_sites(
        self,
        source: bytes,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[CallSite, ...]:
        """Calls inside extracted ScopeUnits + call-form decorators on
        an extracted scope. Module-level calls are skipped per non-goal.
        """
        return self._extract_call_sites_from_tree(self._parse(source), file_path, scope_units)

    def _extract_call_sites_from_tree(
        self,
        tree: Tree,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[CallSite, ...]:
        # Sort deterministically; the shared lookup selects the smallest enclosing span.
        sorted_scopes = sorted(scope_units, key=lambda s: (s.byte_start, -s.byte_end))
        calls: list[CallSite] = []
        for node in self._walk(tree.root_node):
            if node.type != "call":
                continue
            # Only emit if call is inside an extracted scope.
            enclosing = _innermost_scope_containing(sorted_scopes, node.start_byte, node.end_byte)
            if enclosing is None:
                continue
            function_node = node.child_by_field_name("function")
            if function_node is None:
                continue
            # Preserve raw callee text, including newlines and parentheses (spec.md §5.4).
            # Resolution belongs downstream; changing this requires a DECISIONS.md amendment.
            calls.append(
                CallSite(
                    file_path=file_path,
                    line=node.start_point[0] + 1,
                    callee_name=self._node_text(function_node),
                    enclosing_scope_id=enclosing.unit_id,
                )
            )
        return tuple(calls)

    # ------------------------------------------------------------------
    # extract_assignments
    # ------------------------------------------------------------------

    def extract_assignments(
        self,
        source: bytes,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[AssignmentSite, ...]:
        return self._extract_assignments_from_tree(self._parse(source), file_path, scope_units)

    def _extract_assignments_from_tree(
        self,
        tree: Tree,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[AssignmentSite, ...]:
        sorted_scopes = sorted(scope_units, key=lambda s: (s.byte_start, -s.byte_end))
        sites: list[AssignmentSite] = []
        for node in self._walk(tree.root_node):
            if node.type != "assignment":
                continue
            enclosing = _innermost_scope_containing(sorted_scopes, node.start_byte, node.end_byte)
            if enclosing is None:
                continue
            left = node.child_by_field_name("left")
            if left is None:
                continue
            target_name = self._extract_assignment_target_name(left)
            if target_name is None:
                continue
            sites.append(
                AssignmentSite(
                    file_path=file_path,
                    line=node.start_point[0] + 1,
                    target_name=target_name,
                    enclosing_scope_id=enclosing.unit_id,
                )
            )
        return tuple(sites)

    @staticmethod
    def _extract_assignment_target_name(left: Node) -> str | None:
        """Single-identifier targets only for V1.

        Tuple/subscript/attribute targets (`a, b = ...`, `obj.x = ...`,
        `arr[0] = ...`) return None — the trace node's V1 backward-tracing
        is name-keyed, and complex targets need richer modeling.

        Uses `_node_text` for consistency with the rest of the file (rather
        than reading `Node.text` directly). Returns None for identifier
        nodes whose text is empty/missing — that's a degenerate parse
        result, not a valid target.
        """
        if left.type != "identifier":
            return None
        text = PythonAdapter._node_text(left)
        return text or None

    # ------------------------------------------------------------------
    # extract_lexical_bindings
    # ------------------------------------------------------------------

    def extract_lexical_bindings(self, source: bytes, file_path: str) -> tuple[LexicalBinding, ...]:
        """Protocol conformance stub: the Python catalog carries no
        binding rules (`binding=None` everywhere), so the shadowing
        guard has no Python consumer — extraction is the
        shadowing-guard spec's explicit non-goal (FUP-184 owns the
        Python sibling gap)."""
        del source, file_path  # Protocol signature; unused by the stub
        return ()

    # ------------------------------------------------------------------
    # resolve_simple_direct_import
    # ------------------------------------------------------------------

    def resolve_simple_direct_import(
        self, import_ref: ImportRef, import_root: Path
    ) -> ImportResolution:
        if not import_ref.is_simple_direct:
            return ImportResolution(status="unresolved", target_path=None)
        candidates = self._resolver.resolve_candidate_paths(import_ref.module, import_root)
        # Existence check via the symlink-safe primitive per Internal
        # contracts (allowlist: only `.is_file(follow_symlinks=False)`).
        existing: list[Path] = []
        for candidate_path in candidates:
            if (import_root / candidate_path).is_file(follow_symlinks=False):
                existing.append(candidate_path)
        if len(existing) == 1:
            return ImportResolution(
                status="resolved",
                target_path=existing[0].as_posix(),
            )
        if len(existing) >= 2:
            return ImportResolution(status="ambiguous", target_path=None)
        return ImportResolution(status="unresolved", target_path=None)

    # ------------------------------------------------------------------
    # compute_parser_outcome
    # ------------------------------------------------------------------

    def compute_parser_outcome(
        self,
        source: bytes,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[ComputedParserOutcome, dict[str, bool]]:
        """Per-scope `has_error` map keyed by `unit_id`.

        **V1 policy: always returns `("clean", has_error)`.** Tree-sitter
        degrades to ERROR/MISSING nodes per Month 0 spike rather than
        producing no AST at all; the "failed" outcome is the defensive
        forward-compat shape but is unreachable in V1 practice. The
        orchestrator's discard-extracted-tuples branch in `parse_python`
        is correspondingly dead code in V1.

        Pinned by `tests/unit/test_ast_facts_python.py` so a future
        contributor tightening this to "any has_error => failed" (an
        obvious-looking refinement) is forced to update the test and
        therefore acknowledge the policy change. A change to the V1
        policy requires a `DECISIONS.md` entry per spec-fidelity
        discipline.
        """
        return self._compute_parser_outcome_from_tree(self._parse(source), scope_units)

    def _compute_parser_outcome_from_tree(
        self,
        tree: Tree,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[ComputedParserOutcome, dict[str, bool]]:
        # Map each ScopeUnit to its tree-sitter node by byte range. A decorated
        # ScopeUnit's `byte_start` includes its `@decorator`s while the underlying
        # function/class node starts narrower, and the exact inner span is not
        # recoverable from the ScopeUnit alone — so the full stored span is passed
        # as a conservative bound and `_find_node_by_span`'s containment lookup
        # (smallest start_byte / largest end_byte tiebreaker) picks the OUTERMOST
        # scope-defining node enclosed by it.
        # `decorated_definition` is in target_types so a decorator-region
        # syntax error propagates to the wrapping scope's `has_error`.
        # Syntax errors remain per-scope flags; the file verdict stays clean.
        has_error: dict[str, bool] = {}
        for scope in scope_units:
            node = _find_node_by_span(tree.root_node, scope.byte_start, scope.byte_end)
            has_error[scope.unit_id] = bool(node and node.has_error)
        return "clean", has_error

    # ------------------------------------------------------------------
    # Generic tree walker
    # ------------------------------------------------------------------

    @staticmethod
    def _walk(node: Node) -> Iterator[Node]:
        """Pre-order traversal of `node` and all descendants.

        Iterates via `node.children` (NOT `node.named_children`), so
        unnamed tree-sitter nodes — punctuation, keywords — are yielded
        too. Callers filter by `node.type` for the kinds they care about
        (`call`, `assignment`, `import_statement`, etc.).

        Implemented iteratively with an explicit stack so adversarially
        deep parse trees can't exhaust Python's recursion limit. Tree
        depth is independent of the `MAX_PARSE_BYTES` size cap — a
        small file with thousands of nested expressions could otherwise
        blow the stack here. Pre-order is preserved by pushing children
        in reverse so the leftmost is popped first.
        """
        stack: list[Node] = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))


# ---------------------------------------------------------------------------
# Module-level helpers (visible for the orchestrator)
# ---------------------------------------------------------------------------


# Shared geometry (extracted to `scope_search.py` for the JS/TS
# adapters); the private name is kept because tests and this module's
# call sites address it here.
_innermost_scope_containing = innermost_scope_containing

# Python's scope-defining node types for the ScopeUnit→node containment
# lookup. `decorated_definition` is included so a syntax error inside a
# decorator (e.g., `@route(/*malformed*/)\ndef foo(): ...`) propagates
# to the ScopeUnit's `has_error`: the ScopeUnit's span starts at the
# decorator (Month 0 spike), and the decorator's `has_error` lives on
# the decorated_definition wrapper, not the inner function_definition —
# without it the lookup returns the inner (clean) node and downstream
# `degraded` derivation goes blind to decorator-region parse errors.
_PY_OUTCOME_TARGET_TYPES: Final[frozenset[str]] = frozenset(
    {
        "function_definition",
        "class_definition",
        "decorated_definition",
    }
)


def _find_node_by_span(root: Node, byte_start: int, byte_end: int) -> Node | None:
    """Python-typed wrapper over the shared containment lookup
    (`scope_search.find_node_by_span`); semantics documented there."""
    return find_node_by_span(root, byte_start, byte_end, _PY_OUTCOME_TARGET_TYPES)


# ---------------------------------------------------------------------------
# Canonical entry point: parse_python
# ---------------------------------------------------------------------------


def parse_python(source: bytes, file_path: str, resolver: ImportPathResolver) -> ParseResult:
    """Run exclusions, UTF-8 validation, one parse, and ordered extraction.

    Non-bytes input raises TypeError. Skipped and failed results contain no facts.
    Invalid UTF-8 fails; the current adapter reports syntax errors as per-scope
    flags with a clean file verdict. A defensive failed extraction discards facts.
    The adapter stores resolver for import resolution; this pipeline does not call it.
    """
    if not isinstance(source, bytes):
        raise TypeError(
            f"parse_python: source must be bytes, got {type(source).__name__}; "
            f"the consuming node decodes the file once via fetch and passes "
            f"raw bytes through. Decoding to str at this layer would defeat "
            f"the size→pattern→decode→parse DoS pipeline ordering."
        )
    # Step 1: size + path-pattern + content-pattern check.
    skip_reason: SkipReason | None = should_skip(file_path, source)
    if skip_reason is not None:
        return ParseResult(parser_outcome="skipped", skip_reason=skip_reason)
    # Validate UTF-8 without retaining decoded text; tree-sitter consumes raw bytes.
    # Analyze re-encodes already-decoded text, while intake classifies invalid UTF-8
    # as BINARY. Direct byte callers can still reach this failed return (FUP-053).
    # BOM and CRLF bytes remain intact for downstream coordinate translation;
    # tests/unit/test_ast_facts_python.py pins both cases (FUP-058).
    try:
        source.decode("utf-8")
    except UnicodeDecodeError:
        return ParseResult(parser_outcome="failed")
    # Step 3+: parse once, then extract via the from-tree helpers per
    # Internal contracts. The Protocol-public methods each parse
    # internally, but the orchestrator avoids the 5x parse overhead
    # by sharing the Tree across extraction passes. (Reaching into
    # private adapter methods is fine here — `parse_python` is the
    # ast_facts-internal orchestrator, not an external consumer.)
    adapter = PythonAdapter(resolver=resolver)
    tree = adapter._parse(source)
    scope_units = adapter._extract_scopes_from_tree(tree, file_path)
    imports = adapter._extract_imports_from_tree(tree, file_path)
    call_sites = adapter._extract_call_sites_from_tree(tree, file_path, scope_units)
    assignment_sites = adapter._extract_assignments_from_tree(tree, file_path, scope_units)
    outcome, has_error = adapter._compute_parser_outcome_from_tree(tree, scope_units)
    if outcome == "failed":
        # Discard extracted tuples per Internal contracts: the failed-path
        # ParseResult carries the empty-tuples shape regardless of which
        # pipeline stage decided the file is unrecoverable.
        return ParseResult(parser_outcome="failed")
    return ParseResult(
        parser_outcome="clean",
        scope_units=scope_units,
        imports=imports,
        call_sites=call_sites,
        assignment_sites=assignment_sites,
        has_error=has_error,
        error_lines=error_lines_from_tree(tree),
    )
