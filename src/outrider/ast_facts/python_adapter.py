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
        self._walk_for_scopes(
            tree.root_node,
            file_path=file_path,
            qual_path=(),
            in_class=False,
            parent_unit_id=None,
            out=scopes,
        )
        return tuple(scopes)

    def _walk_for_scopes(
        self,
        node: Node,
        *,
        file_path: str,
        qual_path: tuple[str, ...],
        in_class: bool,
        parent_unit_id: str | None,
        out: list[ScopeUnit],
    ) -> None:
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
        stack: list[tuple[Node, tuple[str, ...], bool, str | None]] = [
            (node, qual_path, in_class, parent_unit_id)
        ]
        while stack:
            cur_node, cur_qual_path, cur_in_class, cur_parent_unit_id = stack.pop()
            node_type = cur_node.type

            if node_type == "function_definition":
                name_node = cur_node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = self._node_text(name_node)
                kind = "method" if cur_in_class else "function"
                qualified_name = ".".join(cur_qual_path + (name,))
                byte_start, byte_end = self._scope_byte_range(cur_node)
                line_start, line_end = self._scope_line_range(cur_node)
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
                        decorators=self._decorator_strings(cur_node),
                        parent_scope_id=cur_parent_unit_id,
                    )
                )
                body = cur_node.child_by_field_name("body")
                if body is not None:
                    stack.append((body, cur_qual_path + (name,), False, unit_id))
                continue

            if node_type == "class_definition":
                name_node = cur_node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = self._node_text(name_node)
                qualified_name = ".".join(cur_qual_path + (name,))
                byte_start, byte_end = self._scope_byte_range(cur_node)
                line_start, line_end = self._scope_line_range(cur_node)
                unit_id = compute_unit_id(file_path, kind="class", qualified_name=qualified_name)
                out.append(
                    ScopeUnit(
                        unit_id=unit_id,
                        kind="class",
                        name=name,
                        qualified_name=qualified_name,
                        file_path=file_path,
                        line_start=line_start,
                        line_end=line_end,
                        byte_start=byte_start,
                        byte_end=byte_end,
                        decorators=self._decorator_strings(cur_node),
                        parent_scope_id=cur_parent_unit_id,
                    )
                )
                body = cur_node.child_by_field_name("body")
                if body is not None:
                    stack.append((body, cur_qual_path + (name,), True, unit_id))
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
        # Pre-sort scopes by byte_start ASC, byte_end DESC so the most
        # tightly-enclosing scope is found first by linear search.
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
            # `callee_name` is RAW SOURCE TEXT per canonical spec.md §5.4
            # ("raw text; resolution is a separate concern"). For `obj.method()`
            # this is `"obj.method"`; for multi-line `(\n  chain\n  .step\n)()`
            # it preserves embedded newlines and parens. The trace-node spec
            # (when written) is responsible for normalizing this text when
            # matching against `ImportRef.names` — the field stays raw on the
            # `ast_facts/` side per spec-fidelity discipline. Changing this
            # contract requires a `DECISIONS.md` entry that supersedes §5.4,
            # not a silent shape change here.
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

    def extract_lexical_bindings(
        self,
        source: bytes,  # noqa: ARG002 — Protocol signature
        file_path: str,  # noqa: ARG002 — Protocol signature
    ) -> tuple[LexicalBinding, ...]:
        """Protocol conformance stub: the Python catalog carries no
        binding rules (`binding=None` everywhere), so the shadowing
        guard has no Python consumer — extraction is the
        shadowing-guard spec's explicit non-goal (FUP-184 owns the
        Python sibling gap)."""
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
        existing: list[Path] = [
            c for c in candidates if (import_root / c).is_file(follow_symlinks=False)
        ]
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
        # Map each ScopeUnit to its tree-sitter node by byte range.
        # ScopeUnit byte ranges include decorators when wrapped; the
        # underlying function/class node may have a narrower span.
        # We pass the full ScopeUnit span (via `_inner_span`) to
        # `_find_node_by_span`, which uses containment-based lookup —
        # smallest start_byte / largest end_byte tiebreaker — to locate
        # the OUTERMOST scope-defining node enclosed by the ScopeUnit.
        # `decorated_definition` is in target_types so a decorator-region
        # syntax error propagates to the wrapping scope's `has_error`.
        # NOTE: this currently always returns `"clean"` — syntax errors
        # surface as per-scope-unit `has_error[unit_id] == True`, not as
        # a file-wide "failed" outcome. A future syntax-fatal classifier
        # (e.g., "more than N% of root children are ERROR nodes") could
        # return `"failed"` and would feed analyze's `failed+degraded_llm`
        # outcome alongside the FUP-053 raw-bytes path. Not in V1 scope.
        has_error: dict[str, bool] = {}
        for scope in scope_units:
            inner_start, inner_end = self._inner_span(scope)
            node = _find_node_by_span(tree.root_node, inner_start, inner_end)
            has_error[scope.unit_id] = bool(node and node.has_error)
        return "clean", has_error

    @staticmethod
    def _compute_error_lines_from_tree(tree: Tree) -> frozenset[int]:
        """1-indexed source lines covered by tree-sitter ERROR or MISSING nodes.

        Delegates to the shared `scope_search.error_lines_from_tree`
        (language-agnostic ERROR/MISSING walk); semantics documented
        there. See DECISIONS.md#033.
        """
        return error_lines_from_tree(tree)

    @staticmethod
    def _inner_span(scope: ScopeUnit) -> tuple[int, int]:
        """Return the `ScopeUnit` byte range as the conservative match bound
        for `_find_node_by_span`'s containment lookup.

        For decorated scopes, `byte_start` includes `@decorator`s; the
        underlying `function_definition` / `class_definition` node has
        a narrower start. We can't recover the exact inner span from
        `ScopeUnit` alone, so the ScopeUnit-to-node match in
        `compute_parser_outcome` uses the full stored ScopeUnit span as
        the conservative bound and lets `_find_node_by_span` pick the
        outermost contained scope-defining node from within it.
        """
        return scope.byte_start, scope.byte_end

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
    """Run the size→pattern→decode→parse pipeline per Internal contracts.

    Surfaces a `TypeError` at the top of the pipeline if `source` is not
    `bytes`. Without this guard, `should_skip` would silently iterate
    rules and crash deeper in the loop with a confusing traceback when
    a caller passes `str` by accident.

    Returns:
      * `ParseResult(parser_outcome="skipped", skip_reason=...)` if any
        rule in `EXCLUSION_RULES` matches.
      * `ParseResult(parser_outcome="failed")` if UTF-8 decode fails or
        if `compute_parser_outcome` returns `"failed"` after extraction.
        BOTH paths are V1-unreachable through analyze — see the step-2
        comment below for the decode path and the
        `_compute_parser_outcome_from_tree` comment for the extraction
        path. Future raw-bytes intake (FUP-053) and a future syntax-fatal
        classifier are the respective triggers.
      * `ParseResult(parser_outcome="clean", scope_units=..., ...,
        has_error=...)` on the clean path.

    `resolver` is required by `PythonAdapter`'s constructor (the adapter
    holds it for `resolve_simple_direct_import`); `parse_python` does
    not invoke the resolver on the clean path, but the constructor
    parameter is uniform.
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
    # Step 2: UTF-8 strict decode (validity gate; we do not retain the
    # decoded str, since tree-sitter accepts bytes directly).
    # NOTE: V1 callers (analyze) re-encode a Python `str` to bytes
    # before reaching this gate, and intake's `_classify_or_reserve_decode`
    # rejects invalid UTF-8 upstream with SkipReason.OVERSIZED. The
    # `parser_outcome="failed"` return is therefore V1-unreachable
    # through analyze; kept for the raw-bytes intake path (FUP-053) and
    # for direct `parse_python` consumers that hand it untrusted bytes.
    #
    # BOM / CRLF tolerance (FUP-058 closed 2026-05-21): a UTF-8 BOM
    # (`\xef\xbb\xbf`) is valid UTF-8 (encodes U+FEFF), so strict decode
    # accepts BOM-prefixed source. CRLF line endings are also accepted —
    # tree-sitter parses both line-ending conventions correctly and
    # produces scope-unit byte ranges that include the BOM bytes / CR
    # bytes as-is. Downstream coordinate translation in `coordinates/`
    # consumes those byte ranges with the same byte view, so the bytes
    # round-trip. Pinned by `test_parse_python_handles_utf8_bom_prefix`
    # and `test_parse_python_handles_crlf_line_endings` in
    # `tests/unit/test_ast_facts_python.py`.
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
        error_lines=adapter._compute_error_lines_from_tree(tree),
    )
