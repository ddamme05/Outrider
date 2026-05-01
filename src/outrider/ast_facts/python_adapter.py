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

from typing import TYPE_CHECKING

import tree_sitter_python
from tree_sitter import Language, Node, Parser, Tree

from outrider.ast_facts.models import (
    AssignmentSite,
    CallSite,
    ComputedParserOutcome,
    ImportRef,
    ImportResolution,
    ParseResult,
    ScopeUnit,
    SkipReason,
    compute_unit_id,
)
from outrider.ast_facts.parser_outcome import should_skip

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from outrider.ast_facts.base import ImportPathResolver

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_PY_LANGUAGE = Language(tree_sitter_python.language())
_PARSER = Parser(_PY_LANGUAGE)


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
        tree = self._parse(source)
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
        node_type = node.type

        if node_type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = self._node_text(name_node)
            kind = "method" if in_class else "function"
            qualified_name = ".".join(qual_path + (name,))
            byte_start, byte_end = self._scope_byte_range(node)
            line_start, line_end = self._scope_line_range(node)
            unit_id = compute_unit_id(file_path, kind, qualified_name)
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
            # Recurse into body for nested scopes; nested functions/classes
            # carry the dotted qual path. Methods inside nested functions
            # follow the same rule.
            body = node.child_by_field_name("body")
            if body is not None:
                self._walk_for_scopes(
                    body,
                    file_path=file_path,
                    qual_path=qual_path + (name,),
                    in_class=False,
                    parent_unit_id=unit_id,
                    out=out,
                )
            return

        if node_type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = self._node_text(name_node)
            kind = "class"
            qualified_name = ".".join(qual_path + (name,))
            byte_start, byte_end = self._scope_byte_range(node)
            line_start, line_end = self._scope_line_range(node)
            unit_id = compute_unit_id(file_path, kind, qualified_name)
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
            body = node.child_by_field_name("body")
            if body is not None:
                self._walk_for_scopes(
                    body,
                    file_path=file_path,
                    qual_path=qual_path + (name,),
                    in_class=True,
                    parent_unit_id=unit_id,
                    out=out,
                )
            return

        # decorated_definition: descend, but skip the wrapper itself; the
        # inner function/class node handles decorator-span detection via
        # _scope_byte_range / _decorator_strings.
        for child in node.children:
            self._walk_for_scopes(
                child,
                file_path=file_path,
                qual_path=qual_path,
                in_class=in_class,
                parent_unit_id=parent_unit_id,
                out=out,
            )

    # ------------------------------------------------------------------
    # extract_imports
    # ------------------------------------------------------------------

    def extract_imports(self, source: bytes, file_path: str) -> tuple[ImportRef, ...]:
        tree = self._parse(source)
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

        Each `name` field child is a `dotted_name` or `aliased_import`;
        this collector returns the bound local names (alias if aliased,
        else the dotted name's last segment for the bound symbol — but
        for V1 we keep the full `dotted_name` text since the trace node
        cares about identity, not aliasing).
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
        tree = self._parse(source)
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
        tree = self._parse(source)
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
        """
        if left.type == "identifier":
            return left.text.decode("utf-8") if left.text else None
        return None

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
        tree = self._parse(source)
        # Map each ScopeUnit to its tree-sitter node by byte range.
        # ScopeUnit byte ranges include decorators when wrapped; the
        # underlying function/class node has a possibly-narrower span.
        # We match by the inner span: walk and find the
        # function/class with start_byte equal to the scope's
        # decorator-stripped start (or matching span when no decorators).
        has_error: dict[str, bool] = {}
        for scope in scope_units:
            inner_start, inner_end = self._inner_span(scope)
            node = _find_node_by_span(tree.root_node, inner_start, inner_end)
            has_error[scope.unit_id] = bool(node and node.has_error)
        return "clean", has_error

    @staticmethod
    def _inner_span(scope: ScopeUnit) -> tuple[int, int]:
        """ScopeUnit byte range minus the decorator prefix.

        For decorated scopes, `byte_start` includes `@decorator`s; the
        underlying `function_definition` / `class_definition` node has
        a narrower start. We can't recover the exact inner-start from
        ScopeUnit alone, so the ScopeUnit-to-node match in
        `compute_parser_outcome` searches for any function/class node
        whose span is contained within the ScopeUnit's span. The full
        outer span is the conservative bound.
        """
        return scope.byte_start, scope.byte_end

    # ------------------------------------------------------------------
    # Generic tree walker
    # ------------------------------------------------------------------

    @staticmethod
    def _walk(node: Node) -> Iterator[Node]:
        """Pre-order traversal of named nodes."""
        yield node
        for child in node.children:
            yield from PythonAdapter._walk(child)


# ---------------------------------------------------------------------------
# Module-level helpers (visible for the orchestrator)
# ---------------------------------------------------------------------------


def _innermost_scope_containing(
    sorted_scopes: list[ScopeUnit], byte_start: int, byte_end: int
) -> ScopeUnit | None:
    """Return the smallest scope whose byte range contains the given range,
    or None if no scope contains it (module-level)."""
    candidates = [s for s in sorted_scopes if s.byte_start <= byte_start and byte_end <= s.byte_end]
    if not candidates:
        return None
    # Smallest = largest byte_start (innermost containment).
    return max(candidates, key=lambda s: s.byte_start)


def _find_node_by_span(root: Node, byte_start: int, byte_end: int) -> Node | None:
    """Find the OUTERMOST scope-defining node whose span lies within
    [byte_start, byte_end]. Used by compute_parser_outcome to locate
    the tree-sitter node for a ScopeUnit.

    Target types include `decorated_definition` so that a syntax error
    inside a decorator (e.g., `@route(/*malformed*/)\ndef foo(): ...`)
    propagates to the ScopeUnit's `has_error`. The ScopeUnit's span
    starts at the decorator (per Month 0 spike's decorated_definition
    handling), and the decorator's `has_error` lives on the
    decorated_definition wrapper, not on the inner function_definition.
    Without `decorated_definition` in this set, the search returns the
    inner (clean) function_definition and misclassifies the scope as
    error-free — leaving downstream `degraded` derivation blind to
    decorator-region parse errors.

    "Outermost" = smallest start_byte; on tie, largest end_byte. Each
    ScopeUnit corresponds to ONE scope-defining node — the outer one.
    Picking the deepest contained match would incorrectly attribute a
    nested scope's `has_error` to the outer scope. Nested scopes are
    separate ScopeUnits with their own unit_ids.
    """
    target_types = {
        "function_definition",
        "class_definition",
        "decorated_definition",
    }
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
    try:
        source.decode("utf-8")
    except UnicodeDecodeError:
        return ParseResult(parser_outcome="failed")
    # Step 3+: parse and extract.
    adapter = PythonAdapter(resolver=resolver)
    scope_units = adapter.extract_scopes(source, file_path)
    imports = adapter.extract_imports(source, file_path)
    call_sites = adapter.extract_call_sites(source, file_path, scope_units)
    assignment_sites = adapter.extract_assignments(source, file_path, scope_units)
    outcome, has_error = adapter.compute_parser_outcome(source, file_path, scope_units)
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
    )
