# JavaScript adapter implementing LanguageAdapter Protocol per
# specs/2026-07-02-js-ts-tree-sitter-adapters.md.
"""JavaScript/JSX adapter for tree-sitter-javascript 0.25.0.

Ports `PythonAdapter` method-for-method against the JS grammar (JSX is
native to tree-sitter-javascript, so `.jsx` and JSX-in-`.js` parse
cleanly; Flow-typed `.js` degrades to per-scope `has_error`). Structural
extraction is a tree walk — no `.scm` queries are involved, mirroring
the Python adapter.

Scope-kind fold per the spec (canonical §5.4 enums unchanged):
  * `function_declaration` / `generator_function_declaration`, and
    `arrow_function` / `function_expression` / `generator_function`
    bound to a `variable_declarator` identifier → ``kind="function"``
    (named by the binding).
  * `method_definition` (class body OR object literal), and class-field
    arrows (`field_definition` with a function value) → ``kind="method"``.
  * `class_declaration` and `class` expressions bound to a declarator
    identifier → ``kind="class"``.
  * Anonymous arrows / IIFEs / callbacks and computed-name members are
    not extracted (no stable `qualified_name` — mirrors Python lambdas).
  * Object-literal values nest into `qualified_name` (``obj.method``),
    so same-named members of sibling literals keep distinct unit_ids.

Import-kind fold per the spec: relative specifiers (`./`, `../`) →
``"relative"`` regardless of form; default/named imports, destructured
`require`, and `export … from` re-exports → ``"from"``; namespace
(`import * as ns`), side-effect, bare `require`, and TS
`import x = require(...)` → ``"direct"``; `export * from` → ``"star"``.
Dynamic `import()` is an expression, not a static import — not
extracted. Relative static imports (``import_kind="relative"`` — any
extracted form whose source is `./`-, `../`-, `.`- or `..`-shaped)
carry ``is_simple_direct=True`` and resolve through
`resolve_simple_direct_import` via the injected resolver's
relative-specifier surface (`DECISIONS.md#024`, Amended 2026-07-03);
bare and namespace-package specifiers stay ``False`` (`node_modules`
resolution is out of scope).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import tree_sitter_javascript
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
    error_byte_spans_from_tree,
    error_lines_from_tree,
    innermost_scope_containing,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from outrider.ast_facts.base import ImportPathResolver

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_JS_LANGUAGE: Final = Language(tree_sitter_javascript.language())
_JS_PARSER: Final = Parser(_JS_LANGUAGE)

_RELATIVE_SPECIFIER_PREFIXES: Final = ("./", "../")


# ---------------------------------------------------------------------------
# JavaScriptAdapter
# ---------------------------------------------------------------------------


class JavaScriptAdapter:
    """Implements `LanguageAdapter` for JavaScript/JSX via tree-sitter-javascript.

    The node-type tables are ClassVars so `TypeScriptAdapter` extends them
    (TS adds `abstract_class_declaration`, `public_field_definition`)
    without duplicating the walk.
    """

    _FUNCTION_DECL_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"function_declaration", "generator_function_declaration"}
    )
    _CLASS_DECL_TYPES: ClassVar[frozenset[str]] = frozenset({"class_declaration"})
    _FIELD_DEF_TYPES: ClassVar[frozenset[str]] = frozenset({"field_definition"})
    # Function-valued expressions that make a named binding a scope.
    _FUNCTION_VALUE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"arrow_function", "function_expression", "generator_function"}
    )
    _CLASS_VALUE_TYPES: ClassVar[frozenset[str]] = frozenset({"class"})
    # Method/field name node types with a stable identifier (computed
    # names and string/number keys are skipped — no stable qualified_name).
    _MEMBER_NAME_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"property_identifier", "private_property_identifier"}
    )
    # Lexical-binding visibility frames (shadowing-guard spec). Function
    # frames bound params + hoisted `var`/`function` declarations; block
    # frames bound `let`/`const`/`class`. A function body IS a
    # `statement_block`, so the block set alone resolves block-scoped
    # kinds inside functions; expression-bodied arrows cannot contain
    # declarations.
    _FUNCTION_FRAME_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "function_expression",
            "generator_function",
            "arrow_function",
            "method_definition",
        }
    )
    _BLOCK_FRAME_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"statement_block", "for_statement", "for_in_statement"}
    )
    # Binding-position pattern nodes the identifier collector recurses
    # into. Unknown node types are NOT recursed (under-collection means
    # the shadow guard misses an exotic pattern — the FP persists and
    # JUDGED covers; over-collection would wrongly deny). TS parameter
    # wrappers (`required_parameter`/`optional_parameter`) are handled
    # by field, so `type_annotation` subtrees are never walked.
    _PATTERN_RECURSE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"object_pattern", "array_pattern", "rest_pattern"}
    )

    def __init__(self, resolver: ImportPathResolver) -> None:
        self._resolver = resolver
        self._parser: Parser = _JS_PARSER

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, source: bytes) -> Tree:
        return self._parser.parse(source)

    @staticmethod
    def _node_text(node: Node) -> str:
        # Node.text returns bytes; decode UTF-8. Source has already
        # passed UTF-8 validation per the parse pipeline.
        return node.text.decode("utf-8") if node.text else ""

    @staticmethod
    def _string_text(string_node: Node | None) -> str:
        """Unquoted specifier text of a `string` node: `string_fragment`
        and `escape_sequence` children concatenated in order (`''` has
        none). Escape sequences are preserved as raw source text
        (`\\u002D` stays `\\u002D`, not decoded to `-`) — dropping them
        would corrupt the specifier; decoding is a resolver concern."""
        if string_node is None:
            return ""
        return "".join(
            JavaScriptAdapter._node_text(c)
            for c in string_node.named_children
            if c.type in ("string_fragment", "escape_sequence")
        )

    @staticmethod
    def _decorators_for(node: Node) -> tuple[tuple[str, ...], Node | None]:
        """Decorator text for a scope node, `@` stripped (Python-adapter
        convention), plus the earliest decorator node that lies OUTSIDE
        the scope node's own span (for span widening).

        Two attachment sites, verified against the grammars:
          * decorator children INSIDE the node (TS non-exported decorated
            class) — already inside the span, no widening;
          * immediately preceding named siblings (TS method decorators in
            a class_body; exported-class decorators hang on the
            export_statement as siblings of the declaration).
        """
        inside = [c for c in node.named_children if c.type == "decorator"]
        outside: list[Node] = []
        sib = node.prev_named_sibling
        while sib is not None and sib.type == "decorator":
            outside.append(sib)
            sib = sib.prev_named_sibling
        all_decs = sorted(inside + outside, key=lambda n: n.start_byte)
        texts: list[str] = []
        for dec in all_decs:
            text = JavaScriptAdapter._node_text(dec)
            texts.append(text[1:] if text.startswith("@") else text)
        first_outside = min(outside, key=lambda n: n.start_byte) if outside else None
        return tuple(texts), first_outside

    @staticmethod
    def _walk(node: Node) -> Iterator[Node]:
        """Pre-order traversal of `node` and all descendants (unnamed
        nodes included; callers filter by `node.type`). Iterative so
        adversarially deep parse trees can't exhaust the recursion
        limit — same rationale as `PythonAdapter._walk`.
        """
        stack: list[Node] = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))

    # ------------------------------------------------------------------
    # extract_scopes
    # ------------------------------------------------------------------

    def extract_scopes(self, source: bytes, file_path: str) -> tuple[ScopeUnit, ...]:
        return self._extract_scopes_from_tree(self._parse(source), file_path)

    @staticmethod
    def _push_scope_children(
        stack: list[tuple[Node, tuple[str, ...], bool, str | None]],
        node: Node,
        qual_path: tuple[str, ...],
        in_class: bool,
        parent_unit_id: str | None,
    ) -> None:
        """Generic descent for the scope walk: push all children with the
        inherited frame, reversed so leftmost pops first (pre-order).

        `decorator` subtrees are NOT descended: their contents (config
        object literals, inline arrows) are metadata captured as
        decorator text, not code scopes — descending emitted phantom
        method ScopeUnits from decorator-argument object pairs.
        """
        for child in reversed(node.children):
            if child.type == "decorator":
                continue
            stack.append((child, qual_path, in_class, parent_unit_id))

    def _extract_scopes_from_tree(self, tree: Tree, file_path: str) -> tuple[ScopeUnit, ...]:
        scopes: list[ScopeUnit] = []
        # Frame: (node, qual_path, in_class, parent_unit_id) — the same
        # state the Python walker carries. `in_class` is tracked for
        # shape parity though JS methods are recognized by node type
        # (method_definition), not by position.
        stack: list[tuple[Node, tuple[str, ...], bool, str | None]] = [
            (tree.root_node, (), False, None)
        ]
        while stack:
            node, qual_path, in_class, parent_unit_id = stack.pop()
            node_type = node.type

            if node_type in self._FUNCTION_DECL_TYPES:
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = self._node_text(name_node)
                unit_id = self._emit(
                    scopes,
                    span_node=node,
                    kind="function",
                    name=name,
                    qual_path=qual_path,
                    file_path=file_path,
                    parent_unit_id=parent_unit_id,
                )
                body = node.child_by_field_name("body")
                if body is not None:
                    stack.append((body, (*qual_path, name), False, unit_id))
                continue

            if node_type in self._CLASS_DECL_TYPES:
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = self._node_text(name_node)
                unit_id = self._emit(
                    scopes,
                    span_node=node,
                    kind="class",
                    name=name,
                    qual_path=qual_path,
                    file_path=file_path,
                    parent_unit_id=parent_unit_id,
                )
                body = node.child_by_field_name("body")
                if body is not None:
                    stack.append((body, (*qual_path, name), True, unit_id))
                continue

            if node_type == "method_definition":
                name_node = node.child_by_field_name("name")
                if name_node is None or name_node.type not in self._MEMBER_NAME_TYPES:
                    # Computed/string-keyed member: no stable name for a
                    # ScopeUnit, but the BODY may hold named scopes —
                    # descend generically (dropping the subtree lost
                    # nested functions and mis-attributed their calls).
                    self._push_scope_children(stack, node, qual_path, in_class, parent_unit_id)
                    continue
                name = self._node_text(name_node)
                unit_id = self._emit(
                    scopes,
                    span_node=node,
                    kind="method",
                    name=name,
                    qual_path=qual_path,
                    file_path=file_path,
                    parent_unit_id=parent_unit_id,
                )
                body = node.child_by_field_name("body")
                if body is not None:
                    stack.append((body, (*qual_path, name), False, unit_id))
                continue

            if node_type in self._FIELD_DEF_TYPES:
                # `field_definition` names via `property`; TS
                # `public_field_definition` names via `name`.
                prop = node.child_by_field_name("property") or node.child_by_field_name("name")
                value = node.child_by_field_name("value")
                if (
                    prop is not None
                    and prop.type in self._MEMBER_NAME_TYPES
                    and value is not None
                    and value.type in self._FUNCTION_VALUE_TYPES
                ):
                    name = self._node_text(prop)
                    unit_id = self._emit(
                        scopes,
                        span_node=node,
                        kind="method",
                        name=name,
                        qual_path=qual_path,
                        file_path=file_path,
                        parent_unit_id=parent_unit_id,
                    )
                    body = value.child_by_field_name("body")
                    if body is not None:
                        stack.append((body, (*qual_path, name), False, unit_id))
                    continue
                self._push_scope_children(stack, node, qual_path, in_class, parent_unit_id)
                continue

            if node_type == "pair":
                key = node.child_by_field_name("key")
                value = node.child_by_field_name("value")
                if key is not None and key.type == "property_identifier" and value is not None:
                    name = self._node_text(key)
                    if value.type in self._FUNCTION_VALUE_TYPES:
                        unit_id = self._emit(
                            scopes,
                            span_node=node,
                            kind="method",
                            name=name,
                            qual_path=qual_path,
                            file_path=file_path,
                            parent_unit_id=parent_unit_id,
                        )
                        body = value.child_by_field_name("body")
                        if body is not None:
                            stack.append((body, (*qual_path, name), False, unit_id))
                        continue
                    if value.type == "object":
                        # Nested literal: extend the qualified path so
                        # members keep distinct unit_ids; no scope emitted
                        # for the object itself.
                        stack.append((value, (*qual_path, name), False, parent_unit_id))
                        continue
                self._push_scope_children(stack, node, qual_path, in_class, parent_unit_id)
                continue

            if node_type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value = node.child_by_field_name("value")
                if name_node is not None and name_node.type == "identifier" and value is not None:
                    name = self._node_text(name_node)
                    if value.type in self._FUNCTION_VALUE_TYPES:
                        unit_id = self._emit(
                            scopes,
                            span_node=node,
                            kind="function",
                            name=name,
                            qual_path=qual_path,
                            file_path=file_path,
                            parent_unit_id=parent_unit_id,
                        )
                        body = value.child_by_field_name("body")
                        if body is not None:
                            stack.append((body, (*qual_path, name), False, unit_id))
                        continue
                    if value.type in self._CLASS_VALUE_TYPES:
                        unit_id = self._emit(
                            scopes,
                            span_node=node,
                            kind="class",
                            name=name,
                            qual_path=qual_path,
                            file_path=file_path,
                            parent_unit_id=parent_unit_id,
                        )
                        body = value.child_by_field_name("body")
                        if body is not None:
                            stack.append((body, (*qual_path, name), True, unit_id))
                        continue
                    if value.type == "object":
                        stack.append((value, (*qual_path, name), False, parent_unit_id))
                        continue
                self._push_scope_children(stack, node, qual_path, in_class, parent_unit_id)
                continue

            self._push_scope_children(stack, node, qual_path, in_class, parent_unit_id)
        return tuple(scopes)

    def _emit(
        self,
        out: list[ScopeUnit],
        *,
        span_node: Node,
        kind: str,
        name: str,
        qual_path: tuple[str, ...],
        file_path: str,
        parent_unit_id: str | None,
    ) -> str:
        """Construct + append a ScopeUnit; returns its unit_id.

        Span = the span node, widened to include any decorator nodes
        that precede it as siblings (TS method decorators live in the
        class_body; exported-class decorators on the export_statement).
        """
        decorators, first_outside_dec = self._decorators_for(span_node)
        byte_start, byte_end = span_node.start_byte, span_node.end_byte
        line_start = span_node.start_point[0] + 1
        line_end = span_node.end_point[0] + 1
        if first_outside_dec is not None:
            byte_start = min(byte_start, first_outside_dec.start_byte)
            line_start = min(line_start, first_outside_dec.start_point[0] + 1)
        qualified_name = ".".join((*qual_path, name))
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
                decorators=decorators,
                parent_scope_id=parent_unit_id,
            )
        )
        return unit_id

    # ------------------------------------------------------------------
    # extract_imports
    # ------------------------------------------------------------------

    def extract_imports(self, source: bytes, file_path: str) -> tuple[ImportRef, ...]:
        return self._extract_imports_from_tree(self._parse(source), file_path)

    def _extract_imports_from_tree(self, tree: Tree, file_path: str) -> tuple[ImportRef, ...]:
        imports: list[ImportRef] = []
        for node in self._walk(tree.root_node):
            if node.type == "import_statement":
                imports.append(self._build_import_statement(node, file_path))
            elif node.type == "export_statement" and node.child_by_field_name("source"):
                imports.append(self._build_reexport(node, file_path))
            elif node.type == "variable_declarator":
                ref = self._maybe_require(node, file_path)
                if ref is not None:
                    imports.append(ref)
        return tuple(imports)

    @classmethod
    def _kind_for(cls, module: str, default_kind: str) -> str:
        """Relative specifiers win over the syntactic form, mirroring
        Python where `relative` is its own kind. Bare `.` and `..`
        (directory-index imports) are relative too — Node resolves them
        against the importing file, same as `./`-prefixed forms."""
        if module in (".", "..") or module.startswith(_RELATIVE_SPECIFIER_PREFIXES):
            return "relative"
        return default_kind

    def _build_import_statement(self, node: Node, file_path: str) -> ImportRef:
        line = node.start_point[0] + 1
        # TS legacy `import x = require('m')`.
        req = next((c for c in node.named_children if c.type == "import_require_clause"), None)
        if req is not None:
            ident = next((c for c in req.named_children if c.type == "identifier"), None)
            module = self._string_text(req.child_by_field_name("source"))
            names: tuple[str, ...] = (self._node_text(ident),) if ident is not None else ()
            kind = self._kind_for(module, "direct")
            return ImportRef(
                file_path=file_path,
                line=line,
                import_kind=kind,
                module=module,
                names=names,
                is_simple_direct=kind == "relative",
            )
        module = self._string_text(node.child_by_field_name("source"))
        # TS statement-level type-only import (`import type { X } from 'm'`):
        # a bare `type` keyword token rides between `import` and the clause.
        # Type-space bindings cannot back a runtime call — non-value.
        statement_type_only = any(c.type == "type" for c in node.children)
        clause = next((c for c in node.named_children if c.type == "import_clause"), None)
        defaults: list[str] = []
        namespaces: list[str] = []
        named: list[str] = []
        has_named_imports = False
        if clause is not None:
            for child in clause.named_children:
                if child.type == "identifier":
                    defaults.append(self._node_text(child))
                elif child.type == "namespace_import":
                    namespaces.extend(
                        self._node_text(i) for i in child.named_children if i.type == "identifier"
                    )
                elif child.type == "named_imports":
                    has_named_imports = True
                    for spec in child.named_children:
                        if spec.type != "import_specifier":
                            continue
                        # TS per-specifier type-only (`import { type Pool,
                        # Client }`): the type-space name is not a value
                        # binding — excluded from `names`.
                        if any(c.type == "type" for c in spec.children):
                            continue
                        alias = spec.child_by_field_name("alias")
                        name_field = spec.child_by_field_name("name")
                        target = alias if alias is not None else name_field
                        if target is not None:
                            named.append(self._node_text(target))
        names = tuple(n for n in (*defaults, *namespaces, *named) if n)
        # Namespace-only and side-effect imports bind at most one
        # namespace name — the `import m as ns` analog → "direct";
        # any default/named binding → "from".
        default_kind = "from" if (defaults or has_named_imports) else "direct"
        kind = self._kind_for(module, default_kind)
        # Value iff the statement is not type-only AND it binds something
        # (a side-effect `import "m"` loads the module but binds no name a
        # runtime call can resolve through).
        return ImportRef(
            file_path=file_path,
            line=line,
            import_kind=kind,
            module=module,
            names=names,
            is_simple_direct=kind == "relative",
            is_value_import=not statement_type_only and clause is not None,
        )

    def _build_reexport(self, node: Node, file_path: str) -> ImportRef:
        line = node.start_point[0] + 1
        module = self._string_text(node.child_by_field_name("source"))
        clause = next((c for c in node.named_children if c.type == "export_clause"), None)
        ns_export = next((c for c in node.named_children if c.type == "namespace_export"), None)
        if clause is not None:
            names_list: list[str] = []
            for spec in clause.named_children:
                if spec.type != "export_specifier":
                    continue
                alias = spec.child_by_field_name("alias")
                name_field = spec.child_by_field_name("name")
                target = alias if alias is not None else name_field
                if target is not None:
                    names_list.append(self._node_text(target))
            kind = self._kind_for(module, "from")
            names = tuple(names_list)
        elif ns_export is not None:
            # `export * as ns from 'm'` — still the namespace-polluting
            # wildcard on the source side → "star", with the bound name.
            kind = self._kind_for(module, "star")
            names = tuple(
                self._node_text(i) for i in ns_export.named_children if i.type == "identifier"
            )
        else:
            kind = self._kind_for(module, "star")
            names = ()
        # Re-exports bind NO local name — nothing in this file can call
        # through them, so they are never value imports.
        return ImportRef(
            file_path=file_path,
            line=line,
            import_kind=kind,
            module=module,
            names=names,
            is_simple_direct=kind == "relative",
            is_value_import=False,
        )

    def _maybe_require(self, node: Node, file_path: str) -> ImportRef | None:
        """CommonJS `const x = require('m')` / `const {a, b: c} = require('m')`.

        Only single-string-literal `require` calls bound directly to an
        identifier or object pattern are extracted; anything else
        (member chains, computed specifiers, array patterns) is not an
        import statement shape we can attribute.
        """
        value = node.child_by_field_name("value")
        if value is None or value.type != "call_expression":
            return None
        fn = value.child_by_field_name("function")
        if fn is None or fn.type != "identifier" or self._node_text(fn) != "require":
            return None
        args = value.child_by_field_name("arguments")
        if args is None:
            return None
        arg_nodes = args.named_children
        if len(arg_nodes) != 1 or arg_nodes[0].type != "string":
            return None
        module = self._string_text(arg_nodes[0])
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        line = node.start_point[0] + 1
        if name_node.type == "identifier":
            kind = self._kind_for(module, "direct")
            names: tuple[str, ...] = (self._node_text(name_node),)
        elif name_node.type == "object_pattern":
            kind = self._kind_for(module, "from")
            collected: list[str] = []
            for child in name_node.named_children:
                if child.type == "shorthand_property_identifier_pattern":
                    collected.append(self._node_text(child))
                elif child.type == "pair_pattern":
                    bound = child.child_by_field_name("value")
                    if bound is not None and bound.type == "identifier":
                        collected.append(self._node_text(bound))
            names = tuple(collected)
        else:
            return None
        return ImportRef(
            file_path=file_path,
            line=line,
            import_kind=kind,
            module=module,
            names=names,
            is_simple_direct=kind == "relative",
        )

    # ------------------------------------------------------------------
    # extract_lexical_bindings (shadowing-guard spec)
    # ------------------------------------------------------------------

    def extract_lexical_bindings(self, source: bytes, file_path: str) -> tuple[LexicalBinding, ...]:
        return self._extract_lexical_bindings_from_tree(self._parse(source), file_path)

    def _pattern_identifiers(self, node: Node) -> Iterator[Node]:
        """Binding-POSITION identifiers of a declaration pattern: plain
        identifiers, destructuring shorthands, `{key: local}` values,
        defaults' LEFT sides, rest elements, and TS parameter wrappers
        (by field, so `type_annotation` subtrees are never walked —
        identifiers inside types or default-value EXPRESSIONS are uses,
        not bindings, and must not widen the shadow set)."""
        if node.type in ("identifier", "shorthand_property_identifier_pattern"):
            yield node
        elif node.type == "pair_pattern":
            value = node.child_by_field_name("value")
            if value is not None:
                yield from self._pattern_identifiers(value)
        elif node.type in ("assignment_pattern", "object_assignment_pattern"):
            left = node.child_by_field_name("left")
            if left is not None:
                yield from self._pattern_identifiers(left)
        elif node.type in ("required_parameter", "optional_parameter"):
            pattern = node.child_by_field_name("pattern")
            if pattern is not None:
                yield from self._pattern_identifiers(pattern)
        elif node.type in self._PATTERN_RECURSE_TYPES:
            for child in node.named_children:
                yield from self._pattern_identifiers(child)

    def _enclosing_frame_span(
        self, node: Node, frame_types: frozenset[str]
    ) -> tuple[int, int] | None:
        cur = node.parent
        while cur is not None:
            if cur.type in frame_types:
                return (cur.start_byte, cur.end_byte)
            cur = cur.parent
        return None

    def _extract_lexical_bindings_from_tree(
        self, tree: Tree, file_path: str
    ) -> tuple[LexicalBinding, ...]:
        """Per-kind visibility spans per the shadowing-guard spec:
        params → the enclosing function node; `var` → nearest enclosing
        function frame, else the whole module (hoisting); `let`/`const`
        → nearest block frame, else module; `function`/`class`
        declarations → nearest block/function frame, else module; catch
        params → the catch clause. Module-scope non-import declarations
        DO emit records (a global like `process` is legally shadowable
        at module scope); CJS `require` declarators and import/export
        statements emit none — an import binding must not self-shadow.
        """
        bindings: list[LexicalBinding] = []
        root = tree.root_node
        module_span = (root.start_byte, root.end_byte)
        decl_frames = self._BLOCK_FRAME_TYPES | self._FUNCTION_FRAME_TYPES

        def _add(name_node: Node, kind: str, span: tuple[int, int]) -> None:
            bindings.append(
                LexicalBinding(
                    file_path=file_path,
                    name=self._node_text(name_node),
                    kind=kind,
                    line=name_node.start_point[0] + 1,
                    visibility_byte_start=span[0],
                    visibility_byte_end=span[1],
                )
            )

        for node in self._walk(root):
            if node.type in self._FUNCTION_FRAME_TYPES:
                own_span = (node.start_byte, node.end_byte)
                params = node.child_by_field_name("parameters")
                if params is not None:
                    for child in params.named_children:
                        for ident in self._pattern_identifiers(child):
                            _add(ident, "param", own_span)
                else:
                    # Arrow single-param shorthand: `x => ...`.
                    single = node.child_by_field_name("parameter")
                    if single is not None:
                        for ident in self._pattern_identifiers(single):
                            _add(ident, "param", own_span)
                if node.type in self._FUNCTION_DECL_TYPES:
                    name = node.child_by_field_name("name")
                    if name is not None:
                        _add(
                            name,
                            "function",
                            self._enclosing_frame_span(node, decl_frames) or module_span,
                        )
            elif node.type in ("variable_declaration", "lexical_declaration"):
                if node.type == "variable_declaration":
                    kind = "var"
                    span = (
                        self._enclosing_frame_span(node, self._FUNCTION_FRAME_TYPES) or module_span
                    )
                else:
                    kind = "const" if node.children and node.children[0].type == "const" else "let"
                    span = self._enclosing_frame_span(node, self._BLOCK_FRAME_TYPES) or module_span
                for declarator in node.named_children:
                    if declarator.type != "variable_declarator":
                        continue
                    if self._maybe_require(declarator, file_path) is not None:
                        # A CJS import binding must not shadow itself.
                        continue
                    name_node = declarator.child_by_field_name("name")
                    if name_node is None:
                        continue
                    for ident in self._pattern_identifiers(name_node):
                        _add(ident, kind, span)
            elif node.type == "for_in_statement":
                # `for (const k in obj)` / `for (const v of arr)`: the
                # declaration rides ON the statement (kind + left fields),
                # not as a nested lexical_declaration node.
                kind_node = node.child_by_field_name("kind")
                left = node.child_by_field_name("left")
                if kind_node is not None and left is not None:
                    kind_text = self._node_text(kind_node)
                    if kind_text == "var":
                        span = (
                            self._enclosing_frame_span(node, self._FUNCTION_FRAME_TYPES)
                            or module_span
                        )
                    else:
                        span = (node.start_byte, node.end_byte)
                    kind = kind_text if kind_text in ("var", "let", "const") else "let"
                    for ident in self._pattern_identifiers(left):
                        _add(ident, kind, span)
            elif node.type in self._CLASS_DECL_TYPES:
                name = node.child_by_field_name("name")
                if name is not None:
                    _add(
                        name,
                        "class",
                        self._enclosing_frame_span(node, decl_frames) or module_span,
                    )
            elif node.type == "catch_clause":
                param = node.child_by_field_name("parameter")
                if param is not None:
                    catch_span = (node.start_byte, node.end_byte)
                    for ident in self._pattern_identifiers(param):
                        _add(ident, "catch", catch_span)
        return tuple(bindings)

    # ------------------------------------------------------------------
    # extract_call_sites
    # ------------------------------------------------------------------

    def extract_call_sites(
        self,
        source: bytes,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[CallSite, ...]:
        """Calls inside extracted ScopeUnits. Module-level calls are
        skipped per the §5.4 non-goal (same as Python)."""
        return self._extract_call_sites_from_tree(self._parse(source), file_path, scope_units)

    def _extract_call_sites_from_tree(
        self,
        tree: Tree,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[CallSite, ...]:
        sorted_scopes = sorted(scope_units, key=lambda s: (s.byte_start, -s.byte_end))
        calls: list[CallSite] = []
        for node in self._walk(tree.root_node):
            # `new Pool(cfg)` is a `new_expression` (callee under the
            # `constructor` field), not a `call_expression` — Python
            # parity requires it: `Pool(cfg)` is a `call` node there.
            if node.type == "call_expression":
                function_node = node.child_by_field_name("function")
            elif node.type == "new_expression":
                function_node = node.child_by_field_name("constructor")
            else:
                continue
            enclosing = innermost_scope_containing(sorted_scopes, node.start_byte, node.end_byte)
            if enclosing is None:
                continue
            if function_node is None:
                continue
            # `callee_name` is RAW SOURCE TEXT per canonical §5.4 ("raw
            # text; resolution is a separate concern") — same contract
            # as the Python adapter.
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
        """Single-identifier targets only, mirroring the Python V1 rule:
        `assignment_expression` with an identifier left side, plus
        value-bearing `variable_declarator`s (JS declarations are the
        assignment form Python doesn't have). Declarators whose value is
        a function/class are already represented as ScopeUnits and are
        not double-counted as assignments; destructuring, member, and
        subscript targets return nothing (name-keyed backward tracing).
        """
        sorted_scopes = sorted(scope_units, key=lambda s: (s.byte_start, -s.byte_end))
        scope_value_types = self._FUNCTION_VALUE_TYPES | self._CLASS_VALUE_TYPES
        sites: list[AssignmentSite] = []
        for node in self._walk(tree.root_node):
            if node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                if left is None or left.type != "identifier":
                    continue
                target_name = self._node_text(left)
            elif node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value = node.child_by_field_name("value")
                if (
                    name_node is None
                    or name_node.type != "identifier"
                    or value is None
                    or value.type in scope_value_types
                ):
                    continue
                target_name = self._node_text(name_node)
            else:
                continue
            if not target_name:
                continue
            enclosing = innermost_scope_containing(sorted_scopes, node.start_byte, node.end_byte)
            if enclosing is None:
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

    # ------------------------------------------------------------------
    # resolve_simple_direct_import
    # ------------------------------------------------------------------

    def resolve_simple_direct_import(
        self, import_ref: ImportRef, import_root: Path
    ) -> ImportResolution:
        """Resolve a relative static import via the injected resolver's
        relative-specifier surface (`DECISIONS.md#024`, Amended
        2026-07-03). Mirrors `PythonAdapter.resolve_simple_direct_import`:
        candidates come pre-validated (containment + symlink-safe walk in
        the resolver implementation); existence checks use the
        symlink-safe primitive (allowlist: only
        `.is_file(follow_symlinks=False)`). Resolution is
        importing-file-relative — the ref's own `file_path` anchors the
        specifier. Non-relative refs (`is_simple_direct=False`: bare and
        namespace-package specifiers) stay `unresolved` — `node_modules`
        resolution is out of scope.
        """
        if not import_ref.is_simple_direct:
            return ImportResolution(status="unresolved", target_path=None)
        candidates = self._resolver.resolve_specifier_candidate_paths(
            import_ref.module, import_ref.file_path, import_root
        )
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

        **V1 policy: always returns `("clean", has_error)`** — identical
        to the Python adapter: tree-sitter degrades to ERROR/MISSING
        nodes rather than failing outright (verified for Flow-typed
        `.js`, which localizes ERROR nodes to the affected scope), so
        the "failed" outcome remains the defensive forward-compat shape.
        Pinned by `tests/unit/test_ast_facts_javascript.py`; changing
        the policy is a `DECISIONS.md` matter, as for Python.
        """
        del file_path
        return self._compute_parser_outcome_from_tree(self._parse(source), scope_units)

    def _compute_parser_outcome_from_tree(
        self,
        tree: Tree,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[ComputedParserOutcome, dict[str, bool]]:
        # Error-span intersection instead of Python's ScopeUnit→node
        # containment lookup: JS/TS ScopeUnit spans may start at a
        # preceding decorator sibling (see `_emit`), so no single tree
        # node corresponds to the span — a node lookup either misses the
        # decorator region entirely (top-level exported decorated class)
        # or lands on a decorator-argument node and reads the wrong
        # `has_error`. A scope has an error iff any ERROR/MISSING span
        # intersects its (widened) byte range; zero-width MISSING nodes
        # count as points with an INCLUSIVE end bound — tree-sitter
        # inserts a missing closing token at exactly the recovered
        # scope's byte_end (`class A { m() { return 1; }` with the class
        # brace unclosed puts MISSING at A's and A.m's shared end), so a
        # half-open point check would read the broken scope as clean.
        # Slightly conservative vs Python (an ERROR region enclosing a
        # recovered scope flags it; a boundary MISSING flags the scope
        # it abuts), which is the safe direction under
        # `parse-errors-degrade-to-judged`.
        error_spans = error_byte_spans_from_tree(tree)
        has_error: dict[str, bool] = {}
        for scope in scope_units:
            has_error[scope.unit_id] = any(
                (start < scope.byte_end and end > scope.byte_start)
                or (start == end and scope.byte_start <= start <= scope.byte_end)
                for start, end in error_spans
            )
        return "clean", has_error


# ---------------------------------------------------------------------------
# Shared parse pipeline + canonical entry point
# ---------------------------------------------------------------------------


def _run_parse_pipeline(
    adapter: JavaScriptAdapter, source: bytes, file_path: str, *, entry_point: str
) -> ParseResult:
    """size→pattern→decode→parse pipeline shared by `parse_javascript`
    and `parse_typescript` — the same ordering `parse_python` runs, with
    one parse shared across the extraction passes. `entry_point` names
    the caller in the bytes-guard message.
    """
    if not isinstance(source, bytes):
        raise TypeError(
            f"{entry_point}: source must be bytes, got {type(source).__name__}; "
            f"the consuming node decodes the file once via fetch and passes "
            f"raw bytes through. Decoding to str at this layer would defeat "
            f"the size→pattern→decode→parse DoS pipeline ordering."
        )
    skip_reason: SkipReason | None = should_skip(file_path, source)
    if skip_reason is not None:
        return ParseResult(parser_outcome="skipped", skip_reason=skip_reason)
    try:
        source.decode("utf-8")
    except UnicodeDecodeError:
        return ParseResult(parser_outcome="failed")
    tree = adapter._parse(source)
    scope_units = adapter._extract_scopes_from_tree(tree, file_path)
    imports = adapter._extract_imports_from_tree(tree, file_path)
    call_sites = adapter._extract_call_sites_from_tree(tree, file_path, scope_units)
    assignment_sites = adapter._extract_assignments_from_tree(tree, file_path, scope_units)
    lexical_bindings = adapter._extract_lexical_bindings_from_tree(tree, file_path)
    outcome, has_error = adapter._compute_parser_outcome_from_tree(tree, scope_units)
    if outcome == "failed":
        # Discard extracted tuples: the failed-path ParseResult carries
        # the empty-tuples shape (enforced by the ParseResult validator).
        return ParseResult(parser_outcome="failed")
    return ParseResult(
        parser_outcome="clean",
        scope_units=scope_units,
        imports=imports,
        call_sites=call_sites,
        assignment_sites=assignment_sites,
        lexical_bindings=lexical_bindings,
        has_error=has_error,
        error_lines=error_lines_from_tree(tree),
    )


def parse_javascript(source: bytes, file_path: str, resolver: ImportPathResolver) -> ParseResult:
    """Canonical JS/JSX entry point, mirroring `parse_python`'s contract:
    `skipped` on exclusion-rule match, `failed` on invalid UTF-8, else
    the fully-populated clean `ParseResult`.
    """
    return _run_parse_pipeline(
        JavaScriptAdapter(resolver=resolver),
        source,
        file_path,
        entry_point="parse_javascript",
    )
