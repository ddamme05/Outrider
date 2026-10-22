# JavaScript adapter implementing LanguageAdapter Protocol per
# specs/2026-07-02-js-ts-tree-sitter-adapters.md.
"""JavaScript/JSX adapter for tree-sitter-javascript 0.25.0.

Structural extraction walks the tree without `.scm` queries. JSX parses
natively; Flow-typed `.js` degrades through per-scope `has_error`.
`TypeScriptAdapter` reuses the walk with extended node-type tables.

Canonical §5.4 scope kinds: function declarations and identifier-bound
function expressions are functions; class/object methods and function-valued
class fields are methods; class declarations and identifier-bound class
expressions are classes. Anonymous functions and computed-name members have
no stable scope name. Object-literal bindings qualify members (`obj.method`)
to keep sibling unit_ids distinct.

Import kinds: default/named imports, destructured `require`, and named
re-exports are `from`; namespace/side-effect imports, identifier-bound
`require`, and TS `import x = require(...)` are `direct`; wildcard re-exports
are `star`. Relative specifiers (`./`, `../`, `.`, `..`) override any form
with `relative` and `is_simple_direct=True` (DECISIONS.md#024). Only these
resolve through the injected resolver; `node_modules` resolution is out of
scope. Dynamic `import()` expressions are not extracted.
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

# Shared JavaScript parser.

_JS_LANGUAGE: Final = Language(tree_sitter_javascript.language())
_JS_PARSER: Final = Parser(_JS_LANGUAGE)

_RELATIVE_SPECIFIER_PREFIXES: Final = ("./", "../")


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
    # Computed/string/number keys have no stable qualified_name.
    _MEMBER_NAME_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"property_identifier", "private_property_identifier"}
    )
    # Visibility frames (DECISIONS.md#060): params/var use functions,
    # let/const use blocks, function/class declarations use either.
    # Function bodies are statement_blocks; expression arrows have no declarations.
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
        {"statement_block", "for_statement", "for_in_statement", "switch_body"}
    )
    # Recurse only through known binding patterns. Missing an exotic pattern
    # leaves a false positive for JUDGED; over-collection would wrongly deny it.
    # TS parameter wrappers are handled by field, excluding type annotations.
    _PATTERN_RECURSE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"object_pattern", "array_pattern", "rest_pattern"}
    )

    def __init__(self, resolver: ImportPathResolver) -> None:
        self._resolver = resolver
        self._parser: Parser = _JS_PARSER

    # Internal helpers

    def _parse(self, source: bytes) -> Tree:
        return self._parser.parse(source)

    @staticmethod
    def _node_text(node: Node) -> str:
        # The parse pipeline has already validated source UTF-8.
        return node.text.decode("utf-8") if node.text else ""

    @staticmethod
    def _string_text(string_node: Node | None) -> str:
        """Join a string's fragments and escapes without quotes (`''` is empty).

        Preserve raw escapes (`\\u002D` stays `\\u002D`, not `-`); dropping them
        corrupts the specifier, while decoding belongs to the resolver.
        """
        if string_node is None:
            return ""
        return "".join(
            JavaScriptAdapter._node_text(c)
            for c in string_node.named_children
            if c.type in ("string_fragment", "escape_sequence")
        )

    @staticmethod
    def _decorators_for(node: Node) -> tuple[tuple[str, ...], Node | None]:
        """Return decorator text without `@` and the first span-widening sibling.

        TS non-exported classes hold decorators as children, already in span.
        Methods and exported classes have preceding decorator siblings under
        class_body/export_statement; these widen the scope's start.
        """
        inside: list[Node] = []
        for decorator_child in node.named_children:
            if decorator_child.type == "decorator":
                inside.append(decorator_child)
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
        """Walk all descendants in pre-order, including unnamed nodes.

        Iteration avoids recursion-limit exhaustion on deeply nested trees.
        """
        stack: list[Node] = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))

    # Scope extraction

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
        """Queue children in pre-order with the inherited scope frame.

        Skip decorator subtrees: their objects/arrows are captured metadata,
        and descending them would emit phantom code scopes.
        """
        for child in reversed(node.children):
            if child.type == "decorator":
                continue
            stack.append((child, qual_path, in_class, parent_unit_id))

    def _emit_and_descend(
        self,
        scopes: list[ScopeUnit],
        stack: list[tuple[Node, tuple[str, ...], bool, str | None]],
        *,
        span_node: Node,
        kind: str,
        name: str,
        qual_path: tuple[str, ...],
        file_path: str,
        parent_unit_id: str | None,
        body_owner: Node,
        child_in_class: bool,
    ) -> None:
        """Emit one scope, then queue its body as a child frame.

        `body_owner` is the declaration or its function/class value. A body-less
        node emits a scope without further descent.
        """
        unit_id = self._emit(
            scopes,
            span_node=span_node,
            kind=kind,
            name=name,
            qual_path=qual_path,
            file_path=file_path,
            parent_unit_id=parent_unit_id,
        )
        body = body_owner.child_by_field_name("body")
        if body is not None:
            stack.append((body, (*qual_path, name), child_in_class, unit_id))

    def _extract_scopes_from_tree(self, tree: Tree, file_path: str) -> tuple[ScopeUnit, ...]:
        scopes: list[ScopeUnit] = []
        # in_class preserves the Python frame shape; JS methods use node type,
        # rather than position, for recognition.
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
                self._emit_and_descend(
                    scopes,
                    stack,
                    span_node=node,
                    kind="function",
                    name=name,
                    qual_path=qual_path,
                    file_path=file_path,
                    parent_unit_id=parent_unit_id,
                    body_owner=node,
                    child_in_class=False,
                )
                continue

            if node_type in self._CLASS_DECL_TYPES:
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = self._node_text(name_node)
                self._emit_and_descend(
                    scopes,
                    stack,
                    span_node=node,
                    kind="class",
                    name=name,
                    qual_path=qual_path,
                    file_path=file_path,
                    parent_unit_id=parent_unit_id,
                    body_owner=node,
                    child_in_class=True,
                )
                continue

            if node_type == "method_definition":
                name_node = node.child_by_field_name("name")
                if name_node is None or name_node.type not in self._MEMBER_NAME_TYPES:
                    # Unnamed members still descend: nested named scopes need
                    # extraction so their calls are attributed correctly.
                    self._push_scope_children(stack, node, qual_path, in_class, parent_unit_id)
                    continue
                name = self._node_text(name_node)
                self._emit_and_descend(
                    scopes,
                    stack,
                    span_node=node,
                    kind="method",
                    name=name,
                    qual_path=qual_path,
                    file_path=file_path,
                    parent_unit_id=parent_unit_id,
                    body_owner=node,
                    child_in_class=False,
                )
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
                    self._emit_and_descend(
                        scopes,
                        stack,
                        span_node=node,
                        kind="method",
                        name=name,
                        qual_path=qual_path,
                        file_path=file_path,
                        parent_unit_id=parent_unit_id,
                        body_owner=value,
                        child_in_class=False,
                    )
                    continue
                self._push_scope_children(stack, node, qual_path, in_class, parent_unit_id)
                continue

            if node_type == "pair":
                key = node.child_by_field_name("key")
                value = node.child_by_field_name("value")
                if key is not None and key.type == "property_identifier" and value is not None:
                    name = self._node_text(key)
                    if value.type in self._FUNCTION_VALUE_TYPES:
                        self._emit_and_descend(
                            scopes,
                            stack,
                            span_node=node,
                            kind="method",
                            name=name,
                            qual_path=qual_path,
                            file_path=file_path,
                            parent_unit_id=parent_unit_id,
                            body_owner=value,
                            child_in_class=False,
                        )
                        continue
                    if value.type == "object":
                        # Qualify nested members for distinct unit_ids without
                        # emitting a scope for the object itself.
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
                        self._emit_and_descend(
                            scopes,
                            stack,
                            span_node=node,
                            kind="function",
                            name=name,
                            qual_path=qual_path,
                            file_path=file_path,
                            parent_unit_id=parent_unit_id,
                            body_owner=value,
                            child_in_class=False,
                        )
                        continue
                    if value.type in self._CLASS_VALUE_TYPES:
                        self._emit_and_descend(
                            scopes,
                            stack,
                            span_node=node,
                            kind="class",
                            name=name,
                            qual_path=qual_path,
                            file_path=file_path,
                            parent_unit_id=parent_unit_id,
                            body_owner=value,
                            child_in_class=True,
                        )
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
        """Append a ScopeUnit and return its unit_id.

        Widen the node's span for preceding decorator siblings (TS methods
        and exported classes).
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

    # Static import extraction

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
        """Relative specifiers override syntax, including directory-index `.`/`..`.

        Node resolves these against the importing file, like `./` and `../`.
        """
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
                is_value_import=bool(names),
            )
        module = self._string_text(node.child_by_field_name("source"))
        # TS statement-level `type` precedes the clause; these bindings cannot
        # back runtime calls.
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
                        # TS per-specifier `type` names are not value bindings.
                        if any(c.type == "type" for c in spec.children):
                            continue
                        alias = spec.child_by_field_name("alias")
                        name_field = spec.child_by_field_name("name")
                        target = alias if alias is not None else name_field
                        if target is not None:
                            named.append(self._node_text(target))
        names = tuple(n for n in (*defaults, *namespaces, *named) if n)
        # Default/named forms are "from"; namespace/side-effect forms are "direct".
        default_kind = "from" if (defaults or has_named_imports) else "direct"
        kind = self._kind_for(module, default_kind)
        # Value imports need a surviving runtime binding. Side-effect and
        # all-type-specifier imports bind none, so cannot prove module_presence.
        return ImportRef(
            file_path=file_path,
            line=line,
            import_kind=kind,
            module=module,
            names=names,
            is_simple_direct=kind == "relative",
            is_value_import=not statement_type_only and bool(names),
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
            # `export * as ns` remains a source-side wildcard, with a name.
            kind = self._kind_for(module, "star")
            names = tuple(
                self._node_text(i) for i in ns_export.named_children if i.type == "identifier"
            )
        else:
            kind = self._kind_for(module, "star")
            names = ()
        # Re-exports bind no local callable name, so are never value imports.
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
        """Extract CommonJS require calls bound to identifiers or object patterns.

        Require one string literal; member chains, computed specifiers, and
        array patterns cannot be attributed as import shapes.
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
        # As in ESM, a binding-less require (`const {} = require("pg")`) loads
        # the module but cannot prove module_presence for a runtime call.
        return ImportRef(
            file_path=file_path,
            line=line,
            import_kind=kind,
            module=module,
            names=names,
            is_simple_direct=kind == "relative",
            is_value_import=bool(names),
        )

    # Lexical bindings for the shadowing guard

    def extract_lexical_bindings(self, source: bytes, file_path: str) -> tuple[LexicalBinding, ...]:
        return self._extract_lexical_bindings_from_tree(self._parse(source), file_path)

    def _pattern_identifiers(self, node: Node) -> Iterator[Node]:
        """Collect binding identifiers from patterns and TS parameter wrappers.

        Follow `{key: local}` values, default left sides, and rest elements.
        Type annotations and default-value expressions are uses, not bindings,
        and must not widen the shadow set.
        """
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
        """Extract visibility spans per the shadowing-guard spec (DECISIONS.md#060).

        Params span their function; var hoists to the nearest function;
        let/const use the nearest block; function/class declarations use the
        nearest block or function. Missing frames fall back to module scope;
        catch params span the catch clause. Module declarations can shadow
        globals, but imports/re-exports and CJS require bindings emit no records
        so an import cannot shadow itself.
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
                # for-in/of declarations use the statement's kind/left fields,
                # rather than a nested lexical_declaration.
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

    # Call sites inside extracted scopes

    def extract_call_sites(
        self,
        source: bytes,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[CallSite, ...]:
        """Extract scoped calls; module-level calls are a canonical §5.4 non-goal."""
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
            # Constructors are calls too, with the callee in a different field.
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
            # Canonical §5.4 keeps callee_name as raw text; resolution is separate.
            calls.append(
                CallSite(
                    file_path=file_path,
                    line=node.start_point[0] + 1,
                    callee_name=self._node_text(function_node),
                    enclosing_scope_id=enclosing.unit_id,
                )
            )
        return tuple(calls)

    # Assignments inside extracted scopes

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
        """Extract identifier assignments and value-bearing variable declarators.

        Function/class declarators are scopes, not duplicate assignments.
        Name-keyed backward tracing excludes destructuring and member targets.
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

    # Relative import resolution

    def resolve_simple_direct_import(
        self, import_ref: ImportRef, import_root: Path
    ) -> ImportResolution:
        """Resolve relative static imports through the injected resolver (#024).

        DECISIONS.md#024 anchors specifiers to the importing ref's file_path.
        The resolver validates containment and walks without symlinks; existence
        checks must use the allowlisted `.is_file(follow_symlinks=False)`.
        One existing candidate resolves; multiple existing candidates are ambiguous.
        No existing candidates means unresolved.
        Non-relative refs stay unresolved; node_modules resolution is out of scope.
        """
        if not import_ref.is_simple_direct:
            return ImportResolution(status="unresolved", target_path=None)
        candidates = self._resolver.resolve_specifier_candidate_paths(
            import_ref.module, import_ref.file_path, import_root
        )
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

    # Per-scope parse errors

    def compute_parser_outcome(
        self,
        source: bytes,
        file_path: str,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[ComputedParserOutcome, dict[str, bool]]:
        """Return `("clean", has_error)`, with errors keyed by scope unit_id.

        V1 treats tree-sitter ERROR/MISSING nodes as per-scope degradation,
        including Flow-typed .js. The failed outcome is a defensive future shape.
        Tests pin this policy; changing it requires a DECISIONS.md entry.
        """
        del file_path
        return self._compute_parser_outcome_from_tree(self._parse(source), scope_units)

    def _compute_parser_outcome_from_tree(
        self,
        tree: Tree,
        scope_units: tuple[ScopeUnit, ...],
    ) -> tuple[ComputedParserOutcome, dict[str, bool]]:
        # Intersect error spans with the widened scope: preceding decorators
        # mean no single tree node covers it. Zero-width MISSING points include
        # byte_end, where tree-sitter can insert a missing closing token.
        # Enclosing ERROR regions and boundary MISSING points conservatively
        # degrade affected scopes under `parse-errors-degrade-to-judged`.
        error_spans = error_byte_spans_from_tree(tree)
        has_error: dict[str, bool] = {}
        for scope in scope_units:
            has_error[scope.unit_id] = any(
                (start < scope.byte_end and end > scope.byte_start)
                or (start == end and scope.byte_start <= start <= scope.byte_end)
                for start, end in error_spans
            )
        return "clean", has_error


# Shared parse pipeline and canonical entry point


def _run_parse_pipeline(
    adapter: JavaScriptAdapter, source: bytes, file_path: str, *, entry_point: str
) -> ParseResult:
    """Share one parse across JS/TS extraction in size→pattern→decode→parse order.

    `entry_point` identifies the caller in the bytes-guard error message.
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
        # Failed ParseResults require empty tuples; discard extracted facts.
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
