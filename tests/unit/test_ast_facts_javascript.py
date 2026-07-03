"""Unit tests for `ast_facts/javascript_adapter.py` (JS/JSX).

Consumes only the public `parse_javascript` entry point, the adapter's
Protocol methods, and `ast_facts` domain models — no raw parse trees and
no `tree_sitter` imports, so the boundary-lint test allowlist stays at
its two existing files (per specs/2026-07-02-js-ts-tree-sitter-adapters.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from outrider.ast_facts.javascript_adapter import JavaScriptAdapter, parse_javascript
from outrider.ast_facts.models import ImportRef, ParseResult, ScopeUnit, SkipReason


class _NullResolver:
    """ImportPathResolver stub for extraction tests — extraction never
    consults the resolver; resolution tests inject the real
    `COORDINATES_IMPORT_PATH_RESOLVER` instead."""

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        raise AssertionError("module-form resolution must not be consulted for JS/TS imports")

    def resolve_specifier_candidate_paths(
        self, specifier: str, importing_file_path: str, import_root: Path
    ) -> list[Path]:
        raise AssertionError("extraction paths must not consult the resolver")


def _parse(source: bytes, file_path: str = "src/app.js") -> ParseResult:
    return parse_javascript(source, file_path, _NullResolver())


def _scope(result: ParseResult, qualified_name: str) -> ScopeUnit:
    matches = [s for s in result.scope_units if s.qualified_name == qualified_name]
    assert len(matches) == 1, (
        f"expected exactly one scope {qualified_name!r}, got "
        f"{[s.qualified_name for s in result.scope_units]}"
    )
    return matches[0]


def _import(result: ParseResult, module: str) -> ImportRef:
    matches = [i for i in result.imports if i.module == module]
    assert len(matches) == 1, f"expected one import of {module!r}, got {result.imports}"
    return matches[0]


# ---------------------------------------------------------------------------
# Scope extraction — one test per fold-rule variant (each pinned
# individually per the revert-the-fold rule; a union assertion would
# hide single-variant regressions)
# ---------------------------------------------------------------------------


def test_function_declaration_is_function_scope() -> None:
    result = _parse(b"function top(a, b) { return a + b; }\n")
    scope = _scope(result, "top")
    assert scope.kind == "function"
    assert scope.name == "top"
    assert scope.line_start == 1
    assert scope.byte_end > scope.byte_start


def test_async_and_generator_declarations_are_function_scopes() -> None:
    result = _parse(b"async function fa() {}\nfunction* gen() {}\n")
    assert _scope(result, "fa").kind == "function"
    assert _scope(result, "gen").kind == "function"


def test_arrow_bound_to_const_is_function_named_by_binding() -> None:
    result = _parse(b"const arrow = (x) => x * 2;\n")
    scope = _scope(result, "arrow")
    assert scope.kind == "function"
    assert scope.name == "arrow"


def test_named_function_expression_takes_the_binding_name() -> None:
    """`var fexpr = function inner() {}` — the binding identifier wins
    (spec Resolution 1: "named by the binding identifier")."""
    result = _parse(b"var fexpr = function inner() {};\n")
    assert _scope(result, "fexpr").kind == "function"
    assert not any(s.name == "inner" for s in result.scope_units)


def test_class_with_methods_yields_class_and_method_scopes() -> None:
    src = b"""class Widget extends Base {
  render() { return 1; }
  static create() {}
  get value() { return 2; }
  #priv() {}
}
"""
    result = _parse(src)
    widget = _scope(result, "Widget")
    assert widget.kind == "class"
    for member in ("render", "create", "value", "#priv"):
        scope = _scope(result, f"Widget.{member}")
        assert scope.kind == "method"
        assert scope.parent_scope_id == widget.unit_id


def test_class_field_arrow_is_method_scope() -> None:
    result = _parse(b"class A {\n  handler = (e) => { return e; };\n}\n")
    scope = _scope(result, "A.handler")
    assert scope.kind == "method"
    assert scope.parent_scope_id == _scope(result, "A").unit_id


def test_class_expression_bound_to_const_is_class_scope() -> None:
    result = _parse(b"const C = class { run() {} };\n")
    assert _scope(result, "C").kind == "class"
    assert _scope(result, "C.run").kind == "method"


def test_object_literal_methods_qualify_through_the_binding() -> None:
    """Members nest under the object's binding name so same-named
    members of sibling literals keep distinct unit_ids."""
    src = b"""const obj = { method() { return 1; }, arrowProp: () => 2 };
const other = { method: function () { return 3; } };
"""
    result = _parse(src)
    first = _scope(result, "obj.method")
    second = _scope(result, "other.method")
    assert first.kind == "method"
    assert _scope(result, "obj.arrowProp").kind == "method"
    assert second.kind == "method"
    assert first.unit_id != second.unit_id


def test_nested_object_literals_extend_the_qualified_path() -> None:
    result = _parse(b"const api = { users: { fetch() { return 1; } } };\n")
    assert _scope(result, "api.users.fetch").kind == "method"


def test_anonymous_arrows_and_iifes_are_not_scopes() -> None:
    """No stable qualified_name → not extracted (mirrors Python lambdas)."""
    src = b"""(function () { return 1; })();
(() => 2)();
[1, 2].map((x) => x + 1);
setTimeout(() => {}, 100);
"""
    result = _parse(src)
    assert result.scope_units == ()


def test_computed_member_names_are_not_scopes_but_their_bodies_descend() -> None:
    """A computed-name method gets no ScopeUnit (no stable name), but the
    walker still descends its body: named scopes inside are extracted and
    call sites attribute to them, not to the enclosing class."""
    result = _parse(b'class A { ["computed"]() { function inner() { helper(); } } }\n')
    assert {s.qualified_name for s in result.scope_units} == {"A", "A.inner"}
    inner = _scope(result, "A.inner")
    assert inner.kind == "function"
    assert [(c.callee_name, c.enclosing_scope_id) for c in result.call_sites] == [
        ("helper", inner.unit_id)
    ]


def test_exported_declarations_are_extracted() -> None:
    src = b"""export default function exported() {}
export function namedExport() {}
export const exportedArrow = () => {};
"""
    result = _parse(src)
    assert _scope(result, "exported").kind == "function"
    assert _scope(result, "namedExport").kind == "function"
    assert _scope(result, "exportedArrow").kind == "function"


def test_nested_function_inside_method_is_function_kind() -> None:
    result = _parse(b"class A {\n  m() { function inner() {} }\n}\n")
    inner = _scope(result, "A.m.inner")
    assert inner.kind == "function"
    assert inner.parent_scope_id == _scope(result, "A.m").unit_id


def test_jsx_in_js_and_jsx_files_parses_clean() -> None:
    src = b"function App() { return <div onClick={() => go()}>hi</div>; }\n"
    for path in ("src/app.js", "src/App.jsx"):
        result = _parse(src, path)
        assert result.parser_outcome == "clean"
        assert result.error_lines == frozenset()
        assert _scope(result, "App").kind == "function"


# ---------------------------------------------------------------------------
# Import extraction — one test per mapping row (spec Resolution 2)
# ---------------------------------------------------------------------------


def test_default_import_is_from_kind() -> None:
    ref = _import(_parse(b"import def from 'mod';\n"), "mod")
    assert ref.import_kind == "from"
    assert ref.names == ("def",)
    assert ref.is_simple_direct is False


def test_named_imports_use_alias_over_name() -> None:
    ref = _import(_parse(b"import { a, b as c } from 'mod';\n"), "mod")
    assert ref.import_kind == "from"
    assert ref.names == ("a", "c")


def test_mixed_default_and_named_import() -> None:
    ref = _import(_parse(b"import def2, { d } from 'mod';\n"), "mod")
    assert ref.import_kind == "from"
    assert ref.names == ("def2", "d")


def test_namespace_import_is_direct_kind() -> None:
    """`import * as ns` binds one namespace name — the `import m as ns`
    analog, NOT Python's wildcard (spec Resolution 2)."""
    ref = _import(_parse(b"import * as ns from 'pkg';\n"), "pkg")
    assert ref.import_kind == "direct"
    assert ref.names == ("ns",)


def test_side_effect_import_is_direct_with_no_names() -> None:
    ref = _import(_parse(b"import 'polyfill';\n"), "polyfill")
    assert ref.import_kind == "direct"
    assert ref.names == ()


def test_reexport_named_is_from_kind() -> None:
    ref = _import(_parse(b"export { x, y as z } from 'reexp';\n"), "reexp")
    assert ref.import_kind == "from"
    assert ref.names == ("x", "z")


def test_reexport_star_is_star_kind() -> None:
    ref = _import(_parse(b"export * from 'star-reexp';\n"), "star-reexp")
    assert ref.import_kind == "star"
    assert ref.names == ()


def test_reexport_star_as_namespace_keeps_star_kind_with_name() -> None:
    ref = _import(_parse(b"export * as ns from 'star-ns';\n"), "star-ns")
    assert ref.import_kind == "star"
    assert ref.names == ("ns",)


def test_require_bound_to_identifier_is_direct() -> None:
    ref = _import(_parse(b"const cj = require('commonjs');\n"), "commonjs")
    assert ref.import_kind == "direct"
    assert ref.names == ("cj",)


def test_destructured_require_is_from_with_bound_names() -> None:
    ref = _import(_parse(b"const { e, f: g } = require('mod');\n"), "mod")
    assert ref.import_kind == "from"
    assert ref.names == ("e", "g")


def test_relative_specifier_wins_over_syntactic_form() -> None:
    """Relative precedence applies to every form (spec Resolution 2)."""
    src = b"""import { a } from './rel';
import * as ns from '../up';
const r = require('./local');
export { x } from './re';
"""
    result = _parse(src)
    for module in ("./rel", "../up", "./local", "./re"):
        assert _import(result, module).import_kind == "relative"


def test_bare_dot_and_dotdot_specifiers_are_relative() -> None:
    """Node resolves `.` and `..` (directory-index imports) against the
    importing file, same as `./`-prefixed specifiers."""
    result = _parse(b"import a from '.';\nconst c = require('..');\n")
    assert _import(result, ".").import_kind == "relative"
    assert _import(result, "..").import_kind == "relative"


def test_escape_sequences_in_specifiers_are_preserved_verbatim() -> None:
    """Escape sequences must not be silently deleted from the specifier;
    they are preserved as raw source text (decoding is a resolver
    concern)."""
    result = _parse(rb"import x from './mod\u002Ddata';" + b"\n")
    assert len(result.imports) == 1
    assert result.imports[0].module == "./mod\\u002Ddata"


def test_dynamic_import_is_not_extracted() -> None:
    result = _parse(b"const dyn = import('dynamic');\n")
    assert result.imports == ()


def test_require_with_non_literal_specifier_is_not_extracted() -> None:
    result = _parse(b"const m = require(name);\nconst n = require('a' + 'b');\n")
    assert result.imports == ()


@pytest.mark.parametrize(
    ("src", "module"),
    [
        (b"import { a } from './rel';\n", "./rel"),  # ESM named
        (b"import def from '../up';\n", "../up"),  # ESM default
        (b"import * as ns from './ns';\n", "./ns"),  # ESM namespace
        (b"import './side';\n", "./side"),  # side-effect
        (b"export { x } from './re';\n", "./re"),  # re-export named
        (b"export * from './star';\n", "./star"),  # re-export star
        (b"const cj = require('./cjs');\n", "./cjs"),  # CommonJS
        (b"import x from '.';\n", "."),  # directory index
    ],
)
def test_relative_static_import_is_simple_direct(src: bytes, module: str) -> None:
    """Each relative static form pinned individually (revert-the-fold
    per variant): `import_kind="relative"` implies `is_simple_direct=True`
    per DECISIONS.md#024 (Amended 2026-07-03)."""
    ref = _import(_parse(src), module)
    assert ref.import_kind == "relative"
    assert ref.is_simple_direct is True


@pytest.mark.parametrize(
    ("src", "module"),
    [
        (b"import def from 'mod';\n", "mod"),  # bare ESM
        (b"import * as ns from 'pkg';\n", "pkg"),  # bare namespace
        (b"const cj = require('commonjs');\n", "commonjs"),  # bare CommonJS
        (b"import { a } from '@app/utils';\n", "@app/utils"),  # scoped package
    ],
)
def test_bare_specifier_stays_not_simple_direct(src: bytes, module: str) -> None:
    """Bare / scoped-package specifiers stay `is_simple_direct=False` —
    `node_modules` resolution is out of scope."""
    ref = _import(_parse(src), module)
    assert ref.is_simple_direct is False


def test_bare_specifier_never_resolves(tmp_path: Path) -> None:
    """A non-simple-direct ref returns `unresolved` without consulting
    the resolver (the _NullResolver would raise if consulted)."""
    ref = _import(_parse(b"import def from 'mod';\n"), "mod")
    adapter = JavaScriptAdapter(resolver=_NullResolver())
    resolution = adapter.resolve_simple_direct_import(ref, tmp_path)
    assert resolution.status == "unresolved"
    assert resolution.target_path is None


# ---------------------------------------------------------------------------
# resolve_simple_direct_import — filesystem resolution via the real
# coordinates resolver (root-aware twin, symlink-safe)
# ---------------------------------------------------------------------------


def _resolving_adapter() -> JavaScriptAdapter:
    from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER

    return JavaScriptAdapter(resolver=COORDINATES_IMPORT_PATH_RESOLVER)


def test_relative_import_resolves_single_target(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "rel.js").write_text("export {};\n")
    ref = _import(_parse(b"import { a } from './rel';\n"), "./rel")
    resolution = _resolving_adapter().resolve_simple_direct_import(ref, tmp_path)
    assert resolution.status == "resolved"
    assert resolution.target_path == "src/rel.js"


def test_relative_import_two_extensions_is_ambiguous(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "rel.js").write_text("export {};\n")
    (tmp_path / "src" / "rel.ts").write_text("export {};\n")
    ref = _import(_parse(b"import { a } from './rel';\n"), "./rel")
    resolution = _resolving_adapter().resolve_simple_direct_import(ref, tmp_path)
    assert resolution.status == "ambiguous"
    assert resolution.target_path is None


def test_relative_import_missing_target_is_unresolved(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    ref = _import(_parse(b"import { a } from './rel';\n"), "./rel")
    resolution = _resolving_adapter().resolve_simple_direct_import(ref, tmp_path)
    assert resolution.status == "unresolved"
    assert resolution.target_path is None


def test_relative_import_symlink_target_rejected(tmp_path: Path) -> None:
    """A symlinked candidate is omitted by the root-aware walk AND fails
    the `is_file(follow_symlinks=False)` stat — the attack asserts the
    explicit `unresolved` outcome."""
    (tmp_path / "src").mkdir()
    outside = tmp_path.parent / "outside.js"
    outside.write_text("export {};\n")
    (tmp_path / "src" / "rel.js").symlink_to(outside)
    ref = _import(_parse(b"import { a } from './rel';\n"), "./rel")
    resolution = _resolving_adapter().resolve_simple_direct_import(ref, tmp_path)
    assert resolution.status == "unresolved"
    assert resolution.target_path is None


def test_relative_import_escaping_root_is_unresolved(tmp_path: Path) -> None:
    """`'../../evil'` from a depth-1 file escapes the repo root: the
    construction surface returns zero candidates, so resolution is
    `unresolved` — nothing outside the root is ever statted."""
    (tmp_path / "src").mkdir()
    ref = _import(_parse(b"import { a } from '../../evil';\n"), "../../evil")
    resolution = _resolving_adapter().resolve_simple_direct_import(ref, tmp_path)
    assert resolution.status == "unresolved"
    assert resolution.target_path is None


# ---------------------------------------------------------------------------
# Call sites and assignments
# ---------------------------------------------------------------------------


def test_call_sites_inside_scopes_carry_raw_callee_text() -> None:
    src = b"""function f() {
  compute(1);
  obj.method(2);
}
topLevel();
"""
    result = _parse(src)
    unit_id = _scope(result, "f").unit_id
    callees = {(c.callee_name, c.enclosing_scope_id) for c in result.call_sites}
    assert callees == {("compute", unit_id), ("obj.method", unit_id)}


def test_constructor_invocation_is_a_call_site() -> None:
    """`new Pool(cfg)` parity with Python, where `Pool(cfg)` is a `call`
    node: constructor usage must be visible to same-file tracing."""
    result = _parse(b"function f() { const p = new Pool(cfg); attach(p); }\n")
    assert {c.callee_name for c in result.call_sites} == {"Pool", "attach"}


def test_module_level_calls_are_not_extracted() -> None:
    result = _parse(b"setup();\n")
    assert result.call_sites == ()


def test_assignment_expression_identifier_target() -> None:
    result = _parse(b"function f() { let y; y = 5; }\n")
    assert [(a.target_name, a.enclosing_scope_id) for a in result.assignment_sites] == [
        ("y", _scope(result, "f").unit_id)
    ]


def test_value_bearing_declarator_is_an_assignment() -> None:
    result = _parse(b"function f() { const local = compute(1); }\n")
    assert [a.target_name for a in result.assignment_sites] == ["local"]


def test_declarator_without_value_is_not_an_assignment() -> None:
    result = _parse(b"function f() { let pending; }\n")
    assert result.assignment_sites == ()


def test_member_and_destructuring_targets_are_skipped() -> None:
    result = _parse(b"function f() { obj.x = 1; [a, b] = pair; }\n")
    assert result.assignment_sites == ()


def test_function_valued_declarator_is_scope_not_assignment() -> None:
    """A binding extracted as a ScopeUnit is not double-counted as an
    AssignmentSite."""
    result = _parse(b"function f() { const g = () => 1; }\n")
    assert _scope(result, "f.g").kind == "function"
    assert result.assignment_sites == ()


# ---------------------------------------------------------------------------
# Parser outcome, degradation, skips, guards
# ---------------------------------------------------------------------------


def test_clean_parse_has_no_errors() -> None:
    result = _parse(b"function f() { return 1; }\n")
    assert result.parser_outcome == "clean"
    assert set(result.has_error.values()) == {False}
    assert result.error_lines == frozenset()


def test_flow_types_in_js_degrade_per_scope() -> None:
    """Flow annotations error under the JS grammar; the error stays
    localized to the affected scope (the adapter records it — the
    audited degrade-to-JUDGED consumption is analyze's job, per spec)."""
    src = b"function typed(x: number): number { return x; }\nfunction clean(y) { return y; }\n"
    result = _parse(src, "src/flow.js")
    assert result.parser_outcome == "clean"
    assert result.has_error[_scope(result, "typed").unit_id] is True
    assert result.has_error[_scope(result, "clean").unit_id] is False
    assert 1 in result.error_lines


def test_missing_closing_token_at_scope_end_reaches_has_error() -> None:
    """A missing closing token is a zero-width MISSING node inserted at
    exactly the recovered scope's byte_end — the point check must be
    end-inclusive or the structurally broken scope reads clean."""
    result = _parse(b"class A { m() { return 1; }\n")  # class brace unclosed
    assert result.has_error[_scope(result, "A").unit_id] is True
    assert result.has_error[_scope(result, "A.m").unit_id] is True
    assert result.error_lines


def test_compute_parser_outcome_always_returns_clean() -> None:
    """V1 policy pin, same as Python: tree-sitter degrades via ERROR
    nodes, so the file-level outcome is always "clean" — per-scope
    `has_error` carries the signal. Tightening to "any has_error =>
    failed" is a DECISIONS.md change; this test forces that
    acknowledgment."""
    adapter = JavaScriptAdapter(resolver=_NullResolver())
    src = b"function broken(x: number) { return x; }\n"
    scopes = adapter.extract_scopes(src, "src/flow.js")
    outcome, has_error = adapter.compute_parser_outcome(src, "src/flow.js", scopes)
    assert outcome == "clean"
    assert True in has_error.values()


@pytest.mark.parametrize(
    "path",
    ["dist/app.min.js", "dist/app.min.mjs", "dist/app.min.cjs"],
)
def test_minified_js_family_is_skipped(path: str) -> None:
    result = _parse(b"var x=1;", path)
    assert result.parser_outcome == "skipped"
    assert result.skip_reason is SkipReason.MINIFIED
    assert result.scope_units == ()


def test_minified_skip_is_case_insensitive() -> None:
    """Registry dispatch lowercases extensions, so the suffix skips must
    cover the same case-variant filenames (JQUERY.MIN.JS is legal on
    case-insensitive filesystems)."""
    result = _parse(b"var x=1;", "dist/JQUERY.MIN.JS")
    assert result.parser_outcome == "skipped"
    assert result.skip_reason is SkipReason.MINIFIED


def test_node_modules_is_skipped_as_vendored() -> None:
    result = _parse(b"var x = 1;", "node_modules/pkg/index.js")
    assert result.parser_outcome == "skipped"
    assert result.skip_reason is SkipReason.VENDORED


def test_source_must_be_bytes() -> None:
    with pytest.raises(TypeError, match="parse_javascript: source must be bytes"):
        parse_javascript("not bytes", "src/app.js", _NullResolver())  # type: ignore[arg-type]


def test_invalid_utf8_fails() -> None:
    result = _parse(b"\xff\xfefunction f() {}")
    assert result.parser_outcome == "failed"
    assert result.scope_units == ()
