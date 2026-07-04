"""Unit tests for `ast_facts/typescript_adapter.py` (TS + TSX dialects).

Consumes only the public `parse_typescript` entry point, the adapter's
Protocol methods, and `ast_facts` domain models — no raw parse trees and
no `tree_sitter` imports (per specs/2026-07-02-js-ts-tree-sitter-adapters.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from outrider.ast_facts.models import ParseResult, ScopeUnit, SkipReason
from outrider.ast_facts.typescript_adapter import TypeScriptAdapter, parse_typescript

if TYPE_CHECKING:
    from pathlib import Path


class _NullResolver:
    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        raise AssertionError("module-form resolution must not be consulted for JS/TS imports")

    def resolve_specifier_candidate_paths(
        self, specifier: str, importing_file_path: str, import_root: Path
    ) -> list[Path]:
        raise AssertionError("extraction paths must not consult the resolver")


def _parse(source: bytes, file_path: str = "src/svc.ts") -> ParseResult:
    return parse_typescript(source, file_path, _NullResolver())


def _scope(result: ParseResult, qualified_name: str) -> ScopeUnit:
    matches = [s for s in result.scope_units if s.qualified_name == qualified_name]
    assert len(matches) == 1, (
        f"expected exactly one scope {qualified_name!r}, got "
        f"{[s.qualified_name for s in result.scope_units]}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# TS scope shapes
# ---------------------------------------------------------------------------


def test_typed_generic_function_is_function_scope() -> None:
    result = _parse(b"function typed<T>(x: T): T { return x; }\n")
    scope = _scope(result, "typed")
    assert scope.kind == "function"


def test_typed_arrow_bound_to_const_is_function() -> None:
    result = _parse(b"const tarrow = (x: number): number => x;\n")
    assert _scope(result, "tarrow").kind == "function"


def test_abstract_class_is_class_scope_and_signatures_are_not() -> None:
    src = b"""abstract class Svc<T> implements Api {
  abstract run(): void;
  concrete(y: string): void {}
}
"""
    result = _parse(src)
    assert _scope(result, "Svc").kind == "class"
    assert _scope(result, "Svc.concrete").kind == "method"
    # `abstract run()` is an abstract_method_signature — no body, not a scope.
    assert not any(s.qualified_name == "Svc.run" for s in result.scope_units)


def test_public_field_arrow_is_method_scope() -> None:
    result = _parse(b"class A {\n  private handler = (e: Event) => { return e; };\n}\n")
    scope = _scope(result, "A.handler")
    assert scope.kind == "method"


def test_exported_decorated_class_captures_decorator_and_widens_span() -> None:
    src = b"""@Component({selector: 'app'})
export class Decorated {
  method(): void {}
}
"""
    result = _parse(src)
    scope = _scope(result, "Decorated")
    assert scope.decorators == ("Component({selector: 'app'})",)
    assert scope.line_start == 1  # span starts at the decorator, not `class`


def test_non_exported_decorated_class_captures_decorator() -> None:
    result = _parse(b"@dec\nclass Plain {}\n")
    scope = _scope(result, "Plain")
    assert scope.decorators == ("dec",)
    assert scope.line_start == 1


def test_method_decorators_are_collected_and_widen_the_span() -> None:
    src = b"""class C {
  @a() @b
  decorated(): void {}
  plain(): void {}
}
"""
    result = _parse(src)
    decorated = _scope(result, "C.decorated")
    assert decorated.decorators == ("a()", "b")
    assert decorated.line_start == 2  # widened to the decorator line
    assert _scope(result, "C.plain").decorators == ()


def test_type_only_declarations_are_not_scopes() -> None:
    src = b"""interface Api { run(): void; }
type Alias = { a: number };
enum Color { Red, Green }
declare function ambient(): void;
"""
    result = _parse(src)
    assert result.scope_units == ()


def test_function_inside_namespace_is_extracted_without_namespace_segment() -> None:
    """Namespaces (`internal_module`) are not ScopeUnits per the spec;
    functions inside them extract with an unprefixed qualified_name.
    Pinned so the shape is explicit, not accidental."""
    result = _parse(b"namespace NS { export function inner() {} }\n")
    scope = _scope(result, "inner")
    assert scope.kind == "function"
    assert scope.parent_scope_id is None


def test_decorator_arguments_do_not_produce_phantom_scopes() -> None:
    """Object-literal pairs inside decorator ARGUMENTS are metadata, not
    code scopes — the walker must not descend decorator subtrees."""
    result = _parse(b"class Svc {\n  @dec({m: () => 1})\n  run() { return 2; }\n}\n")
    assert {s.qualified_name for s in result.scope_units} == {"Svc", "Svc.run"}


def test_syntax_error_in_decorated_method_body_reaches_has_error() -> None:
    """The decorator-widened span must not blind per-scope has_error: an
    error in the method BODY flags the method even when a clean
    decorator-argument node starts earlier in the widened span."""
    result = _parse(b"class Svc {\n  @dec({m: () => 1})\n  run() { return (1 +; }\n}\n")
    assert result.has_error[_scope(result, "Svc.run").unit_id] is True
    assert result.error_lines


def test_broken_decorator_on_exported_toplevel_class_reaches_has_error() -> None:
    """Python parity: a decorator-region parse error must reach the
    decorated scope's has_error even with no enclosing scope to absorb
    it (the exported top-level class case)."""
    result = _parse(b"@Injectable(1 +)\nexport class Svc {\n  run() { return 1; }\n}\n")
    assert result.has_error[_scope(result, "Svc").unit_id] is True


# ---------------------------------------------------------------------------
# TS import shapes
# ---------------------------------------------------------------------------


def test_type_only_imports_are_marked_non_value() -> None:
    """Shadowing-guard spec: type-space names cannot back a runtime call.
    A statement-level `import type` is marked non-value (its names ride
    but admission ignores non-value refs); a per-specifier `type U` is
    excluded from `names` while the value sibling keeps the ref a value
    import. Kind classification is unchanged from the adapters arc."""
    result = _parse(b"import type { T } from 'mod';\nimport { type U, val } from './rel';\n")
    type_only = next(i for i in result.imports if i.module == "mod")
    assert type_only.import_kind == "from"
    assert type_only.is_value_import is False
    mixed = next(i for i in result.imports if i.module == "./rel")
    assert mixed.import_kind == "relative"
    assert mixed.names == ("val",)  # `type U` is type-space — excluded
    assert mixed.is_value_import is True


def test_legacy_import_require_is_direct() -> None:
    result = _parse(b"import x = require('legacy');\n")
    assert len(result.imports) == 1
    ref = result.imports[0]
    assert ref.import_kind == "direct"
    assert ref.module == "legacy"
    assert ref.names == ("x",)
    assert ref.is_simple_direct is False


def test_export_assignment_is_not_an_import() -> None:
    result = _parse(b"export = thing;\n")
    assert result.imports == ()


# ---------------------------------------------------------------------------
# Dialect dispatch — the `<A>b` cast vs JSX divergence (mirrored corpus:
# type_assertion under :language(typescript), jsx under :language(tsx))
# ---------------------------------------------------------------------------


def test_type_assertion_parses_clean_as_ts_but_errors_as_tsx() -> None:
    src = b"const a = <A>b;\n"
    as_ts = _parse(src, "src/cast.ts")
    as_tsx = _parse(src, "src/cast.tsx")
    assert as_ts.error_lines == frozenset()
    assert as_tsx.error_lines != frozenset()


def test_jsx_return_parses_clean_as_tsx_but_errors_as_ts() -> None:
    src = b"export function Page() { return <main/>; }\n"
    as_tsx = _parse(src, "src/Page.tsx")
    as_ts = _parse(src, "src/Page.ts")
    assert as_tsx.error_lines == frozenset()
    assert _scope(as_tsx, "Page").kind == "function"
    assert as_ts.error_lines != frozenset()


def test_tsx_component_with_typed_props_extracts() -> None:
    src = b"""export function Page({ title }: Props): JSX.Element {
  return <main>{title}</main>;
}
const Card = (p: CardProps) => <div>{p.body}</div>;
"""
    result = _parse(src, "src/Page.tsx")
    assert _scope(result, "Page").kind == "function"
    assert _scope(result, "Card").kind == "function"
    assert result.error_lines == frozenset()


def test_extension_dispatch_is_case_insensitive() -> None:
    src = b"export function Page() { return <main/>; }\n"
    result = _parse(src, "src/Page.TSX")
    assert result.error_lines == frozenset()


def test_dialect_argument_is_validated() -> None:
    with pytest.raises(ValueError, match="dialect must be 'typescript' or 'tsx'"):
        TypeScriptAdapter(_NullResolver(), dialect="flow")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Skips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["types/index.d.ts", "types/index.d.mts", "types/index.d.cts"],
)
def test_declaration_files_are_skipped_as_generated(path: str) -> None:
    result = _parse(b"export declare function f(): void;\n", path)
    assert result.parser_outcome == "skipped"
    assert result.skip_reason is SkipReason.GENERATED_FILENAME
    assert result.scope_units == ()


def test_source_must_be_bytes() -> None:
    with pytest.raises(TypeError, match="parse_typescript: source must be bytes"):
        parse_typescript("not bytes", "src/x.ts", _NullResolver())  # type: ignore[arg-type]


def test_legacy_import_require_relative_is_simple_direct() -> None:
    """TS `import x = require('./legacy')` with a relative source is a
    relative static import: kind "relative", `is_simple_direct=True` per
    DECISIONS.md#024 (Amended 2026-07-03)."""
    result = _parse(b"import x = require('./legacy');\n")
    assert len(result.imports) == 1
    ref = result.imports[0]
    assert ref.import_kind == "relative"
    assert ref.is_simple_direct is True


def test_relative_import_resolves_through_dialect_adapter(tmp_path: Path) -> None:
    """The TS dialect adapter inherits the JS resolution path: a relative
    import from a .ts file resolves against the fan-out (ts target)."""
    from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "dep.ts").write_text("export {};\n")
    result = _parse(b"import { d } from './dep';\n")
    ref = next(i for i in result.imports if i.module == "./dep")
    assert ref.is_simple_direct is True
    adapter = TypeScriptAdapter(resolver=COORDINATES_IMPORT_PATH_RESOLVER)
    resolution = adapter.resolve_simple_direct_import(ref, tmp_path)
    assert resolution.status == "resolved"
    assert resolution.target_path == "src/dep.ts"
