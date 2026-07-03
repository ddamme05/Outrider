"""Unit tests for `ast_facts/registry.py` — the per-language adapter seam.

First dedicated registry test file (the registry shipped in V1 with a
single `.py` entry and no direct tests; the JS/TS extension makes the
dispatch table load-bearing enough to pin).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from outrider.ast_facts.javascript_adapter import JavaScriptAdapter
from outrider.ast_facts.python_adapter import PythonAdapter
from outrider.ast_facts.registry import get_adapter_factory, supported_extensions
from outrider.ast_facts.typescript_adapter import TypeScriptAdapter

if TYPE_CHECKING:
    from pathlib import Path


class _NullResolver:
    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return []


# `.pyi` routes to PythonAdapter so stub files keep their established
# `skipped+GENERATED_FILENAME` audit record under the registry gate
# (dispatch spec, open question 3).
_ALL_EXTENSIONS = (".cjs", ".cts", ".js", ".jsx", ".mjs", ".mts", ".py", ".pyi", ".ts", ".tsx")


def test_supported_extensions_lists_the_full_set() -> None:
    assert supported_extensions() == _ALL_EXTENSIONS


@pytest.mark.parametrize("extension", _ALL_EXTENSIONS)
def test_every_registered_extension_resolves(extension: str) -> None:
    factory = get_adapter_factory(extension)
    assert factory is not None
    adapter = factory(_NullResolver())
    # LanguageAdapter isn't runtime_checkable; the six Protocol methods
    # being present is the constructible contract.
    for method in (
        "extract_scopes",
        "extract_imports",
        "extract_call_sites",
        "extract_assignments",
        "resolve_simple_direct_import",
        "compute_parser_outcome",
    ):
        assert callable(getattr(adapter, method))


@pytest.mark.parametrize("extension", [".PY", ".Js", ".TSX", ".MJS"])
def test_extension_matching_is_case_insensitive(extension: str) -> None:
    assert get_adapter_factory(extension) is not None


def test_adapter_classes_route_by_extension_group() -> None:
    resolver = _NullResolver()
    py = get_adapter_factory(".py")
    js = get_adapter_factory(".js")
    ts = get_adapter_factory(".ts")
    assert py is not None and js is not None and ts is not None
    assert isinstance(py(resolver), PythonAdapter)
    js_adapter = js(resolver)
    assert isinstance(js_adapter, JavaScriptAdapter)
    assert not isinstance(js_adapter, TypeScriptAdapter)
    assert isinstance(ts(resolver), TypeScriptAdapter)


def test_ts_and_tsx_factories_bind_different_dialects() -> None:
    """Behavioral proof the two extensions bind different grammars: a
    JSX-returning function extracts under the tsx dialect, while the
    typescript dialect mis-parses it badly enough that no scope is
    recovered (and vice versa `<A>b` casts only parse under typescript —
    pinned via error_lines in test_ast_facts_typescript.py)."""
    resolver = _NullResolver()
    ts_factory = get_adapter_factory(".ts")
    tsx_factory = get_adapter_factory(".tsx")
    assert ts_factory is not None and tsx_factory is not None
    src = b"export function Page() { return <main/>; }\n"
    ts_scopes = ts_factory(resolver).extract_scopes(src, "x.ts")
    tsx_scopes = tsx_factory(resolver).extract_scopes(src, "x.tsx")
    assert [s.name for s in tsx_scopes] == ["Page"]
    assert ts_scopes == ()


@pytest.mark.parametrize("extension", ["", ".rs", "py", ".go", ".vue", ".svelte"])
def test_unknown_or_malformed_extensions_return_none(extension: str) -> None:
    assert get_adapter_factory(extension) is None
