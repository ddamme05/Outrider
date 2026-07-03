"""Structural eval scenario: `parse_source` extension dispatch.

LLM-free per `docs/conventions.md` — pins the registry's language-generic
parse dispatch directly (dispatch spec): each extension group routes to
its owning parser, `.pyi` keeps its GENERATED_FILENAME audit record,
case variants dispatch identically, the two dispatch structures stay in
lockstep, and unregistered extensions raise the typed error.
"""

from unittest.mock import MagicMock

import pytest

from outrider.ast_facts import UnsupportedExtensionError, parse_source
from outrider.ast_facts.models import SkipReason
from outrider.ast_facts.registry import _LANGUAGE_ADAPTERS, _PARSE_FN_LOADERS


def test_python_routes_to_python_adapter() -> None:
    result = parse_source(b"def f():\n    return 1\n", "src/app.py", MagicMock())
    assert result.parser_outcome == "clean"
    assert [s.kind for s in result.scope_units] == ["function"]


def test_pyi_keeps_generated_filename_skip() -> None:
    """`.pyi` routes to the Python adapter, whose suffix rule skips stub
    files as GENERATED_FILENAME — the audit record the registry entry
    exists to preserve (dispatch spec, open question 3)."""
    result = parse_source(b"def f() -> int: ...\n", "stubs/app.pyi", MagicMock())
    assert result.parser_outcome == "skipped"
    assert result.skip_reason is SkipReason.GENERATED_FILENAME


def test_javascript_routes_to_javascript_adapter() -> None:
    result = parse_source(b"function f() { return 1; }\n", "src/app.js", MagicMock())
    assert result.parser_outcome == "clean"
    assert [s.kind for s in result.scope_units] == ["function"]


def test_tsx_routes_to_tsx_dialect() -> None:
    """JSX-returning source parses clean only under the tsx grammar —
    proof the `.tsx` route binds the right dialect."""
    src = b"export function Page() { return <main/>; }\n"
    result = parse_source(src, "src/Page.tsx", MagicMock())
    assert result.parser_outcome == "clean"
    assert result.error_lines == frozenset()
    assert [s.name for s in result.scope_units] == ["Page"]


def test_typescript_cast_routes_to_typescript_dialect() -> None:
    """`<A>b` casts parse clean only under the typescript grammar —
    proof `.ts` does NOT bind the tsx dialect."""
    result = parse_source(b"const a = <A>b;\n", "src/cast.ts", MagicMock())
    assert result.error_lines == frozenset()


def test_dispatch_is_case_insensitive() -> None:
    src = b"function f() { return 1; }\n"
    lower = parse_source(src, "src/app.js", MagicMock())
    upper = parse_source(src, "dist/APP.JS", MagicMock())
    assert upper.parser_outcome == lower.parser_outcome == "clean"
    assert [s.name for s in upper.scope_units] == [s.name for s in lower.scope_units]


def test_unregistered_extension_raises_typed_error() -> None:
    with pytest.raises(
        UnsupportedExtensionError, match=r"no registered adapter for extension '\.go'"
    ):
        parse_source(b"func main() {}\n", "src/main.go", MagicMock())


def test_dispatch_structures_stay_in_lockstep() -> None:
    """The adapter map and the parse-fn loader map must cover the same
    extension set — the import-time assert enforces it; this pin keeps
    the property visible in the suite and localizes the failure."""
    assert set(_LANGUAGE_ADAPTERS) == set(_PARSE_FN_LOADERS)
