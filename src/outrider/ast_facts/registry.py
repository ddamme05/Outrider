# LanguageAdapter registry per
# specs/2026-04-30-ast-facts-module.md.
"""Per-language adapter registry keyed by file extension.

Entries: `.py` → `PythonAdapter`; `.js`/`.jsx`/`.mjs`/`.cjs` →
`JavaScriptAdapter`; `.ts`/`.mts`/`.cts` → typescript-dialect
`TypeScriptAdapter`; `.tsx` → tsx-dialect `TypeScriptAdapter` (per
specs/2026-07-02-js-ts-tree-sitter-adapters.md — the registry is the
single grammar-dispatch point; `typescript` and `tsx` are distinct
grammars, so each extension group registers its own factory).

`get_adapter_factory(extension)` returns an adapter factory — typed as
`Callable[[ImportPathResolver], LanguageAdapter]` — not an adapter
instance. The caller constructs the adapter with their own
`ImportPathResolver` per `nodes-receive-deps-via-closure`. A factory is
any callable accepting an `ImportPathResolver`: the `PythonAdapter` /
`JavaScriptAdapter` classes themselves, or the dialect-binding
functions for TypeScript.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from outrider.ast_facts.javascript_adapter import JavaScriptAdapter
from outrider.ast_facts.python_adapter import PythonAdapter
from outrider.ast_facts.typescript_adapter import TypeScriptAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.ast_facts.base import ImportPathResolver, LanguageAdapter


def _typescript_factory(resolver: ImportPathResolver) -> LanguageAdapter:
    return TypeScriptAdapter(resolver=resolver, dialect="typescript")


def _tsx_factory(resolver: ImportPathResolver) -> LanguageAdapter:
    return TypeScriptAdapter(resolver=resolver, dialect="tsx")


# Maps file extension (with leading dot) → adapter factory. Typed
# against the Protocol (not concrete classes) so new languages register
# without rewriting this type or the `get_adapter_factory` signature.
_LANGUAGE_ADAPTERS: Final[dict[str, Callable[[ImportPathResolver], LanguageAdapter]]] = {
    ".py": PythonAdapter,
    ".js": JavaScriptAdapter,
    ".jsx": JavaScriptAdapter,
    ".mjs": JavaScriptAdapter,
    ".cjs": JavaScriptAdapter,
    ".ts": _typescript_factory,
    ".mts": _typescript_factory,
    ".cts": _typescript_factory,
    ".tsx": _tsx_factory,
}


def get_adapter_factory(
    extension: str,
) -> Callable[[ImportPathResolver], LanguageAdapter] | None:
    """Return the adapter factory for a given file extension, or None
    if no adapter is registered. Caller constructs with their resolver.

    Extension matching is case-insensitive: `.py`, `.PY`, `.Py` all
    resolve to the same adapter. Without normalization, a caller passing
    `Path(file).suffix` for a file named `Foo.PY` (legal on
    case-insensitive filesystems and across some ingest paths) would
    silently get `None` and skip analyzing the file. Empty string or
    extensions without a leading dot return `None` (consistent with the
    "unsupported language" semantics).
    """
    if not extension:
        return None
    return _LANGUAGE_ADAPTERS.get(extension.lower())


def supported_extensions() -> tuple[str, ...]:
    """Sorted tuple of registered file extensions."""
    return tuple(sorted(_LANGUAGE_ADAPTERS))
