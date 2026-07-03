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
`ImportPathResolver` per `nodes-receive-deps-via-closure`. Every
factory lazily imports its adapter module inside the call, keeping this
module import-light (`DECISIONS.md#018` point 6): importing the
registry loads no grammar, and each language's grammar loads on first
dispatch of that language only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.ast_facts.base import ImportPathResolver, LanguageAdapter

# The single extension→dialect mapping for the two TypeScript grammars.
# `parse_typescript` derives its dialect from THIS table (imported from
# here), so direct entry-point callers and registry consumers can never
# disagree on which grammar parses a given extension.
TYPESCRIPT_DIALECT_BY_EXTENSION: Final[dict[str, Literal["typescript", "tsx"]]] = {
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
}

# Factories import their adapter module INSIDE the call (mirroring the
# package __init__'s lazy __getattr__): importing the registry must not
# load any grammar C extension, and dispatching one language must not
# pay for — or fail on — the wheels of the others. A broken
# tree-sitter-typescript install therefore cannot make
# `get_adapter_factory(".py")` unreachable.


def _python_factory(resolver: ImportPathResolver) -> LanguageAdapter:
    from outrider.ast_facts.python_adapter import PythonAdapter

    return PythonAdapter(resolver)


def _javascript_factory(resolver: ImportPathResolver) -> LanguageAdapter:
    from outrider.ast_facts.javascript_adapter import JavaScriptAdapter

    return JavaScriptAdapter(resolver)


def _typescript_factory(resolver: ImportPathResolver) -> LanguageAdapter:
    from outrider.ast_facts.typescript_adapter import TypeScriptAdapter

    return TypeScriptAdapter(resolver=resolver, dialect="typescript")


def _tsx_factory(resolver: ImportPathResolver) -> LanguageAdapter:
    from outrider.ast_facts.typescript_adapter import TypeScriptAdapter

    return TypeScriptAdapter(resolver=resolver, dialect="tsx")


# Maps file extension (with leading dot) → adapter factory. Typed
# against the Protocol (not concrete classes) so new languages register
# without rewriting this type or the `get_adapter_factory` signature.
# TS entries are derived from TYPESCRIPT_DIALECT_BY_EXTENSION so the
# dialect mapping exists exactly once.
_LANGUAGE_ADAPTERS: Final[dict[str, Callable[[ImportPathResolver], LanguageAdapter]]] = {
    ".py": _python_factory,
    ".js": _javascript_factory,
    ".jsx": _javascript_factory,
    ".mjs": _javascript_factory,
    ".cjs": _javascript_factory,
    **{
        ext: (_tsx_factory if dialect == "tsx" else _typescript_factory)
        for ext, dialect in TYPESCRIPT_DIALECT_BY_EXTENSION.items()
    },
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
