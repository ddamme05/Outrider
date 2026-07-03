# LanguageAdapter registry per
# specs/2026-04-30-ast-facts-module.md.
"""Per-language adapter registry keyed by file extension.

Entries: `.py`/`.pyi` → `PythonAdapter`; `.js`/`.jsx`/`.mjs`/`.cjs` →
`JavaScriptAdapter`; `.ts`/`.mts`/`.cts` → typescript-dialect
`TypeScriptAdapter`; `.tsx` → tsx-dialect `TypeScriptAdapter` (per
specs/2026-07-02-js-ts-tree-sitter-adapters.md — the registry is the
single grammar-dispatch point; `typescript` and `tsx` are distinct
grammars, so each extension group registers its own factory).
`parse_source` is the language-generic parse dispatch over the same
extension groups (dispatch spec).

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

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final, Literal

from outrider.ast_facts.errors import UnsupportedExtensionError

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.ast_facts.base import ImportPathResolver, LanguageAdapter
    from outrider.ast_facts.models import ParseResult

# Extension groups — each group shares one adapter and one `parse_*`
# entry point; `_LANGUAGE_ADAPTERS` and `parse_source` both derive from
# these so registration and parse dispatch cannot disagree.
# `.pyi` routes to the Python adapter so stub files keep their
# established audit record (`skipped+GENERATED_FILENAME` from the
# parser's suffix rule) instead of flipping to UNSUPPORTED_LANGUAGE
# under the registry gate (dispatch spec, open question 3).
PYTHON_EXTENSIONS: Final[tuple[str, ...]] = (".py", ".pyi")
JAVASCRIPT_EXTENSIONS: Final[tuple[str, ...]] = (".js", ".jsx", ".mjs", ".cjs")

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
    **dict.fromkeys(PYTHON_EXTENSIONS, _python_factory),
    **dict.fromkeys(JAVASCRIPT_EXTENSIONS, _javascript_factory),
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


def parse_source(source: bytes, file_path: str, resolver: ImportPathResolver) -> ParseResult:
    """Language-generic parse dispatch: route `file_path` by extension to
    the owning `parse_*` entry point (per the dispatch spec).

    Same lazy-import discipline as the adapter factories — the grammar
    for a language loads on first parse of that language only. Raises
    `UnsupportedExtensionError` for unregistered extensions; the analyze
    path never sees it (the node gates on `get_adapter_factory` first
    and routes unregistered extensions to the UNSUPPORTED_LANGUAGE
    skip), so it fires only for direct callers that skipped the gate.
    """
    suffix = PurePosixPath(file_path.lower()).suffix
    if suffix in PYTHON_EXTENSIONS:
        from outrider.ast_facts.python_adapter import parse_python

        return parse_python(source, file_path, resolver)
    if suffix in JAVASCRIPT_EXTENSIONS:
        from outrider.ast_facts.javascript_adapter import parse_javascript

        return parse_javascript(source, file_path, resolver)
    if suffix in TYPESCRIPT_DIALECT_BY_EXTENSION:
        from outrider.ast_facts.typescript_adapter import parse_typescript

        return parse_typescript(source, file_path, resolver)
    raise UnsupportedExtensionError(
        f"no registered adapter for extension {suffix!r} (from {file_path!r}); "
        f"supported: {', '.join(supported_extensions())}"
    )
