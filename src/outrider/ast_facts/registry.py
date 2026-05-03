# LanguageAdapter registry per
# specs/2026-04-30-ast-facts-module.md.
"""Per-language adapter registry keyed by file extension.

V1: single entry `.py` → `PythonAdapter`. The seam V1.5 will extend
when JS/TS support arrives.

`get_adapter_factory(extension)` returns an adapter factory — typed as
`Callable[[ImportPathResolver], LanguageAdapter]` — not an adapter
instance. The caller constructs the adapter with their own
`ImportPathResolver` per `nodes-receive-deps-via-closure`. For V1 the
factory is the `PythonAdapter` class itself (a class is callable);
V1.5's JS/TS adapter classes register the same way as long as they
accept an `ImportPathResolver` in the constructor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from outrider.ast_facts.python_adapter import PythonAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.ast_facts.base import ImportPathResolver, LanguageAdapter

# Maps file extension (with leading dot) → adapter factory.
# An "adapter factory" is any callable taking an `ImportPathResolver`
# and returning a `LanguageAdapter`-shaped instance — typed against the
# Protocol (not the concrete `PythonAdapter`) so V1.5 can register
# `.js` / `.ts` adapters here without rewriting this type or the
# `get_adapter_factory` signature.
_LANGUAGE_ADAPTERS: Final[dict[str, Callable[[ImportPathResolver], LanguageAdapter]]] = {
    ".py": PythonAdapter,
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
