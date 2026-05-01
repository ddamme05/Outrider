# LanguageAdapter registry per
# specs/2026-04-30-ast-facts-module.md.
"""Per-language adapter registry keyed by file extension.

V1: single entry `.py` → `PythonAdapter`. The seam V1.5 will extend
when JS/TS support arrives.

The registry is a factory of adapter constructors, not adapter
instances — each call to `get_adapter_factory(extension)` returns the
class so the caller can construct it with their own `ImportPathResolver`
per `nodes-receive-deps-via-closure`.
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
    """
    return _LANGUAGE_ADAPTERS.get(extension)


def supported_extensions() -> tuple[str, ...]:
    """Sorted tuple of registered file extensions."""
    return tuple(sorted(_LANGUAGE_ADAPTERS))
