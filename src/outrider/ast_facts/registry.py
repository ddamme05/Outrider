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

from typing import Final

from outrider.ast_facts.python_adapter import PythonAdapter

# Maps file extension (with leading dot) → adapter class.
# Adding JS/TS in V1.5 means registering `.js` / `.ts` here without
# changing any caller's interface.
_LANGUAGE_ADAPTERS: Final[dict[str, type[PythonAdapter]]] = {
    ".py": PythonAdapter,
}


def get_adapter_factory(extension: str) -> type[PythonAdapter] | None:
    """Return the adapter class for a given file extension, or None
    if no adapter is registered. Caller constructs with their resolver.
    """
    return _LANGUAGE_ADAPTERS.get(extension)


def supported_extensions() -> tuple[str, ...]:
    """Sorted tuple of registered file extensions."""
    return tuple(sorted(_LANGUAGE_ADAPTERS))
