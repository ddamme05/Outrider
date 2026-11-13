"""Extract source facts from TypeScript and TSX.

The registry selects the TypeScript or TSX grammar for each extension; using
the wrong grammar would confuse casts with JSX. This adapter extends the
JavaScript tree walk with TypeScript node types and decorators.

See specs/2026-07-02-js-ts-tree-sitter-adapters.md for the extraction rules.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar, Final, Literal

import tree_sitter_typescript
from tree_sitter import Language, Parser

from outrider.ast_facts.javascript_adapter import JavaScriptAdapter, _run_parse_pipeline

if TYPE_CHECKING:
    from outrider.ast_facts.base import ImportPathResolver
    from outrider.ast_facts.models import ParseResult

# ---------------------------------------------------------------------------
# Module-level singletons — one per dialect
# ---------------------------------------------------------------------------

_TS_LANGUAGE: Final = Language(tree_sitter_typescript.language_typescript())
_TSX_LANGUAGE: Final = Language(tree_sitter_typescript.language_tsx())
_TS_PARSER: Final = Parser(_TS_LANGUAGE)
_TSX_PARSER: Final = Parser(_TSX_LANGUAGE)

TypeScriptDialect = Literal["typescript", "tsx"]


# ---------------------------------------------------------------------------
# TypeScriptAdapter
# ---------------------------------------------------------------------------


class TypeScriptAdapter(JavaScriptAdapter):
    """Implements `LanguageAdapter` for TypeScript and TSX.

    Dialect is fixed at construction (extension-only routing per the
    spec's grammar-dispatch non-goal); the registry registers one
    factory per dialect rather than sniffing content here.
    """

    _CLASS_DECL_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"class_declaration", "abstract_class_declaration"}
    )
    _FIELD_DEF_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"field_definition", "public_field_definition"}
    )

    def __init__(
        self, resolver: ImportPathResolver, dialect: TypeScriptDialect = "typescript"
    ) -> None:
        super().__init__(resolver=resolver)
        if dialect not in ("typescript", "tsx"):
            raise ValueError(
                f"TypeScriptAdapter dialect must be 'typescript' or 'tsx', got {dialect!r}"
            )
        self._dialect: TypeScriptDialect = dialect
        self._parser = _TSX_PARSER if dialect == "tsx" else _TS_PARSER


# ---------------------------------------------------------------------------
# Canonical entry point
# ---------------------------------------------------------------------------


def parse_typescript(source: bytes, file_path: str, resolver: ImportPathResolver) -> ParseResult:
    """Canonical TS/TSX entry point, mirroring `parse_python`'s contract.

    Dialect comes from the registry's `TYPESCRIPT_DIALECT_BY_EXTENSION`
    — the single extension→dialect mapping — so direct callers and
    registry consumers provably parse a given extension with the same
    grammar. Unknown extensions default to the typescript dialect.
    """
    from outrider.ast_facts.registry import TYPESCRIPT_DIALECT_BY_EXTENSION

    suffix = PurePosixPath(file_path.lower()).suffix
    dialect: TypeScriptDialect = TYPESCRIPT_DIALECT_BY_EXTENSION.get(suffix, "typescript")
    return _run_parse_pipeline(
        TypeScriptAdapter(resolver=resolver, dialect=dialect),
        source,
        file_path,
        entry_point="parse_typescript",
    )
