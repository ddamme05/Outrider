# AST-facts public surface per specs/2026-04-30-ast-facts-module.md.
"""Expose domain models eagerly and load parsers only when requested.

Models, Protocols, errors, and enums import without tree-sitter. Parse entry
points and triviality helpers load their owning grammar through __getattr__
(DECISIONS.md#018 point 6). Adapter classes stay in their own modules; callers
can construct them through registry.get_adapter_factory.

tests/integration/test_ast_facts_query_registry.py checks the import-light
contract in a fresh subprocess, including audit consumers of SkipReason.
"""

from outrider.ast_facts.base import ImportPathResolver, LanguageAdapter
from outrider.ast_facts.errors import (
    AstFactsError,
    ParseError,
    TraceResolutionError,
    UnknownQueryMatchId,
    UnsupportedExtensionError,
)
from outrider.ast_facts.models import (
    AssignmentSite,
    CallSite,
    ChangedRegion,
    ComputedParserOutcome,
    ExclusionRule,
    ImportRef,
    ImportResolution,
    LexicalBinding,
    ParseResult,
    ParserOutcome,
    QueryCaptureSpan,
    QueryMatchSpan,
    ResolutionStatus,
    ScopeUnit,
    SkipReason,
    Span,
    SymbolCandidate,
    TrivialityReason,
    compute_unit_id,
)

__all__ = [
    # Domain models (§5.4 + new types added by this spec)
    "AssignmentSite",
    "CallSite",
    "ChangedRegion",
    "ImportRef",
    "ImportResolution",
    "LexicalBinding",
    "ParseResult",
    "QueryCaptureSpan",
    "QueryMatchSpan",
    "ScopeUnit",
    "Span",
    "SymbolCandidate",
    # Literal types and enums
    "ComputedParserOutcome",
    "ExclusionRule",
    "ParserOutcome",
    "ResolutionStatus",
    "SkipReason",
    "TrivialityReason",
    # Protocols
    "ImportPathResolver",
    "LanguageAdapter",
    # Errors
    "AstFactsError",
    "ParseError",
    "TraceResolutionError",
    "UnknownQueryMatchId",
    "UnsupportedExtensionError",
    # Helpers
    "compute_unit_id",
    # Lazy-loaded entry points (load tree_sitter on first access)
    "FileTrivialityContext",
    "TRIVIAL_FILTER_VERSION",
    "TrivialityVerdict",
    "build_triviality_context",
    "classify_scope_triviality",
    "parse_javascript",
    "parse_python",
    "parse_source",
    "parse_typescript",
]

# Names served by the lazy __getattr__ below. `triviality.py` imports
# tree_sitter at module load (same as python_adapter), so its surface
# stays behind the same import-light gate; only `TrivialityReason`
# (models.py) is eager for audit-side consumers.
_TRIVIALITY_LAZY: frozenset[str] = frozenset(
    {
        "FileTrivialityContext",
        "TRIVIAL_FILTER_VERSION",
        "TrivialityVerdict",
        "build_triviality_context",
        "classify_scope_triviality",
    }
)


def __getattr__(name: str) -> object:
    """Load and cache a requested parser/helper without eager grammar imports."""
    if name == "parse_python":
        from outrider.ast_facts.python_adapter import parse_python

        # Cache so subsequent lookups are direct, preserving import-light:
        # tree_sitter is only loaded on first access, not on every access.
        globals()["parse_python"] = parse_python
        return parse_python
    if name == "parse_javascript":
        from outrider.ast_facts.javascript_adapter import parse_javascript

        globals()["parse_javascript"] = parse_javascript
        return parse_javascript
    if name == "parse_typescript":
        from outrider.ast_facts.typescript_adapter import parse_typescript

        globals()["parse_typescript"] = parse_typescript
        return parse_typescript
    if name == "parse_source":
        # The registry module itself is import-light (factories and this
        # dispatcher lazy-import per language), but the lazy gate here
        # keeps the package root's contract uniform: every parse entry
        # point loads on first access.
        from outrider.ast_facts.registry import parse_source

        globals()["parse_source"] = parse_source
        return parse_source
    if name in _TRIVIALITY_LAZY:
        from outrider.ast_facts import triviality

        value = getattr(triviality, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'outrider.ast_facts' has no attribute {name!r}")
