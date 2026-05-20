# AST-facts public surface per specs/2026-04-30-ast-facts-module.md.
"""AST-facts module — Python adapter and shared domain models.

**Light types eager-import** at module load (Pydantic models, Protocols,
errors, literal types, the SkipReason enum). These have no `tree_sitter`
dependency and are safe for `outrider.audit.events` and other consumers
to import without paying tree-sitter init cost.

**`parse_python` lazy-imports** via module-level `__getattr__` per
`DECISIONS.md#018` point 6: accessing `outrider.ast_facts.parse_python`
triggers loading `python_adapter.py`, which in turn loads `tree_sitter`.
This keeps `from outrider.ast_facts.models import SkipReason` cheap for
audit-side consumers — `python_adapter.py` is loaded only when the
adapter is actually needed.

The import-light contract is enforced by a subprocess-isolated
regression test in `tests/integration/test_ast_facts_query_registry.py`
that asserts `tree_sitter not in sys.modules` after
`from outrider.ast_facts.models import SkipReason`.
"""

from outrider.ast_facts.base import ImportPathResolver, LanguageAdapter
from outrider.ast_facts.errors import (
    AstFactsError,
    ParseError,
    TraceResolutionError,
    UnknownQueryMatchId,
)
from outrider.ast_facts.models import (
    AssignmentSite,
    CallSite,
    ChangedRegion,
    ComputedParserOutcome,
    ExclusionRule,
    ImportRef,
    ImportResolution,
    ParseResult,
    ParserOutcome,
    QueryCaptureSpan,
    QueryMatchSpan,
    ResolutionStatus,
    ScopeUnit,
    SkipReason,
    Span,
    SymbolCandidate,
    compute_unit_id,
)

__all__ = [
    # Domain models (§5.4 + new types added by this spec)
    "AssignmentSite",
    "CallSite",
    "ChangedRegion",
    "ImportRef",
    "ImportResolution",
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
    # Protocols
    "ImportPathResolver",
    "LanguageAdapter",
    # Errors
    "AstFactsError",
    "ParseError",
    "TraceResolutionError",
    "UnknownQueryMatchId",
    # Helpers
    "compute_unit_id",
    # Lazy-loaded entry point (loads tree_sitter on first access)
    "parse_python",
]


def __getattr__(name: str) -> object:
    """Lazy-load `parse_python` to keep light-type imports tree-sitter-free.

    Per `DECISIONS.md#018` point 6: this `__getattr__` is the gate that
    keeps `from outrider.ast_facts.models import SkipReason` cheap for
    audit consumers. `parse_python` lives in `python_adapter.py` which
    imports `tree_sitter`; loading it on every `from outrider.ast_facts`
    import would pull tree-sitter into every consumer's module graph.
    """
    if name == "parse_python":
        from outrider.ast_facts.python_adapter import parse_python

        # Cache so subsequent lookups are direct, preserving import-light:
        # tree_sitter is only loaded on first access, not on every access.
        globals()["parse_python"] = parse_python
        return parse_python
    raise AttributeError(f"module 'outrider.ast_facts' has no attribute {name!r}")
