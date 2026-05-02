# Typed exceptions for the ast_facts/ surface per docs/conventions.md.
"""Exception hierarchy.

Per `docs/conventions.md`: "Functions that can fail with meaningful
variants raise typed exceptions. No `raise Exception(...)` and no
returning `None` for 'didn't work.'"

`UnknownQueryMatchId` is raised by `queries/registry.py` per the V1
ast_facts/ spec's Implementation Sketch — defined here (not in
`queries/errors.py`) so callers across `ast_facts/`, `queries/`, and
`audit/` import a single error module.
"""


class AstFactsError(Exception):
    """Base for all ast_facts/ typed exceptions."""


class ParseError(AstFactsError):
    """Raised when the parser cannot recover from malformed source.

    The V1 parse-failure path returns `parser_outcome="failed"` with
    empty tuples on `ParseResult` — this exception is for unrecoverable
    cases the orchestrator surfaces only at construction-error or
    contract-violation boundaries (e.g., the parser library itself
    raising), not for the routine error-tree case (which tree-sitter
    produces as ERROR/MISSING nodes per the Month 0 spike).
    """


class TraceResolutionError(AstFactsError):
    """Raised when import resolution hits a contract violation.

    Not used for the legitimate `unresolved` / `ambiguous` outcomes
    (those are returned via `ImportResolution.status`); reserved for
    cases like the injected `ImportPathResolver` Protocol violating
    its contract.
    """


class UnknownQueryMatchId(AstFactsError):  # noqa: N818  # name pinned by V1 ast_facts/ spec
    """Raised by `queries/registry.py` when called with an id that is
    not present in the current registry or the deprecation ledger.

    Distinct from "registered query, zero matches" which returns the
    empty tuple from `match(...)`. The two outcomes are caller-
    distinguishable so `audit/replay.py` can tell registry drift apart
    from genuine zero-match results — per the V1 ast_facts/ spec's
    Internal contracts (the Unknown-id behavior bullet).
    """
