# Tree-sitter query registry per
# specs/2026-04-30-ast-facts-module.md Internal contracts.
"""Query-id registry and execution surface.

Owns:
  * The `query_match_id` → query-body mapping (file-stem decoupled
    per Internal contracts: renaming a `.scm` file does not churn ids).
  * The compiled `tree_sitter.Query` cache (built at module load).
  * Two public functions:
      - `get_query_source(id) -> str` for documentation / audit-trail use.
      - `match(id, source) -> tuple[QueryMatchSpan, ...]` for replay
        and analyze-node use; returns fully domain-modeled results so
        no `tree_sitter.Query`/`Node`/`QueryCursor` ever leaves
        `queries/` per `docs/trust-boundaries.md` §4 (AST firewall).

Mandatory-capture rejection runs at module-load time per Internal
contracts: a registered pattern with zero `@` captures, or with all
captures quantified as optional (`?`/`*`), has an undefined envelope
and raises `ValueError` at import, not at runtime. The check requires
at least one MANDATORY capture (quantifier `''` or `'+'`) per pattern.

Sort order per Internal contracts:
  * Within a match, captures are flattened sorted by
    `(byte_start, byte_end, name)` ascending.
  * Across matches, the returned tuple is sorted by
    `(byte_start, byte_end)` ascending, with a primitive-projection
    tiebreaker on the captures (Pydantic models lack `__lt__`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor

from outrider.ast_facts.errors import UnknownQueryMatchId
from outrider.ast_facts.models import QueryCaptureSpan, QueryMatchSpan

# ---------------------------------------------------------------------------
# Compiled language and parser (module-level singletons)
# ---------------------------------------------------------------------------

_PY_LANGUAGE: Final = Language(tree_sitter_python.language())
_PARSER: Final = Parser(_PY_LANGUAGE)


# ---------------------------------------------------------------------------
# Id → .scm filename mapping (file-stem decoupled per Internal contracts:
# the id is the authoritative name; filenames are implementation detail).
# ---------------------------------------------------------------------------

_QUERIES_DIR: Final = Path(__file__).parent / "python"

# `capture_quantifier(p, c)` returns the quantifier as a string:
# `''` = mandatory (one), `'+'` = one-or-more (also mandatory),
# `'?'` = zero-or-one, `'*'` = zero-or-more.
_MANDATORY_QUANTIFIERS: Final[frozenset[str]] = frozenset({"", "+"})

_QUERY_ID_TO_FILENAME: Final[dict[str, str]] = {
    "python.function_definition": "function_definition.scm",
    "python.class_definition": "class_definition.scm",
    "python.import_statement": "import_statement.scm",
    "python.import_from_statement": "import_from_statement.scm",
}

# V1: empty. Populated when a query's semantics change and a new id
# alongside the old one is needed for replay of historical reviews
# per Internal contracts.
_DEPRECATED_QUERY_ID_TO_BODY: Final[dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Module-load: read .scm files, compile queries, run captureless-query
# rejection per Internal contracts.
# ---------------------------------------------------------------------------


def _load_and_compile() -> tuple[dict[str, str], dict[str, Query]]:
    bodies: dict[str, str] = {}
    compiled: dict[str, Query] = {}
    for query_id, filename in _QUERY_ID_TO_FILENAME.items():
        body = (_QUERIES_DIR / filename).read_text(encoding="utf-8")
        bodies[query_id] = body
        compiled[query_id] = _compile_and_validate(query_id, body, filename)
    # Deprecated bodies also compile and validate.
    for query_id, body in _DEPRECATED_QUERY_ID_TO_BODY.items():
        bodies[query_id] = body
        compiled[query_id] = _compile_and_validate(query_id, body, source="deprecated_ledger")
    return bodies, compiled


def _compile_and_validate(query_id: str, body: str, source: str | None = None) -> Query:
    """Compile a query body and reject any pattern lacking a mandatory capture.

    Per Internal contracts: every registered pattern MUST produce at
    least one capture per match (envelope rule). A pattern with zero
    captures, or with all captures quantified as optional (`?`/`*`),
    has an undefined envelope and would crash `match(...)` at runtime
    when `min()` sees empty captures.

    Validation walks each pattern via tree-sitter's per-pattern
    introspection (`capture_quantifier(pattern_index, capture_index)`
    raises when the capture isn't part of that pattern). Multi-pattern
    files are permitted — the envelope rule applies per-pattern, not
    per-file. Single-pattern is the V1 convention but not enforced.
    """
    where = f" (loaded from {source})" if source else ""
    query = Query(_PY_LANGUAGE, body)
    # tree-sitter's type stubs declare these as `Callable[[], int]` but
    # at runtime they're int attributes — cast for mypy.
    pattern_count = cast("int", query.pattern_count)
    capture_count = cast("int", query.capture_count)
    if pattern_count < 1:
        raise ValueError(
            f"Query {query_id!r}{where} has pattern_count=0; the body "
            f"must define at least one pattern."
        )
    # Per-pattern check: each pattern must have at least one MANDATORY
    # capture. Optional quantifiers (`'?'`/`'*'`) might fire zero times
    # at runtime, leaving an empty captures tuple, which crashes
    # `QueryMatchSpan`'s envelope `min`/`max` over empty captures.
    # Per Internal contracts' optional-captures residual edge: V1's
    # non-empty-match guarantee depends on mandatory captures. A pattern
    # whose captures are ALL optional fails registration here rather
    # than crashing at first match. `capture_quantifier(p, c)` raises
    # (SystemError or similar) when capture c isn't in pattern p; the
    # broad Exception catch handles tree-sitter binding variation
    # without rejecting otherwise-valid queries.
    for p in range(pattern_count):
        pattern_mandatory_count = 0
        for c in range(capture_count):
            try:
                quantifier = query.capture_quantifier(p, c)
            except Exception:  # noqa: BLE001, S112 - tree-sitter raises various types; the negative case is by-design (capture not in pattern), not an error to log
                continue
            if quantifier in _MANDATORY_QUANTIFIERS:
                pattern_mandatory_count += 1
        if pattern_mandatory_count < 1:
            raise ValueError(
                f"Query {query_id!r}{where} pattern {p} has no "
                f"mandatory captures (all captures are optional/star "
                f"quantified). The envelope rule per Internal contracts "
                f"(specs/2026-04-30-ast-facts-module.md) requires every "
                f"registered pattern to produce at least one capture "
                f"per match; optional-only patterns might fire with "
                f"empty captures at runtime."
            )
    return query


_QUERY_BODIES, _COMPILED_QUERIES = _load_and_compile()


def _all_known_ids() -> set[str]:
    return set(_QUERY_ID_TO_FILENAME) | set(_DEPRECATED_QUERY_ID_TO_BODY)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def get_query_source(query_match_id: str) -> str:
    """Return the raw `.scm` body for a registered or deprecated id.

    Raises `UnknownQueryMatchId` if the id is not in either ledger.
    """
    if query_match_id not in _QUERY_BODIES:
        raise UnknownQueryMatchId(
            f"query_match_id {query_match_id!r} is not in the registry "
            f"(known ids: {sorted(_all_known_ids())})"
        )
    return _QUERY_BODIES[query_match_id]


def match(query_match_id: str, source: bytes) -> tuple[QueryMatchSpan, ...]:
    """Run the named query against `source`; return domain-modeled spans.

    Empty tuple = registered query, zero matches against this source.
    Raises `UnknownQueryMatchId` if `query_match_id` is unknown.
    """
    if query_match_id not in _COMPILED_QUERIES:
        raise UnknownQueryMatchId(
            f"query_match_id {query_match_id!r} is not in the registry "
            f"(known ids: {sorted(_all_known_ids())})"
        )
    query = _COMPILED_QUERIES[query_match_id]
    tree = _PARSER.parse(source)

    raw_matches: list[QueryMatchSpan] = []
    for _pattern_index, captures in QueryCursor(query).matches(tree.root_node):
        # captures: dict[str, list[Node]] per Month 0 spike findings
        # (canonical docs say bare Node; runtime returns list[Node]).
        flat: list[QueryCaptureSpan] = []
        for capture_name, nodes in captures.items():
            for node in nodes:
                flat.append(
                    QueryCaptureSpan(
                        name=capture_name,
                        byte_start=node.start_byte,
                        byte_end=node.end_byte,
                    )
                )
        # Per Internal contracts: sort captures by (byte_start, byte_end, name).
        flat.sort(key=lambda c: (c.byte_start, c.byte_end, c.name))
        capture_tuple = tuple(flat)
        # Envelope per Internal contracts.
        envelope_start = min(c.byte_start for c in capture_tuple)
        envelope_end = max(c.byte_end for c in capture_tuple)
        raw_matches.append(
            QueryMatchSpan(
                byte_start=envelope_start,
                byte_end=envelope_end,
                captures=capture_tuple,
            )
        )

    # Sort matches by (byte_start, byte_end) with captures-projection tiebreaker
    # per Internal contracts (Pydantic models lack a default `__lt__`).
    def _sort_key(m: QueryMatchSpan) -> tuple[int, int, tuple[tuple[int, int, str], ...]]:
        cap_proj = tuple((c.byte_start, c.byte_end, c.name) for c in m.captures)
        return (m.byte_start, m.byte_end, cap_proj)

    raw_matches.sort(key=_sort_key)
    return tuple(raw_matches)
