# FUP-170: post-cost-gate single-parse bundle for the analyze hot path.
"""One head parse → triviality + parameterized-call facts, for analyze.

Both the trivial-scope side table (`triviality.py`) and the execute-like scan
(`parameterized_calls.py`) parse the same head bytes; on the analyze hot path
that's two redundant head parses (FUP-170). This post-cost-gate bundle parses the
head ONCE and feeds that single tree to both extractors' tree-accepting cores.
The base side (a different file) is parsed separately by the triviality side.

The pre-cost-gate parse (`parse_source` dispatch) deliberately stays SEPARATE: it feeds
`decide_degradation` + the token estimate, both of which run BEFORE the cost gate,
and the trivial-scope spec pins COST_BUDGET_EXHAUSTED-before-classification — so
carrying that tree across the gate is out of scope (see
specs/2026-06-13-analyze-parse-consolidation.md, the "why partial" rationale: a
parse is ms while the LLM call is seconds, so the extra saved parse isn't worth a
settled-spec reconciliation or a firewall handle). This bundle is therefore
called STRICTLY AFTER the cost gate, so a cost-skipped file never reaches it and
the cost-gate-precedence holds by construction.

AST firewall: the raw tree never leaves this module; only the
`FileTrivialityContext` + `ParameterizedCallScan` domain models cross. Composes
the two extractors' private tree-accepting cores (cross-module private import is
the established `ast_facts`/`coordinates` pattern, e.g. `coordinates.translator`
consuming `diff_parser._wrap_github_hunks_with_headers`).
"""

from __future__ import annotations

from typing import Final

import tree_sitter_python
from tree_sitter import Language, Parser

from outrider.ast_facts.parameterized_calls import ParameterizedCallScan, _scan_from_tree
from outrider.ast_facts.triviality import (
    FileTrivialityContext,
    _build_side_table,
    _build_side_table_from_tree,
)

__all__ = ["extract_triviality_and_scan"]

_PY_LANGUAGE: Final = Language(tree_sitter_python.language())
_PARSER: Final = Parser(_PY_LANGUAGE)


def extract_triviality_and_scan(
    head_source: bytes,
    base_source: bytes | None,
    *,
    compute_triviality: bool,
    degraded: bool,
) -> tuple[FileTrivialityContext | None, ParameterizedCallScan | None]:
    """Parse the head ONCE; derive the parameterized scan + (optionally) triviality.

    Mirrors the analyze node's two existing gates so behavior is byte-identical:

    - `degraded=True` → no trustworthy parse tree: returns `(None, None)` with NO
      parse at all (matches the node's `None if degraded_mode` scan gate AND the
      fact that degraded files are never trivial-classified — neither parse ran
      today).
    - clean → parses the head ONCE; the scan always derives from that tree; the
      triviality context derives from the SAME tree (+ a separate base parse)
      only when `compute_triviality` (the node builds it only for files with a
      patch and included scopes). The base side (different bytes) parses
      separately, and only when triviality is wanted.

    Must be called AFTER the cost gate (cost-skipped files never reach it),
    preserving the trivial-scope spec's COST_BUDGET_EXHAUSTED-before-classification
    ordering. Equivalence (the FUP-170 acceptance bar): the returned values match
    `build_triviality_context(head_source, base_source)` (when `compute_triviality`)
    and `scan_parameterized_calls(head_source)` (or `None` when degraded) called
    separately — the only difference is one shared head parse instead of two.
    """
    if degraded:
        return None, None
    head_tree = _PARSER.parse(head_source)
    triviality: FileTrivialityContext | None = None
    if compute_triviality:
        triviality = FileTrivialityContext(
            head=_build_side_table_from_tree(head_tree, head_source),
            base=_build_side_table(base_source) if base_source is not None else None,
        )
    scan = _scan_from_tree(head_tree)
    return triviality, scan
