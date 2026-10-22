# FUP-170: post-cost-gate single-parse bundle for the analyze hot path.
"""Share one post-cost-gate head parse between triviality and SQL-call facts.

The base source is parsed separately only when triviality needs it. The earlier
parse_source call still supplies degradation and token estimates before the cost
gate; this bundle must run afterward to preserve COST_BUDGET_EXHAUSTED precedence
(specs/2026-06-13-analyze-parse-consolidation.md, FUP-170).

Raw trees stay inside ast_facts; callers receive FileTrivialityContext and
ParameterizedCallScan domain models.
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
    """Return the SQL-call scan and optional triviality context after the cost gate.

    Degraded input returns (None, None) without parsing. Otherwise one head tree
    supplies both extractors; compute_triviality also enables the separate base
    parse when base_source is present. Results match the individual extractors.
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
