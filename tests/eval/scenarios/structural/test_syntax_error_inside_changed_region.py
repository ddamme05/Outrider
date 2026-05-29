"""Structural eval scenario: parse-degraded fallback on syntax error inside diff.

Per docs/spec.md §11.2 + the `parse-errors-degrade-to-judged` invariant:
a file with a syntax error INSIDE the changed region should eventually
trigger a parse-degraded fallback. Findings produced under degraded mode
get downgraded to JUDGED tier per the invariant.

V1 ast_facts contract (verified empirically):
- `parser_outcome="clean"` even when source has syntax errors. The
  "failed" outcome is reserved for UTF-8 decode failures and the
  defensive `compute_parser_outcome → "failed"` branch (unreachable in
  V1 practice — see `python_adapter.compute_parser_outcome` policy).
- `scope_units` contains whatever scopes tree-sitter could extract.
  An unrecoverable region (broken function header where the parser
  can't form a `function_definition` node at all) yields NO ScopeUnit
  for that region — the error is invisible to scope-level reasoning.
- Per-scope `has_error` flags indicate error nodes WITHIN recovered
  scopes, but contain no entry for regions that didn't yield a scope.

Coordinates contract: `diff_line_to_scope(diff_line, scope_units)`
returns None for any line outside the extracted scope set — including
lines in unparseable code where no scope was extracted at all. The
None return is INDISTINGUISHABLE from "diff line is at module level"
in a clean file. Distinguishing the two is the analyze-node's job.

Still skipped post-coordinates because the degraded-marker derivation
isn't a coordinates concern. The eventual analyze-node logic must
combine: (1) `diff_line_to_scope == None` AND (2) a separate
"tree-level error overlaps the diff line" signal — the latter is NOT
in coordinates and NOT in ast_facts' current public surface, and would
need a new ast_facts method (e.g., `tree_has_error_at_line`) before
the degraded derivation can be implemented. This scenario flips when
both that ast_facts extension AND an LLM-free importable degraded-
derivation helper land (the analyze node shipped, but its degraded
logic is inline + LLM-coupled).
"""

from __future__ import annotations

import pytest

# `parse_python` and `diff_line_to_scope` imports moved into the test body
# below — `outrider.ast_facts.__getattr__` lazy-loads tree_sitter on first
# attribute access, and a top-level import here triggers that load during
# pytest collection even though the module is skipped. Keeping heavy
# imports behind the skipmark preserves the import-light contract for
# skipped scenarios.

pytestmark = pytest.mark.skip(
    reason="requires (1) the analyze-node degraded derivation as an LLM-free "
    "importable helper AND (2) a public ast_facts tree-level error surface (e.g. "
    "tree_has_error_at_line) for no-scope syntax errors — per-scope has_error "
    "doesn't cover regions that yield no scope"
)

# Syntax error is inside the changed region; the diff line itself lies
# in unparseable code (broken_function's header is missing the closing
# paren on line 5; tree-sitter can't extract it as a function_definition).
SOURCE = """\
def healthy_function():
    return 1


def broken_function(  # diff hunk lives here, line 5 — parser fails on this region
    return None
"""

DIFF_LINE = 5

EXPECTED_DEGRADED_MARKER = True


def test_syntax_error_inside_diff_triggers_parse_degraded_fallback() -> None:
    """V1 ast_facts + coordinates behavior on a partial-parse file with the
    diff line inside the unparseable region.

    Verifies the actual current contract (live assertions):
    - parse_python returns parser_outcome="clean" (V1 policy)
    - scope_units contains the recoverable healthy_function only
    - diff_line_to_scope returns None for the diff line (which lies in
      unparseable code, outside every extracted scope)

    Pending (commented): the analyze-node-derived degraded=True marker.
    Deriving it requires a tree-level error-overlap signal not yet in
    ast_facts' public surface — see module docstring.
    """
    from unittest.mock import MagicMock

    from outrider.ast_facts import parse_python
    from outrider.coordinates import diff_line_to_scope

    parse_result = parse_python(
        source=SOURCE.encode("utf-8"),
        file_path="test.py",
        resolver=MagicMock(),
    )

    # ast_facts V1 contract: parser_outcome stays "clean" for syntax errors.
    # "failed" is reserved for UTF-8 decode failures.
    assert parse_result.parser_outcome == "clean"

    # Tree-sitter recovers the parseable scope; the broken function header
    # can't be formed as a function_definition node, so no scope is
    # extracted for it. Scope-level reasoning loses the broken region.
    scope_names = tuple(s.name for s in parse_result.scope_units)
    assert "healthy_function" in scope_names
    assert "broken_function" not in scope_names

    # coordinates contract: line 5 is outside every extracted scope (the
    # broken function isn't there to contain it), so diff_line_to_scope
    # returns None. This is the SAME return as "module-level diff line in
    # a clean file" — coordinates can't distinguish the two cases, and
    # shouldn't try to. The distinction is the analyze-node's responsibility.
    scope = diff_line_to_scope(
        file_path="test.py",
        diff_line=DIFF_LINE,
        scope_units=list(parse_result.scope_units),
    )
    assert scope is None

    # Pending analyze-node + ast_facts tree-level error signal: the
    # degraded=True marker requires combining the None return above with a
    # "tree-error overlaps diff line" signal that doesn't exist in V1
    # ast_facts. Lands when both extensions ship.
    # assert analyze_node_derive_degraded(parse_result, DIFF_LINE) is EXPECTED_DEGRADED_MARKER
