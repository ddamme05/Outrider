"""Structural eval scenario: degrade-don't-skip on syntax error inside diff.

Per docs/spec.md §11.2 + the `parse-errors-degrade-to-judged` invariant: a file
with a syntax error INSIDE the changed region — where the error breaks a scope's
header so tree-sitter recovers NO scope for it — must DEGRADE to JUDGED-only review,
not be silently skipped. This is the no-scope case of DECISIONS.md#033.

LLM-free, exercising the real surfaces:
- `parse_python` ships `ParseResult.error_lines` — the 1-indexed tree ERROR/MISSING
  lines, scope-INDEPENDENT, so the broken header's line IS flagged even though no
  `ScopeUnit` (and thus no `has_error` entry) was recovered for it.
- `coordinates.diff_line_to_scope` returns None for the diff line (no scope contains
  it) — indistinguishable, at the coordinates layer, from a module-level line.
- `agent/nodes/degradation.decide_degradation` combines them: clean parse, no changed
  scope unit, but an addable changed line intersects `error_lines` → `mode="degraded"`
  with `degradation_reason="tree_has_error_no_scope"`. That is the derivation that was
  inline + LLM-coupled before Phase 2; now pure + importable.
"""

from __future__ import annotations

# Heavy imports (`parse_python` lazy-loads tree_sitter) live in the test body.

# Syntax error INSIDE the changed region: `broken_function`'s header is missing the
# closing paren on line 5, so tree-sitter forms no `function_definition` node there.
SOURCE = """\
def healthy_function():
    return 1


def broken_function(  # diff hunk lives here, line 5 — parser fails on this region
    return None
"""

# Hunks-only patch (GitHub wire shape — coordinates synthesizes the headers) that
# ADDS line 5 (the broken header). Context lines 3-4 (blank) + 6 ("    return None").
PATCH = (
    "@@ -3,3 +3,4 @@\n"
    " \n"
    " \n"
    "+def broken_function(  # diff hunk lives here, line 5 — parser fails on this region\n"
    "     return None\n"
)

DIFF_LINE = 5

EXPECTED_DECISION_MODE = "degraded"
EXPECTED_DEGRADATION_REASON = "tree_has_error_no_scope"


def test_syntax_error_inside_diff_triggers_parse_degraded_fallback() -> None:
    """No-scope syntax error on a changed line → degrade (not skip), JUDGED-only."""
    from unittest.mock import MagicMock

    from outrider.agent.nodes.degradation import decide_degradation
    from outrider.ast_facts import parse_python
    from outrider.coordinates import diff_line_to_scope, lookup_patched_file

    parse_result = parse_python(
        source=SOURCE.encode("utf-8"),
        file_path="test.py",
        resolver=MagicMock(),
    )

    # ast_facts V1 contract: parser_outcome stays "clean" for syntax errors; the
    # broken function yields NO scope, so `has_error` (keyed by scope) can't see it.
    assert parse_result.parser_outcome == "clean"
    scope_names = tuple(s.name for s in parse_result.scope_units)
    assert "healthy_function" in scope_names
    assert "broken_function" not in scope_names

    # ...but `error_lines` (scope-independent) DOES flag the broken header line.
    assert DIFF_LINE in parse_result.error_lines

    # coordinates: line 5 is outside every extracted scope → None (the same return as
    # a module-level line; distinguishing the two is the degradation decision's job).
    assert (
        diff_line_to_scope(
            file_path="test.py", diff_line=DIFF_LINE, scope_units=list(parse_result.scope_units)
        )
        is None
    )

    # The real degradation decision (LLM-free): the changed line adds the broken
    # header, which intersects `error_lines` with no recovered scope → degrade, with
    # the no-scope reason — NOT a silent NO_CHANGED_SCOPE_UNITS skip.
    patched_file = lookup_patched_file(PATCH, "test.py")
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == EXPECTED_DECISION_MODE
    assert decision.degradation_reason == EXPECTED_DEGRADATION_REASON
    # No scope recovered → no scope context; the degraded prompt uses bounded hunks.
    assert decision.included_scope_units == ()
