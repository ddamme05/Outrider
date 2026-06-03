"""Structural eval scenario: graceful degradation on syntax error outside diff.

Per docs/spec.md §11.2 + the `parse-errors-degrade-to-judged` invariant: a file
with a syntax error OUTSIDE the changed region degrades gracefully — the analyze
node's degradation decision is `clean` (NOT degraded) because the CHANGED scope
unit is parseable even though the file as a whole has an error elsewhere.

LLM-free: `ast_facts` ships the parse result (with whatever `scope_units`
tree-sitter recovers from the parseable region); `coordinates` ships
`diff_line_to_scope`; and the degradation decision that combines a `ParseResult`
+ the patch into a `clean`/`degraded`/`skip` outcome is the pure
`agent/nodes/degradation.decide_degradation` (extracted from the analyze node so a
structural scenario can call it without the LLM). This scenario exercises the real
decision — the changed scope has added lines and no tree error → `mode="clean"`.
"""

from __future__ import annotations

# `parse_python` / `diff_line_to_scope` / `decide_degradation` imports live in the
# test body — `outrider.ast_facts.__getattr__` lazy-loads tree_sitter on first
# attribute access, and a top-level import would trigger that load during pytest
# collection. Keeping heavy imports in the body preserves the import-light contract.

# The syntax error is at the bottom of the file (unmatched paren); the diff hunk
# lies in the parseable region above (line 2, inside `changed_function`).
SOURCE = """\
def changed_function():
    return 42  # diff hunk lives here, line 2


def broken_function(  # syntax error: unmatched paren, line 5
    return None
"""

# Hunks-only patch (GitHub `/pulls/{n}/files` wire shape — `coordinates`
# synthesizes the `--- a/` / `+++ b/` headers). Changes line 2 inside
# `changed_function`, so that scope unit has an added line.
PATCH = (
    "@@ -1,2 +1,2 @@\n"
    " def changed_function():\n"
    "-    return 0\n"
    "+    return 42  # diff hunk lives here, line 2\n"
)

DIFF_LINE = 2

EXPECTED_RESOLVED_NAME = "changed_function"
EXPECTED_DECISION_MODE = "clean"  # graceful degradation, NOT full degraded mode


def test_syntax_error_outside_diff_degrades_gracefully() -> None:
    """Diff line in the parseable region → ScopeUnit resolves, and the analyze
    degradation decision is `clean` because the CHANGED scope has no tree error
    (the syntax error lives in a different, unchanged scope).
    """
    from unittest.mock import MagicMock

    from outrider.agent.nodes.degradation import decide_degradation
    from outrider.ast_facts import parse_python
    from outrider.coordinates import diff_line_to_scope, lookup_patched_file

    parse_result = parse_python(
        source=SOURCE.encode("utf-8"),
        file_path="test.py",
        resolver=MagicMock(),
    )

    # ast_facts contract: tree-sitter recovers the parseable region's scopes;
    # the diff line maps to a real ScopeUnit via diff_line_to_scope.
    scope = diff_line_to_scope(
        file_path="test.py",
        diff_line=DIFF_LINE,
        scope_units=list(parse_result.scope_units),
    )
    assert scope is not None
    assert scope.name == EXPECTED_RESOLVED_NAME

    # The real analyze degradation decision (LLM-free): the changed scope unit
    # carries no tree error, so the file is reviewed cleanly, NOT degraded —
    # even though the whole-file parse has an error in `broken_function`.
    patched_file = lookup_patched_file(PATCH, "test.py")
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == EXPECTED_DECISION_MODE
    assert decision.degradation_reason is None
    # The changed scope is the one carried for the (clean) review prompt.
    assert any(su.name == EXPECTED_RESOLVED_NAME for su in decision.included_scope_units)
