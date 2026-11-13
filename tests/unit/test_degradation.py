"""Check analysis decisions for clean, degraded, and unusable parses.

Construct parse results without a model or database; real-source parsing is
covered separately. Cover caller skip checks and files without usable scopes."""

from __future__ import annotations

from difflib import unified_diff

import pytest

from outrider.agent.nodes.degradation import DegradationDecision, decide_degradation
from outrider.ast_facts.models import ParseResult, SkipReason
from outrider.ast_facts.registry import parse_source
from outrider.coordinates import (
    COORDINATES_IMPORT_PATH_RESOLVER,
    changed_line_spans,
    extract_line_range_body,
    line_range_diff_hunks,
    lookup_patched_file,
    unscoped_changed_line_ranges,
)

_GUARD_BASE = (
    "def read_secret(user):\n"
    "    if not user.is_admin:\n"
    "        raise PermissionError()\n"
    '    return "秘密"\n'
)
_GUARD_HEAD = 'def read_secret(user):\n    return "秘密"\n'


@pytest.mark.parametrize(
    ("base", "head", "expected_scopes", "context_lines"),
    [
        pytest.param(_GUARD_BASE, _GUARD_HEAD, (("read_secret", 1),), 3, id="guard"),
        pytest.param(_GUARD_BASE, _GUARD_HEAD, (("read_secret", 1),), 0, id="guard-no-context"),
        pytest.param(
            '@require_auth\ndef read_secret(user):\n    return "秘密"\n',
            _GUARD_HEAD,
            (("read_secret", 1),),
            0,
            id="decorator",
        ),
        pytest.param(
            'def removed():\n    return "secret"\n\ndef neighbor():\n    return 1\n',
            "def neighbor():\n    return 1\n",
            (),
            0,
            id="deleted-function-does-not-select-neighbor",
        ),
        pytest.param(
            "def read_secret(user):\n    return 1\n\n" + _GUARD_BASE,
            "def read_secret(user):\n    return 1\n\n" + _GUARD_HEAD,
            (("read_secret", 4),),
            0,
            id="repeated-function-name",
        ),
        pytest.param(
            "def first():\n    guard()\n    return 1\n\ndef second():\n    return 1\n",
            "def first():\n    return 1\n\ndef second():\n    return 2\n",
            (("first", 1), ("second", 4)),
            3,
            id="deletion-and-addition-in-different-functions",
        ),
        pytest.param(
            "class Store:\n    def read(self):\n        guard()\n        return 1\n",
            "class Store:\n    def read(self):\n        return 1\n",
            (("Store", 1), ("Store.read", 2)),
            0,
            id="method-and-containing-class",
        ),
        pytest.param(
            'def read_secret():\n    audit()\n    return "secret"\n',
            "def read_secret():\n    audit()\n",
            (("read_secret", 1),),
            0,
            id="last-statement",
        ),
    ],
)
def test_deleted_changes_select_the_surviving_scope(
    base: str, head: str, expected_scopes: tuple[tuple[str, int], ...], context_lines: int
) -> None:
    path = "x.py"
    resolver = COORDINATES_IMPORT_PATH_RESOLVER
    base_parse = parse_source(base.encode(), path, resolver)
    head_parse = parse_source(head.encode(), path, resolver)
    patch = "".join(
        unified_diff(
            base.splitlines(keepends=True),
            head.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=context_lines,
        )
    )
    patched = lookup_patched_file(patch, path)
    assert patched is not None
    decision = decide_degradation(head_parse, patched, base_scope_units=base_parse.scope_units)
    assert (
        tuple((scope.qualified_name, scope.line_start) for scope in decision.included_scope_units)
        == expected_scopes
    )
    if expected_scopes:
        assert decision.mode == "clean"
    for scope, hunks in zip(
        decision.included_scope_units, decision.included_clipped_hunks, strict=True
    ):
        changed = changed_line_spans(
            scope,
            patched,
            head_source=head,
            base_source=base,
            base_scope_units=base_parse.scope_units,
        )
        rendered_removals = [
            line[1:] for hunk in hunks for line in hunk.splitlines()[1:] if line.startswith("-")
        ]
        assert rendered_removals == [
            base.splitlines()[entry.line_no - 1] for entry in changed.base_removed
        ]
        assert hunks


@pytest.mark.parametrize(
    ("base", "head", "has_base_scopes", "expected_ranges"),
    [
        pytest.param(
            "ALLOW_ADMIN = False\n\ndef read():\n    return 1\n",
            "ALLOW_ADMIN = True\n\ndef read():\n    return 2\n",
            True,
            ((1, 3),),
            id="setting-alongside-changed-function",
        ),
        pytest.param(_GUARD_BASE, _GUARD_HEAD, True, (), id="owned-guard-removal"),
        pytest.param(_GUARD_BASE, _GUARD_HEAD, False, (), id="known-head-interior-without-base"),
        pytest.param(
            "def read():\n    guard()\n\ndef read():\n    return 1\n",
            "def read():\n    return 1\n",
            True,
            ((1, 2),),
            id="deleted-definition-with-repeated-name",
        ),
        pytest.param("VALUE = 1\nOLD = 2\n", "VALUE = 1\n", True, ((1, 1),), id="eof"),
        pytest.param("VALUE = 1\n", "", True, (), id="empty-head"),
        pytest.param(
            "A = 0\nB = 0\nC = 0\nD = 0\n",
            "A = 1\nB = 0\nC = 0\nD = 1\n",
            True,
            ((1, 4),),
            id="merge-nearby-module-changes",
        ),
    ],
)
def test_primary_regions_cover_changes_outside_selected_scopes(
    base: str,
    head: str,
    has_base_scopes: bool,
    expected_ranges: tuple[tuple[int, int], ...],
) -> None:
    path = "x.py"
    resolver = COORDINATES_IMPORT_PATH_RESOLVER
    base_parse = parse_source(base.encode(), path, resolver)
    head_parse = parse_source(head.encode(), path, resolver)
    patch = "".join(
        unified_diff(
            base.splitlines(keepends=True),
            head.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=0,
        )
    )
    patched = lookup_patched_file(patch, path)
    assert patched is not None
    ranges = unscoped_changed_line_ranges(
        patched,
        head_parse.scope_units,
        base_scope_units=base_parse.scope_units if has_base_scopes else (),
        head_line_count=len(head.splitlines()),
    )
    assert ranges == expected_ranges
    for start, end in ranges:
        assert extract_line_range_body(start, end, head) == "".join(
            head.splitlines(keepends=True)[start - 1 : end]
        )
        hunks = line_range_diff_hunks(patched, start, end, head_line_count=len(head.splitlines()))
        assert any(line.startswith("-") for hunk in hunks for line in hunk.splitlines()[1:])
        assert lookup_patched_file("".join(hunks), path) is not None


# ---------------------------------------------------------------------------
# DegradationDecision typed-contract guard (__post_init__)
# ---------------------------------------------------------------------------


def test_decision_skip_requires_skip_reason() -> None:
    with pytest.raises(ValueError, match="mode='skip' requires"):
        DegradationDecision(mode="skip", parse_status="clean")


def test_decision_degraded_requires_degradation_reason() -> None:
    with pytest.raises(ValueError, match="mode='degraded' requires"):
        DegradationDecision(mode="degraded", parse_status="degraded")


def test_decision_clean_must_not_carry_skip_reason() -> None:
    with pytest.raises(ValueError, match="must not carry skip_reason"):
        DegradationDecision(mode="clean", parse_status="clean", skip_reason=SkipReason.VENDORED)


def test_decision_clean_must_not_carry_degradation_reason() -> None:
    with pytest.raises(ValueError, match="must not carry degradation_reason"):
        DegradationDecision(mode="clean", parse_status="clean", degradation_reason="parse_failed")


def test_decision_valid_skip_constructs() -> None:
    d = DegradationDecision(
        mode="skip", parse_status="clean", skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS
    )
    assert d.mode == "skip"
    assert d.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS
    assert d.included_scope_units == ()


# ---------------------------------------------------------------------------
# decide_degradation
# ---------------------------------------------------------------------------


def test_decide_degradation_raises_on_parser_skipped() -> None:
    # Parser-stage skips are the node's responsibility (handled BEFORE this is
    # called, because lookup_patched_file can raise on a skipped file's patch).
    # Passing one here is a caller-contract violation.
    skipped = ParseResult(parser_outcome="skipped", skip_reason=SkipReason.VENDORED)
    with pytest.raises(RuntimeError, match="parser-skipped result"):
        decide_degradation(skipped, None)


def test_decide_degradation_clean_no_patch_skips_no_changed_scope_units() -> None:
    decision = decide_degradation(ParseResult(parser_outcome="clean"), None)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS
    assert decision.parse_status == "clean"


def test_decide_degradation_clean_no_scope_units_skips_no_changed_scope_units() -> None:
    # patched_file present but the parse has no scope units → empty intersection
    # → NO_CHANGED_SCOPE_UNITS (exercises _intersect_changed_scope_units's empty path).
    patched_file = lookup_patched_file("@@ -1 +1 @@\n-a\n+b\n", "x.py")
    assert patched_file is not None
    decision = decide_degradation(ParseResult(parser_outcome="clean"), patched_file)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS


@pytest.mark.parametrize(
    ("error_lines", "observed_candidate", "expected_mode", "expected_reason"),
    [
        (frozenset(), False, "clean", None),
        (frozenset({1}), False, "degraded", "tree_has_error_no_scope"),
        (frozenset({5}), False, "skip", None),
        (frozenset(), True, "degraded", "module_level_observed_match"),
    ],
)
def test_primary_changed_regions_preserve_parse_and_observed_precedence(
    error_lines: frozenset[int],
    observed_candidate: bool,
    expected_mode: str,
    expected_reason: str | None,
) -> None:
    patched = lookup_patched_file("@@ -1 +1 @@\n-ALLOW = False\n+ALLOW = True\n", "x.py")
    assert patched is not None
    decision = decide_degradation(
        ParseResult(parser_outcome="clean", error_lines=error_lines),
        patched,
        module_level_observed_candidate=observed_candidate,
        primary_line_ranges=((1, 1),),
    )
    assert decision.mode == expected_mode
    assert decision.degradation_reason == expected_reason
    assert decision.included_primary_line_ranges == (((1, 1),) if expected_mode == "clean" else ())


# The "failed" parser outcome is V1-unreachable (intake gates invalid UTF-8) but
# decide_degradation handles it for the raw-bytes intake path (FUP-053); pin both
# branches directly since the node-level tests can't reach them in V1.


def test_decide_degradation_failed_no_patch_skips_no_reviewable_context() -> None:
    decision = decide_degradation(ParseResult(parser_outcome="failed"), None)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_REVIEWABLE_CONTEXT
    assert decision.parse_status == "failed"


def test_decide_degradation_failed_with_added_lines_degrades_parse_failed() -> None:
    # A failed parse WITH addable text → degraded(parse_failed), no scope context
    # (the failed path never intersects scopes).
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.py")
    assert patched_file is not None
    decision = decide_degradation(ParseResult(parser_outcome="failed"), patched_file)
    assert decision.mode == "degraded"
    assert decision.degradation_reason == "parse_failed"
    assert decision.parse_status == "failed"
    assert decision.included_scope_units == ()
    assert decision.included_clipped_hunks == ()


# ---------------------------------------------------------------------------
# decide_degradation — no-scope syntax-error degrade-don't-skip (DECISIONS#033)
# ---------------------------------------------------------------------------


def test_decide_degradation_no_scope_error_line_degrades() -> None:
    # clean parse, NO changed scope unit, but an ADDED line (2) intersects a tree
    # error line (error_lines={2}) → degrade with tree_has_error_no_scope, not skip.
    parse_result = ParseResult(parser_outcome="clean", error_lines=frozenset({2}))
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.py")  # adds line 2
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == "degraded"
    assert decision.degradation_reason == "tree_has_error_no_scope"
    assert decision.parse_status == "degraded"
    # No scope recovered → no scope context; the degraded prompt uses bounded hunks.
    assert decision.included_scope_units == ()


def test_decide_degradation_module_candidate_degrades_with_clean_parse_status() -> None:
    """Module-scope routing (DECISIONS.md#062):
    a module-only diff (clean parse, no changed scope units) with the
    precomputed eligible-match flag degrades as `module_level_observed_match`
    — and `parse_status` stays truthfully `clean` (a routing choice, not a
    parse defect)."""
    parse_result = ParseResult(parser_outcome="clean")
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.js")
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file, module_level_observed_candidate=True)
    assert decision.mode == "degraded"
    assert decision.degradation_reason == "module_level_observed_match"
    assert decision.parse_status == "clean"
    assert decision.included_scope_units == ()


def test_decide_degradation_without_module_candidate_skip_unchanged() -> None:
    """Revert-the-fold control: the flag defaulting False keeps today's
    NO_CHANGED_SCOPE_UNITS skip for module-only diffs with no eligible match."""
    parse_result = ParseResult(parser_outcome="clean")
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.js")
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS


def test_decide_degradation_parse_error_precedence_over_module_candidate() -> None:
    """Parse-error precedence: a syntax-error file (error_lines intersecting
    the added line) degrades as `tree_has_error_no_scope` even when the module
    candidate flag is set — the error branch runs first, so OBSERVED stays off
    error-recovered trees (revert-the-precedence: evaluating the module branch
    first fails this pin)."""
    parse_result = ParseResult(parser_outcome="clean", error_lines=frozenset({2}))
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.js")
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file, module_level_observed_candidate=True)
    assert decision.mode == "degraded"
    assert decision.degradation_reason == "tree_has_error_no_scope"
    assert decision.parse_status == "degraded"


def test_decide_degradation_no_scope_deletion_only_stays_skip() -> None:
    # An error line exists but the change is a pure DELETION (no added target line)
    # → no addable line intersects error_lines → still NO_CHANGED_SCOPE_UNITS skip
    # (the addable-lines-only non-goal: we do not solve deletion-only changes).
    parse_result = ParseResult(parser_outcome="clean", error_lines=frozenset({2}))
    patched_file = lookup_patched_file("@@ -1,2 +1,1 @@\n a\n-b\n", "x.py")  # deletes line 2
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS


def test_decide_degradation_no_scope_added_line_not_an_error_line_stays_skip() -> None:
    # The added line (2) does NOT intersect error_lines (error on line 5) → skip.
    parse_result = ParseResult(parser_outcome="clean", error_lines=frozenset({5}))
    patched_file = lookup_patched_file("@@ -1 +1,2 @@\n a\n+b\n", "x.py")  # adds line 2 only
    assert patched_file is not None
    decision = decide_degradation(parse_result, patched_file)
    assert decision.mode == "skip"
    assert decision.skip_reason == SkipReason.NO_CHANGED_SCOPE_UNITS
