"""Unit tests for `coordinates.diff_line_to_scope`.

Per docs/spec.md §5.6 — covers the six edge cases enumerated in the Month 0
spike `spikes/tree_sitter/demos/demo_q6_diff_line_to_scope.py` (decorator
line, first/last line of nested function, class-body line between methods,
module-level comment, line outside file range), plus the multi-file
scope-list disambiguation case and the repeated-call idempotence property.

See DECISIONS.md#006-two-month-0-spikes-not-five for the test discipline.
"""

from __future__ import annotations

from typing import Literal

import pytest

from outrider.ast_facts.base import ImportPathResolver
from outrider.ast_facts.models import ScopeUnit
from outrider.coordinates import (
    COORDINATES_IMPORT_PATH_RESOLVER,
    CoordinateError,
    diff_line_to_scope,
    lookup_patched_file,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_scope(
    *,
    unit_id: str,
    kind: Literal["function", "method", "class"],
    name: str,
    file_path: str,
    line_start: int,
    line_end: int,
    parent_scope_id: str | None = None,
) -> ScopeUnit:
    """Construct a ScopeUnit for line-range testing.

    `byte_start`/`byte_end` are not consulted by `diff_line_to_scope`; set to
    placeholder values that satisfy the model's constraints.
    """
    return ScopeUnit(
        unit_id=unit_id,
        kind=kind,
        name=name,
        qualified_name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        byte_start=0,
        byte_end=1,
        parent_scope_id=parent_scope_id,
    )


# ----------------------------------------------------------------------------
# Spike edge case 1: decorator line above a method → method's enclosing scope
# ----------------------------------------------------------------------------


def test_decorator_line_above_method_resolves_to_method() -> None:
    """Per Q2 spike: tree-sitter's `decorated_definition` node spans from the
    decorator through the function body. A diff line on the decorator should
    resolve to the method (the innermost decorated scope), not the class.

    Fixture:
        class MyClass:        # line 1
            @decorator        # line 2 ← diff line
            def my_method(self):  # line 3
                return 1      # line 4
    """
    scope_units = [
        _make_scope(
            unit_id="cls",
            kind="class",
            name="MyClass",
            file_path="x.py",
            line_start=1,
            line_end=4,
        ),
        _make_scope(
            unit_id="m",
            kind="method",
            name="my_method",
            file_path="x.py",
            line_start=2,
            line_end=4,
            parent_scope_id="cls",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=2, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "m"


# ----------------------------------------------------------------------------
# Spike edge case 2: first line of function body → that function
# ----------------------------------------------------------------------------


def test_first_line_of_function_body_resolves_to_function() -> None:
    """Diff line on the first body line of a function → that function.

    Fixture:
        def func():        # line 1
            x = 42         # line 2 ← diff line (first body line)
    """
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=1,
            line_end=2,
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=2, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "f"


# ----------------------------------------------------------------------------
# Spike edge case 3: last line of nested function → nested function (innermost)
# ----------------------------------------------------------------------------


def test_last_line_of_nested_function_resolves_to_nested_not_outer() -> None:
    """Innermost-scope rule: a line shared by outer + nested scopes resolves
    to the nested (smallest line span wins).

    Fixture:
        def outer_function():            # line 1
            def nested_helper():         # line 2
                return 42                # line 3 ← diff line (last line of nested)
            return nested_helper()       # line 4
    """
    scope_units = [
        _make_scope(
            unit_id="outer",
            kind="function",
            name="outer_function",
            file_path="x.py",
            line_start=1,
            line_end=4,
        ),
        _make_scope(
            unit_id="nested",
            kind="function",
            name="nested_helper",
            file_path="x.py",
            line_start=2,
            line_end=3,
            parent_scope_id="outer",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=3, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "nested"


# ----------------------------------------------------------------------------
# Spike edge case 4: class-body line between methods → the class
# ----------------------------------------------------------------------------


def test_class_body_line_between_methods_resolves_to_class() -> None:
    """A diff line in class-level code between two method bodies → the class
    scope (only the class contains it; methods don't).

    Fixture:
        class MyClass:           # line 1
            def method_a(self):  # line 2
                return 1         # line 3
                                 # line 4 ← diff line (between methods)
            def method_b(self):  # line 5
                return 2         # line 6
    """
    scope_units = [
        _make_scope(
            unit_id="cls",
            kind="class",
            name="MyClass",
            file_path="x.py",
            line_start=1,
            line_end=6,
        ),
        _make_scope(
            unit_id="ma",
            kind="method",
            name="method_a",
            file_path="x.py",
            line_start=2,
            line_end=3,
            parent_scope_id="cls",
        ),
        _make_scope(
            unit_id="mb",
            kind="method",
            name="method_b",
            file_path="x.py",
            line_start=5,
            line_end=6,
            parent_scope_id="cls",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=4, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "cls"


# ----------------------------------------------------------------------------
# Spike edge case 5: module-level comment line → None
# ----------------------------------------------------------------------------


def test_module_level_comment_line_returns_none() -> None:
    """A diff line outside any function/method/class scope → None.

    Fixture:
        # A module-level comment        # line 1 ← diff line
        def func():                     # line 2
            return 1                    # line 3
    """
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=2,
            line_end=3,
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=1, scope_units=scope_units)
    assert result is None


# ----------------------------------------------------------------------------
# Spike edge case 6: line outside the file's line range → None (deterministic)
# ----------------------------------------------------------------------------


def test_line_far_past_file_end_returns_none_deterministically() -> None:
    """A diff line outside the file's line range → None (no scope matches);
    must be deterministic, not raise.
    """
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=1,
            line_end=10,
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=999, scope_units=scope_units)
    assert result is None


def test_line_zero_raises_coordinate_error() -> None:
    """`diff_line=0` raises CoordinateError — surfaces caller kind-confusion
    (e.g., a 0-indexed tree-sitter row passed without conversion) instead of
    silently returning None (which would conflate the kind error with the
    legitimate "no enclosing scope" answer).
    """
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=1,
            line_end=10,
        ),
    ]
    with pytest.raises(CoordinateError, match="not a valid 1-indexed source line"):
        diff_line_to_scope(file_path="x.py", diff_line=0, scope_units=scope_units)


def test_negative_diff_line_raises_coordinate_error() -> None:
    """Negative `diff_line` raises CoordinateError — same kind-confusion guard."""
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=1,
            line_end=10,
        ),
    ]
    with pytest.raises(CoordinateError, match="not a valid 1-indexed source line"):
        diff_line_to_scope(file_path="x.py", diff_line=-1, scope_units=scope_units)


# ----------------------------------------------------------------------------
# Multi-file scope list disambiguation (per spec line 84-85)
# ----------------------------------------------------------------------------


def test_multifile_scope_list_filters_by_file_path() -> None:
    """A scope_units list spanning multiple files: only scopes where
    ScopeUnit.file_path matches the file_path argument are eligible.

    Fixture: two scopes in two files, both covering line 5. file_path="a.py"
    should match a.py's scope, NOT b.py's, even though both contain line 5.
    """
    scope_units = [
        _make_scope(
            unit_id="a_func",
            kind="function",
            name="a_func",
            file_path="a.py",
            line_start=1,
            line_end=10,
        ),
        _make_scope(
            unit_id="b_func",
            kind="function",
            name="b_func",
            file_path="b.py",
            line_start=1,
            line_end=10,
        ),
    ]
    result = diff_line_to_scope(file_path="a.py", diff_line=5, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "a_func"
    assert result.file_path == "a.py"


def test_multifile_scope_list_returns_none_when_only_other_file_has_match() -> None:
    """If the file_path argument has no matching scopes but another file does,
    return None — never accidentally fall through to the other file's scope.
    """
    scope_units = [
        _make_scope(
            unit_id="b_func",
            kind="function",
            name="b_func",
            file_path="b.py",
            line_start=1,
            line_end=10,
        ),
    ]
    result = diff_line_to_scope(file_path="a.py", diff_line=5, scope_units=scope_units)
    assert result is None


# ----------------------------------------------------------------------------
# Idempotence (per spec test scenarios)
# ----------------------------------------------------------------------------


def test_repeated_calls_are_idempotent() -> None:
    """Same inputs → same unit_id returned across calls (or both None)."""
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=1,
            line_end=5,
        ),
    ]
    a = diff_line_to_scope(file_path="x.py", diff_line=3, scope_units=scope_units)
    b = diff_line_to_scope(file_path="x.py", diff_line=3, scope_units=scope_units)
    assert a is not None and b is not None
    assert a.unit_id == b.unit_id


def test_repeated_calls_idempotent_for_none_case() -> None:
    """Same inputs that miss → both calls return None."""
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=10,
            line_end=20,
        ),
    ]
    a = diff_line_to_scope(file_path="x.py", diff_line=5, scope_units=scope_units)
    b = diff_line_to_scope(file_path="x.py", diff_line=5, scope_units=scope_units)
    assert a is None
    assert b is None


# ----------------------------------------------------------------------------
# Empty inputs
# ----------------------------------------------------------------------------


def test_empty_scope_units_returns_none() -> None:
    """Empty `scope_units` list → None deterministically."""
    result = diff_line_to_scope(file_path="x.py", diff_line=5, scope_units=[])
    assert result is None


# ----------------------------------------------------------------------------
# Innermost-scope tiebreaker stress: deeply nested scopes
# ----------------------------------------------------------------------------


def test_three_level_nesting_returns_innermost() -> None:
    """class > method > nested_inner_function: innermost wins.

    Fixture:
        class C:                          # line 1
            def m(self):                  # line 2
                def inner():              # line 3
                    return 1              # line 4 ← diff line
                return inner()            # line 5
    """
    scope_units = [
        _make_scope(
            unit_id="c",
            kind="class",
            name="C",
            file_path="x.py",
            line_start=1,
            line_end=5,
        ),
        _make_scope(
            unit_id="m",
            kind="method",
            name="m",
            file_path="x.py",
            line_start=2,
            line_end=5,
            parent_scope_id="c",
        ),
        _make_scope(
            unit_id="inner",
            kind="function",
            name="inner",
            file_path="x.py",
            line_start=3,
            line_end=4,
            parent_scope_id="m",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=4, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "inner"


def test_innermost_when_scope_order_in_list_is_outer_first() -> None:
    """Result independent of scope_units list order: outer-first input still
    returns innermost.
    """
    scope_units = [
        _make_scope(
            unit_id="outer",
            kind="function",
            name="outer",
            file_path="x.py",
            line_start=1,
            line_end=5,
        ),
        _make_scope(
            unit_id="inner",
            kind="function",
            name="inner",
            file_path="x.py",
            line_start=2,
            line_end=3,
            parent_scope_id="outer",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=2, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "inner"


def test_innermost_when_scope_order_in_list_is_inner_first() -> None:
    """Result independent of scope_units list order: inner-first input also
    returns innermost. Catches an implementation that returns the first match.
    """
    scope_units = [
        _make_scope(
            unit_id="inner",
            kind="function",
            name="inner",
            file_path="x.py",
            line_start=2,
            line_end=3,
            parent_scope_id="outer",
        ),
        _make_scope(
            unit_id="outer",
            kind="function",
            name="outer",
            file_path="x.py",
            line_start=1,
            line_end=5,
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=2, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "inner"


# ---------------------------------------------------------------------------
# lookup_patched_file: three None branches + happy path + duplicate-entry raise
# ---------------------------------------------------------------------------


_VALID_PATCH = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,1 +1,2 @@\n def foo():\n+    return 42\n"


def test_lookup_patched_file_empty_patch_returns_none() -> None:
    """Empty string + None patch input → None (boolean-helper policy)."""
    assert lookup_patched_file("", "src/foo.py") is None
    assert lookup_patched_file(None, "src/foo.py") is None


def test_lookup_patched_file_invalid_path_returns_none() -> None:
    """validate_diff_path failure (`..` traversal, shell metachar, absolute,
    `.git/` first-component) collapses to None rather than raising. Matches
    the `file_in_patch` boolean-helper policy so a malformed caller path
    routes downstream as "not in patch" — the security gate lives at
    `validate_diff_path`, not at this membership query."""
    assert lookup_patched_file(_VALID_PATCH, "../etc/passwd") is None
    assert lookup_patched_file(_VALID_PATCH, "/abs/path.py") is None
    assert lookup_patched_file(_VALID_PATCH, "foo;rm -rf /.py") is None
    assert lookup_patched_file(_VALID_PATCH, ".git/HEAD") is None


def test_lookup_patched_file_absent_from_patch_returns_none() -> None:
    """Well-formed patch + valid path BUT file is not in the patch → None.
    Distinguishes the "no diff content for this file" case from the
    "couldn't validate path" case in downstream control flow (both collapse
    to None per the documented boolean-helper policy)."""
    assert lookup_patched_file(_VALID_PATCH, "src/bar.py") is None


def test_lookup_patched_file_present_returns_patched_file() -> None:
    """Happy path: well-formed patch + path that IS present returns the
    unidiff PatchedFile object so the caller can iterate its hunks."""
    result = lookup_patched_file(_VALID_PATCH, "src/foo.py")
    assert result is not None
    # PatchedFile.path is the canonical (target-side) path for additions/
    # modifications/renames.
    assert result.path == "src/foo.py"


# ---------------------------------------------------------------------------
# GitHub wire format — hunks-only patch (regression for the
# UnidiffParseError surfaced by the analyze smoke test against a real
# PR. The single V1 production caller is analyze.py:708 — it passes
# `(changed_file.patch, changed_file.path)` where the patch IS the
# hunks for that file by construction. The wrapper synthesizes file
# headers using the queried path; lookup succeeds with the matching
# PatchedFile.)
# ---------------------------------------------------------------------------


# Exactly what GitHub's PR-files API returns for `patch` (hunks only,
# no `--- a/...` / `+++ b/...` headers, no `diff --git` line).
_GITHUB_API_PATCH = (
    # Hunk-body line counts (3 on each side) must match the header.
    # Source = context_above + old_line + context_below = 3 lines.
    # Target = context_above + new_line + context_below = 3 lines.
    "@@ -30,3 +30,3 @@ def search_users(prefix: str) -> list[dict]:\n"
    "   context_above\n"
    "-  old_line\n"
    "+  new_line\n"
    "   context_below\n"
)


def test_lookup_patched_file_handles_github_hunks_only_shape() -> None:
    """Pin the smoke-test regression: previously raised
    UnidiffParseError('Unexpected hunk found'). The wrapper synthesizes
    `--- a/<path>` / `+++ b/<path>` headers around the hunks so unidiff
    can parse them. Returns a PatchedFile whose `.path` matches the
    queried path."""
    result = lookup_patched_file(_GITHUB_API_PATCH, "src/foo.py")
    assert result is not None
    assert result.path == "src/foo.py"


def test_lookup_patched_file_diff_git_prefixed_full_diff_still_parses() -> None:
    """A `diff --git`-prefixed full unified diff (the existing fixture
    shape used by other tests) passes through the wrapper unchanged
    because the first non-blank line starts with `diff`, not `@@`.
    Pin: the detector doesn't over-trigger on full-diff shapes."""
    diff_git_patch = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,1 +1,2 @@\n"
        " original\n"
        "+added\n"
    )
    result = lookup_patched_file(diff_git_patch, "src/foo.py")
    assert result is not None
    assert result.path == "src/foo.py"


def test_lookup_patched_file_malformed_patch_still_raises() -> None:
    """Pin: the wrapper isn't a recovery mechanism for arbitrarily
    broken inputs. A patch whose hunk-line counts don't match the body
    still raises CoordinateError (wrapped from `unidiff.UnidiffParseError`).
    The wrapper only handles the canonical GitHub hunks-only shape;
    body-vs-header mismatches are propagated as malformed input."""
    # Hunk header claims 5 lines on each side but the body has 1 line.
    # unidiff raises "Hunk is shorter than expected" → wrapped as
    # CoordinateError("malformed patch input: ...").
    malformed = "@@ -1,5 +1,5 @@\n one line only\n"
    with pytest.raises(CoordinateError, match="malformed patch input"):
        lookup_patched_file(malformed, "src/foo.py")


def test_lookup_patched_file_handles_utf8_bom_prefix() -> None:
    """Pin the adversarial-audit finding: a hunks-only patch beginning
    with a U+FEFF BOM (a file authored with BOM that GitHub echoes in
    the diff payload) MUST be detected as hunks-only and wrapped. Before
    the BOM-strip fix, `str.lstrip()` left the BOM in place; the
    detector saw `﻿@@` and missed the hunks-only shape, the wrapper
    passed unchanged, `unidiff.PatchSet` produced an empty PatchSet, and
    the file silently downgraded to NO_REVIEWABLE_CONTEXT at the
    consumer. Pin: with the BOM-aware lstrip, this returns a usable
    PatchedFile."""
    bom_patch = "﻿" + ("@@ -1,1 +1,2 @@\n original\n+added\n")
    result = lookup_patched_file(bom_patch, "src/foo.py")
    assert result is not None, (
        "BOM-prefixed hunks-only patch silently downgraded to None — "
        "the BOM-strip in _wrap_github_hunks_with_headers regressed"
    )
    assert result.path == "src/foo.py"


def test_lookup_patched_file_handles_bom_after_whitespace() -> None:
    """Belt-and-suspenders: BOM that appears AFTER leading whitespace
    (`'  \\n\\ufeff@@ ...'`) must also be stripped. The detector handles
    `lstrip().removeprefix('\\ufeff').lstrip()` to tolerate the BOM in
    either order relative to whitespace, since real-world emitters vary."""
    patch_with_ws_and_bom = "  \n﻿" + ("@@ -1,1 +1,2 @@\n original\n+added\n")
    result = lookup_patched_file(patch_with_ws_and_bom, "src/foo.py")
    assert result is not None
    assert result.path == "src/foo.py"


# NOTE: The two defensive error paths in `lookup_patched_file` —
# malformed-unidiff wrap (`UnidiffParseError → CoordinateError`) and
# duplicate-entries detection — are NOT pinned here because `unidiff`
# is extremely lenient on garbage text input (silently produces an
# empty/single PatchSet rather than raising) and consolidates duplicate
# paths during parsing. The defensive code is correct shape but the
# triggering inputs require lower-level injection (e.g., monkeypatching
# `PatchSet` or constructing a `PatchSet` programmatically) rather
# than text fixtures. The same discipline applies to `file_in_patch`
# in `test_coordinates_file_in_patch.py`, which also doesn't try to
# exercise those branches from text input.


# ---------------------------------------------------------------------------
# COORDINATES_IMPORT_PATH_RESOLVER: Protocol satisfaction + statelessness
# ---------------------------------------------------------------------------


def test_coordinates_import_path_resolver_satisfies_protocol() -> None:
    """Singleton instance satisfies `ImportPathResolver` Protocol via
    `isinstance` (PEP 544 runtime-checkable). `build_graph` uses this
    check as the structural gate; pinning here catches a regression
    that would otherwise surface only at lifespan start."""
    assert isinstance(COORDINATES_IMPORT_PATH_RESOLVER, ImportPathResolver)


def test_coordinates_import_path_resolver_is_stateless() -> None:
    """Singleton docstring claims 'Stateless; safe to share across
    concurrent reviews.' Pin: no instance attributes. If a future
    refactor adds per-instance state, this fails — at which point the
    concurrent-safety claim needs re-evaluation."""
    assert vars(COORDINATES_IMPORT_PATH_RESOLVER) == {}
