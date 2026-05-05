"""Unit tests for `coordinates.file_in_patch` — file-membership helper.

Per docs/spec.md §4.1.7 and specs/2026-05-04-coordinates-module.md (judgment
call 2): the publisher uses this helper to distinguish `unchanged_region`
(file in patch but span outside any hunk) from `non_diffed_file` (file
absent from patch entirely) WITHOUT inlining patch-membership math —
keeps trust boundary #3 intact.

Comparison via `unidiff.PatchedFile.path` (normalized — `a/` / `b/`
prefix stripped). For rename hunks, `pf.path` returns the target
(head-side) path per unidiff's source — matches the F8 fold commitment
("match `to_file` only").
"""

from __future__ import annotations

import pytest

from outrider.coordinates import CoordinateError, file_in_patch

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


SIMPLE_PATCH = (
    "diff --git a/src/foo.py b/src/foo.py\n"
    "--- a/src/foo.py\n"
    "+++ b/src/foo.py\n"
    "@@ -1,2 +1,3 @@\n"
    " line_one\n"
    "+added_line\n"
    " line_three\n"
)


# ----------------------------------------------------------------------------
# Happy path — present / absent
# ----------------------------------------------------------------------------


def test_returns_true_for_path_present_in_patch() -> None:
    """Path appearing as a PatchedFile target → True."""
    assert file_in_patch("src/foo.py", SIMPLE_PATCH) is True


def test_returns_false_for_path_absent_from_patch() -> None:
    """Path not in patch → False (trace-discovered finding case)."""
    assert file_in_patch("src/other.py", SIMPLE_PATCH) is False


def test_returns_false_for_empty_patch() -> None:
    """Empty patch → False for any file_path (no files to match)."""
    assert file_in_patch("src/foo.py", "") is False
    assert file_in_patch("anything.py", "") is False


# ----------------------------------------------------------------------------
# Multi-file patch
# ----------------------------------------------------------------------------


def test_multifile_patch_finds_each_file() -> None:
    """A patch with multiple files: each file present returns True; absent files False."""
    multifile_patch = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,1 +1,2 @@\n"
        " original_foo\n"
        "+added_in_foo\n"
        "diff --git a/bar.py b/bar.py\n"
        "--- a/bar.py\n"
        "+++ b/bar.py\n"
        "@@ -1,1 +1,2 @@\n"
        " original_bar\n"
        "+added_in_bar\n"
    )
    assert file_in_patch("foo.py", multifile_patch) is True
    assert file_in_patch("bar.py", multifile_patch) is True
    assert file_in_patch("baz.py", multifile_patch) is False


# ----------------------------------------------------------------------------
# Prefix discipline — raw `b/` header text is NOT a valid path argument
# ----------------------------------------------------------------------------


def test_raw_b_prefix_not_matched() -> None:
    """`b/src/foo.py` (raw header text) → False; canonical paths carry no prefix.

    Catches the implementation that compares raw `+++` header text against
    the canonical `file_path` (without the `b/` prefix).
    """
    assert file_in_patch("b/src/foo.py", SIMPLE_PATCH) is False


def test_raw_a_prefix_not_matched() -> None:
    """`a/src/foo.py` (raw header text, source side) → False."""
    assert file_in_patch("a/src/foo.py", SIMPLE_PATCH) is False


# ----------------------------------------------------------------------------
# Rename hunks: F8 fold — match target (head-side) only
# ----------------------------------------------------------------------------


def test_rename_hunk_matches_target_path() -> None:
    """For a rename hunk where from_file != to_file, `pf.path` returns the
    target (head-side) path — file_in_patch returns True for the target.
    """
    rename_patch = (
        "diff --git a/old_name.py b/new_name.py\n"
        "similarity index 80%\n"
        "rename from old_name.py\n"
        "rename to new_name.py\n"
        "--- a/old_name.py\n"
        "+++ b/new_name.py\n"
        "@@ -1,2 +1,2 @@\n"
        " line_one\n"
        "-old_line\n"
        "+new_line\n"
    )
    assert file_in_patch("new_name.py", rename_patch) is True


def test_rename_hunk_does_not_match_source_path() -> None:
    """For a rename hunk, the from_file (base-side path) is NOT a member;
    file_in_patch returns False for it.

    V1 findings reference head-side paths per ast_facts conventions; from_file
    membership is not part of the contract (per the F8 fold commitment).
    """
    rename_patch = (
        "diff --git a/old_name.py b/new_name.py\n"
        "similarity index 80%\n"
        "rename from old_name.py\n"
        "rename to new_name.py\n"
        "--- a/old_name.py\n"
        "+++ b/new_name.py\n"
        "@@ -1,2 +1,2 @@\n"
        " line_one\n"
        "-old_line\n"
        "+new_line\n"
    )
    assert file_in_patch("old_name.py", rename_patch) is False


# ----------------------------------------------------------------------------
# Malformed patch input — wrap UnidiffParseError as CoordinateError
# ----------------------------------------------------------------------------


def test_malformed_patch_raises_coordinate_error() -> None:
    """Hunk shorter than declared → unidiff raises UnidiffParseError;
    file_in_patch wraps it as CoordinateError without leaking the underlying
    type.
    """
    malformed_patch = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,5 +1,5 @@\n"
        " only_one_line_provided_but_five_declared\n"
    )
    with pytest.raises(CoordinateError, match="malformed patch input"):
        file_in_patch("x.py", malformed_patch)


def test_lenient_garbage_patch_returns_false() -> None:
    """Free-text input that unidiff parses leniently to an empty PatchSet
    → False (no files in the empty PatchSet match).
    """
    assert file_in_patch("x.py", "not a real diff at all just garbage") is False


# ----------------------------------------------------------------------------
# Idempotence
# ----------------------------------------------------------------------------


def test_repeated_calls_idempotent() -> None:
    """Same inputs → same boolean result across repeated calls."""
    a = file_in_patch("src/foo.py", SIMPLE_PATCH)
    b = file_in_patch("src/foo.py", SIMPLE_PATCH)
    assert a == b is True


def test_repeated_calls_idempotent_for_absent() -> None:
    """Same inputs that miss → both calls return False."""
    a = file_in_patch("missing.py", SIMPLE_PATCH)
    b = file_in_patch("missing.py", SIMPLE_PATCH)
    assert a == b is False


# ----------------------------------------------------------------------------
# Empty / degenerate file_path
# ----------------------------------------------------------------------------


def test_empty_file_path_returns_false() -> None:
    """Empty file_path → False (no PatchedFile.path is empty)."""
    assert file_in_patch("", SIMPLE_PATCH) is False
