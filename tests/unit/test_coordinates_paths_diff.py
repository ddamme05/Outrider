"""Unit tests for `coordinates.validate_diff_path` — diff-side path validator.

Per docs/spec.md §10.1 + docs/trust-boundaries.md §5.3 — the
`paths-validated-before-use` invariant (security-critical) requires file
paths from the diff to be validated as relative-only, free of `..`
traversal, free of shell metacharacters, in POSIX form, before they reach
the GitHub comment API.
"""

from __future__ import annotations

import pytest

from outrider.coordinates import CoordinateError, validate_diff_path

# ----------------------------------------------------------------------------
# Valid paths pass through and are returned in POSIX form
# ----------------------------------------------------------------------------


def test_simple_relative_path_passes() -> None:
    """Simple repo-relative path is returned unchanged in POSIX form."""
    assert validate_diff_path("src/foo.py") == "src/foo.py"


def test_single_segment_path_passes() -> None:
    """Top-level file with no directory components."""
    assert validate_diff_path("foo.py") == "foo.py"


def test_deeply_nested_path_passes() -> None:
    """Deep nesting is fine as long as no component violates the rules."""
    assert validate_diff_path("a/b/c/d/e/f.py") == "a/b/c/d/e/f.py"


def test_dot_in_path_passes() -> None:
    """A leading `./` or interior `.` segment normalizes via PurePosixPath."""
    assert validate_diff_path("./foo.py") == "foo.py"


def test_dot_segments_collapse() -> None:
    """`a/./b` collapses to `a/b` per PurePosixPath normalization."""
    assert validate_diff_path("a/./b.py") == "a/b.py"


def test_double_slash_collapses() -> None:
    """`a//b` collapses to `a/b` per PurePosixPath normalization."""
    assert validate_diff_path("a//b.py") == "a/b.py"


def test_path_with_unicode_passes() -> None:
    """Non-ASCII filenames (e.g., `α.py`) are valid — Unicode is allowed."""
    assert validate_diff_path("src/α.py") == "src/α.py"


# ----------------------------------------------------------------------------
# Rejections — empty / absolute
# ----------------------------------------------------------------------------


def test_empty_string_rejected() -> None:
    """Empty path → CoordinateError."""
    with pytest.raises(CoordinateError, match="empty"):
        validate_diff_path("")


def test_absolute_posix_path_rejected() -> None:
    """`/etc/passwd` (POSIX absolute) → CoordinateError."""
    with pytest.raises(CoordinateError, match="absolute"):
        validate_diff_path("/etc/passwd")


def test_absolute_path_with_subdirs_rejected() -> None:
    """`/home/user/project/foo.py` → CoordinateError."""
    with pytest.raises(CoordinateError, match="absolute"):
        validate_diff_path("/home/user/project/foo.py")


# ----------------------------------------------------------------------------
# Rejections — `..` traversal
# ----------------------------------------------------------------------------


def test_leading_double_dot_rejected() -> None:
    """`../foo.py` → CoordinateError."""
    with pytest.raises(CoordinateError, match=r"'\.\.'"):
        validate_diff_path("../foo.py")


def test_interior_double_dot_rejected() -> None:
    """`a/../b.py` → CoordinateError (interior traversal)."""
    with pytest.raises(CoordinateError, match=r"'\.\.'"):
        validate_diff_path("a/../b.py")


def test_trailing_double_dot_rejected() -> None:
    """`a/..` → CoordinateError."""
    with pytest.raises(CoordinateError, match=r"'\.\.'"):
        validate_diff_path("a/..")


def test_chained_double_dots_rejected() -> None:
    """`../../etc/passwd` (multi-level escape) → CoordinateError."""
    with pytest.raises(CoordinateError, match=r"'\.\.'"):
        validate_diff_path("../../etc/passwd")


# ----------------------------------------------------------------------------
# Rejections — backslash (Windows separator)
# ----------------------------------------------------------------------------


def test_backslash_rejected() -> None:
    """`a\\b.py` → CoordinateError (Windows separator; GitHub paths are POSIX)."""
    with pytest.raises(CoordinateError, match="backslash"):
        validate_diff_path("a\\b.py")


def test_windows_absolute_rejected_via_backslash() -> None:
    """`C:\\Users\\file.py` → CoordinateError (caught by the backslash check)."""
    with pytest.raises(CoordinateError, match="backslash"):
        validate_diff_path("C:\\Users\\file.py")


# ----------------------------------------------------------------------------
# Rejections — Windows drive-letter prefix (forward-slash form)
# ----------------------------------------------------------------------------


def test_windows_drive_forward_slash_rejected() -> None:
    """`C:/Users/file.py` → CoordinateError. `PurePosixPath("C:/...").is_absolute()`
    is False (POSIX considers absolute = leading `/`), so the standard
    `is_absolute()` check would silently let drive-prefixed paths reach
    the GitHub comment API. The explicit drive-prefix rejection closes
    that gap.
    """
    with pytest.raises(CoordinateError, match="drive-letter prefix"):
        validate_diff_path("C:/Users/file.py")


def test_windows_drive_lowercase_rejected() -> None:
    """`d:/foo.py` (lowercase drive letter) → CoordinateError."""
    with pytest.raises(CoordinateError, match="drive-letter prefix"):
        validate_diff_path("d:/foo.py")


def test_windows_drive_relative_rejected() -> None:
    """`C:foo.py` (drive-relative, no separator after colon) → CoordinateError.
    Drive-relative paths inherit the current directory of the named drive
    on Windows; treating them as repo-relative would be wrong.
    """
    with pytest.raises(CoordinateError, match="drive-letter prefix"):
        validate_diff_path("C:foo.py")


def test_colon_mid_path_not_drive_treated_as_metachar() -> None:
    """A colon NOT in drive-letter position is caught as a shell metacharacter
    if the colon falls in the metachar reject set, or otherwise must not
    be confused for a drive prefix. `foo:bar.py` is shell-suspicious; the
    drive-prefix regex anchors on the start of the string, so the colon
    here is evaluated by the shell-metachar gate (today: passes; if the
    metachar set ever includes `:`, it's caught there). Either way, this
    is NOT a drive-prefix rejection — confirms the regex is anchored.
    """
    # Today: `:` is not in the shell-metachar reject set, so this passes.
    # The test pins that the drive-prefix regex doesn't false-positive on
    # mid-path colons.
    assert validate_diff_path("foo:bar.py") == "foo:bar.py"


# ----------------------------------------------------------------------------
# Rejections — shell metacharacters
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metachar",
    [";", "&", "|", "`", "$", "(", ")", "<", ">", "*", "?", "~", "[", "]", "{", "}", "'", '"'],
)
def test_shell_metacharacter_rejected(metachar: str) -> None:
    """Every shell metacharacter in the conservative reject set → CoordinateError."""
    with pytest.raises(CoordinateError, match="shell metacharacters"):
        validate_diff_path(f"foo{metachar}bar.py")


def test_newline_in_path_rejected() -> None:
    """Newline in path → CoordinateError (header-injection prevention)."""
    with pytest.raises(CoordinateError, match="shell metacharacters"):
        validate_diff_path("foo\nbar.py")


def test_carriage_return_in_path_rejected() -> None:
    """`\\r` in path → CoordinateError."""
    with pytest.raises(CoordinateError, match="shell metacharacters"):
        validate_diff_path("foo\rbar.py")


def test_nul_byte_in_path_rejected() -> None:
    """NUL byte in path → CoordinateError (null-byte attack prevention)."""
    with pytest.raises(CoordinateError, match="shell metacharacters"):
        validate_diff_path("foo\x00bar.py")
