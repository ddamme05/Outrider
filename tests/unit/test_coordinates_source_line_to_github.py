"""Check source-line conversion at file boundaries.

The last source line is valid; a range past EOF must raise BYTE_OFFSET_INVALID
rather than be silently shortened. Cover inverted and out-of-range starts too."""

from __future__ import annotations

import pytest

from outrider.coordinates import (
    CoordinateError,
    GitHubCommentLocation,
    source_line_to_github,
)
from outrider.coordinates.errors import CoordinateErrorKind

# Three-line file with a 2-line added hunk at lines 2-3 of head.
# Head: line 1 = "first\n", line 2 = "second\n", line 3 = "third\n".
_HEAD = "first\nsecond\nthird\n"
_PATCH = "@@ -0,0 +1,3 @@\n+first\n+second\n+third\n"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_source_line_to_github_mid_file_range_succeeds() -> None:
    """A valid mid-file line range translates cleanly."""
    location = source_line_to_github(
        file_path="src/foo.py",
        line_start=2,
        line_end=2,
        head_content=_HEAD,
        patch=_PATCH,
    )
    assert isinstance(location, GitHubCommentLocation)
    assert location.file_path == "src/foo.py"
    assert location.line == 2
    assert location.side == "RIGHT"


def test_source_line_to_github_legitimate_last_line_succeeds() -> None:
    """line_end == total source lines is the LEGITIMATE last-line case.

    The implementation's inner except-on-`line_end + 1` fires here,
    and byte_end = len(head_bytes) is the correct value. Distinguished
    from past-EOF by the pre-validation of `line_end` itself.
    """
    location = source_line_to_github(
        file_path="src/foo.py",
        line_start=3,
        line_end=3,
        head_content=_HEAD,
        patch=_PATCH,
    )
    assert location.line == 3


def test_source_line_to_github_multiline_range_succeeds() -> None:
    """A multi-line range (V1 collapses to line_start internally)."""
    location = source_line_to_github(
        file_path="src/foo.py",
        line_start=2,
        line_end=3,
        head_content=_HEAD,
        patch=_PATCH,
    )
    # tree_sitter_to_github collapses to line_start per its semantics.
    assert location.line == 2


# ---------------------------------------------------------------------------
# Past-EOF rejections (the Codex-review fix)
# ---------------------------------------------------------------------------


def test_source_line_to_github_line_end_past_eof_raises() -> None:
    """line_end past EOF MUST raise BYTE_OFFSET_INVALID.

    Regression pin: an earlier implementation caught every
    CoordinateError from `_line_to_byte_offset(line_end + 1)` and
    silently treated it as "last line of file", letting past-EOF
    findings publish on EOF anyway. The shipped implementation splits
    the validation: line_end itself is checked first; past-EOF raises
    immediately.
    """
    with pytest.raises(CoordinateError) as exc_info:
        source_line_to_github(
            file_path="src/foo.py",
            line_start=2,
            line_end=99999,  # Way past EOF
            head_content=_HEAD,
            patch=_PATCH,
        )
    assert exc_info.value.kind is CoordinateErrorKind.BYTE_OFFSET_INVALID
    assert "exceeds source-line count" in str(exc_info.value)


def test_source_line_to_github_line_start_past_eof_raises() -> None:
    """line_start past EOF raises via the first `_line_to_byte_offset` call."""
    with pytest.raises(CoordinateError) as exc_info:
        source_line_to_github(
            file_path="src/foo.py",
            line_start=99999,
            line_end=99999,
            head_content=_HEAD,
            patch=_PATCH,
        )
    assert exc_info.value.kind is CoordinateErrorKind.BYTE_OFFSET_INVALID


def test_source_line_to_github_line_end_one_past_last_line_raises() -> None:
    """The boundary case: line_end is total_lines + 1 (one past the actual end).

    A 3-line file means line_end=4 is past EOF. This is the case most
    likely to slip past a naïve bounds check; pin it explicitly.
    """
    with pytest.raises(CoordinateError) as exc_info:
        source_line_to_github(
            file_path="src/foo.py",
            line_start=2,
            line_end=4,  # one past the last line
            head_content=_HEAD,
            patch=_PATCH,
        )
    assert exc_info.value.kind is CoordinateErrorKind.BYTE_OFFSET_INVALID


# ---------------------------------------------------------------------------
# Argument-shape rejections
# ---------------------------------------------------------------------------


def test_source_line_to_github_line_end_less_than_line_start_raises() -> None:
    """Inverted range raises BYTE_OFFSET_INVALID."""
    with pytest.raises(CoordinateError) as exc_info:
        source_line_to_github(
            file_path="src/foo.py",
            line_start=3,
            line_end=2,
            head_content=_HEAD,
            patch=_PATCH,
        )
    assert exc_info.value.kind is CoordinateErrorKind.BYTE_OFFSET_INVALID
    assert "must be >=" in str(exc_info.value)


def test_source_line_to_github_line_start_zero_raises() -> None:
    """line_start < 1 raises BYTE_OFFSET_INVALID (1-indexed)."""
    with pytest.raises(CoordinateError) as exc_info:
        source_line_to_github(
            file_path="src/foo.py",
            line_start=0,
            line_end=1,
            head_content=_HEAD,
            patch=_PATCH,
        )
    assert exc_info.value.kind is CoordinateErrorKind.BYTE_OFFSET_INVALID
