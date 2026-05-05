"""Unit tests for `coordinates.tree_sitter_to_github` translator.

Per docs/spec.md §5.6 — covers happy-path translation, error paths, byte
boundary cases, UTF-8 / line-ending semantics, multi-file disambiguation,
multi-line span collapse, and patch-input edge cases.

See DECISIONS.md#006-two-month-0-spikes-not-five for the off-by-one test
discipline these cases honor — coordinate math correctness is enforced by
exhaustive boundary tests, not by spike work.
"""

from __future__ import annotations

import pytest

from outrider.coordinates import CoordinateError, GitHubCommentLocation, tree_sitter_to_github

# ----------------------------------------------------------------------------
# Fixtures: minimal unified diffs for common shapes
# ----------------------------------------------------------------------------


SIMPLE_HEAD = "line_one\nadded_line\nline_three\n"
# Hunk replaces 2 lines (line_one + line_three) with 3 lines (line_one + added_line + line_three).
SIMPLE_PATCH = (
    "diff --git a/src/foo.py b/src/foo.py\n"
    "--- a/src/foo.py\n"
    "+++ b/src/foo.py\n"
    "@@ -1,2 +1,3 @@\n"
    " line_one\n"
    "+added_line\n"
    " line_three\n"
)
SIMPLE_FILE_PATH = "src/foo.py"


# ----------------------------------------------------------------------------
# Happy-path: span fully inside an added-line region → INLINE_COMMENT
# ----------------------------------------------------------------------------


def test_span_in_added_line_returns_right_side_location() -> None:
    """Span fully inside an added line → GitHubCommentLocation(line, side=RIGHT)."""
    # SIMPLE_HEAD bytes: "line_one\n" = 9 bytes; "added_line\n" starts at byte 9.
    # byte_start in [9, 19] = inside "added_line" content.
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=9,
        byte_end=19,
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.file_path == SIMPLE_FILE_PATH
    assert loc.line == 2
    assert loc.side == "RIGHT"
    assert loc.start_line is None
    assert loc.start_side is None


def test_span_at_first_byte_of_added_line() -> None:
    """byte_start at the first byte of the added line → line 2."""
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=9,  # first byte of "added_line"
        byte_end=10,
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.line == 2


def test_span_in_context_line_returns_correct_line() -> None:
    """Span on a context (unchanged) line within a hunk's target range — INLINE_COMMENT.

    Context lines ARE in target_length of the hunk, so they're reviewable on the
    head side. line_one (bytes 0-8) is a context line of the SIMPLE_PATCH hunk.
    """
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=0,
        byte_end=8,
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.line == 1
    assert loc.side == "RIGHT"


# ----------------------------------------------------------------------------
# Hunk-boundary: first and last reviewable lines of a hunk
# ----------------------------------------------------------------------------


def test_span_on_first_line_of_hunk() -> None:
    """byte_start on the first reviewable line of the hunk → that line."""
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=0,
        byte_end=4,
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.line == 1


def test_span_on_last_line_of_hunk() -> None:
    """byte_start on the last reviewable line of the hunk → that line."""
    # "line_three" starts at byte 20 (after "line_one\nadded_line\n" = 9 + 11 = 20).
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=20,
        byte_end=29,
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.line == 3


# ----------------------------------------------------------------------------
# Error paths: span in unchanged code, file not in patch
# ----------------------------------------------------------------------------


def test_span_in_unchanged_code_within_diffed_file_raises() -> None:
    """Span in an unchanged region of a diffed file → CoordinateError."""
    # head_content: 4 lines; only the first 3 are in the hunk.
    # Byte for line 4 is past the hunk's target range.
    head = "line_one\nadded_line\nline_three\nunchanged_outside_hunk\n"
    # Same patch as SIMPLE_PATCH (only 3 lines in target range).
    # Byte 31 is start of "unchanged_outside_hunk" (line 4); past hunk end.
    with pytest.raises(CoordinateError, match="not in any hunk"):
        tree_sitter_to_github(
            file_path=SIMPLE_FILE_PATH,
            byte_start=31,
            byte_end=50,
            head_content=head,
            patch=SIMPLE_PATCH,
        )


def test_span_for_file_not_in_patch_raises() -> None:
    """File not present in patch → CoordinateError 'not present in the patch'."""
    with pytest.raises(CoordinateError, match="not present in the patch"):
        tree_sitter_to_github(
            file_path="other_file.py",  # not the path in SIMPLE_PATCH
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


# ----------------------------------------------------------------------------
# Idempotence (same inputs → same outputs)
# ----------------------------------------------------------------------------


def test_repeated_calls_are_idempotent() -> None:
    """Same inputs → equal GitHubCommentLocation across repeated calls."""
    args = {
        "file_path": SIMPLE_FILE_PATH,
        "byte_start": 9,
        "byte_end": 19,
        "head_content": SIMPLE_HEAD,
        "patch": SIMPLE_PATCH,
    }
    first = tree_sitter_to_github(**args)
    second = tree_sitter_to_github(**args)
    assert first == second


# ----------------------------------------------------------------------------
# V1 commitment: side=RIGHT only
# ----------------------------------------------------------------------------


def test_v1_translator_only_produces_right_side() -> None:
    """V1 returns side=RIGHT for every successful translation.

    LEFT-side commenting on deleted base-only content would require a
    base-side parse, which is out of V1 scope per the §5.6 signature.
    """
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=9,
        byte_end=19,
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.side == "RIGHT"


# ----------------------------------------------------------------------------
# UTF-8 byte-offset coverage (per spikes/tree_sitter/NOTES.md line 97)
# ----------------------------------------------------------------------------


def test_utf8_multibyte_byte_offset_resolves_correct_line() -> None:
    """PEP 3131 identifier `α` (2 bytes UTF-8) — line counting must use bytes,
    not codepoints.

    "def α():\\n    return 42\\n" → 23 codepoints, 24 bytes.
    Byte 10 (start of "    return 42") is on line 2; codepoint 9 (the codepoint
    at the same logical position after the multibyte expansion) is also on
    line 2, but an implementation that mistakenly indexed into `head_content`
    by bytes would land at codepoint 10 ("r"), still line 2. To distinguish
    the bug, byte 5 (the second byte of `α`) must NOT be a valid line
    boundary — `head_content[5]` would be "(" (a different codepoint), not
    `α`'s second byte. The translator works on bytes throughout, so it never
    indexes head_content by codepoint.
    """
    head = "def α():\n    return 42\n"
    patch = (
        "diff --git a/u.py b/u.py\n"
        "--- a/u.py\n"
        "+++ b/u.py\n"
        "@@ -1,1 +1,2 @@\n"
        " def α():\n"
        "+    return 42\n"
    )
    # byte 10 = first byte of "    return 42" (line 2)
    loc = tree_sitter_to_github(
        file_path="u.py",
        byte_start=10,
        byte_end=24,
        head_content=head,
        patch=patch,
    )
    assert loc.line == 2


# ----------------------------------------------------------------------------
# Multi-line span collapse to single-line (per non-goal #1)
# ----------------------------------------------------------------------------


def test_multiline_span_collapses_to_byte_start_line() -> None:
    """byte_start and byte_end on different lines → collapses to byte_start's line.

    V1 commitment: start_line / start_side stay None even for multi-line spans.
    """
    # SIMPLE_HEAD: byte 0 is line 1; byte 19 is end of line 2.
    # Span covers lines 1-2; should collapse to byte_start's line = 1.
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=0,
        byte_end=19,
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.line == 1
    assert loc.start_line is None
    assert loc.start_side is None


# ----------------------------------------------------------------------------
# Multi-file patch disambiguation
# ----------------------------------------------------------------------------


def test_multifile_patch_uses_normalized_target_path() -> None:
    """With two files in the patch having overlapping line numbers, only the
    hunk whose `unidiff.PatchedFile.path` matches `file_path` is considered.

    Catches the implementation that compares raw `+++ b/foo.py` text against
    the canonical `file_path` (without the `b/` prefix).
    """
    # Two files, both with hunks at line 1.
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
    # Asking for foo.py — should resolve via foo's hunks, not bar's.
    foo_head = "original_foo\nadded_in_foo\n"
    loc = tree_sitter_to_github(
        file_path="foo.py",
        byte_start=13,  # start of "added_in_foo"
        byte_end=25,
        head_content=foo_head,
        patch=multifile_patch,
    )
    assert loc.file_path == "foo.py"
    assert loc.line == 2

    # Asking for bar.py — different file, should resolve via bar's hunks.
    bar_head = "original_bar\nadded_in_bar\n"
    loc = tree_sitter_to_github(
        file_path="bar.py",
        byte_start=13,
        byte_end=25,
        head_content=bar_head,
        patch=multifile_patch,
    )
    assert loc.file_path == "bar.py"
    assert loc.line == 2


def test_multifile_patch_rejects_raw_header_path() -> None:
    """`b/foo.py` (raw header text) is NOT a valid `file_path` argument.

    The canonical `file_path` carries no `a/`/`b/` prefix; an implementation
    that compared raw `+++` header text would accidentally accept `b/foo.py`.
    """
    with pytest.raises(CoordinateError, match="not present in the patch"):
        tree_sitter_to_github(
            file_path="b/src/foo.py",  # canonical strips the b/ prefix
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


# ----------------------------------------------------------------------------
# Patch-input edge cases: malformed, empty
# ----------------------------------------------------------------------------


def test_lenient_garbage_patch_raises_coordinate_error() -> None:
    """Garbage that unidiff parses leniently → CoordinateError via the
    file-not-found path.

    `unidiff` returns an empty PatchSet for free-text input rather than
    raising; we still surface CoordinateError because the file isn't in
    the (empty) patch.
    """
    with pytest.raises(CoordinateError, match="not present in the patch"):
        tree_sitter_to_github(
            file_path=SIMPLE_FILE_PATH,
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch="not a real diff at all just garbage\nand more garbage",
        )


def test_malformed_hunk_raises_coordinate_error_via_unidiff_parse_error() -> None:
    """Hunk shorter than declared → unidiff raises UnidiffParseError;
    coordinates wraps it as CoordinateError without leaking the underlying type.

    Per `unidiff/patch.py` (aegis-docs), `UnidiffParseError` is raised when
    the hunk falls short of its declared line count ("Hunk is shorter than
    expected"). Construct a patch declaring 5 source / 5 target lines but
    providing only 1 context line.
    """
    malformed_patch = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,5 +1,5 @@\n"
        " only_one_line_provided_but_five_declared\n"
    )
    with pytest.raises(CoordinateError, match="malformed patch input"):
        tree_sitter_to_github(
            file_path="x.py",
            byte_start=0,
            byte_end=8,
            head_content="only_one_line_provided_but_five_declared\n",
            patch=malformed_patch,
        )


def test_empty_patch_raises_coordinate_error() -> None:
    """Empty patch (`patch=""`) → CoordinateError for any file_path.

    No hunks means nothing is reviewable.
    """
    with pytest.raises(CoordinateError, match="not present in the patch"):
        tree_sitter_to_github(
            file_path=SIMPLE_FILE_PATH,
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch="",
        )


# ----------------------------------------------------------------------------
# Boundary byte-span coverage (DECISIONS.md#006 — off-by-one test discipline)
# ----------------------------------------------------------------------------


def test_zero_width_span_in_hunk_succeeds() -> None:
    """Zero-width span (byte_start == byte_end) treated as a point at byte_start.

    Returns a valid GitHubCommentLocation if the byte falls in a hunk.
    """
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=9,
        byte_end=9,  # zero-width point at start of "added_line"
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.line == 2


def test_byte_start_zero_in_hunk_succeeds() -> None:
    """byte_start == 0 (file start) → line 1 if line 1 is in a hunk."""
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=0,
        byte_end=4,
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.line == 1


def test_byte_end_at_head_byte_length_in_hunk_succeeds() -> None:
    """byte_end == len(head_bytes) → boundary is in-bounds (half-open interval)."""
    head_byte_length = len(SIMPLE_HEAD.encode("utf-8"))
    # byte_start on line 3 ("line_three"), byte_end at EOF.
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=20,  # start of "line_three"
        byte_end=head_byte_length,  # EOF
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert loc.line == 3


def test_byte_start_out_of_bounds_raises() -> None:
    """byte_start > len(head_bytes.encode('utf-8')) → CoordinateError, not IndexError."""
    head_byte_length = len(SIMPLE_HEAD.encode("utf-8"))
    with pytest.raises(CoordinateError, match="byte_start.*out of bounds"):
        tree_sitter_to_github(
            file_path=SIMPLE_FILE_PATH,
            byte_start=head_byte_length + 1,
            byte_end=head_byte_length + 5,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


def test_byte_start_at_eof_rejected() -> None:
    """byte_start == head_byte_length → CoordinateError (start at EOF rejected).

    Half-open interval: `byte_start ∈ [0, head_byte_length)`. A start at EOF
    has no reviewable byte and would otherwise map to a ghost line past the
    last real line on newline-terminated files. Catches the implementation
    that uses `byte_start > head_byte_length` and silently accepts EOF starts.
    """
    head_byte_length = len(SIMPLE_HEAD.encode("utf-8"))
    with pytest.raises(CoordinateError, match="byte_start.*out of bounds"):
        tree_sitter_to_github(
            file_path=SIMPLE_FILE_PATH,
            byte_start=head_byte_length,
            byte_end=head_byte_length,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


def test_empty_head_content_rejects_any_span() -> None:
    """Empty `head_content` (`head_byte_length == 0`) → every span rejected.

    No reviewable bytes means no reviewable lines; the `byte_start >= head_byte_length`
    rule rejects byte_start=0 when head is empty.
    """
    with pytest.raises(CoordinateError, match="byte_start.*out of bounds"):
        tree_sitter_to_github(
            file_path=SIMPLE_FILE_PATH,
            byte_start=0,
            byte_end=0,
            head_content="",
            patch=SIMPLE_PATCH,
        )


def test_byte_end_out_of_bounds_raises() -> None:
    """byte_end > len(head_bytes.encode('utf-8')) → CoordinateError, not IndexError.

    A byte_start that is in-bounds with a byte_end that overshoots EOF is
    still a contract violation — the span extends past reviewable content.
    Catches the bug where byte_start passes validation but the span as a
    whole is unbounded.
    """
    head_byte_length = len(SIMPLE_HEAD.encode("utf-8"))
    with pytest.raises(CoordinateError, match="byte_end.*out of bounds"):
        tree_sitter_to_github(
            file_path=SIMPLE_FILE_PATH,
            byte_start=0,  # in-bounds
            byte_end=head_byte_length + 1,  # over-bounds
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


def test_negative_byte_start_raises() -> None:
    """byte_start < 0 → CoordinateError."""
    with pytest.raises(CoordinateError, match="out of bounds"):
        tree_sitter_to_github(
            file_path=SIMPLE_FILE_PATH,
            byte_start=-1,
            byte_end=5,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


def test_inverted_byte_span_raises() -> None:
    """byte_end < byte_start → CoordinateError (half-open interval rule)."""
    with pytest.raises(CoordinateError, match="must be >="):
        tree_sitter_to_github(
            file_path=SIMPLE_FILE_PATH,
            byte_start=10,
            byte_end=5,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


# ----------------------------------------------------------------------------
# Line-ending coverage: \n, \r\n, lone \r (str.splitlines() semantics)
# ----------------------------------------------------------------------------


def test_lf_line_endings_count_correctly() -> None:
    """Standard Unix `\\n` line endings — sanity check for the common case."""
    head = "a\nb\nc\n"
    patch = "diff --git a/lf.py b/lf.py\n--- a/lf.py\n+++ b/lf.py\n@@ -1,2 +1,3 @@\n a\n+b\n c\n"
    # byte 2 = "b" (start of line 2)
    loc = tree_sitter_to_github(
        file_path="lf.py",
        byte_start=2,
        byte_end=3,
        head_content=head,
        patch=patch,
    )
    assert loc.line == 2


def test_crlf_line_endings_count_correctly() -> None:
    """CRLF line endings — `\\r\\n` is one line terminator (Windows files).

    Catches the implementation that uses `head_content.split("\\n")` and
    miscounts CRLF files (where each line ends with `\\r\\n` but `\\n` alone
    is mis-split into `["a\\r", "b\\r", ...]`).
    """
    head = "a\r\nb\r\nc\r\n"
    # head_bytes: "a"=0, "\r"=1, "\n"=2, "b"=3, "\r"=4, "\n"=5, "c"=6, "\r"=7, "\n"=8
    # byte 3 = "b" (start of line 2)
    patch = (
        "diff --git a/crlf.py b/crlf.py\n"
        "--- a/crlf.py\n"
        "+++ b/crlf.py\n"
        "@@ -1,2 +1,3 @@\n"
        " a\n"
        "+b\n"
        " c\n"
    )
    loc = tree_sitter_to_github(
        file_path="crlf.py",
        byte_start=3,
        byte_end=4,
        head_content=head,
        patch=patch,
    )
    assert loc.line == 2


def test_lone_cr_treated_as_data_per_git_semantics() -> None:
    """Lone `\\r` mid-content is NOT a line terminator — git diff convention
    counts only `\\n` (LF). Catches the prior `bytes.splitlines()` impl
    (which DOES split on lone `\\r`) and would have shifted line numbers
    relative to `unidiff.Hunk.target_start`'s `\\n`-only line numbering.

    Source `b"a\\rb\\rc\\n"`: git considers this one line terminated by `\\n`.
    Bytes 0-4 (`a`, `\\r`, `b`, `\\r`, `c`) all map to L1src=1. The patch's
    hunk reflects the same view: one line (`target_length=1`).
    """
    head = "a\rb\rc\n"
    # head_bytes: a=0, \r=1, b=2, \r=3, c=4, \n=5
    patch = (
        "diff --git a/cr.py b/cr.py\n--- a/cr.py\n+++ b/cr.py\n@@ -1,1 +1,1 @@\n-old\n+a\rb\rc\n"
    )
    # byte 2 ("b") and byte 4 ("c") are both on line 1 per git semantics.
    loc = tree_sitter_to_github(
        file_path="cr.py",
        byte_start=2,
        byte_end=3,
        head_content=head,
        patch=patch,
    )
    assert loc.line == 1


# ----------------------------------------------------------------------------
# Path validation gate — paths-validated-before-use [security-critical]
# ----------------------------------------------------------------------------


def test_invalid_file_path_rejected_before_any_other_work() -> None:
    """tree_sitter_to_github calls validate_diff_path() on file_path FIRST.

    Per the `paths-validated-before-use` invariant (docs/spec.md §10.1),
    coordinates enforces validation before any path reaches the GitHub
    comment API or is stored in a returned `GitHubCommentLocation`. An
    absolute path is rejected with CoordinateError mentioning 'absolute',
    matching `validate_diff_path`'s rejection message — the rejection
    fires from the path validator, not from a later byte-bounds or
    patch-membership check.
    """
    with pytest.raises(CoordinateError, match="absolute"):
        tree_sitter_to_github(
            file_path="/etc/passwd",
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


def test_traversal_in_file_path_rejected() -> None:
    """`..` traversal in file_path → CoordinateError from validate_diff_path."""
    with pytest.raises(CoordinateError, match=r"'\.\.'"):
        tree_sitter_to_github(
            file_path="../../etc/passwd",
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


def test_shell_metachar_in_file_path_rejected() -> None:
    """Shell metacharacter in file_path → CoordinateError from validate_diff_path."""
    with pytest.raises(CoordinateError, match="shell metacharacters"):
        tree_sitter_to_github(
            file_path="foo;rm.py",
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


def test_windows_drive_path_rejected() -> None:
    """Windows drive-qualified paths → CoordinateError from validate_diff_path's
    drive-letter rejection. Pins the propagation: the validator's
    Windows-drive rejection flows through to `tree_sitter_to_github`
    BEFORE any byte-bounds or patch-membership work, matching the
    "validate path first" contract of the security-critical
    `paths-validated-before-use` invariant.
    """
    with pytest.raises(CoordinateError, match="drive-letter prefix"):
        tree_sitter_to_github(
            file_path="C:/Users/file.py",
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )
    with pytest.raises(CoordinateError, match="drive-letter prefix"):
        tree_sitter_to_github(
            file_path="C:relative.py",
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )
    # Backslash form is caught by the backslash rule (which fires before
    # the drive-letter rule); same CoordinateError class, different message.
    with pytest.raises(CoordinateError, match="backslash"):
        tree_sitter_to_github(
            file_path="C:\\Users\\file.py",
            byte_start=0,
            byte_end=8,
            head_content=SIMPLE_HEAD,
            patch=SIMPLE_PATCH,
        )


# ----------------------------------------------------------------------------
# Return type is a GitHubCommentLocation (not a tuple, dict, etc.)
# ----------------------------------------------------------------------------


def test_return_value_is_github_comment_location_instance() -> None:
    """The return type is the canonical Pydantic model, not a raw dict/tuple."""
    loc = tree_sitter_to_github(
        file_path=SIMPLE_FILE_PATH,
        byte_start=9,
        byte_end=19,
        head_content=SIMPLE_HEAD,
        patch=SIMPLE_PATCH,
    )
    assert isinstance(loc, GitHubCommentLocation)


# ----------------------------------------------------------------------------
# Misuse-resistance: keyword-only arguments
# ----------------------------------------------------------------------------


def test_translator_rejects_positional_args() -> None:
    """`tree_sitter_to_github` is keyword-only — five same-type positional
    parameters (`file_path`/`head_content`/`patch` are all `str`,
    `byte_start`/`byte_end` are both `int`) make accidental swaps invisible.
    Keyword-only forces explicit parameter naming at the call site.
    """
    with pytest.raises(TypeError, match="positional"):
        tree_sitter_to_github(  # type: ignore[misc]
            SIMPLE_FILE_PATH,
            9,
            19,
            SIMPLE_HEAD,
            SIMPLE_PATCH,
        )


# ----------------------------------------------------------------------------
# Path-normalization symmetry: validated `./foo.py` matches unidiff path
# ----------------------------------------------------------------------------


def test_validated_path_matches_unidiff_normalized_path() -> None:
    """`validate_diff_path("./foo.py")` collapses to `"foo.py"`. The translator
    normalizes `unidiff.PatchedFile.path` symmetrically (via
    `PurePosixPath(...).as_posix()`) before comparison, so both halves of
    the path-equality use the same canonical form.

    Catches the implementation that compares the validated path against
    `pf.path` raw — which would silently miss a match when unidiff's
    parsed path retains a `./` prefix from a synthetic patch header.
    """
    # Synthetic patch header with `b/./foo.py`: unidiff's `pf.path` keeps
    # the `./` after stripping `b/`. Without symmetric normalization the
    # match against validated `"foo.py"` would fail.
    patch_with_dot = (
        "diff --git a/foo.py b/./foo.py\n"
        "--- a/foo.py\n"
        "+++ b/./foo.py\n"
        "@@ -1,1 +1,2 @@\n"
        " orig\n"
        "+added\n"
    )
    head = "orig\nadded\n"
    loc = tree_sitter_to_github(
        file_path="foo.py",
        byte_start=5,  # start of "added"
        byte_end=10,
        head_content=head,
        patch=patch_with_dot,
    )
    assert loc.file_path == "foo.py"
    assert loc.line == 2


def test_caller_dot_prefix_normalized_to_canonical() -> None:
    """`tree_sitter_to_github(file_path="./foo.py", ...)` against a patch
    whose `pf.path` is already canonical `"foo.py"` matches via
    `validate_diff_path`'s caller-side normalization. Returned
    `GitHubCommentLocation.file_path` is the canonical form `"foo.py"`,
    not the input form `"./foo.py"`.

    Locks the composition, not just the helper: the prior commit's bug
    in `file_in_patch` (raw caller path compared to normalized unidiff
    path) didn't reach `_find_patched_file` because `tree_sitter_to_github`
    canonicalizes via `validate_diff_path` BEFORE handing off. This test
    pins that composition so a future refactor can't accidentally bypass
    the canonicalization step.
    """
    canonical_patch = (
        "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1,1 +1,2 @@\n orig\n+added\n"
    )
    head = "orig\nadded\n"
    loc = tree_sitter_to_github(
        file_path="./foo.py",
        byte_start=5,
        byte_end=10,
        head_content=head,
        patch=canonical_patch,
    )
    assert loc.file_path == "foo.py"
    assert loc.line == 2


def test_caller_double_slash_normalized_to_canonical() -> None:
    """`tree_sitter_to_github(file_path="a//b.py", ...)` matches a patch
    whose `pf.path` is `"a/b.py"` — `validate_diff_path` collapses the
    double slash via `PurePosixPath`. Returned location carries the
    canonical form.
    """
    canonical_patch = (
        "diff --git a/a/b.py b/a/b.py\n--- a/a/b.py\n+++ b/a/b.py\n@@ -1,1 +1,2 @@\n orig\n+added\n"
    )
    head = "orig\nadded\n"
    loc = tree_sitter_to_github(
        file_path="a//b.py",
        byte_start=5,
        byte_end=10,
        head_content=head,
        patch=canonical_patch,
    )
    assert loc.file_path == "a/b.py"
    assert loc.line == 2


# ----------------------------------------------------------------------------
# Duplicate-path detection: webhook-attacker reject
# ----------------------------------------------------------------------------


def test_duplicate_patched_file_entries_rejected() -> None:
    """Two `+++ b/foo.py` blocks in one patch → `CoordinateError` instead of
    silently first-matching. unidiff allows duplicate `PatchedFile.path`
    entries; the translator rejects them as ambiguous routing input
    (webhook-attacker reachable per trust boundary #5).
    """
    patch_with_dups = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,1 +1,2 @@\n"
        " a\n"
        "+b\n"
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -10,1 +10,2 @@\n"
        " x\n"
        "+y\n"
    )
    head = "a\nb\nc\nd\ne\nf\ng\nh\ni\nx\ny\n"
    with pytest.raises(CoordinateError, match="duplicate entries"):
        tree_sitter_to_github(
            file_path="foo.py",
            byte_start=0,
            byte_end=1,
            head_content=head,
            patch=patch_with_dups,
        )


# ----------------------------------------------------------------------------
# GitHubCommentLocation: side / start_side cross-field validator
# ----------------------------------------------------------------------------


def test_github_comment_location_mixed_side_multiline_rejected() -> None:
    """Multi-line comment with `start_side != side` raises ValidationError.
    GitHub's review API rejects mixed-side multi-line with an opaque 422;
    catching it at model construction makes the failure local.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="start_side.*must.*equal side"):
        GitHubCommentLocation(
            file_path="foo.py",
            line=10,
            side="RIGHT",
            start_line=5,
            start_side="LEFT",
        )


def test_github_comment_location_same_side_multiline_accepted() -> None:
    """Multi-line comment with matching sides constructs successfully."""
    loc = GitHubCommentLocation(
        file_path="foo.py",
        line=10,
        side="RIGHT",
        start_line=5,
        start_side="RIGHT",
    )
    assert loc.start_side == loc.side == "RIGHT"
