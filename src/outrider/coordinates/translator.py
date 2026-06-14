"""tree_sitter_to_github translator + GitHubCommentLocation + CoordinateError.

Per docs/spec.md §5.6 (the two §5.6 functions and the failure-mode contract)
and docs/spec.md §7.2 (the GitHubCommentLocation canonical shape).

See DECISIONS.md#006-two-month-0-spikes-not-five for the off-by-one test
discipline this translator honors — coordinate math correctness is enforced
by exhaustive boundary tests on this file's surface, not by spike work.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from outrider.coordinates.diff_parser import _wrap_github_hunks_with_headers, validate_diff_path
from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind

if TYPE_CHECKING:
    from unidiff.patch import PatchedFile


class GitHubCommentLocation(BaseModel):
    """Output of coordinate translation per docs/spec.md §7.2.

    V1 returns single-line locations only — `start_line` / `start_side` stay
    None for every successful translation, regardless of whether the byte
    span covers a single line or multiple lines (multi-line spans collapse
    to the line containing `byte_start`). Multi-line GitHub-comment support
    is a follow-up.

    `file_path` is treated as opaque at the type layer — format validation
    (relative-only, no `..` traversal, no shell metacharacters) is
    `validate_diff_path()`'s job per docs/spec.md §10.1. Constructing
    `GitHubCommentLocation` with an unvalidated path is a caller-side
    contract violation, not a model-level error.
    """

    model_config = ConfigDict(extra="forbid")

    file_path: str
    line: int = Field(ge=1)
    side: Literal["LEFT", "RIGHT"]
    start_line: int | None = Field(default=None, ge=1)
    start_side: Literal["LEFT", "RIGHT"] | None = None

    @model_validator(mode="after")
    def _enforce_multiline_pairing(self) -> Self:
        # start_line and start_side are paired: both set, or both None.
        if (self.start_line is None) != (self.start_side is None):
            raise ValueError(
                "GitHubCommentLocation: start_line and start_side must both be set or both be None"
            )
        if self.start_line is not None:
            # Multi-line range points upward to `line`.
            if self.start_line > self.line:
                raise ValueError(
                    f"GitHubCommentLocation: start_line ({self.start_line}) must "
                    f"be <= line ({self.line})"
                )
            # Multi-line range stays on a single side. GitHub's review API rejects
            # mixed-side multi-line comments with an opaque 422; catching it here
            # makes the failure local to model construction.
            if self.start_side != self.side:
                raise ValueError(
                    f"GitHubCommentLocation: start_side ({self.start_side!r}) must "
                    f"equal side ({self.side!r}) for multi-line comments"
                )
        return self


def tree_sitter_to_github(
    *,
    file_path: str,
    byte_start: int,
    byte_end: int,
    head_content: str,
    patch: str,
) -> GitHubCommentLocation:
    """Convert a tree-sitter byte span to a GitHub review comment location.

    Returns the line number and `side="RIGHT"` for posting (V1 produces
    head-side comments only; LEFT-side commenting on deleted base-only
    content is out of V1 scope per the §5.6 signature, which accepts only
    `head_content`).

    Raises `CoordinateError` if the span does not map to a reviewable line
    (e.g., the span is in unchanged code outside any hunk, the file is not
    present in the patch, or the byte offsets are out of bounds).

    Per docs/spec.md §5.6 — V1 returns single-line locations only;
    multi-line spans collapse to the line containing `byte_start`.

    `file_path` is validated via `validate_diff_path()` BEFORE any path
    reaches the GitHub comment API or is stored in the returned
    `GitHubCommentLocation`, per the `paths-validated-before-use`
    invariant (docs/spec.md §10.1, security-critical).
    """
    validated_path = validate_diff_path(file_path)
    head_bytes = head_content.encode("utf-8")
    _validate_byte_span(byte_start, byte_end, len(head_bytes))

    head_line = _byte_offset_to_line(head_bytes, byte_start)
    matched_file = _find_patched_file(patch, validated_path)

    for hunk in matched_file:
        # target_start / target_length give the 1-indexed head-side line range
        # (added + context lines; deletions don't count toward target_length).
        if hunk.target_start <= head_line < hunk.target_start + hunk.target_length:
            return GitHubCommentLocation(
                file_path=validated_path,
                line=head_line,
                side="RIGHT",
            )

    raise CoordinateError(
        f"head line {head_line} for file {validated_path!r} is not in any hunk's "
        f"reviewable range (span is in unchanged code within a diffed file)",
        kind=CoordinateErrorKind.UNCHANGED_REGION,
    )


def query_span_to_source_lines(
    *,
    byte_start: int,
    byte_end: int,
    head_content: str,
) -> tuple[int, int]:
    """Convert a `QueryMatchSpan` byte envelope to a 1-indexed INCLUSIVE source
    line range `(line_start, line_end)`.

    The OBSERVED-tier producer's bridge from `ast_facts.QueryMatchSpan` (byte
    offsets) to `ReviewFinding.line_start`/`line_end` (source lines), so byte→line
    math never inlines at the analyze call site (trust boundary #3).

    Half-open input: `byte_end` is EXCLUSIVE (one-past-the-last byte, matching
    tree-sitter and `QueryMatchSpan`), so `line_end` is the line of the last
    CONTENT byte (`byte_end - 1`); deriving it from `byte_end` directly would
    spill onto the next line when a span ends exactly at a `\\n`.

    Requires a NON-EMPTY span (`byte_start < byte_end`). `QueryMatchSpan` admits
    `byte_start == byte_end` (its validator rejects only `byte_end < byte_start`),
    but a zero-width match has no reviewable line range and would underflow
    `byte_end - 1`; it raises `CoordinateError(kind=BYTE_OFFSET_INVALID)`.
    Out-of-bounds offsets raise the same kind via `_validate_byte_span`.

    See `DECISIONS.md#047`.
    """
    if byte_end <= byte_start:
        raise CoordinateError(
            f"query span must be non-empty: byte_end {byte_end} must be > "
            f"byte_start {byte_start} (a zero-width OBSERVED match span has no "
            f"reviewable source-line range)",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )
    head_bytes = head_content.encode("utf-8")
    _validate_byte_span(byte_start, byte_end, len(head_bytes))
    line_start = _byte_offset_to_line(head_bytes, byte_start)
    line_end = _byte_offset_to_line(head_bytes, byte_end - 1)
    return (line_start, line_end)


# ----------------------------------------------------------------------------
# Internal helpers — package-private, no boundary surface
# ----------------------------------------------------------------------------


def _validate_byte_span(byte_start: int, byte_end: int, head_byte_length: int) -> None:
    """Reject out-of-bounds or inverted byte spans with a CoordinateError.

    Half-open interval semantics matching tree-sitter's
    `Node.start_byte` / `Node.end_byte`:
    - `byte_start ∈ [0, head_byte_length)` — start is inclusive on the
      first byte; `byte_start == head_byte_length` is "starts at EOF"
      and rejected because there is no reviewable byte at that offset
      (would otherwise map to a ghost line past the last real line on
      newline-terminated files).
    - `byte_end ∈ [byte_start, head_byte_length]` — end is exclusive,
      so `byte_end == head_byte_length` is in-bounds (canonical
      one-past-the-last-byte for spans that run to EOF).

    Empty files (`head_byte_length == 0`) have no reviewable bytes;
    every span is rejected by the `byte_start >= head_byte_length` rule.
    """
    if byte_start < 0 or byte_start >= head_byte_length:
        raise CoordinateError(
            f"byte_start {byte_start} out of bounds for head_content ({head_byte_length} bytes)",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )
    if byte_end > head_byte_length:
        raise CoordinateError(
            f"byte_end {byte_end} out of bounds for head_content ({head_byte_length} bytes)",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )
    if byte_end < byte_start:
        raise CoordinateError(
            f"byte_end {byte_end} must be >= byte_start {byte_start} (half-open interval)",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )


def _byte_offset_to_line(head_bytes: bytes, byte_offset: int) -> int:
    """Return the 1-indexed line number containing `byte_offset` in `head_bytes`.

    Uses git-diff line semantics: a line is delimited by `\\n` (LF). `\\r\\n`
    is treated as `\\n` (the `\\r` is just data preceding the LF). Lone `\\r`
    mid-content is NOT a line terminator — git diff agrees, so this function
    aligns with `unidiff.Hunk.target_start`'s line numbering.

    UTF-8 byte offsets land on character boundaries per tree-sitter's invariant,
    so the byte slice is safe.
    """
    return head_bytes[:byte_offset].count(b"\n") + 1


def _line_to_byte_offset(head_bytes: bytes, line_number: int) -> int:
    """Inverse of `_byte_offset_to_line`: return the byte offset of the FIRST
    byte of line `line_number` (1-indexed) in `head_bytes`.

    Used by `source_line_to_github` to translate a `ReviewFinding`'s line
    coords to the byte coords the tree-sitter translation path consumes.

    Line 1 → byte 0. Line N (N ≥ 2) → byte just after the (N-1)th `\\n`.
    A `line_number` beyond the last line raises `CoordinateError(kind=
    BYTE_OFFSET_INVALID)` — the caller has a finding pointing past EOF,
    which is a producer-side bug (model hallucinated line number OR
    head_content drifted from the version the finding was anchored
    against).
    """
    if line_number < 1:
        raise CoordinateError(
            f"line_number {line_number} must be >= 1 (1-indexed source lines)",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )
    if line_number == 1:
        return 0
    # Find the position right after the (line_number - 1)th newline.
    newlines_needed = line_number - 1
    pos = 0
    for _ in range(newlines_needed):
        next_newline = head_bytes.find(b"\n", pos)
        if next_newline == -1:
            raise CoordinateError(
                f"line_number {line_number} exceeds source-line count "
                f"({head_bytes.count(b'\\n') + 1} lines in head_content)",
                kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
            )
        pos = next_newline + 1
    return pos


def source_line_to_github(
    *,
    file_path: str,
    line_start: int,
    line_end: int,
    head_content: str,
    patch: str,
) -> GitHubCommentLocation:
    """Source-line publisher entry point — line coords → GitHub comment location.

    The byte-based `tree_sitter_to_github` is the canonical translator
    (analyze produces tree-sitter byte spans). `ReviewFinding` carries
    `line_start` / `line_end` instead — the publish node uses this
    surface to bridge to GitHub's line-based comment API without
    inlining the line→byte math (which would violate
    `coordinates-module-is-sole-translator`).

    Translates the source line range to a byte span via
    `_line_to_byte_offset` and delegates to `tree_sitter_to_github`.
    Same `CoordinateError(kind=...)` taxonomy applies: out-of-bounds
    lines raise `BYTE_OFFSET_INVALID`; unchanged-region spans raise
    `UNCHANGED_REGION`; etc.

    V1 collapses multi-line findings to the line containing
    `line_start` per the existing `tree_sitter_to_github` semantics
    (`docs/spec.md` §5.6).
    """
    head_bytes = head_content.encode("utf-8")
    byte_start = _line_to_byte_offset(head_bytes, line_start)
    if line_end < line_start:
        raise CoordinateError(
            f"line_end {line_end} must be >= line_start {line_start}",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )
    # Validate `line_end` is REACHABLE AND has content in head_content
    # BEFORE the half-open byte_end lookup. The two-step check
    # distinguishes:
    #   (a) line_end IS the last content line → byte_end = end-of-buffer
    #   (b) line_end is PAST the last content line → raise
    #       BYTE_OFFSET_INVALID. An over-broad `except CoordinateError`
    #       here would silently truncate past-EOF findings to EOF and
    #       let them publish, so the explicit bounds check is required.
    #
    # The `byte_at_line_end >= len(head_bytes)` check additionally
    # rejects the "trailing empty line" case: a file ending with `\n`
    # has a structurally-reachable line N+1 at position end-of-buffer
    # with zero content, which is not legitimately commentable. The
    # naive `_line_to_byte_offset(line_end)` check alone would admit
    # this case because the position IS computable (= EOB); the
    # content-presence check rejects.
    byte_at_line_end = _line_to_byte_offset(head_bytes, line_end)
    if byte_at_line_end >= len(head_bytes):
        raise CoordinateError(
            f"line_end {line_end} has no content in head_content "
            f"({len(head_bytes)} bytes total); cannot anchor inline comment "
            f"past the last content line",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )
    # `line_end` is 1-indexed inclusive in `ReviewFinding`; the half-open
    # byte interval ends at the START of line_end + 1, OR at end-of-buffer
    # when line_end is the legitimate last line of the file.
    try:
        byte_end = _line_to_byte_offset(head_bytes, line_end + 1)
    except CoordinateError:
        # line_end is the LAST line of the file (the validation above
        # confirmed line_end itself is reachable, so this except can
        # only fire when line_end + 1 would be past EOF — the
        # legitimate end-of-buffer case).
        byte_end = len(head_bytes)
    return tree_sitter_to_github(
        file_path=file_path,
        byte_start=byte_start,
        byte_end=byte_end,
        head_content=head_content,
        patch=patch,
    )


def _find_patched_file(patch: str, file_path: str) -> PatchedFile:
    """Parse `patch` and find the `PatchedFile` matching `file_path`.

    Comparison uses `unidiff.PatchedFile.path` — the operation-dependent
    canonical path with `a/`/`b/` prefix stripped: additions,
    modifications, and renames return the **target** (head-side) path;
    deletions return the **source** path because the target is `/dev/null`.
    Normalized via `PurePosixPath(...).as_posix()` so surface forms like
    `./foo.py` and `foo.py` match — `validate_diff_path` applies the same
    normalization on its side, and the two halves must agree. Raises
    `CoordinateError` for malformed patches (any underlying `unidiff`
    exception is wrapped), for patches that don't contain `file_path`,
    and for patches that contain duplicate file entries with the same
    normalized path (webhook-attacker input per trust boundary #5;
    deterministic systems reject ambiguous routing input).

    Hunks-only normalization (per `vendor-payloads-normalized-at-boundary`):
    the raw `patch` string is run through `_wrap_github_hunks_with_headers`
    before `PatchSet` so GitHub's `/pulls/{n}/files` shape (hunks-only,
    no `--- a/...` headers) parses cleanly. Publish is `tree_sitter_to_github`'s
    first consumer; without this wrap, every call against the wire shape
    GitHub actually returns would raise `MALFORMED_PATCH`.
    """
    wrapped = _wrap_github_hunks_with_headers(patch, file_path)
    try:
        patchset = PatchSet(wrapped)
    except UnidiffParseError as e:
        raise CoordinateError(
            f"malformed patch input: {e}",
            kind=CoordinateErrorKind.MALFORMED_PATCH,
        ) from e

    matches = [pf for pf in patchset if PurePosixPath(pf.path).as_posix() == file_path]
    if len(matches) > 1:
        raise CoordinateError(
            f"patch contains {len(matches)} duplicate entries for {file_path!r}",
            kind=CoordinateErrorKind.DUPLICATE_FILE_ENTRY,
        )
    if not matches:
        raise CoordinateError(
            f"file_path {file_path!r} is not present in the patch ({len(patchset)} files in patch)",
            kind=CoordinateErrorKind.FILE_NOT_IN_PATCH,
        )
    return matches[0]
