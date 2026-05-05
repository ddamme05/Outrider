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

from outrider.coordinates.diff_parser import validate_diff_path
from outrider.coordinates.errors import CoordinateError

if TYPE_CHECKING:
    from unidiff.patch import PatchedFile


class GitHubCommentLocation(BaseModel):
    """Output of coordinate translation per docs/spec.md §7.2 line 1017.

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
        f"reviewable range (span is in unchanged code within a diffed file)"
    )


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
            f"byte_start {byte_start} out of bounds for head_content ({head_byte_length} bytes)"
        )
    if byte_end > head_byte_length:
        raise CoordinateError(
            f"byte_end {byte_end} out of bounds for head_content ({head_byte_length} bytes)"
        )
    if byte_end < byte_start:
        raise CoordinateError(
            f"byte_end {byte_end} must be >= byte_start {byte_start} (half-open interval)"
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
    """
    try:
        patchset = PatchSet(patch)
    except UnidiffParseError as e:
        raise CoordinateError(f"malformed patch input: {e}") from e

    matches = [pf for pf in patchset if PurePosixPath(pf.path).as_posix() == file_path]
    if len(matches) > 1:
        raise CoordinateError(f"patch contains {len(matches)} duplicate entries for {file_path!r}")
    if not matches:
        raise CoordinateError(
            f"file_path {file_path!r} is not present in the patch ({len(patchset)} files in patch)"
        )
    return matches[0]
