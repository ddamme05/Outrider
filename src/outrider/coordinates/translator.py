"""tree_sitter_to_github translator + GitHubCommentLocation + CoordinateError.

Per docs/spec.md §5.6 (the two §5.6 functions and the failure-mode contract)
and docs/spec.md §7.2 (the GitHubCommentLocation canonical shape).
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CoordinateError(Exception):
    """Raised when coordinate translation cannot produce a reviewable location.

    Single failure mode for the coordinates module per docs/spec.md §5.6.
    Coordinates' contract: any patch-parse failure, span-out-of-hunk, or
    span-out-of-bounds surfaces as `CoordinateError`, never as an underlying
    `unidiff` parse exception or `IndexError` leak.
    """


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
        # When both set, the multi-line range points upward to `line`.
        if self.start_line is not None and self.start_line > self.line:
            raise ValueError(
                f"GitHubCommentLocation: start_line ({self.start_line}) must "
                f"be <= line ({self.line})"
            )
        return self


def tree_sitter_to_github(
    file_path: str,
    byte_start: int,
    byte_end: int,
    head_content: str,
    patch: str,
) -> GitHubCommentLocation:
    """Convert a tree-sitter byte span to a GitHub review comment location.

    Returns the line number and side (LEFT/RIGHT) for posting.
    Raises CoordinateError if the span does not map to a reviewable line
    (e.g., the span is in unchanged code outside any hunk).

    Per docs/spec.md §5.6 — V1 returns single-line locations only;
    multi-line spans collapse to the line containing `byte_start`.
    """
    raise NotImplementedError(
        "tree_sitter_to_github lands in the next commit per the implementation sequence"
    )
