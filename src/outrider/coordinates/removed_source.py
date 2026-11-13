"""Validate removed BASE source before analysis or review-body placement."""

from __future__ import annotations

from typing import TYPE_CHECKING

from outrider.coordinates.diff_parser import _parse_and_match, validate_diff_path
from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind
from outrider.coordinates.spans import (
    line_range_to_span,
    source_line_count,
    span_is_nonempty,
    span_within_file,
)

if TYPE_CHECKING:
    from unidiff.patch import PatchedFile


def validate_removed_file(
    *,
    file_path: str,
    status: str,
    base_content: str | None,
    head_content: str | None,
    patch: str | None,
) -> PatchedFile:
    """Require a true removal whose complete patch reproduces the captured BASE.

    GitHub's hunks-only patch is supported. Partial/truncated removals, renames,
    retained HEAD content and source mismatches cannot establish BASE evidence.
    """
    path = validate_diff_path(file_path)

    def invalid() -> CoordinateError:
        return CoordinateError(
            f"file {path!r} lacks a complete removal patch matching its BASE source",
            kind=CoordinateErrorKind.MALFORMED_PATCH,
        )

    if status != "removed" or head_content is not None or not base_content or not patch:
        raise invalid()
    _, removed_file = _parse_and_match(patch, path)
    if removed_file is None:
        raise CoordinateError(
            f"removed file {path!r} is absent from its patch",
            kind=CoordinateErrorKind.FILE_NOT_IN_PATCH,
        )
    if removed_file.source_file not in {path, f"a/{path}"} or not removed_file:
        raise invalid()

    removed_lines: list[str] = []
    no_final_newline = False
    for hunk in removed_file:
        if (
            hunk.target_start != 0
            or hunk.target_length != 0
            or hunk.source_start != len(removed_lines) + 1
        ):
            raise invalid()
        for line in hunk:
            if line.is_removed:
                if no_final_newline or line.source_line_no != len(removed_lines) + 1:
                    raise invalid()
                removed_lines.append(line.value)
            elif (
                line.line_type == "\\"
                and line.value.strip() == "No newline at end of file"
                and removed_lines
                and not no_final_newline
            ):
                removed_lines[-1] = removed_lines[-1].removesuffix("\n")
                no_final_newline = True
            else:
                raise invalid()
    if "".join(removed_lines) != base_content:
        raise invalid()
    return removed_file


def removed_source_to_review_body(
    *,
    file_path: str,
    line_start: int,
    line_end: int,
    status: str,
    base_content: str | None,
    head_content: str | None,
    patch: str | None,
) -> str:
    """Validate a BASE range and return its canonical review-body path.

    This surface never creates a LEFT or RIGHT inline-comment location.
    """
    validate_removed_file(
        file_path=file_path,
        status=status,
        base_content=base_content,
        head_content=head_content,
        patch=patch,
    )
    if base_content is None:  # The validated removal requires BASE content.
        raise CoordinateError(
            "removed source has no BASE", kind=CoordinateErrorKind.MALFORMED_PATCH
        )
    span = line_range_to_span(line_start, line_end, base_content)
    if (
        line_end > source_line_count(base_content)
        or not span_is_nonempty(span)
        or not span_within_file(span, file_byte_length=len(base_content.encode("utf-8")))
    ):
        raise CoordinateError(
            "removed-source range is outside BASE content",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )
    return validate_diff_path(file_path)
