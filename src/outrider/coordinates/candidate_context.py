"""Map a review claim to captured BASE evidence without guessing line offsets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from outrider.coordinates.diff_parser import lookup_patched_file, validate_diff_path
from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind
from outrider.coordinates.spans import line_range_diff_hunks, line_range_to_span, source_line_count

if TYPE_CHECKING:
    from unidiff import PatchedFile


@dataclass(frozen=True, slots=True)
class CandidateChangeContext:
    """Captured change evidence, not a verdict that a claim is a PR regression."""

    availability: Literal[
        "available", "added_file", "patch_unavailable", "invalid_patch", "source_mismatch"
    ]
    head_path: str
    base_path: str | None
    base_ranges: tuple[tuple[int, int], ...] = ()
    diff: str = ""
    diff_truncated: bool = False
    head_lines_added: bool | None = None
    base_lines_removed: bool | None = None


def candidate_change_context(
    *,
    file_path: str,
    previous_path: str | None,
    status: Literal["added", "modified", "removed", "renamed"],
    patch: str | None,
    base_source: str | None,
    head_source: str,
    line_start: int,
    line_end: int,
    max_diff_bytes: int,
) -> CandidateChangeContext:
    """Map HEAD lines and anchored removals only after both captures verify.

    Validate unchanged gaps as well as every patch line. A partial GitHub patch
    cannot silently turn an omitted modification into unchanged BASE evidence.
    Missing or inconsistent evidence yields an explicit unavailable result.
    """
    head_path = validate_diff_path(file_path)
    base_path = None if status == "added" else validate_diff_path(previous_path or file_path)
    line_range_to_span(line_start, line_end, head_source)
    if max_diff_bytes < 0:
        raise CoordinateError(
            "candidate change diff byte limit must be non-negative",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    unavailable = CandidateChangeContext(
        availability="patch_unavailable", head_path=head_path, base_path=base_path
    )
    if patch is None:
        return unavailable
    if base_source is None and status != "added":
        return replace(unavailable, availability="source_mismatch")
    try:
        parsed = lookup_patched_file(patch, head_path)
    except CoordinateError:
        return replace(unavailable, availability="invalid_patch")
    if parsed is None:
        if patch == "" and base_source == head_source:
            return replace(
                unavailable,
                availability="available",
                base_ranges=((line_start, line_end),),
                head_lines_added=False,
                base_lines_removed=False,
            )
        return replace(unavailable, availability="invalid_patch")
    verified = _mapped_base_lines(
        parsed, base_source or "", head_source, line_start=line_start, line_end=line_end
    )
    if verified is None:
        return replace(unavailable, availability="source_mismatch")
    base_lines, head_added, base_removed = verified
    hunks = line_range_diff_hunks(
        parsed, line_start, line_end, head_line_count=source_line_count(head_source)
    )
    diff, truncated = _bound_diff("".join(hunks), max_diff_bytes)
    return replace(
        unavailable,
        availability="added_file" if status == "added" else "available",
        base_ranges=_contiguous_ranges(base_lines),
        diff=diff,
        diff_truncated=truncated,
        head_lines_added=head_added,
        base_lines_removed=base_removed,
    )


def _physical_lines(source: str) -> list[str]:
    parts = source.split("\n")
    return [line + "\n" for line in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _mapped_base_lines(
    patch: PatchedFile, base: str, head: str, *, line_start: int, line_end: int
) -> tuple[list[int], bool, bool] | None:
    base_lines, head_lines = _physical_lines(base), _physical_lines(head)
    head_line_count = source_line_count(head)
    base_cursor = head_cursor = 1
    selected: list[int] = []
    head_added = base_removed = False

    def unchanged_gap(base_end: int, head_end: int) -> bool:
        if base_end < base_cursor or head_end < head_cursor:
            return False
        if base_lines[base_cursor - 1 : base_end - 1] != head_lines[head_cursor - 1 : head_end - 1]:
            return False
        selected.extend(
            base_cursor + number - head_cursor
            for number in range(max(line_start, head_cursor), min(line_end + 1, head_end))
        )
        return True

    for hunk in patch:
        base_start = hunk.source_start + (hunk.source_length == 0)
        head_start = hunk.target_start + (hunk.target_length == 0)
        if not unchanged_gap(base_start, head_start):
            return None
        base_cursor, head_cursor = base_start, head_start
        for line in hunk:
            if (line.is_context or line.is_removed) and (
                line.source_line_no != base_cursor
                or base_cursor > len(base_lines)
                or base_lines[base_cursor - 1].removesuffix("\n") != line.value.removesuffix("\n")
            ):
                return None
            if (line.is_context or line.is_added) and (
                line.target_line_no != head_cursor
                or head_cursor > len(head_lines)
                or head_lines[head_cursor - 1].removesuffix("\n") != line.value.removesuffix("\n")
            ):
                return None
            if line.is_context:
                if line_start <= head_cursor <= line_end:
                    selected.append(base_cursor)
                base_cursor += 1
                head_cursor += 1
            elif line.is_removed:
                anchor = min(max(head_cursor, 1), head_line_count)
                if line_start <= anchor <= line_end:
                    selected.append(base_cursor)
                    base_removed = True
                base_cursor += 1
            elif line.is_added:
                head_added |= line_start <= head_cursor <= line_end
                head_cursor += 1
        if (
            base_cursor != base_start + hunk.source_length
            or head_cursor != head_start + hunk.target_length
        ):
            return None
    if not unchanged_gap(len(base_lines) + 1, len(head_lines) + 1):
        return None
    return selected, head_added, base_removed


def _contiguous_ranges(lines: list[int]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for line in sorted(set(lines)):
        if ranges and line == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], line)
        else:
            ranges.append((line, line))
    return tuple(ranges)


def _bound_diff(text: str, maximum: int) -> tuple[str, bool]:
    pieces: list[str] = []
    size = 0
    for line in _physical_lines(text):
        size += len(line.encode())
        if size > maximum:
            return "".join(pieces), True
        pieces.append(line)
    return text, False
