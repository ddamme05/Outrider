"""Check source ranges and prepare bounded patch text for analysis.

Byte ranges are half-open [byte_start, byte_end); source line ranges are
1-indexed and inclusive. All translation stays in coordinates/ and raw patch
lines remain private (coordinates-module-is-sole-translator).

See specs/2026-05-19-analyze-foundation.md §4 and
docs/trust-boundaries.md#3-coordinate-translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from outrider.ast_facts.models import (
    Span,  # noqa: TC001 — constructed at runtime by line_range_to_span
)
from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind

if TYPE_CHECKING:
    from unidiff import Hunk, PatchedFile
    from unidiff.patch import Line

    from outrider.ast_facts.models import ScopeUnit
    from outrider.ast_facts.parameterized_calls import ParameterizedCallScan


class ChangedLineSpan(BaseModel):
    """One changed line in one side's source: 1-indexed line number plus
    its whole-line byte `Span` (via `line_range_to_span`, half-open).

    The side (head vs base) is carried by which `ScopeChangedLineSpans`
    field holds the entry, not by this model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    line_no: int = Field(ge=1)
    span: Span


class ScopeChangedLineSpans(BaseModel):
    """Per-side changed-line data for one scope unit — the trivial-scope
    filter's veto input (see `changed_line_spans`).

    `head_added` lines verify against the head parse; `base_removed`
    lines verify against the base parse. Coordinates-owned domain shape:
    raw `unidiff.Line` objects never leave `coordinates/`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    head_added: tuple[ChangedLineSpan, ...]
    base_removed: tuple[ChangedLineSpan, ...]


def span_within_file(span: Span, file_byte_length: int) -> bool:
    """Check the file-size upper bound on a half-open byte span.

    `Span` validates non-negative, ordered offsets; this adds
    `byte_end <= file_byte_length`. A negative file length raises CoordinateError.
    Zero-width spans may pass. Degraded finding admission also requires
    `span_within_degraded_context` to anchor the span to changed content.
    """
    if file_byte_length < 0:
        raise CoordinateError(
            f"span_within_file: file_byte_length must be non-negative, got {file_byte_length}",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    return span.byte_end <= file_byte_length


def span_within_degraded_context(
    span: Span,
    addable_diff_byte_ranges: tuple[tuple[int, int], ...],
) -> bool:
    """Test non-empty overlap with at least one added-content byte range.

    Ranges are half-open: `[a, b)` intersects `[c, d)` iff `a < d and c < b`.
    Touching boundaries do not overlap. Empty spans return False, even inside
    an added range; ranges with `c >= d` are skipped. No ranges returns False.
    This anchors degraded JUDGED findings to added content rather than context.
    """
    if span.byte_start >= span.byte_end:
        return False
    return any(
        c < d and span.byte_start < d and c < span.byte_end for c, d in addable_diff_byte_ranges
    )


def line_range_within_scope_unit(line_start: int, line_end: int, scope_unit: ScopeUnit) -> bool:
    """An ordered range must fit the scope's inclusive line bounds.

    Line numbers are 1-indexed. Containment is
    `scope_unit.line_start <= line_start` and `line_end <= scope_unit.line_end`.
    Use line coordinates for model findings:
    whole-line byte spans start at column zero, before an indented scope's
    token-based byte_start. Byte containment would reject valid findings.
    See specs/2026-06-01-analyze-span-frame-mismatch.md.
    """
    return scope_unit.line_start <= line_start and line_end <= scope_unit.line_end


def line_range_vetoed_by_parameterized_call(
    line_start: int,
    line_end: int,
    scan: ParameterizedCallScan,
) -> bool:
    """Test whether an inclusive line range is safe to veto (FUP-162).

    The range must fit inside a safe parameterized call and overlap no unsafe
    call. Spanning both kinds keeps the finding eligible for HITL. Compare
    source lines, as in `line_range_within_scope_unit`, so indentation does
    not prevent the veto (FUP-126).
    See specs/2026-06-12-sqli-parameterized-call-veto.md.
    """
    contained_in_safe = any(
        site.line_start <= line_start and line_end <= site.line_end
        for site in scan.safe_parameterized_calls
    )
    if not contained_in_safe:
        return False
    # The scan preserves unsafe-call multiplicity, including two calls on one
    # line (FUP-170). Any unsafe overlap prevents the veto.
    return not any(
        # Inclusive interval intersection with each unsafe site.
        not (line_end < site.line_start or site.line_end < line_start)
        for site in scan.unsafe_parameterized_calls
    )


def line_range_to_span(line_start: int, line_end: int, source: str) -> Span:
    """Map a 1-indexed inclusive line range to whole-line UTF-8 byte offsets.

    The half-open span starts at the first byte of line_start and ends at the
    next line's start or EOF, including line_end's newline when present.
    A trailing newline creates a final empty line; selecting it alone returns
    a zero-width span at the byte length of source. Callers that forbid empty
    spans must check `span_is_nonempty`.

    Raise CoordinateError for line_start < 1, reversed ranges, or line_end
    past the source's line count; never clamp a proposed finding past EOF.
    Used for proposal hashes and degraded file bounds (DECISIONS.md#022).
    """
    if line_start < 1 or line_end < line_start:
        raise CoordinateError(
            f"line_range_to_span: invalid line range ({line_start}, {line_end})",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    source_bytes = source.encode("utf-8")
    # Each newline starts another line, including an empty final line at EOF.
    line_starts = [0]
    for i, byte in enumerate(source_bytes):
        if byte == 0x0A:  # '\n'
            line_starts.append(i + 1)
    total_lines = len(line_starts)
    if line_end > total_lines:
        raise CoordinateError(
            f"line_range_to_span: line_end={line_end} exceeds source line count {total_lines}",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )
    byte_start = line_starts[line_start - 1]
    byte_end = line_starts[line_end] if line_end < total_lines else len(source_bytes)
    return Span(byte_start=byte_start, byte_end=byte_end)


def scope_unit_has_added_lines(scope_unit: ScopeUnit, patched_file: PatchedFile) -> bool:
    """Test for an added target line inside the scope's inclusive line bounds.

    Clipped hunks may contain only context, so their presence alone must not
    pull an unchanged neighboring scope into the prompt. Pure deletion edits
    are handled separately by `scope_unit_has_changed_lines`.
    """
    return any(
        line.is_added
        and line.target_line_no is not None
        and scope_unit.line_start <= line.target_line_no <= scope_unit.line_end
        for hunk in patched_file
        for line in hunk
    )


def scope_unit_has_changed_lines(
    scope_unit: ScopeUnit,
    patched_file: PatchedFile,
    *,
    base_scope_units: tuple[ScopeUnit, ...] = (),
) -> bool:
    """Include additions in the head scope or removals from its surviving base owner.

    Matching the surviving first line prevents a deletion immediately before a
    scope from making that unchanged neighbor appear edited.
    """
    if scope_unit_has_added_lines(scope_unit, patched_file):
        return True
    owner = _surviving_base_scope(scope_unit, patched_file, base_scope_units)
    if owner is None:
        return False
    return any(
        line.is_removed
        and line.source_line_no is not None
        and owner.line_start <= line.source_line_no <= owner.line_end
        for hunk in patched_file
        for line in hunk
    )


def _surviving_base_scope(
    scope_unit: ScopeUnit,
    patched_file: PatchedFile,
    base_scope_units: tuple[ScopeUnit, ...],
) -> ScopeUnit | None:
    source_line = _source_line_for_target(scope_unit.line_start, patched_file)
    if source_line is None:
        return None
    owners = [
        base
        for base in base_scope_units
        if base.file_path == scope_unit.file_path
        and base.kind == scope_unit.kind
        and base.qualified_name == scope_unit.qualified_name
        and base.line_start <= source_line <= base.line_end
    ]
    return owners[0] if len(owners) == 1 else None


def _source_line_for_target(target_line: int, patched_file: PatchedFile) -> int | None:
    """Map a surviving target line to its source line; additions have no source."""
    offset = 0
    for hunk in patched_file:
        # Empty ranges name the line before the change.
        first_target = hunk.target_start if hunk.target_length else hunk.target_start + 1
        if target_line < first_target:
            return target_line + offset
        for line in hunk:
            if line.target_line_no == target_line:
                source_line: int | None = line.source_line_no
                return source_line
        source_after = hunk.source_start + max(hunk.source_length, 1)
        target_after = hunk.target_start + max(hunk.target_length, 1)
        offset = source_after - target_after
    return target_line + offset


def unscoped_changed_line_ranges(
    patched_file: PatchedFile,
    included_scope_units: tuple[ScopeUnit, ...],
    *,
    base_scope_units: tuple[ScopeUnit, ...] = (),
    head_line_count: int,
    context_lines: int = 2,
) -> tuple[tuple[int, int], ...]:
    """Bound changed head regions not already owned by selected scopes.

    Removed text anchors to the following surviving head line, or the last line
    at EOF. Empty heads have no current source range. Results are merged,
    inclusive 1-based ranges; source text is never invented for deleted lines.
    """
    if head_line_count < 0 or context_lines < 0:
        raise CoordinateError(
            "head line count and context lines must be non-negative",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    if head_line_count == 0:
        return ()
    base_owners = tuple(
        owner
        for scope in included_scope_units
        if (owner := _surviving_base_scope(scope, patched_file, base_scope_units)) is not None
    )
    changed_lines: set[int] = set()
    for hunk in patched_file:
        target_cursor = hunk.target_start if hunk.target_length else hunk.target_start + 1
        for line in hunk:
            target_line = line.target_line_no
            if target_line is not None:
                if line.is_added and not 1 <= target_line <= head_line_count:
                    raise CoordinateError(
                        f"added line {target_line} exceeds head line count {head_line_count}",
                        kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
                    )
                if line.is_added and not any(
                    scope.line_start <= target_line <= scope.line_end
                    for scope in included_scope_units
                ):
                    changed_lines.add(target_line)
                target_cursor = target_line + 1
                continue
            if not line.is_removed:
                continue
            if any(
                line.source_line_no is not None
                and owner.line_start <= line.source_line_no <= owner.line_end
                for owner in base_owners
            ):
                continue
            if any(
                scope.line_start < target_cursor <= scope.line_end for scope in included_scope_units
            ):
                continue
            changed_lines.add(min(max(target_cursor, 1), head_line_count))
    ranges: list[tuple[int, int]] = []
    for line_no in sorted(changed_lines):
        start = max(1, line_no - context_lines)
        end = min(head_line_count, line_no + context_lines)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return tuple(ranges)


def span_is_nonempty(span: Span) -> bool:
    """Test byte_start < byte_end; zero-width spans cannot anchor a finding.

    Span permits empty ranges, so finding admission needs this separate gate.
    """
    return span.byte_start < span.byte_end


def patched_file_has_added_lines(patched_file: PatchedFile) -> bool:
    """Test whether any hunk has added text for a degraded analysis call.

    This handles files without a recovered ScopeUnit while keeping raw
    unidiff access inside coordinates/.
    """
    return any(line.is_added for hunk in patched_file for line in hunk)


def patched_file_has_removed_lines(patched_file: PatchedFile) -> bool:
    """Test for removed text before the trivial-scope filter runs.

    Removed lines without base content prevent any scope from being classified
    trivial; raw unidiff access stays inside coordinates/.
    """
    return any(line.is_removed for hunk in patched_file for line in hunk)


def source_line_count(source: str) -> int:
    """Count physical LF-delimited lines without inventing a trailing EOF line."""
    return source.count("\n") + int(bool(source) and not source.endswith("\n"))


def added_line_byte_ranges(patched_file: PatchedFile, source: str) -> tuple[tuple[int, int], ...]:
    """Return half-open whole-line byte ranges for added lines in head source.

    Merge contiguous target lines into runs. Degraded JUDGED spans must overlap
    these ranges; module-scope OBSERVED matches must fit wholly inside one
    (DECISIONS.md#062). Both use the same deterministic diff anchor.

    Pure deletions have no target line and produce no range (FUP-050).
    Every target line must exist physically in HEAD, including the end of a
    contiguous added run. The empty position after a final newline is not source.
    """
    added = sorted(added_line_numbers(patched_file))
    if not added:
        return ()
    head_lines = source_line_count(source)
    if any(not 1 <= line <= head_lines for line in added):
        raise CoordinateError(
            f"added target lines exceed physical HEAD line bounds (1-{head_lines})",
            kind=CoordinateErrorKind.BYTE_OFFSET_INVALID,
        )
    ranges: list[tuple[int, int]] = []
    run_start = run_end = added[0]
    for line_no in added[1:]:
        if line_no == run_end + 1:
            run_end = line_no
            continue
        span = line_range_to_span(run_start, run_end, source)
        ranges.append((span.byte_start, span.byte_end))
        run_start = run_end = line_no
    span = line_range_to_span(run_start, run_end, source)
    ranges.append((span.byte_start, span.byte_end))
    return tuple(ranges)


def added_line_numbers(patched_file: PatchedFile) -> frozenset[int]:
    """Return 1-indexed added target lines in the head file.

    Used for line-space intersections such as parser error lines
    (DECISIONS.md#033). Pure deletions contribute no target lines (FUP-050).
    """
    return frozenset(
        line.target_line_no
        for hunk in patched_file
        for line in hunk
        if line.is_added and line.target_line_no is not None
    )


def extract_scope_unit_body(scope_unit: ScopeUnit, source_bytes: bytes) -> str:
    """Decode the scope's byte range as UTF-8.

    Parser adapters supply character-boundary offsets. Replacement decoding
    keeps an invalid producer offset from crashing prompt assembly; malformed
    bytes become U+FFFD.
    """
    return source_bytes[scope_unit.byte_start : scope_unit.byte_end].decode(
        "utf-8", errors="replace"
    )


def scope_unit_body_owners(selected_scope_units: tuple[ScopeUnit, ...]) -> tuple[int, ...]:
    """Assign each selected scope a complete rendered body, in original order.

    Sort containing starts before children; keep the rendered body reaching furthest.
    Partial overlaps retain both bodies. Equal byte spans use the first selected unit.
    """
    owners = list(range(len(selected_scope_units)))
    ordered = sorted(
        owners,
        key=lambda index: (
            selected_scope_units[index].file_path,
            selected_scope_units[index].byte_start,
            -selected_scope_units[index].byte_end,
            index,
        ),
    )
    owner_index: int | None = None
    for index in ordered:
        scope = selected_scope_units[index]
        if owner_index is None:
            owner_index = index
            continue
        owner = selected_scope_units[owner_index]
        if scope.file_path != owner.file_path or scope.byte_end > owner.byte_end:
            owner_index = index
            continue
        owners[index] = owner_index
    return tuple(owners)


def scope_unit_source_span(
    scope_unit: ScopeUnit, source_bytes: bytes, *, include_trailing_newline: bool = False
) -> Span:
    """Return the body span, optionally including an actual following rendered LF."""
    end = scope_unit.byte_end
    if (
        include_trailing_newline
        and end < len(source_bytes)
        and source_bytes[end : end + 1] == b"\n"
    ):
        end += 1
    return Span(byte_start=scope_unit.byte_start, byte_end=end)


def extract_line_range_body(line_start: int, line_end: int, source: str) -> str:
    """Extract complete UTF-8 source lines after validating their inclusive bounds."""
    span = line_range_to_span(line_start, line_end, source)
    return source.encode("utf-8")[span.byte_start : span.byte_end].decode("utf-8")


def exact_quote_overlaps_span(
    quote: str,
    source: str,
    *,
    required_overlap: Span,
    visible_byte_ranges: tuple[tuple[int, int], ...],
) -> bool:
    """Find an exact visible source quote that overlaps a declared source span."""
    source_bytes, quote_bytes = source.encode("utf-8"), quote.encode("utf-8")
    if not quote_bytes or not 0 <= required_overlap.byte_start < required_overlap.byte_end <= len(
        source_bytes
    ):
        return False
    visible = tuple(
        (start, end) for start, end in visible_byte_ranges if 0 <= start < end <= len(source_bytes)
    )
    start = max(0, required_overlap.byte_start - len(quote_bytes) + 1)
    stop = min(len(source_bytes), required_overlap.byte_end + len(quote_bytes) - 1)
    position = source_bytes.find(quote_bytes, start, stop)
    while position >= 0:
        if any(low <= position and position + len(quote_bytes) <= high for low, high in visible):
            return True
        position = source_bytes.find(quote_bytes, position + 1, stop)
    return False


@dataclass(frozen=True, slots=True)
class BoundedDiffContext:
    """Bounded rendered diff and the actual HEAD bytes represented within it."""

    text: str
    head_byte_ranges: tuple[tuple[int, int], ...]


def bound_diff_hunks_text(
    patched_file: PatchedFile,
    *,
    max_lines: int,
    max_chars: int,
) -> str:
    """Concatenate hunk body lines within caller-supplied prompt caps.

    max_lines counts unidiff Line objects, including added, removed, and context
    lines; max_chars bounds the returned string, including any truncation marker.
    Negative caps raise CoordinateError. Either cap stops assembly: one bounds
    many short lines, the other bounds a few long lines.

    The analyze layer sets degraded prompt limits (100 lines, 8192 characters);
    coordinates/ accepts the limits without choosing policy values.
    """
    return _render_bounded_diff(patched_file, max_lines=max_lines, max_chars=max_chars)[0]


def bound_diff_context(
    patched_file: PatchedFile, source: str, *, max_lines: int, max_chars: int
) -> BoundedDiffContext:
    """Render the same bounded diff while recording shown HEAD prefixes only.

    Removed lines and the truncation marker are not current source. A shown
    prefix counts only when it exactly matches that physical HEAD line.
    """
    text, shown_lines = _render_bounded_diff(patched_file, max_lines=max_lines, max_chars=max_chars)
    ranges: list[tuple[int, int]] = []
    for line, rendered in shown_lines:
        if line.target_line_no is None or not (line.is_added or line.is_context):
            continue
        prefix = rendered[1:]
        if not prefix:
            continue
        try:
            span = line_range_to_span(line.target_line_no, line.target_line_no, source)
            body = extract_line_range_body(line.target_line_no, line.target_line_no, source)
        except CoordinateError:
            continue
        if body.startswith(prefix):
            shown = prefix
        elif prefix.startswith(body):
            # Diff framing can terminate a physical HEAD line that has no LF.
            shown = body
        else:
            continue
        if shown:
            ranges.append((span.byte_start, span.byte_start + len(shown.encode("utf-8"))))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(set(ranges)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return BoundedDiffContext(text=text, head_byte_ranges=tuple(merged))


def _render_bounded_diff(
    patched_file: PatchedFile, *, max_lines: int, max_chars: int
) -> tuple[str, tuple[tuple[Line, str], ...]]:
    if max_lines < 0:
        raise CoordinateError(
            f"bound_diff_hunks_text: max_lines must be non-negative, got {max_lines}",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    if max_chars < 0:
        raise CoordinateError(
            f"bound_diff_hunks_text: max_chars must be non-negative, got {max_chars}",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    truncation_sentinel = "\n[truncated: prompt budget cap reached]\n"
    # Content can use the full budget. On truncation, fit the marker into
    # remaining space, shortening it if necessary; never exceed max_chars.
    pieces: list[str] = []
    shown_lines: list[tuple[Line, str]] = []
    total_chars = 0
    line_count = 0

    def _append_sentinel_and_return() -> tuple[str, tuple[tuple[Line, str], ...]]:
        remaining = max_chars - total_chars
        if remaining > 0:
            sentinel = truncation_sentinel[:remaining]
            pieces.append(sentinel)
        return "".join(pieces), tuple(shown_lines)

    for hunk in patched_file:
        for line in hunk:
            if line_count >= max_lines:
                return _append_sentinel_and_return()
            line_text = str(line)
            if total_chars + len(line_text) > max_chars:
                # Prioritize the truncation marker over a partial overflowing
                # line so the prompt exposes truncation when room remains.
                remaining = max_chars - total_chars
                if remaining > 0:
                    marker_room = min(len(truncation_sentinel), remaining)
                    line_room = remaining - marker_room
                    if line_room > 0:
                        pieces.append(line_text[:line_room])
                        shown_lines.append((line, line_text[:line_room]))
                        total_chars += line_room
                return _append_sentinel_and_return()
            pieces.append(line_text)
            shown_lines.append((line, line_text))
            total_chars += len(line_text)
            line_count += 1
    return "".join(pieces), tuple(shown_lines)


def scope_unit_diff_hunks(
    scope_unit: ScopeUnit,
    patched_file: PatchedFile,
    *,
    base_scope_units: tuple[ScopeUnit, ...] = (),
) -> tuple[str, ...]:
    """Return one rendered string per hunk clipped to the scope's line range.

    Filter added and context lines by inclusive target bounds. Hunk overlap
    alone would leak changed lines outside the scope into the prompt. Removed
    lines ride along when either neighboring target line is in range, using
    `_clip_hunk_lines`' shared rule. Rewrite headers to match the clipped
    source/target counts. A known base owner also retains its removals when
    the hunk supplies no target-side context. Return () when no lines survive.
    """
    base_scope = _surviving_base_scope(scope_unit, patched_file, base_scope_units)
    removed_line_range = (
        (base_scope.line_start, base_scope.line_end) if base_scope is not None else None
    )
    surviving: list[str] = []
    for hunk in patched_file:
        clipped = _clip_hunk_to_line_range(
            hunk,
            line_start=scope_unit.line_start,
            line_end=scope_unit.line_end,
            removed_line_range=removed_line_range,
        )
        if clipped is not None:
            surviving.append(clipped)
    return tuple(surviving)


def line_range_diff_hunks(
    patched_file: PatchedFile,
    line_start: int,
    line_end: int,
    *,
    head_line_count: int,
) -> tuple[str, ...]:
    """Clip authentic hunks to a primary head range, retaining anchored removals."""
    if line_start < 1 or line_end < line_start or line_end > head_line_count:
        raise CoordinateError(
            f"invalid primary head range ({line_start}, {line_end}) for {head_line_count} lines",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    surviving: list[str] = []
    for hunk in patched_file:
        target_cursor = hunk.target_start if hunk.target_length else hunk.target_start + 1
        removed_lines: list[int] = []
        for line in hunk:
            if line.target_line_no is not None:
                target_cursor = line.target_line_no + 1
            elif line.is_removed and line.source_line_no is not None:
                anchor = min(max(target_cursor, 1), head_line_count)
                if line_start <= anchor <= line_end:
                    removed_lines.append(line.source_line_no)
        removed_line_range = (removed_lines[0], removed_lines[-1]) if removed_lines else None
        clipped = _clip_hunk_to_line_range(
            hunk,
            line_start=line_start,
            line_end=line_end,
            removed_line_range=removed_line_range,
        )
        if clipped is not None:
            surviving.append(clipped)
    return tuple(surviving)


def changed_line_spans(
    scope_unit: ScopeUnit,
    patched_file: PatchedFile,
    *,
    head_source: str,
    base_source: str | None,
    base_scope_units: tuple[ScopeUnit, ...] = (),
) -> ScopeChangedLineSpans:
    """Build per-side changed-line spans for the trivial-scope veto (DECISIONS.md#044).

    Head additions use target lines inside the scope and whole-line spans in
    head_source. Base removals use the exact kept-removed set from
    `_clip_hunk_lines`, matching the prompt's clipped hunks, including deletions
    that ride along outside the scope's base range. Base-range containment
    would miss content that the prompt exposes and the veto must examine.

    Kept removed lines without base_source raise CoordinateError as a misuse guard;
    the filter wiring must refuse trivial classification before calling when
    base content is absent.
    """
    added_nos = sorted(
        line.target_line_no
        for hunk in patched_file
        for line in hunk
        if line.is_added
        and line.target_line_no is not None
        and scope_unit.line_start <= line.target_line_no <= scope_unit.line_end
    )
    head_added = tuple(
        ChangedLineSpan(line_no=n, span=line_range_to_span(n, n, head_source)) for n in added_nos
    )

    base_scope = _surviving_base_scope(scope_unit, patched_file, base_scope_units)
    removed_line_range = (
        (base_scope.line_start, base_scope.line_end) if base_scope is not None else None
    )
    removed_nos: set[int] = set()
    for hunk in patched_file:
        kept = _clip_hunk_lines(
            hunk,
            line_start=scope_unit.line_start,
            line_end=scope_unit.line_end,
            removed_line_range=removed_line_range,
        )
        for line in kept:
            src_no = getattr(line, "source_line_no", None)
            if getattr(line, "is_removed", False) and src_no is not None:
                removed_nos.add(src_no)

    if removed_nos and base_source is None:
        raise CoordinateError(
            "changed_line_spans: kept-removed lines exist but base_source is None",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    base_removed: tuple[ChangedLineSpan, ...] = ()
    if base_source is not None:
        base_removed = tuple(
            ChangedLineSpan(line_no=n, span=line_range_to_span(n, n, base_source))
            for n in sorted(removed_nos)
        )
    return ScopeChangedLineSpans(head_added=head_added, base_removed=base_removed)


def _clip_hunk_lines(
    hunk: Hunk,
    *,
    line_start: int,
    line_end: int,
    removed_line_range: tuple[int, int] | None = None,
) -> list[Line]:
    """Select raw lines for an inclusive target-side range.

    Added and context lines need an in-range target number. Removed lines ride
    along if the preceding or following target neighbor is kept. Rendering and
    `changed_line_spans` share this selection so the veto sees every removal
    shown by the clipped hunks. Raw unidiff lines stay inside coordinates/.
    """
    kept: list[Line] = []
    pending_removed: list[Line] = []
    any_kept_in_block = False
    for line in hunk:
        target_no = line.target_line_no
        if target_no is not None:
            in_range = line_start <= target_no <= line_end
            if in_range:
                kept.extend(pending_removed)
                pending_removed = []
                kept.append(line)
                any_kept_in_block = True
            else:
                # Pending removals may still ride with the previous kept neighbor.
                if any_kept_in_block:
                    kept.extend(pending_removed)
                pending_removed = []
                any_kept_in_block = False
        else:
            # Buffer removals until a neighboring target line decides their fate.
            pending_removed.append(line)
    # Trailing removals ride with the last target neighbor if it was kept.
    if any_kept_in_block:
        kept.extend(pending_removed)
    if removed_line_range is not None:
        kept_ids = {id(line) for line in kept}
        source_start, source_end = removed_line_range
        kept = [
            line
            for line in hunk
            if id(line) in kept_ids
            or (
                line.is_removed
                and line.source_line_no is not None
                and source_start <= line.source_line_no <= source_end
            )
        ]
    return kept


def _clip_hunk_to_line_range(
    hunk: Hunk,
    *,
    line_start: int,
    line_end: int,
    removed_line_range: tuple[int, int] | None = None,
) -> str | None:
    """Render the selected lines with corrected source/target header counts.

    Use `_clip_hunk_lines` for selection; return None when no lines survive.
    """
    hunk_lines: list[Line] = list(hunk)
    kept = _clip_hunk_lines(
        hunk,
        line_start=line_start,
        line_end=line_end,
        removed_line_range=removed_line_range,
    )

    if not kept:
        return None

    # Recompute header counts. Source side counts removed + context;
    # target side counts added + context.
    source_lines: list[Line] = []
    target_lines: list[Line] = []
    for line in kept:
        if line.is_removed or line.is_context:
            source_lines.append(line)
        if line.is_added or line.is_context:
            target_lines.append(line)
    if target_lines:
        target_start = target_lines[0].target_line_no
    else:
        target_start = (
            hunk.target_start if hunk.target_length == 0 else max(0, hunk.target_start - 1)
        )
        for original_line in hunk_lines:
            if original_line is kept[0]:
                break
            if original_line.target_line_no is not None:
                target_start = original_line.target_line_no
    source_count = len(source_lines)
    target_count = len(target_lines)

    if source_lines:
        source_start = source_lines[0].source_line_no
    else:
        # With no kept source lines, use the preceding source line as the
        # insertion anchor rather than defaulting to the new-file header shape.
        # A parent with source_length == 0 already supplies source_start as its
        # anchor; otherwise initialize max(0, source_start - 1). Walk its prefix
        # to the last source line before the first kept addition.
        hunk_source_length = getattr(hunk, "source_length", 0)
        hunk_source_start = getattr(hunk, "source_start", 0)
        if hunk_source_length == 0:
            last_source_line = hunk_source_start
        else:
            last_source_line = max(0, hunk_source_start - 1)
        first_kept_added = kept[0]
        for original_line in hunk_lines:
            if original_line is first_kept_added:
                break
            source_line_number = getattr(original_line, "source_line_no", None)
            if source_line_number is not None:
                last_source_line = source_line_number
        source_start = last_source_line

    header = f"@@ -{source_start},{source_count} +{target_start},{target_count} @@"
    section_header = getattr(hunk, "section_header", "")
    if section_header:
        header = f"{header} {section_header}"
    body = "".join(str(kept_line) for kept_line in kept)
    # str(Line) typically includes the trailing newline; header gets one too.
    return f"{header}\n{body}"
