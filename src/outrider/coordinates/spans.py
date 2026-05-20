# See specs/2026-05-19-analyze-foundation.md §4.
"""Span-based coordinate helpers.

Per the `coordinates-module-is-sole-translator` invariant: all
translation among diff coordinates, tree-sitter byte spans, GitHub
comment locations, and source line numbers lives in the `coordinates/`
module. The five functions below cover span-containment, byte-to-line
translation, and scope-unit-bounded diff slicing for the analyze
sister spec's parser + node body.

Interval semantics: all `Span` instances are half-open
`[byte_start, byte_end)` — `byte_end` is exclusive (a 4-byte span at
offset 0 has `byte_start=0, byte_end=4`, covering bytes 0/1/2/3). Same
convention applies to `addable_diff_byte_ranges` tuples consumed by
`span_within_degraded_context`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from outrider.coordinates.errors import CoordinateError

if TYPE_CHECKING:
    from unidiff import PatchedFile

    from outrider.ast_facts.models import ScopeUnit, Span


def span_within_scope_unit(span: Span, scope_unit: ScopeUnit) -> bool:
    """True iff `span` is contained within `scope_unit`'s byte range.

    Half-open containment: a `Span(s, e)` is contained in
    `ScopeUnit(byte_start=u_s, byte_end=u_e)` iff
    `u_s <= s` AND `e <= u_e`. An empty span `Span(a, a)` is contained
    in any scope unit whose range straddles `a`.

    Consumed by the sister analyze-implementation spec's parser to admit
    OBSERVED-tier and INFERRED-tier findings only when their spans land
    inside the scope unit they reference. Without this, the model could
    propose a finding whose span points outside any scope (e.g., the
    file header) and bypass the proof-boundary's structural-evidence
    requirement.
    """
    return scope_unit.byte_start <= span.byte_start and span.byte_end <= scope_unit.byte_end


def span_within_file(span: Span, file_byte_length: int) -> bool:
    """True iff `span.byte_end <= file_byte_length`.

    `Span` already enforces `byte_start >= 0` and `byte_start <= byte_end`
    at construction; this helper adds the file-size upper bound.

    Safety floor only — NOT sufficient for degraded-mode admission.
    Per the sister spec's degraded admission rule, a degraded-mode
    finding's span must ALSO intersect at least one addable diff hunk
    (see `span_within_degraded_context`). `span_within_file` admits
    the entire file's bytes including comments, license headers, and
    unchanged code; degraded findings restricted to that alone would
    let the model fabricate JUDGED-tier findings against arbitrary
    in-file bytes the patch references but doesn't touch.
    """
    if file_byte_length < 0:
        raise CoordinateError(
            f"span_within_file: file_byte_length must be non-negative, got {file_byte_length}"
        )
    return span.byte_end <= file_byte_length


def span_within_degraded_context(
    span: Span,
    addable_diff_byte_ranges: tuple[tuple[int, int], ...],
) -> bool:
    """True iff `span` intersects at least one of the addable diff hunks.

    Required for degraded-mode admission per post-split audit S8: a
    JUDGED-tier degraded finding's span MUST land within content the
    patch actually added/modified, not arbitrary in-file bytes the
    model fabricates from prompt-included context.

    Each entry in `addable_diff_byte_ranges` is a half-open
    `(byte_start, byte_end)` tuple matching `Span`'s native semantics.
    Intersection: span `[a, b)` intersects range `[c, d)` iff
    `a < d AND c < b`. Boundary-touch behavior is documented in the
    spec §4: a span ending exactly at a range's start (or starting at
    its end) does NOT intersect — touching only the boundary is not
    overlap.

    Empty `addable_diff_byte_ranges` returns False (no degraded
    context to anchor against).
    """
    # Half-open intersection: [a, b) intersects [c, d) iff a < d AND c < b.
    return any(span.byte_start < d and c < span.byte_end for c, d in addable_diff_byte_ranges)


def span_to_line_range(span: Span, source: str) -> tuple[int, int]:
    """Convert a byte `Span` to a 1-indexed `(line_start, line_end)` range.

    `source` is the full source text as a UTF-8 `str`. Both line numbers
    are 1-indexed (matching GitHub comment + `ReviewFinding.line_start`
    conventions); `line_end` is the line containing the LAST byte of the
    span, NOT exclusive (so a span covering only line 10 returns
    `(10, 10)`, not `(10, 11)`).

    For a half-open empty span `Span(a, a)`, both ends return the line
    containing byte `a` (the position just before the empty span). This
    matches how text editors render zero-width cursors.

    Raises `CoordinateError` if the span points past end-of-source —
    fail-loud per project convention. Silent round-up would mask model
    fabrications.

    Important: line counting walks bytes via `source.encode("utf-8")`
    not characters, because `Span` measures in bytes. A `str` index would
    miscount multibyte characters.
    """
    source_bytes = source.encode("utf-8")
    if span.byte_end > len(source_bytes):
        raise CoordinateError(
            f"span_to_line_range: span byte_end={span.byte_end} exceeds "
            f"source length {len(source_bytes)} bytes"
        )

    # Newline at byte position N means the NEXT byte starts a new line.
    # Line 1 starts at byte 0; line K starts at the byte after the
    # (K-1)th \n.
    def line_for_byte(b: int) -> int:
        # 1-indexed: byte 0 is on line 1.
        return source_bytes[:b].count(b"\n") + 1

    if span.byte_start == span.byte_end:
        # Empty span: the line containing the position just before it.
        # Treat as the line that would contain byte_start.
        line = line_for_byte(span.byte_start)
        return (line, line)

    line_start = line_for_byte(span.byte_start)
    # The last byte of the span is at byte_end - 1 (half-open).
    line_end = line_for_byte(span.byte_end - 1)
    return (line_start, line_end)


def scope_unit_diff_hunks(scope_unit: ScopeUnit, patched_file: PatchedFile) -> tuple[str, ...]:
    """Clip a unified-diff `PatchedFile`'s hunks to lines inside `scope_unit`.

    Returns the scope-unit-bounded hunk text as a tuple of strings
    (one per surviving hunk). Used by the sister analyze-implementation
    spec's node body to assemble the `user_prompt` content with
    scope-unit-bounded diff hunks rather than the full PR diff —
    keeps the model focused on changes inside the function it's
    reviewing.

    A hunk survives only if at least one of its target-side lines falls
    within `[scope_unit.line_start, scope_unit.line_end]` (inclusive,
    1-indexed). The returned text reproduces the unidiff format
    (`@@ ... @@` header + body) but only for surviving hunks; if a
    hunk straddles the scope boundary, its full body is kept (clipping
    INSIDE a hunk's body would corrupt the diff format).

    Empty tuple return means the scope unit has no overlapping diff
    changes — caller handles that case (e.g., skips the file).
    """
    surviving: list[str] = []
    for hunk in patched_file:
        # `hunk.target_start` is the first line of the hunk on the head
        # side (1-indexed); `hunk.target_length` is the number of lines.
        hunk_first_line = hunk.target_start
        hunk_last_line = hunk.target_start + hunk.target_length - 1
        # Inclusive overlap with scope range:
        overlaps = (
            hunk_first_line <= scope_unit.line_end and scope_unit.line_start <= hunk_last_line
        )
        if overlaps:
            surviving.append(str(hunk))
    return tuple(surviving)
