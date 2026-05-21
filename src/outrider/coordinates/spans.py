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

    Required for degraded-mode admission: a JUDGED-tier degraded
    finding's span MUST land within content the patch actually
    added/modified, not arbitrary in-file bytes the model fabricates
    from prompt-included context.

    Each entry in `addable_diff_byte_ranges` is a half-open
    `(byte_start, byte_end)` tuple matching `Span`'s native semantics.
    Intersection: span `[a, b)` intersects range `[c, d)` iff
    `a < d AND c < b`. Boundary-touch behavior is documented in the
    spec §4: a span ending exactly at a range's start (or starting at
    its end) does NOT intersect — touching only the boundary is not
    overlap.

    Empty `addable_diff_byte_ranges` returns False (no degraded
    context to anchor against). An empty `span` (`byte_start == byte_end`)
    also returns False: a zero-width span covers no bytes, so it cannot
    meaningfully anchor a JUDGED-tier finding to changed content. Without
    this gate, the half-open check `a < d AND c < a` would admit a
    zero-width span sitting strictly inside a changed range, which is
    exactly the fabrication the gate is supposed to refuse. Empty
    ranges in `addable_diff_byte_ranges` are likewise skipped: a range
    with `c >= d` carries no bytes.
    """
    if span.byte_start >= span.byte_end:
        return False
    # Half-open intersection: [a, b) intersects [c, d) iff a < d AND c < b.
    return any(
        c < d and span.byte_start < d and c < span.byte_end for c, d in addable_diff_byte_ranges
    )


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


def scope_unit_has_added_lines(scope_unit: ScopeUnit, patched_file: PatchedFile) -> bool:
    """True iff `patched_file` contains at least one added line whose
    target line number falls within `scope_unit`'s 1-indexed line range.

    `scope_unit_diff_hunks` clips by `target_line_no` and keeps context
    lines in range; a scope unit that contains ONLY context lines (no
    actual changes) yields a non-empty clipped result. Callers that
    drive changed-region intersection need the stronger "has added
    lines in range" check so a comment-line edit doesn't drag a
    neighboring untouched function into the prompt.

    V1 limitation: removed-only changes (pure deletion within an
    otherwise-unchanged function, no added text) are not detected by
    this rule. The common case (modifications + additions) is. A
    future head-side line mapping for pure deletions lives at this
    surface alongside `scope_unit_diff_hunks` because both depend on
    the same unidiff Line-attribute semantics.
    """
    return any(
        line.is_added
        and line.target_line_no is not None
        and scope_unit.line_start <= line.target_line_no <= scope_unit.line_end
        for hunk in patched_file
        for line in hunk
    )


def bound_diff_hunks_text(
    patched_file: PatchedFile,
    *,
    max_lines: int,
    max_chars: int,
) -> str:
    """Concatenate `patched_file`'s lines as text, truncated at the
    first cap hit.

    `max_lines` caps the total `unidiff.Line` object count (added +
    removed + context combined); `max_chars` caps the character total.
    Either cap closes the gate via early-return — the line cap
    prevents many-tiny-lines fan-out, the char cap prevents pathological
    few-very-long-lines blowup.

    Used by the analyze node body's degraded-mode prompt assembly,
    where the spec pins both caps (≤100 Line, ≤8192 chars). Caps are
    kwargs rather than module constants because the coordinates layer
    is policy-free — the analyze layer pins the cap values per its
    spec §7 step 3c.
    """
    truncation_sentinel = "\n[truncated: prompt budget cap reached]\n"
    # Reserve sentinel-sized headroom so the marker stays inside
    # max_chars when truncation fires. If max_chars is smaller than the
    # sentinel, the sentinel itself is truncated to fit; the returned
    # string never exceeds max_chars.
    sentinel_room = min(len(truncation_sentinel), max_chars)
    sentinel = truncation_sentinel[:sentinel_room]
    content_budget = max_chars - sentinel_room
    pieces: list[str] = []
    total_chars = 0
    line_count = 0
    for hunk in patched_file:
        for line in hunk:
            if line_count >= max_lines:
                pieces.append(sentinel)
                return "".join(pieces)
            line_text = str(line)
            if total_chars + len(line_text) > content_budget:
                # Truncate to remaining content budget (the sentinel space
                # was reserved above) rather than silently emit empty
                # bounded_hunks. A single >content_budget line at first
                # iteration would otherwise return "" with no audit signal
                # that the diff was lost, and the degraded LLM call would
                # see "no changes to review" instead of "diff was too big."
                remaining = content_budget - total_chars
                if remaining > 0:
                    pieces.append(line_text[:remaining])
                pieces.append(sentinel)
                return "".join(pieces)
            pieces.append(line_text)
            total_chars += len(line_text)
            line_count += 1
    return "".join(pieces)


def scope_unit_diff_hunks(scope_unit: ScopeUnit, patched_file: PatchedFile) -> tuple[str, ...]:
    """Clip a unified-diff `PatchedFile`'s hunks to lines inside `scope_unit`.

    Returns the scope-unit-bounded hunk text as a tuple of strings
    (one per surviving hunk). Used by the sister analyze-implementation
    spec's node body to assemble the `user_prompt` content with
    scope-unit-bounded diff hunks rather than the full PR diff —
    keeps the model focused on changes inside the function it's
    reviewing.

    A hunk's body lines are FILTERED to those whose target-side line
    number falls within `[scope_unit.line_start, scope_unit.line_end]`
    (inclusive, 1-indexed). The hunk header (`@@ -A,B +C,D @@`) is
    rewritten to reflect the clipped body's source/target line counts
    so the emitted text remains a valid unified diff. keeping the full overlapping hunk leaked
    changed lines outside the eligible scope unit into the analyze
    prompt — true scope-unit-bounded clipping requires this in-hunk
    filter, not just hunk-level overlap.

    Removed-side (`-` prefix) lines have no target line number; they
    are kept iff at least one neighboring target-side line in the same
    contiguous chunk falls in range. The simpler equivalent that this
    implementation uses: keep a removed line iff the next kept-or-
    skippable target line is in range, OR the previous one was — i.e.,
    they ride along with the surrounding target context. Context (` `
    prefix) lines have a target line number; they get the standard
    in-range filter.

    Empty tuple return means the scope unit has no overlapping diff
    changes — caller handles that case (e.g., skips the file).
    """
    surviving: list[str] = []
    for hunk in patched_file:
        clipped = _clip_hunk_to_line_range(
            hunk,
            line_start=scope_unit.line_start,
            line_end=scope_unit.line_end,
        )
        if clipped is not None:
            surviving.append(clipped)
    return tuple(surviving)


def _clip_hunk_to_line_range(
    hunk: object,
    *,
    line_start: int,
    line_end: int,
) -> str | None:
    """Return a clipped hunk text with header recomputed, or None if empty.

    Filters body lines by target-side line number. Removed lines carry
    along with adjacent in-range target lines: a removed line is
    included iff the surrounding context puts it inside the kept block
    (in practice: between two kept lines, or adjacent to a kept added/
    context line). Header `@@ -src_start,src_len +tgt_start,tgt_len @@`
    is rewritten to match the surviving line counts.
    """
    # unidiff's Hunk is iterable over Line objects; each Line has
    # `is_added`, `is_removed`, `is_context`, `source_line_no`,
    # `target_line_no`, and `value`.
    hunk_lines: list[object] = list(hunk)  # type: ignore[call-overload]

    # First pass: decide which lines to keep, walking forward. A removed
    # line is kept iff its immediate target neighbor (next kept added/
    # context line forward, or previous one backward) was kept.
    kept: list[object] = []
    pending_removed: list[object] = []
    any_kept_in_block = False
    for line in hunk_lines:
        target_no: int | None = getattr(line, "target_line_no", None)
        if target_no is not None:
            # Added or context line — has a target line number.
            in_range = line_start <= target_no <= line_end
            if in_range:
                # Flush pending removed-lines that rode along this block.
                kept.extend(pending_removed)
                pending_removed = []
                kept.append(line)
                any_kept_in_block = True
            else:
                # Out of range — flush pending removed-lines IFF any
                # earlier kept neighbor in this contiguous block exists
                # (they rode along that block's tail context).
                if any_kept_in_block:
                    kept.extend(pending_removed)
                pending_removed = []
                any_kept_in_block = False
        else:
            # Removed line — defer; ride with the next neighbor's verdict.
            pending_removed.append(line)
    # Trailing removed lines without a following target neighbor: include
    # iff the last contiguous block ended with kept lines.
    if any_kept_in_block:
        kept.extend(pending_removed)

    if not kept:
        return None

    # Recompute header counts. Source side counts removed + context;
    # target side counts added + context.
    src_lines = [
        ln for ln in kept if getattr(ln, "is_removed", False) or getattr(ln, "is_context", False)
    ]
    tgt_lines = [
        ln for ln in kept if getattr(ln, "is_added", False) or getattr(ln, "is_context", False)
    ]
    tgt_start = tgt_lines[0].target_line_no if tgt_lines else 0  # type: ignore[attr-defined]
    src_len = len(src_lines)
    tgt_len = len(tgt_lines)

    if src_lines:
        src_start = src_lines[0].source_line_no  # type: ignore[attr-defined]
    else:
        # Pure-added clipped subset: there are no source-side lines in
        # the kept set, so `src_lines[0]` isn't available. `src_start = 0`
        # is a forbidden coordinate for in-file insertions —
        # `@@ -0,0 +N,M @@` is the new-file shape, NOT the in-file
        # insertion shape. For an in-file insertion the unidiff
        # convention is `@@ -K,0 +N,M @@` where K is the source line
        # BEFORE which the insertion happens.
        #
        # Derive K by walking the ORIGINAL hunk forward up to the first
        # kept added line, tracking the most recent source_line_no
        # encountered. That value is the source line that immediately
        # precedes the insertion point.
        #
        # Init value: for a parent hunk with source-side lines
        # (`source_length > 0`), the line "before the hunk" is
        # `source_start - 1`. For a parent hunk that is itself a pure
        # insertion (`source_length == 0`), `hunk.source_start` already
        # names the insertion anchor (no -1 — there's no "first source
        # line in hunk" to subtract from).
        hunk_source_length = getattr(hunk, "source_length", 0)
        hunk_source_start = getattr(hunk, "source_start", 0)
        if hunk_source_length == 0:
            last_src_seen = hunk_source_start
        else:
            last_src_seen = max(0, hunk_source_start - 1)
        first_kept_added = kept[0]
        for original_line in hunk_lines:
            if original_line is first_kept_added:
                break
            src_no = getattr(original_line, "source_line_no", None)
            if src_no is not None:
                last_src_seen = src_no
        src_start = last_src_seen

    header = f"@@ -{src_start},{src_len} +{tgt_start},{tgt_len} @@"
    section_header = getattr(hunk, "section_header", "")
    if section_header:
        header = f"{header} {section_header}"
    body = "".join(str(ln) for ln in kept)
    # str(Line) typically includes the trailing newline; header gets one too.
    return f"{header}\n{body}"
