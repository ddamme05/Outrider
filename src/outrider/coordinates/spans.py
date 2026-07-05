# See specs/2026-05-19-analyze-foundation.md §4.
"""Span-based coordinate helpers.

Per the `coordinates-module-is-sole-translator` invariant: all
translation among diff coordinates, tree-sitter byte spans, GitHub
comment locations, and source line numbers lives in the `coordinates/`
module. The helpers below cover span-containment, byte-to-line
translation, scope-unit-bounded diff slicing, and added-line detection
for the analyze sister spec's parser + node body. See the public
exports + each helper's docstring for the full surface.

Interval semantics: all `Span` instances are half-open
`[byte_start, byte_end)` — `byte_end` is exclusive (a 4-byte span at
offset 0 has `byte_start=0, byte_end=4`, covering bytes 0/1/2/3). Same
convention applies to `addable_diff_byte_ranges` tuples consumed by
`span_within_degraded_context`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from outrider.ast_facts.models import (
    Span,  # noqa: TC001 — constructed at runtime by line_range_to_span
)
from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind

if TYPE_CHECKING:
    from unidiff import PatchedFile

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
            f"span_within_file: file_byte_length must be non-negative, got {file_byte_length}",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
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


def line_range_within_scope_unit(line_start: int, line_end: int, scope_unit: ScopeUnit) -> bool:
    """True iff the 1-indexed inclusive line range is contained in `scope_unit`.

    Contained iff `scope_unit.line_start <= line_start AND line_end <=
    scope_unit.line_end` (inclusive both ends).

    The analyze parser's admission gate for findings the model anchors by source
    LINE number. Line space — not byte space — is the correct containment frame:
    the model is shown each scope
    unit's line range (and the diff `@@` line numbers), and `ScopeUnit` carries
    `line_start`/`line_end`. A whole-line byte span starts at column 0, BEFORE a
    nested/indented scope unit's token-based `byte_start`, so byte containment
    spuriously rejects valid in-scope findings — the silent findings-loss bug.
    See `specs/2026-06-01-analyze-span-frame-mismatch.md`.
    """
    return scope_unit.line_start <= line_start and line_end <= scope_unit.line_end


def line_range_vetoed_by_parameterized_call(
    line_start: int,
    line_end: int,
    scan: ParameterizedCallScan,
) -> bool:
    """True iff a finding's claimed 1-indexed inclusive line range lands on a
    provably-parameterized execute call (specs/2026-06-12-sqli-parameterized-
    call-veto.md, FUP-162).

    Vetoed iff BOTH: (a) the range is contained in some
    `safe_parameterized_calls` site (inclusive both ends, the
    `line_range_within_scope_unit` semantics), AND (b) it intersects no
    `unsafe_parameterized_calls` site (the materialized multiset `all − safe`,
    FUP-170) — a range spanning a safe and an unsafe call is NOT vetoed and
    flows through to HITL.

    Line space — not byte space — is the comparison frame, for the same
    reason as `line_range_within_scope_unit` above: a whole-line byte span
    starts at column 0, before an indented call node's token-based byte
    start, so byte containment would silently never veto indented
    `cursor.execute(...)` calls (the FUP-126 frame-mismatch class).
    """
    contained_in_safe = any(
        site.line_start <= line_start and line_end <= site.line_end
        for site in scan.safe_parameterized_calls
    )
    if not contained_in_safe:
        return False
    # The unsafe set (multiset `all − safe`) is materialized at scan time as
    # `ParameterizedCallScan.unsafe_parameterized_calls` (FUP-170 — the multiset
    # discipline that handles two-calls-on-one-line now lives in the producer).
    # A range spanning a safe and an unsafe call is NOT vetoed, so it must
    # intersect no unsafe site.
    return not any(
        # Inclusive interval intersection with each unsafe site.
        not (line_end < site.line_start or site.line_end < line_start)
        for site in scan.unsafe_parameterized_calls
    )


def line_range_to_span(line_start: int, line_end: int, source: str) -> Span:
    """Convert a 1-indexed inclusive line range to a whole-line byte `Span`.

    `byte_start` is the first byte of `line_start`; `byte_end` (exclusive) is the
    first byte of the line after `line_end`, or end-of-source when `line_end` is
    the last line — so the span covers `line_start` through `line_end` inclusive
    (including `line_end`'s trailing newline when present).

    Edge case — the trailing empty line: a trailing `\\n` creates a phantom final
    empty line that shares the end-of-source `byte_end` of the line before it.
    When the range is that trailing empty line alone (`line_start == line_end ==
    total_lines`) the result is a zero-width `Span` (`byte_start == byte_end ==
    len(source)`); this function does NOT reject zero-width — callers that forbid
    it (the analyze degraded path) gate with `span_is_nonempty`.

    `source` is the full file text; offsets are UTF-8 bytes (`Span` is
    byte-based). Raises `CoordinateError` if `line_start < 1`, `line_end <
    line_start`, or `line_end` exceeds the source's line count — fail-loud per
    project convention; a model-proposed range past EOF must not silently
    clamp (the analyze parser maps that raise to a `span_outside_file`
    rejection). Lives in `coordinates/` per `coordinates-module-is-sole-translator`.
    See `DECISIONS.md#022` (span-key amendment 2026-06-01 — the proposal hash + the
    degraded file-bounds check fold the raw line range through this helper).
    """
    if line_start < 1 or line_end < line_start:
        raise CoordinateError(
            f"line_range_to_span: invalid line range ({line_start}, {line_end})",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    source_bytes = source.encode("utf-8")
    # line_starts[k-1] = byte offset where 1-indexed line k begins. Line 1 at
    # byte 0; line K begins just after the (K-1)th '\n'. A trailing '\n' yields
    # a final (empty) line whose start == len(source_bytes).
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


def span_is_nonempty(span: Span) -> bool:
    """True iff `span` covers at least one byte (`byte_start < byte_end`).

    `Span` admits `byte_end >= byte_start` at construction (half-open
    interval, zero-width spans are schema-legal because `ScopeUnit`-like
    consumers can have legitimate empty ranges). For finding admission
    the analyze parser requires a span that actually anchors to bytes —
    a zero-width finding doesn't point at any code. Lives here so the
    span-shape decision is owned by `coordinates/` per
    `coordinates-module-is-sole-translator`.
    """
    return span.byte_start < span.byte_end


def patched_file_has_added_lines(patched_file: PatchedFile) -> bool:
    """True iff any hunk in `patched_file` carries at least one added line.

    Sibling to `scope_unit_has_added_lines` for the case where the caller
    has a `PatchedFile` but no `ScopeUnit` to scope against (e.g., parse-
    failed files where the analyze node still wants to know "is there
    addable diff text worth a degraded LLM call"). Lives in `coordinates/`
    because reading `unidiff.Line` attributes is the boundary owner's job
    per `docs/trust-boundaries.md#3-coordinate-translation`.
    """
    return any(line.is_added for hunk in patched_file for line in hunk)


def patched_file_has_removed_lines(patched_file: PatchedFile) -> bool:
    """True iff any hunk in `patched_file` carries at least one removed line.

    Sibling of `patched_file_has_added_lines` for the trivial-scope
    filter's fail-closed pre-check (removed lines with no base content →
    classify nothing trivial). Lives in `coordinates/` because reading
    `unidiff.Line` attributes is the boundary owner's job per
    `docs/trust-boundaries.md#3-coordinate-translation`.
    """
    return any(line.is_removed for hunk in patched_file for line in hunk)


def added_line_byte_ranges(patched_file: PatchedFile, source: str) -> tuple[tuple[int, int], ...]:
    """Half-open `(byte_start, byte_end)` ranges in HEAD `source` covering every line
    the patch ADDS (target side).

    This is the producer for the `addable_diff_byte_ranges` that
    `span_within_degraded_context` consumes — the deterministic degraded context the
    degraded JUDGED admission gates against, so a degraded finding's span must overlap
    content the patch actually added/modified, not arbitrary in-file bytes the model
    fabricates from prompt context. The SAME ranges anchor the OBSERVED producer's
    module-scope admission arm (`analyze_observed._module_level_admits`,
    specs/2026-07-04-module-scope-admission-arm.md): a `module_scope_eligible`
    query's scope-disjoint match must sit fully inside one range — both consumers
    gate against one deterministic diff anchor. Reads `unidiff.Line` (`is_added` /
    `target_line_no`), the boundary owner's job per
    `coordinates-module-is-sole-translator`.

    Contiguous added target lines are merged into runs; each run maps to a whole-line
    byte span via `line_range_to_span`, so a span overlapping any added line intersects
    the returned ranges. Pure additions only — a pure deletion carries no
    `target_line_no` (the same V1 limitation as `scope_unit_has_added_lines`, FUP-050).
    `line_range_to_span` raises `CoordinateError` if a target line exceeds `source`;
    that is a patch/source-misalignment bug (the patch's target side IS `source`), left
    to fail loud per project convention.
    """
    added = sorted(
        line.target_line_no
        for hunk in patched_file
        for line in hunk
        if line.is_added and line.target_line_no is not None
    )
    if not added:
        return ()
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
    """The 1-indexed target (HEAD) line numbers the patch ADDS.

    Line-space sibling of `added_line_byte_ranges`, for callers that intersect added
    lines against another line set rather than a byte span — e.g. the degraded
    no-scope decision intersects these with `ParseResult.error_lines` (per
    DECISIONS.md#033). Reads `unidiff.Line` (`is_added` / `target_line_no`), the
    boundary owner's job per `coordinates-module-is-sole-translator`. Pure additions
    only — a pure deletion carries no `target_line_no` (FUP-050 limitation).
    """
    return frozenset(
        line.target_line_no
        for hunk in patched_file
        for line in hunk
        if line.is_added and line.target_line_no is not None
    )


def extract_scope_unit_body(scope_unit: ScopeUnit, source_bytes: bytes) -> str:
    """Decode the UTF-8 bytes of `scope_unit`'s byte range from `source_bytes`.

    The `parse_*` entry points (`parse_python`, `parse_javascript`,
    `parse_typescript`) guarantee `ScopeUnit.byte_start` / `byte_end`
    land on UTF-8 char boundaries (tree-sitter byte offsets respect the
    source encoding). `errors="replace"` is defense-in-depth: under the producer
    contract the decoded text is round-trip valid; if a future producer
    bug emits a non-boundary offset, the prompt sees U+FFFD rather than
    crashing the analyze pass mid-render.

    Lives in `coordinates/` because byte-span slicing on `ScopeUnit` is
    coordinate-translation surface per
    `coordinates-module-is-sole-translator`.
    """
    return source_bytes[scope_unit.byte_start : scope_unit.byte_end].decode(
        "utf-8", errors="replace"
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
    # `max_chars` is the hard cap on the returned string. Content uses
    # the full budget; the sentinel is fitted into whatever room is left
    # at truncation-time (truncated itself if necessary). When no
    # truncation fires, the sentinel doesn't appear at all and content
    # gets the full `max_chars`.
    pieces: list[str] = []
    total_chars = 0
    line_count = 0

    def _append_sentinel_and_return() -> str:
        remaining = max_chars - total_chars
        if remaining > 0:
            sentinel = truncation_sentinel[:remaining]
            pieces.append(sentinel)
        return "".join(pieces)

    for hunk in patched_file:
        for line in hunk:
            if line_count >= max_lines:
                return _append_sentinel_and_return()
            line_text = str(line)
            if total_chars + len(line_text) > max_chars:
                # Fit as much of the truncated line as possible while
                # leaving room for the full sentinel. If the remaining
                # budget is smaller than the sentinel, line content
                # yields to the sentinel — the marker IS the audit signal
                # that truncation happened, so preserving it beats one
                # more partial line of content.
                remaining = max_chars - total_chars
                if remaining > 0:
                    marker_room = min(len(truncation_sentinel), remaining)
                    line_room = remaining - marker_room
                    if line_room > 0:
                        pieces.append(line_text[:line_room])
                        total_chars += line_room
                return _append_sentinel_and_return()
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


def changed_line_spans(
    scope_unit: ScopeUnit,
    patched_file: PatchedFile,
    *,
    head_source: str,
    base_source: str | None,
) -> ScopeChangedLineSpans:
    """Per-side changed-line spans for `scope_unit` — the trivial-scope
    filter's veto input. See DECISIONS.md#044.

    HEAD side: added lines whose `target_line_no` falls in the scope's
    line range (the same rule that admits the scope via
    `scope_unit_has_added_lines`), each as a whole-line span in
    `head_source`.

    BASE side: the kept-removed set — removed lines that survive
    `_clip_hunk_lines` clipping to the scope's range, i.e. EXACTLY the
    deletions the prompt's clipped hunks show (ride-along semantics
    included). Per the trivial-scope-filter spec, base-range containment
    attribution is rejected: a deletion adjacent to the scope rides into
    the prompt while mapping outside the scope's base range, and
    anything the prompt would have shown the model must pass the veto.

    `base_source=None` with a non-empty kept-removed set raises
    `CoordinateError` — unreachable under the intake contract (modified
    files carry `content_base`; added files have no removed lines;
    removed files never pass admission), so this is a misuse guard, not
    a control path. The filter wiring fail-closes BEFORE calling when
    base content is genuinely absent.
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

    removed_nos: set[int] = set()
    for hunk in patched_file:
        kept = _clip_hunk_lines(
            hunk, line_start=scope_unit.line_start, line_end=scope_unit.line_end
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
    hunk: object,
    *,
    line_start: int,
    line_end: int,
) -> list[object]:
    """Structured core of hunk clipping: the kept `unidiff.Line` objects.

    Single source of truth for which lines survive clipping to the
    inclusive target-side range `[line_start, line_end]`. Removed lines
    have no target line number; they ride along with adjacent kept
    target context (kept iff the next kept-or-skippable target line is
    in range, OR the previous one was). Consumed by BOTH
    `_clip_hunk_to_line_range` (prompt rendering) and
    `changed_line_spans` (the trivial-scope veto), so the veto sees
    exactly the lines the prompt shows — one clipping decision, two
    consumers. Private: raw `unidiff.Line` objects never leave
    `coordinates/`.
    """
    # unidiff's Hunk is iterable over Line objects; each Line has
    # `is_added`, `is_removed`, `is_context`, `source_line_no`,
    # `target_line_no`, and `value`.
    kept: list[object] = []
    pending_removed: list[object] = []
    any_kept_in_block = False
    for line in list(hunk):  # type: ignore[call-overload]
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
    return kept


def _clip_hunk_to_line_range(
    hunk: object,
    *,
    line_start: int,
    line_end: int,
) -> str | None:
    """Return a clipped hunk text with header recomputed, or None if empty.

    Line selection is `_clip_hunk_lines` (the shared structured core);
    this renderer rewrites the header `@@ -src_start,src_len
    +tgt_start,tgt_len @@` to match the surviving line counts and
    stringifies.
    """
    hunk_lines: list[object] = list(hunk)  # type: ignore[call-overload]
    kept = _clip_hunk_lines(hunk, line_start=line_start, line_end=line_end)

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
