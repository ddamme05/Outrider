# See specs/2026-05-19-analyze-foundation.md §4.
"""Span-based coordinate helper tests.

Pins the half-open interval semantics (byte_end exclusive), the file-size
safety floor, the addable-diff intersection rule for degraded-mode admission,
the 1-indexed-line → whole-line-byte-span conversion (including multibyte
UTF-8 safety), and the scope-unit-bounded hunk clipping.
"""

from __future__ import annotations

import pytest
from unidiff import PatchSet

from outrider.ast_facts.models import ScopeUnit, Span
from outrider.coordinates import (
    CoordinateError,
    added_line_byte_ranges,
    added_line_numbers,
    bound_diff_hunks_text,
    changed_line_spans,
    extract_scope_unit_body,
    line_range_to_span,
    line_range_within_scope_unit,
    patched_file_has_added_lines,
    scope_unit_diff_hunks,
    scope_unit_has_added_lines,
    span_is_nonempty,
    span_within_degraded_context,
    span_within_file,
)


def _scope_unit(
    *, byte_start: int = 100, byte_end: int = 200, line_start: int = 10, line_end: int = 20
) -> ScopeUnit:
    return ScopeUnit(
        unit_id="a" * 64,
        kind="function",
        name="foo",
        qualified_name="module.foo",
        file_path="src/foo.py",
        line_start=line_start,
        line_end=line_end,
        byte_start=byte_start,
        byte_end=byte_end,
    )


# ---------------------------------------------------------------------------
# span_within_file
# ---------------------------------------------------------------------------


def test_span_is_nonempty_strict_inequality() -> None:
    """`span_is_nonempty` returns True iff `byte_start < byte_end`. The
    `Span` carrier admits `byte_end >= byte_start`; this helper carries
    the parser's stricter rule (a zero-width finding doesn't anchor to
    bytes) at the coordinates boundary.
    """
    assert span_is_nonempty(Span(byte_start=10, byte_end=11)) is True
    assert span_is_nonempty(Span(byte_start=10, byte_end=10)) is False
    assert span_is_nonempty(Span(byte_start=0, byte_end=0)) is False


def test_span_within_file_exact_end_admitted() -> None:
    assert span_within_file(Span(byte_start=0, byte_end=100), file_byte_length=100) is True


def test_span_within_file_past_end_rejected() -> None:
    assert span_within_file(Span(byte_start=0, byte_end=101), file_byte_length=100) is False


def test_span_within_file_rejects_negative_file_length() -> None:
    """Defensive: a negative file size is a caller bug."""
    with pytest.raises(CoordinateError, match="non-negative"):
        span_within_file(Span(byte_start=0, byte_end=10), file_byte_length=-1)


# ---------------------------------------------------------------------------
# span_within_degraded_context
# ---------------------------------------------------------------------------


def test_span_within_degraded_context_intersects_single_range() -> None:
    assert (
        span_within_degraded_context(
            Span(byte_start=50, byte_end=70),
            addable_diff_byte_ranges=((40, 80),),
        )
        is True
    )


def test_span_within_degraded_context_intersects_one_of_many() -> None:
    """Multiple addable hunks — span hits one."""
    ranges = ((10, 20), (50, 70), (200, 250))
    assert span_within_degraded_context(Span(byte_start=60, byte_end=65), ranges) is True


def test_span_within_degraded_context_no_intersection() -> None:
    """Span sits entirely in a gap between addable ranges."""
    ranges = ((10, 20), (50, 70))
    assert span_within_degraded_context(Span(byte_start=25, byte_end=40), ranges) is False


def test_span_within_degraded_context_empty_ranges() -> None:
    """No addable hunks → no degraded context to anchor against."""
    assert span_within_degraded_context(Span(byte_start=0, byte_end=100), ()) is False


def test_span_within_degraded_context_boundary_touch_right_excluded() -> None:
    """Half-open: span ending exactly at range start does NOT intersect.

    Span [10, 50) and range [50, 100): the boundary touches but
    overlap is empty. Per §4 locked semantics, this is rejected.
    """
    assert (
        span_within_degraded_context(
            Span(byte_start=10, byte_end=50),
            addable_diff_byte_ranges=((50, 100),),
        )
        is False
    )


def test_span_within_degraded_context_boundary_touch_left_excluded() -> None:
    """Span [50, 100) and range [10, 50): boundary touches but empty overlap."""
    assert (
        span_within_degraded_context(
            Span(byte_start=50, byte_end=100),
            addable_diff_byte_ranges=((10, 50),),
        )
        is False
    )


def test_span_within_degraded_context_span_fully_equal_range() -> None:
    """Span == range: same half-open interval, full overlap, admitted."""
    assert (
        span_within_degraded_context(
            Span(byte_start=10, byte_end=50),
            addable_diff_byte_ranges=((10, 50),),
        )
        is True
    )


def test_span_within_degraded_context_empty_span_rejected() -> None:
    """An empty span `[a, a)` covers zero bytes — it cannot anchor a
    JUDGED-tier finding to changed content. Even when `a` sits strictly
    inside an addable range, the gate rejects: a zero-width span the
    model fabricated to "land inside" the diff would otherwise slip
    through degraded-mode admission.
    """
    assert (
        span_within_degraded_context(
            Span(byte_start=20, byte_end=20),
            addable_diff_byte_ranges=((10, 50),),
        )
        is False
    )


def test_span_within_degraded_context_empty_range_skipped() -> None:
    """A range with `c >= d` carries no bytes; the gate skips it rather
    than admitting overlap by accident. Without this, a malformed
    `addable_diff_byte_ranges` tuple with `(50, 50)` would let a span
    at `byte_start=49, byte_end=51` slip through via `49 < 50 AND 50 < 51`.
    """
    # Span overlaps a real range, so we know rejection here is specifically
    # because the empty range is skipped, not because the span itself is empty.
    assert (
        span_within_degraded_context(
            Span(byte_start=49, byte_end=51),
            addable_diff_byte_ranges=((50, 50),),
        )
        is False
    )


# ---------------------------------------------------------------------------
# scope_unit_diff_hunks
# ---------------------------------------------------------------------------


_PATCH_TEXT = """\
diff --git a/src/foo.py b/src/foo.py
index abc..def 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
 def first():
     return 1
+    raise NotImplementedError

@@ -10,3 +11,4 @@
 def second():
     return 2
+    # added

"""


def _patched_file() -> object:
    """Build a unidiff PatchedFile from the canned _PATCH_TEXT."""
    patch = PatchSet.from_string(_PATCH_TEXT)
    return patch[0]


def test_scope_unit_diff_hunks_keeps_overlapping() -> None:
    """A scope unit covering target lines 1–5 keeps the first hunk only."""
    su = _scope_unit(line_start=1, line_end=5)
    hunks = scope_unit_diff_hunks(su, _patched_file())  # type: ignore[arg-type]
    assert len(hunks) == 1
    assert "first()" in hunks[0]


def test_scope_unit_diff_hunks_keeps_both_when_scope_spans_both() -> None:
    """A scope unit covering target lines 1–20 keeps both hunks."""
    su = _scope_unit(line_start=1, line_end=20)
    hunks = scope_unit_diff_hunks(su, _patched_file())  # type: ignore[arg-type]
    assert len(hunks) == 2


def test_scope_unit_diff_hunks_keeps_second_only() -> None:
    """A scope unit covering target lines 10–14 keeps the second hunk only."""
    su = _scope_unit(line_start=10, line_end=14)
    hunks = scope_unit_diff_hunks(su, _patched_file())  # type: ignore[arg-type]
    assert len(hunks) == 1
    assert "second()" in hunks[0]


def test_scope_unit_diff_hunks_empty_when_disjoint() -> None:
    """A scope unit covering target lines 100–110 has no overlapping hunks."""
    su = _scope_unit(line_start=100, line_end=110)
    hunks = scope_unit_diff_hunks(su, _patched_file())  # type: ignore[arg-type]
    assert hunks == ()


# ---------------------------------------------------------------------------
# Post-foundation push/PR review fold: pure-added clipped subset must NOT
# emit `@@ -0,0 +N,M @@` (the new-file shape) for an in-file insertion.
# Convention is `@@ -K,0 +N,M @@` where K names the source line BEFORE
# which the insertion happens.
# ---------------------------------------------------------------------------


# Patch with a hunk whose body mixes context + an inserted line. Scope-
# unit clipping that drops the context (because it's outside the scope
# unit's line range) and keeps ONLY the added line produces a
# pure-added clipped subset — the case the residual fix targets.
_PATCH_WITH_INSERTION_MID_HUNK = """\
diff --git a/src/foo.py b/src/foo.py
index abc..def 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,3 +10,4 @@ def context_header():
     existing_line_one
+    inserted_line
     existing_line_two
     existing_line_three
"""


def test_scope_unit_diff_hunks_pure_added_clip_uses_correct_src_start() -> None:
    """Clipped hunk keeping only added lines must emit `@@ -K,0 +N,M @@`
    where K is the source line BEFORE the insertion — NOT 0 (which is
    the new-file shape).

    Post-foundation push/PR review (medium confidence): the prior
    fallback `src_start = 0` would silently emit `@@ -0,0 +N,M @@` for
    any clipped subset that retained only added lines, contradicting the
    helper's "valid unified diff" contract for in-file insertions.

    Setup: original hunk at `@@ -10,3 +10,4 @@` with body
      ` existing_line_one     (source 10, target 10)`
      `+inserted_line         (target 11)`
      ` existing_line_two     (source 11, target 12)`
      ` existing_line_three   (source 12, target 13)`
    Scope covers target line 11 only — drops the context lines, keeps
    only the inserted line. Walking the original hunk forward stops
    at the `+inserted_line`; the last source_line_no seen before it is
    10 (the first context line). So src_start MUST be 10, src_len = 0.
    """
    patched = PatchSet.from_string(_PATCH_WITH_INSERTION_MID_HUNK)[0]
    su = _scope_unit(line_start=11, line_end=11)
    hunks = scope_unit_diff_hunks(su, patched)  # type: ignore[arg-type]
    assert len(hunks) == 1, f"expected one clipped hunk, got {hunks!r}"
    header = hunks[0].splitlines()[0]
    # The header must NOT be the new-file shape.
    assert header.startswith("@@ -10,0 +"), (
        f"pure-added clipped hunk must anchor src to the line before the "
        f"insertion (10), not 0; got header={header!r}"
    )
    # src_len must be 0; tgt_len must be 1 (one inserted line).
    assert ",0 " in header, f"src_len must be 0 for pure-added clip; got {header!r}"


def test_scope_unit_diff_hunks_pure_added_clip_at_hunk_start_uses_pre_hunk_anchor() -> None:
    """If the first kept added line is at the very START of the original
    hunk (no preceding source line in the hunk's body), src_start falls
    back to `hunk.source_start - 1` — the line just BEFORE the hunk
    began. Defends the boundary case the regression-fix init handles.
    """
    patch_text = """\
diff --git a/src/foo.py b/src/foo.py
index abc..def 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,3 +10,4 @@ def context_header():
+    inserted_at_top
     existing_line_one
     existing_line_two
     existing_line_three
"""
    patched = PatchSet.from_string(patch_text)[0]
    # Scope covers ONLY the inserted line's target (line 10).
    su = _scope_unit(line_start=10, line_end=10)
    hunks = scope_unit_diff_hunks(su, patched)  # type: ignore[arg-type]
    assert len(hunks) == 1
    header = hunks[0].splitlines()[0]
    # Original hunk source_start=10; line just before = 9. Clipped
    # pure-added hunk anchors at src=9, src_len=0.
    assert header.startswith("@@ -9,0 +"), (
        f"clipped pure-added hunk with no preceding source line in body must "
        f"anchor at source_start - 1 = 9; got header={header!r}"
    )


# ---------------------------------------------------------------------------
# scope_unit_has_added_lines
# ---------------------------------------------------------------------------


def test_scope_unit_has_added_lines_true_when_added_in_range() -> None:
    """Default fixture: hunks add lines at target 3 (first) and target 13
    (second). A scope unit covering 1-5 sees added line 3 → True."""
    su = _scope_unit(line_start=1, line_end=5)
    assert scope_unit_has_added_lines(su, _patched_file()) is True  # type: ignore[arg-type]


def test_scope_unit_has_added_lines_false_when_only_context_in_range() -> None:
    """A scope unit covering target lines 1-2 sees only the context
    lines (`def first():`, `return 1`) — no added line in that subrange.
    `scope_unit_diff_hunks` would return non-empty (kept the context),
    but the stricter check returns False so the unit doesn't enter the
    intersection set."""
    su = _scope_unit(line_start=1, line_end=2)
    assert scope_unit_has_added_lines(su, _patched_file()) is False  # type: ignore[arg-type]


def test_scope_unit_has_added_lines_false_when_disjoint() -> None:
    """A scope unit covering target lines 100-110 has nothing in range."""
    su = _scope_unit(line_start=100, line_end=110)
    assert scope_unit_has_added_lines(su, _patched_file()) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# patched_file_has_added_lines
# ---------------------------------------------------------------------------


def test_patched_file_has_added_lines_true_when_any_addition() -> None:
    """A patch containing at least one `+` line returns True. Sibling
    to `scope_unit_has_added_lines` for callers without a scope unit.
    """
    text = """\
diff --git a/f b/f
--- a/f
+++ b/f
@@ -1,1 +1,2 @@
 keep
+added
"""
    patched = PatchSet.from_string(text)[0]
    assert patched_file_has_added_lines(patched) is True  # type: ignore[arg-type]


def test_patched_file_has_added_lines_false_for_pure_deletion() -> None:
    """A patch with no added lines (pure deletion or no diff) returns
    False — the discriminator analyze uses to route between
    NO_REVIEWABLE_CONTEXT and failed+degraded_llm.
    """
    text = """\
diff --git a/f b/f
--- a/f
+++ b/f
@@ -1,2 +1,1 @@
 keep
-removed
"""
    patched = PatchSet.from_string(text)[0]
    assert patched_file_has_added_lines(patched) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_scope_unit_body
# ---------------------------------------------------------------------------


def test_extract_scope_unit_body_decodes_byte_range() -> None:
    """Decode the bytes inside a `ScopeUnit`'s `[byte_start, byte_end)`
    range. Caller passes pre-encoded `source_bytes`; helper returns
    decoded text.
    """
    source = "def first():\n    return 1\n\ndef second():\n    return 2\n"
    source_bytes = source.encode("utf-8")
    su = _scope_unit(byte_start=0, byte_end=25)  # covers "def first():\n    return 1"
    body = extract_scope_unit_body(su, source_bytes)
    assert body == "def first():\n    return 1"


def test_extract_scope_unit_body_handles_invalid_utf8_with_replacement() -> None:
    """If the producer hands us a byte range that lands on a non-UTF-8
    boundary, `errors="replace"` surfaces U+FFFD rather than crashing
    the analyze pass mid-render. Defense-in-depth — under the producer
    contract, tree-sitter offsets land on char boundaries.
    """
    # Construct bytes with a multi-byte UTF-8 char (€ = 0xE2 0x82 0xAC),
    # then point a ScopeUnit at the middle of that sequence.
    source_bytes = "€abc".encode()  # 6 bytes: 0xE2 0x82 0xAC 0x61 0x62 0x63
    su = _scope_unit(byte_start=1, byte_end=4)  # mid-€ to past it
    body = extract_scope_unit_body(su, source_bytes)
    # U+FFFD appears somewhere; helper does not raise.
    assert "�" in body


# ---------------------------------------------------------------------------
# bound_diff_hunks_text
# ---------------------------------------------------------------------------


def test_bound_diff_hunks_text_under_cap_returns_full_text() -> None:
    """Both caps far above content → entire patch text returned."""
    text = bound_diff_hunks_text(_patched_file(), max_lines=100, max_chars=8192)  # type: ignore[arg-type]
    # Both hunks present.
    assert "first()" in text
    assert "second()" in text


def test_bound_diff_hunks_text_truncates_when_line_cap_exceeded() -> None:
    """`max_lines` cap closes the gate when reached; a sentinel marks
    the truncation so the audit/prompt isn't a silent empty."""
    text = bound_diff_hunks_text(_patched_file(), max_lines=2, max_chars=8192)  # type: ignore[arg-type]
    assert "truncated" in text
    # First two lines present; later lines absent.
    assert "second()" not in text


def test_bound_diff_hunks_text_first_line_exceeding_char_cap_is_partial_not_empty() -> None:
    """Regression: a single hunk line exceeding `max_chars` used to
    return `""` silently. Behavior now: emit the line truncated to the
    remaining char budget + a sentinel. The audit row reflects "diff was
    too big to fit" rather than "no diff to review."""
    # Construct a patch with a single hunk line > max_chars.
    huge_line = "a" * 200
    patch_text = f"""\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,1 +1,2 @@
 def first():
+{huge_line}
"""
    patched = PatchSet.from_string(patch_text)[0]
    text = bound_diff_hunks_text(patched, max_lines=100, max_chars=50)  # type: ignore[arg-type]
    # Result is NOT empty (the prior silent-fail behavior).
    assert text != ""
    # Truncation sentinel present.
    assert "truncated" in text
    # Some content of the bounded prefix is present.
    assert text != "\n[truncated: prompt budget cap reached]\n"


def test_bound_diff_hunks_text_result_never_exceeds_max_chars_on_char_cap() -> None:
    """Hard cap: truncation must keep the result within `max_chars`. The
    sentinel previously appended after the content budget was filled,
    so output exceeded the cap by the sentinel length.
    """
    huge_line = "a" * 1000
    patch_text = f"""\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,1 +1,2 @@
 def first():
+{huge_line}
"""
    patched = PatchSet.from_string(patch_text)[0]
    for cap in (50, 100, 200, 8192):
        text = bound_diff_hunks_text(patched, max_lines=100, max_chars=cap)  # type: ignore[arg-type]
        assert len(text) <= cap, f"max_chars={cap}: result len={len(text)}"


def test_bound_diff_hunks_text_result_never_exceeds_max_chars_on_line_cap() -> None:
    """Hard cap: when `max_lines` fires, appending the sentinel must
    not push the total past `max_chars`. With `max_lines=1` and a small
    `max_chars` cap, the second-line iteration triggers the line cap;
    the result is one short line + sentinel and must fit.
    """
    patch_text = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,2 +1,4 @@
 def first():
+    x = 1
+    y = 2
 def second():
"""
    patched = PatchSet.from_string(patch_text)[0]
    # Small caps to stress the budget; line cap fires on the 2nd line.
    for cap in (60, 100, 200):
        text = bound_diff_hunks_text(patched, max_lines=1, max_chars=cap)  # type: ignore[arg-type]
        assert "truncated" in text, f"cap={cap}: expected line-cap truncation"
        assert len(text) <= cap, f"line-cap fired with cap={cap}, result len={len(text)}"


def test_bound_diff_hunks_text_content_uses_full_max_chars_when_no_truncation() -> None:
    """Content should use the FULL `max_chars` budget when no truncation
    fires. Previously the impl pre-reserved sentinel space, lowering the
    real cap by ~46 chars even when content would have fit. CodeRabbit
    flagged this as premature truncation.
    """
    # Patch where the entire text is slightly under max_chars (no
    # truncation should fire). Verify the sentinel does NOT appear.
    text = """\
diff --git a/f b/f
--- a/f
+++ b/f
@@ -1,1 +1,2 @@
 keep
+added
"""
    patched = PatchSet.from_string(text)[0]
    full = bound_diff_hunks_text(patched, max_lines=100, max_chars=10_000)  # type: ignore[arg-type]
    # Sentinel absent when no truncation.
    assert "[truncated:" not in full


def test_bound_diff_hunks_text_rejects_negative_max_chars() -> None:
    """Public helper: a negative `max_chars` makes the cap contract
    nonsensical. The sentinel/headroom math assumes non-negative bounds;
    fail-fast at the helper boundary rather than producing a garbage
    return string. Analyze passes positive constants today, but
    `bound_diff_hunks_text` is part of the coordinates public surface.
    """
    patched = PatchSet.from_string("diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1,0 +1,1 @@\n+x\n")[0]
    with pytest.raises(CoordinateError, match="max_chars must be non-negative"):
        bound_diff_hunks_text(patched, max_lines=10, max_chars=-1)  # type: ignore[arg-type]


def test_bound_diff_hunks_text_rejects_negative_max_lines() -> None:
    """Symmetric to max_chars: a negative `max_lines` is nonsensical."""
    patched = PatchSet.from_string("diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1,0 +1,1 @@\n+x\n")[0]
    with pytest.raises(CoordinateError, match="max_lines must be non-negative"):
        bound_diff_hunks_text(patched, max_lines=-1, max_chars=100)  # type: ignore[arg-type]


def test_bound_diff_hunks_text_handles_max_chars_smaller_than_sentinel() -> None:
    """Defensive edge: if `max_chars` is smaller than the sentinel
    itself, the sentinel is truncated to fit rather than blowing the
    cap. Not a production path (caps are >> sentinel length) but the
    invariant `len(result) <= max_chars` must hold universally.
    """
    patched = PatchSet.from_string("diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1,0 +1,1 @@\n+x\n")[0]
    text = bound_diff_hunks_text(patched, max_lines=1, max_chars=10)  # type: ignore[arg-type]
    assert len(text) <= 10


# ---------------------------------------------------------------------------
# line_range_within_scope_unit (line-space containment, FUP-126)
# ---------------------------------------------------------------------------


def test_line_range_within_scope_unit_inside() -> None:
    su = _scope_unit(line_start=10, line_end=20)
    assert line_range_within_scope_unit(12, 15, su) is True


def test_line_range_within_scope_unit_boundaries_inclusive() -> None:
    su = _scope_unit(line_start=10, line_end=20)
    # Whole-scope, and single line at each edge — the scope's FIRST line is the
    # case byte containment got wrong (a whole-line byte span starts before the
    # indented token's byte_start); line space admits it.
    assert line_range_within_scope_unit(10, 20, su) is True
    assert line_range_within_scope_unit(10, 10, su) is True
    assert line_range_within_scope_unit(20, 20, su) is True


def test_line_range_within_scope_unit_starts_before_rejected() -> None:
    su = _scope_unit(line_start=10, line_end=20)
    assert line_range_within_scope_unit(9, 12, su) is False


def test_line_range_within_scope_unit_extends_past_end_rejected() -> None:
    su = _scope_unit(line_start=10, line_end=20)
    assert line_range_within_scope_unit(18, 21, su) is False


# ---------------------------------------------------------------------------
# line_range_to_span (1-indexed line range → whole-line byte Span, FUP-126)
# ---------------------------------------------------------------------------

_SRC = "def f():\n    x = 1\n    y = 2"  # 3 lines, no trailing newline


def test_line_range_to_span_middle_line() -> None:
    # Line 2 "    x = 1" begins at byte 9; line 3 begins at byte 19.
    assert line_range_to_span(2, 2, _SRC) == Span(byte_start=9, byte_end=19)


def test_line_range_to_span_last_line_runs_to_eof() -> None:
    assert line_range_to_span(3, 3, _SRC) == Span(byte_start=19, byte_end=len(_SRC.encode()))


def test_line_range_to_span_multi_line() -> None:
    assert line_range_to_span(1, 3, _SRC) == Span(byte_start=0, byte_end=len(_SRC.encode()))


def test_line_range_to_span_multibyte_safe() -> None:
    # α and β are 2-byte UTF-8; line 2 "β = 2" begins at byte 7 (after "α = 1\n").
    src = "α = 1\nβ = 2"
    span = line_range_to_span(2, 2, src)
    assert span == Span(byte_start=7, byte_end=len(src.encode()))
    assert src.encode()[span.byte_start : span.byte_end].decode() == "β = 2"


def test_line_range_to_span_past_eof_raises() -> None:
    with pytest.raises(CoordinateError):
        line_range_to_span(2, 4, _SRC)  # only 3 lines


def test_line_range_to_span_invalid_range_raises() -> None:
    with pytest.raises(CoordinateError):
        line_range_to_span(0, 1, _SRC)  # line_start < 1
    with pytest.raises(CoordinateError):
        line_range_to_span(3, 2, _SRC)  # line_end < line_start


def test_line_range_to_span_range_ending_on_trailing_empty_line() -> None:
    """A trailing `\\n` creates a phantom final empty line. A range ending on it is
    admissible: `line_range_to_span(1, total_lines)` maps to the whole-file span
    (the phantom line shares the prior line's end-of-source `byte_end`).
    """
    src = "a\nb\nc\n"  # 4 lines: "a", "b", "c", and the trailing empty line 4
    full = line_range_to_span(1, 4, src)
    assert full == Span(byte_start=0, byte_end=len(src.encode()))


def test_line_range_to_span_trailing_empty_line_alone_is_zero_width() -> None:
    """The trailing empty line by itself maps to a zero-width span; the helper
    does not reject it (callers that forbid zero-width gate with
    `span_is_nonempty`).
    """
    src = "a\nb\nc\n"
    span = line_range_to_span(4, 4, src)
    assert span == Span(byte_start=len(src.encode()), byte_end=len(src.encode()))


# ---------------------------------------------------------------------------
# added_line_byte_ranges — producer for span_within_degraded_context (FUP-138)
# ---------------------------------------------------------------------------


def _patched_from(patch_text: str) -> object:
    return PatchSet.from_string(patch_text)[0]


def test_added_line_byte_ranges_single_added_line() -> None:
    # HEAD source "a\nADDED\nc\n"; the patch adds target line 2 ("ADDED").
    # Line 2 whole-line byte span is [2,8) ("ADDED\n").
    src = "a\nADDED\nc\n"
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,3 @@\n a\n+ADDED\n c\n"
    ranges = added_line_byte_ranges(_patched_from(patch), src)  # type: ignore[arg-type]
    assert ranges == ((2, 8),)
    # The range a finding on line 2 would intersect.
    assert span_within_degraded_context(line_range_to_span(2, 2, src), ranges)


def _byte_range(line_start: int, line_end: int, src: str) -> tuple[int, int]:
    span = line_range_to_span(line_start, line_end, src)
    return (span.byte_start, span.byte_end)


def test_added_line_byte_ranges_contiguous_lines_merge() -> None:
    # Two adjacent added lines (2 and 3) merge into ONE run → one byte span.
    src = "a\nX\nY\nd\n"  # lines: a(0-1) X(2-3) Y(4-5) d(6-7)
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,4 @@\n a\n+X\n+Y\n d\n"
    ranges = added_line_byte_ranges(_patched_from(patch), src)  # type: ignore[arg-type]
    assert ranges == (_byte_range(2, 3, src),)


def test_added_line_byte_ranges_non_contiguous_lines_split() -> None:
    # Added lines 2 and 4 (line 3 unchanged) → TWO separate ranges.
    src = "a\nX\nc\nY\ne\n"
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,5 @@\n a\n+X\n c\n+Y\n e\n"
    ranges = added_line_byte_ranges(_patched_from(patch), src)  # type: ignore[arg-type]
    assert len(ranges) == 2
    assert ranges[0] == _byte_range(2, 2, src)
    assert ranges[1] == _byte_range(4, 4, src)


def test_added_line_byte_ranges_pure_deletion_is_empty() -> None:
    # A pure deletion carries no target_line_no → no addable ranges.
    src = "a\nc\n"
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,2 @@\n a\n-b\n c\n"
    ranges = added_line_byte_ranges(_patched_from(patch), src)  # type: ignore[arg-type]
    assert ranges == ()


def test_added_line_numbers_returns_added_target_lines() -> None:
    # Adds target lines 2 and 4 (line 3 unchanged).
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,5 @@\n a\n+X\n c\n+Y\n e\n"
    assert added_line_numbers(_patched_from(patch)) == frozenset({2, 4})  # type: ignore[arg-type]


def test_added_line_numbers_pure_deletion_is_empty() -> None:
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,2 @@\n a\n-b\n c\n"
    assert added_line_numbers(_patched_from(patch)) == frozenset()  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# changed_line_spans — the trivial-scope-filter veto input.
# Per specs/2026-06-10-trivial-scope-filter.md: the base side is the
# kept-removed set from the shared clipping core (`_clip_hunk_lines`),
# i.e. exactly the deletions the prompt's clipped hunks show — NOT
# base-range containment.
# ---------------------------------------------------------------------------

# Base:                          Head:
#   1  @require_auth               1  def foo():
#   2  def foo():                  2      return 1
#   3      return 1                3      # note
# The deletion of `@require_auth` (base line 1) maps OUTSIDE foo's base
# range, but rides along into foo's clipped hunk (flushed by the kept
# in-range `def foo():` target line). The veto must see it.
_DECORATOR_DELETION_PATCH = (
    "--- a/x.py\n+++ b/x.py\n"
    "@@ -1,3 +1,3 @@\n"
    "-@require_auth\n"
    " def foo():\n"
    "     return 1\n"
    "+    # note\n"
)
_DECORATOR_DELETION_HEAD = "def foo():\n    return 1\n    # note\n"
_DECORATOR_DELETION_BASE = "@require_auth\ndef foo():\n    return 1\n"


def test_changed_line_spans_head_added_lines_in_scope() -> None:
    """Added lines within the scope's head range land in head_added with
    whole-line spans from the head source."""
    su = _scope_unit(line_start=1, line_end=3)
    result = changed_line_spans(
        su,
        _patched_from(_DECORATOR_DELETION_PATCH),  # type: ignore[arg-type]
        head_source=_DECORATOR_DELETION_HEAD,
        base_source=_DECORATOR_DELETION_BASE,
    )
    assert [e.line_no for e in result.head_added] == [3]
    expected = line_range_to_span(3, 3, _DECORATOR_DELETION_HEAD)
    assert result.head_added[0].span == expected


def test_changed_line_spans_ride_along_deletion_is_in_base_removed() -> None:
    """THE over-skip seam (security review Critical): a deletion adjacent
    to the scope — outside its base range — rides into the prompt's
    clipped hunk, so it MUST appear in base_removed. Base-range
    attribution would miss it; the kept-removed frame catches it."""
    su = _scope_unit(line_start=1, line_end=3)
    result = changed_line_spans(
        su,
        _patched_from(_DECORATOR_DELETION_PATCH),  # type: ignore[arg-type]
        head_source=_DECORATOR_DELETION_HEAD,
        base_source=_DECORATOR_DELETION_BASE,
    )
    assert [e.line_no for e in result.base_removed] == [1]
    expected = line_range_to_span(1, 1, _DECORATOR_DELETION_BASE)
    assert result.base_removed[0].span == expected


def test_changed_line_spans_agrees_with_rendered_clipped_hunks() -> None:
    """One clipping decision, two consumers: every removed line in
    base_removed appears in the RENDERED clipped hunks, and vice versa —
    the veto sees exactly what the prompt shows."""
    su = _scope_unit(line_start=1, line_end=3)
    pf = _patched_from(_DECORATOR_DELETION_PATCH)
    rendered = scope_unit_diff_hunks(su, pf)  # type: ignore[arg-type]
    rendered_removed = [
        line[1:]
        for hunk_text in rendered
        for line in hunk_text.splitlines()[1:]  # skip the @@ header
        if line.startswith("-")
    ]
    result = changed_line_spans(
        su,
        pf,  # type: ignore[arg-type]
        head_source=_DECORATOR_DELETION_HEAD,
        base_source=_DECORATOR_DELETION_BASE,
    )
    base_lines = _DECORATOR_DELETION_BASE.splitlines()
    veto_removed = [base_lines[e.line_no - 1] for e in result.base_removed]
    assert veto_removed == rendered_removed == ["@require_auth"]


def test_changed_line_spans_disjoint_hunk_deletions_excluded() -> None:
    """Deletions in a hunk with no kept lines for the scope do not ride
    into base_removed — the prompt would not show them either."""
    from outrider.coordinates import changed_line_spans

    patch = (
        "--- a/x.py\n+++ b/x.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def foo():\n"
        "+    # note\n"
        "     return 1\n"
        "@@ -10,3 +11,2 @@\n"
        " def bar():\n"
        "-    check()\n"
        "     return 2\n"
    )
    head = "def foo():\n    # note\n    return 1\n" + "\n" * 7 + "def bar():\n    return 2\n"
    base = "def foo():\n    return 1\n" + "\n" * 7 + "def bar():\n    check()\n    return 2\n"
    su = _scope_unit(line_start=1, line_end=3)
    result = changed_line_spans(
        su,
        _patched_from(patch),  # type: ignore[arg-type]
        head_source=head,
        base_source=base,
    )
    assert [e.line_no for e in result.head_added] == [2]
    assert result.base_removed == ()


def test_changed_line_spans_missing_base_with_removed_lines_raises() -> None:
    """Misuse guard: kept-removed lines + base_source=None is unreachable
    under the intake contract; reaching it is a caller bug, fail loud."""
    su = _scope_unit(line_start=1, line_end=3)
    with pytest.raises(CoordinateError):
        changed_line_spans(
            su,
            _patched_from(_DECORATOR_DELETION_PATCH),  # type: ignore[arg-type]
            head_source=_DECORATOR_DELETION_HEAD,
            base_source=None,
        )


def test_changed_line_spans_added_file_no_base_ok() -> None:
    """Added-status files (base_source=None, no removed lines) classify
    fine: empty base_removed, no error."""
    from outrider.coordinates import changed_line_spans

    patch = "--- /dev/null\n+++ b/x.py\n@@ -0,0 +1,2 @@\n+def foo():\n+    return 1\n"
    su = _scope_unit(line_start=1, line_end=2)
    result = changed_line_spans(
        su,
        _patched_from(patch),  # type: ignore[arg-type]
        head_source="def foo():\n    return 1\n",
        base_source=None,
    )
    assert [e.line_no for e in result.head_added] == [1, 2]
    assert result.base_removed == ()
