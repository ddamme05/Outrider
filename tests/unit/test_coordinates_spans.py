# See specs/2026-05-19-analyze-foundation.md §4.
"""Span-based coordinate helper tests.

Five helpers, one file. Pins the half-open interval semantics
(byte_end exclusive), the file-size safety floor, the addable-diff
intersection rule for degraded-mode admission, the byte→1-indexed-line
conversion (including multibyte UTF-8 safety), and the scope-unit-
bounded hunk clipping.
"""

from __future__ import annotations

import pytest
from unidiff import PatchSet

from outrider.ast_facts.models import ScopeUnit, Span
from outrider.coordinates import (
    CoordinateError,
    bound_diff_hunks_text,
    extract_scope_unit_body,
    patched_file_has_added_lines,
    scope_unit_diff_hunks,
    scope_unit_has_added_lines,
    span_to_line_range,
    span_within_degraded_context,
    span_within_file,
    span_within_scope_unit,
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
# span_within_scope_unit
# ---------------------------------------------------------------------------


def test_span_within_scope_unit_strictly_inside() -> None:
    su = _scope_unit(byte_start=100, byte_end=200)
    assert span_within_scope_unit(Span(byte_start=120, byte_end=180), su) is True


def test_span_within_scope_unit_equal_bounds() -> None:
    """Span equal to scope unit bounds is contained (half-open: end is exclusive)."""
    su = _scope_unit(byte_start=100, byte_end=200)
    assert span_within_scope_unit(Span(byte_start=100, byte_end=200), su) is True


def test_span_within_scope_unit_extends_past_end_rejected() -> None:
    su = _scope_unit(byte_start=100, byte_end=200)
    assert span_within_scope_unit(Span(byte_start=100, byte_end=201), su) is False


def test_span_within_scope_unit_starts_before_start_rejected() -> None:
    su = _scope_unit(byte_start=100, byte_end=200)
    assert span_within_scope_unit(Span(byte_start=99, byte_end=150), su) is False


def test_span_within_scope_unit_empty_span_inside() -> None:
    """Empty span [a, a) at any point inside the scope: contained."""
    su = _scope_unit(byte_start=100, byte_end=200)
    assert span_within_scope_unit(Span(byte_start=150, byte_end=150), su) is True


# ---------------------------------------------------------------------------
# span_within_file
# ---------------------------------------------------------------------------


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
# span_to_line_range
# ---------------------------------------------------------------------------


def test_span_to_line_range_single_line() -> None:
    source = "line one\nline two\nline three\n"
    # Bytes 0-7 are "line one"; line_start = line_end = 1.
    assert span_to_line_range(Span(byte_start=0, byte_end=8), source) == (1, 1)


def test_span_to_line_range_multi_line() -> None:
    source = "line one\nline two\nline three\n"
    # Bytes 5-15 span "ne\nline t" — lines 1 to 2.
    assert span_to_line_range(Span(byte_start=5, byte_end=15), source) == (1, 2)


def test_span_to_line_range_starts_at_newline_boundary() -> None:
    source = "aaa\nbbb\nccc\n"
    # Byte 4 is 'b' (first byte of line 2); span (4, 7) = "bbb" on line 2.
    assert span_to_line_range(Span(byte_start=4, byte_end=7), source) == (2, 2)


def test_span_to_line_range_empty_span_returns_single_line() -> None:
    source = "aaa\nbbb\nccc\n"
    # Empty span at byte 5 (inside "bbb" on line 2).
    assert span_to_line_range(Span(byte_start=5, byte_end=5), source) == (2, 2)


def test_span_to_line_range_past_end_raises() -> None:
    source = "short\n"
    with pytest.raises(CoordinateError, match="exceeds source length"):
        span_to_line_range(Span(byte_start=0, byte_end=999), source)


def test_span_to_line_range_multibyte_safe() -> None:
    """Span bytes must be counted against UTF-8 encoded source.

    "café" is 4 chars / 5 bytes (é is 0xC3 0xA9). A naive `source[start:end]`
    str-index would miscount; this test ensures byte counting is correct.
    """
    source = "café\nbar\n"  # bytes: "c","a","f", 0xC3,0xA9, "\n", "b","a","r","\n"
    # Total source bytes = 10 (caf=3, é=2, \n=1, bar=3, \n=1).
    # Byte 6 is 'b' (first byte of line 2).
    assert span_to_line_range(Span(byte_start=6, byte_end=9), source) == (2, 2)


def test_span_to_line_range_includes_trailing_newline() -> None:
    """A span ending exactly at end-of-source is admitted (boundary, not past)."""
    source = "abc\n"
    # Length = 4 bytes. Span (0, 4) is admitted.
    line_range = span_to_line_range(Span(byte_start=0, byte_end=4), source)
    # Byte 3 is '\n' — still on line 1 (the newline terminates it).
    assert line_range == (1, 1)


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
