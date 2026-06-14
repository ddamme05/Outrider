# Tests for `coordinates.query_span_to_source_lines` — the OBSERVED-tier
# producer's byte-envelope → source-line bridge (Cost Lever 3 spec).
"""Pin `query_span_to_source_lines` semantics:

- a `QueryMatchSpan` byte envelope (half-open, `byte_end` EXCLUSIVE) maps to a
  1-indexed INCLUSIVE source line range `(line_start, line_end)`;
- a span that ends exactly at a `\\n` does NOT spill onto the next line
  (`line_end` is derived from the last CONTENT byte, `byte_end - 1`);
- a ZERO-WIDTH span (`byte_start == byte_end`, which `QueryMatchSpan` admits)
  is rejected — it has no reviewable line range and would underflow `byte_end - 1`;
- inverted / out-of-bounds offsets raise `BYTE_OFFSET_INVALID`;
- offsets are BYTE offsets, not character offsets (multi-byte UTF-8).
"""

from __future__ import annotations

import pytest

from outrider.coordinates import CoordinateError, query_span_to_source_lines
from outrider.coordinates.errors import CoordinateErrorKind

# Three-line file. Byte layout (19 bytes total):
#   line 1 "first\n"  → bytes 0..5  (\n at 5)
#   line 2 "second\n" → bytes 6..12 (\n at 12)
#   line 3 "third\n"  → bytes 13..18 (\n at 18)
_HEAD = "first\nsecond\nthird\n"


# ---------------------------------------------------------------------------
# Happy paths — inclusive line range
# ---------------------------------------------------------------------------


def test_single_line_span_returns_same_start_and_end() -> None:
    """ "first" (bytes 0..5, exclusive of the \\n) is wholly on line 1."""
    assert query_span_to_source_lines(byte_start=0, byte_end=5, head_content=_HEAD) == (1, 1)


def test_single_line_span_on_a_later_line() -> None:
    """ "second" (bytes 6..12, exclusive of the \\n) is wholly on line 2."""
    assert query_span_to_source_lines(byte_start=6, byte_end=12, head_content=_HEAD) == (2, 2)


def test_multi_line_span_spans_inclusive_line_range() -> None:
    """ "first\\nsecond" (bytes 0..12) covers lines 1 through 2 inclusive."""
    assert query_span_to_source_lines(byte_start=0, byte_end=12, head_content=_HEAD) == (1, 2)


def test_span_running_to_eof() -> None:
    """ "third\\n" (bytes 13..19, byte_end == len, the EOF-exclusive offset)
    stays on line 3 — the trailing \\n does not invent a line 4."""
    assert query_span_to_source_lines(byte_start=13, byte_end=19, head_content=_HEAD) == (3, 3)


# ---------------------------------------------------------------------------
# Ends-on-newline — the exclusive-byte_end non-spill case (F2)
# ---------------------------------------------------------------------------


def test_span_ending_exactly_at_newline_does_not_spill_to_next_line() -> None:
    """A span covering "first\\n" (bytes 0..6, byte_end exclusive one past the
    \\n at index 5) must report line_end == 1, NOT 2.

    Regression pin: deriving line_end from `byte_end` directly would count the
    \\n at index 5 and land on line 2. The helper derives from `byte_end - 1`
    (the last content byte) precisely to prevent this spill.
    """
    assert query_span_to_source_lines(byte_start=0, byte_end=6, head_content=_HEAD) == (1, 1)


def test_span_of_only_the_trailing_newline_stays_on_its_line() -> None:
    """A 1-byte span over just line 1's \\n (bytes 5..6) maps to line 1."""
    assert query_span_to_source_lines(byte_start=5, byte_end=6, head_content=_HEAD) == (1, 1)


# ---------------------------------------------------------------------------
# Zero-width rejection (F3) — QueryMatchSpan admits byte_start == byte_end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0, 5, 6, 18])
def test_zero_width_span_is_rejected(offset: int) -> None:
    """`byte_start == byte_end` has no reviewable line range and would underflow
    `byte_end - 1`; it must raise BYTE_OFFSET_INVALID rather than silently
    return a bogus range."""
    with pytest.raises(CoordinateError) as exc_info:
        query_span_to_source_lines(byte_start=offset, byte_end=offset, head_content=_HEAD)
    assert exc_info.value.kind is CoordinateErrorKind.BYTE_OFFSET_INVALID


def test_inverted_span_is_rejected() -> None:
    """`byte_end < byte_start` is rejected by the same non-empty guard."""
    with pytest.raises(CoordinateError) as exc_info:
        query_span_to_source_lines(byte_start=10, byte_end=5, head_content=_HEAD)
    assert exc_info.value.kind is CoordinateErrorKind.BYTE_OFFSET_INVALID


# ---------------------------------------------------------------------------
# Out-of-bounds offsets
# ---------------------------------------------------------------------------


def test_byte_start_past_eof_is_rejected() -> None:
    """`byte_start >= len(head)` is out of bounds (no reviewable byte there)."""
    with pytest.raises(CoordinateError) as exc_info:
        query_span_to_source_lines(byte_start=100, byte_end=101, head_content=_HEAD)
    assert exc_info.value.kind is CoordinateErrorKind.BYTE_OFFSET_INVALID


def test_byte_end_past_eof_is_rejected() -> None:
    """`byte_end > len(head)` is out of bounds."""
    with pytest.raises(CoordinateError) as exc_info:
        query_span_to_source_lines(byte_start=0, byte_end=100, head_content=_HEAD)
    assert exc_info.value.kind is CoordinateErrorKind.BYTE_OFFSET_INVALID


# ---------------------------------------------------------------------------
# Byte offsets, not character offsets (multi-byte UTF-8)
# ---------------------------------------------------------------------------

# "café\nbar\n" — é is 2 UTF-8 bytes (0xC3 0xA9), so the byte layout is:
#   line 1 "café\n" → bytes 0..5 (c=0,a=1,f=2,é=3-4,\n=5)
#   line 2 "bar\n"  → bytes 6..9 (b=6,a=7,r=8,\n=9)
_HEAD_UTF8 = "café\nbar\n"


def test_offsets_are_byte_offsets_line_one_with_multibyte_char() -> None:
    """ "café" is 5 BYTES (not 4 chars) on line 1; the span maps to line 1."""
    assert query_span_to_source_lines(byte_start=0, byte_end=5, head_content=_HEAD_UTF8) == (1, 1)


def test_offsets_are_byte_offsets_line_two_after_multibyte_char() -> None:
    """ "bar" begins at byte 6 (after the 2-byte é + \\n), on line 2."""
    assert query_span_to_source_lines(byte_start=6, byte_end=9, head_content=_HEAD_UTF8) == (2, 2)
