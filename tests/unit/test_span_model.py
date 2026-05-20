# See specs/2026-05-19-analyze-foundation.md §1.
"""`Span` byte-range domain model tests.

Pins (a) `byte_start <= byte_end` invariant, (b) `byte_start >= 0`,
(c) JS-safe-int upper bound at `2^53 - 1`, (d) frozen + `extra='forbid'`
discipline matches the cross-boundary convention.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.ast_facts import Span


def test_span_admits_well_formed() -> None:
    span = Span(byte_start=10, byte_end=20)
    assert span.byte_start == 10
    assert span.byte_end == 20


def test_span_admits_empty_range() -> None:
    """Half-open interval [a, a) is a zero-length span — admitted."""
    span = Span(byte_start=5, byte_end=5)
    assert span.byte_start == span.byte_end == 5


def test_span_rejects_negative_byte_start() -> None:
    with pytest.raises(ValidationError):
        Span(byte_start=-1, byte_end=10)


def test_span_rejects_negative_byte_end() -> None:
    with pytest.raises(ValidationError):
        Span(byte_start=0, byte_end=-1)


def test_span_rejects_end_before_start() -> None:
    """A descending byte range is a node-side bug — fail at construction."""
    with pytest.raises(ValidationError, match=r"byte_end \(5\) must be >= byte_start \(10\)"):
        Span(byte_start=10, byte_end=5)


def test_span_admits_js_safe_int_ceiling() -> None:
    """2^53 - 1 is the JS-safe-int ceiling, the documented upper bound."""
    ceiling = 2**53 - 1
    span = Span(byte_start=0, byte_end=ceiling)
    assert span.byte_end == ceiling


def test_span_rejects_above_js_safe_int_ceiling() -> None:
    """Above the ceiling, JS dashboard consumers would silently truncate.

    Fail-loud at construction so a misconfigured-or-malicious giant file
    surfaces at the schema boundary rather than at the dashboard layer.
    """
    with pytest.raises(ValidationError):
        Span(byte_start=0, byte_end=2**53)


def test_span_frozen_rejects_mutation() -> None:
    """`ConfigDict(frozen=True)` blocks runtime mutation."""
    span = Span(byte_start=0, byte_end=10)
    with pytest.raises(ValidationError):
        span.byte_start = 99  # type: ignore[misc]


def test_span_rejects_extra_fields() -> None:
    """`ConfigDict(extra='forbid')` matches cross-boundary convention."""
    with pytest.raises(ValidationError):
        Span(byte_start=0, byte_end=10, extra_attr="oops")  # type: ignore[call-arg]
