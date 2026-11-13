"""Check that model citations are exact, visible source near the claimed location."""

from typing import Literal

from outrider.coordinates import CoordinateError, exact_quote_overlaps_span, line_range_to_span

QuoteRejectionDetail = Literal[
    "empty_quote", "not_in_source", "outside_primary_claim_span", "invalid_claimed_span"
]


def quote_rejection_detail(
    evidence: str,
    source: str,
    *,
    line_start: int,
    line_end: int,
    visible_byte_ranges: tuple[tuple[int, int], ...],
) -> QuoteRejectionDetail | None:
    """Ground the citation without deciding whether its interpretation is correct."""
    if not evidence.strip():
        return "empty_quote"
    if evidence not in source:
        return "not_in_source"
    try:
        claimed_span = line_range_to_span(line_start, line_end, source)
    except CoordinateError:
        return "invalid_claimed_span"
    if exact_quote_overlaps_span(
        evidence, source, required_overlap=claimed_span, visible_byte_ranges=visible_byte_ranges
    ):
        return None
    return "outside_primary_claim_span"
