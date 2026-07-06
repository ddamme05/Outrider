"""Offset/limit pagination helpers for the REST list endpoints.

Turns a raw ``(page, page_size, total_items)`` request into a :class:`Page`
descriptor the API layer serializes into the ``pagination`` envelope: the SQL
``OFFSET``/``LIMIT`` window to feed the query builder, the human-readable range
label ("Showing 21-40 of 195"), and the prev/next navigation flags the client
uses to enable or disable the page controls.

Storage-agnostic by design — the caller already knows ``total_items`` (usually
from a cheap ``COUNT(*)``) and only needs this module to work out the slice
bounds and the display metadata. No database access happens here.
"""

from dataclasses import dataclass
from datetime import datetime

DEFAULT_PAGE_SIZE = 20


@dataclass(frozen=True)
class Page:
    """A single resolved page of a paginated list response."""

    number: int
    size: int
    total_items: int
    total_pages: int
    offset: int
    limit: int
    has_previous: bool
    has_next: bool
    label: str
    generated_at: str


def page_count(total_items: int, page_size: int) -> int:
    """Return the number of pages needed to cover ``total_items``.

    Ceil-divides so a partial final page still counts as a whole page.
    """
    full_pages = total_items // page_size
    remainder = total_items % page_size
    return full_pages + (1 if remainder else 0)


def slice_bounds(page: int, page_size: int) -> tuple[int, int]:
    """Return the ``(offset, limit)`` window for the requested page.

    The offset is zero-based for the SQL query; the limit is simply the page
    size. Page numbers are treated as one-based to match the API contract.
    """
    offset = (page - 1) * page_size
    return offset, page_size


def range_label(page: int, page_size: int, total_items: int) -> str:
    """Return a "Showing X-Y of N" label for the current page."""
    if total_items == 0:
        return "No results"
    first = (page - 1) * page_size + 1
    last = min(page * page_size, total_items)
    return f"Showing {first}-{last} of {total_items}"


def paginate(page: int, page_size: int, total_items: int) -> Page:
    """Resolve a pagination request into a fully-populated :class:`Page`.

    ``page`` is one-based. Callers pass the total row count they already have;
    this returns the slice bounds plus all the display metadata the frontend
    needs to render the pager.
    """
    total_pages = page_count(total_items, page_size)
    offset, limit = slice_bounds(page, page_size)
    return Page(
        number=page,
        size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        offset=offset,
        limit=limit,
        has_previous=page > 1,
        has_next=page < total_pages,
        label=range_label(page, page_size, total_items),
        generated_at=datetime.utcnow().isoformat() + "Z",
    )
