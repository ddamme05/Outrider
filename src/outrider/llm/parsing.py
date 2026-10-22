"""Remove an optional outer Markdown fence before structured-output validation.

Hosts may wrap JSON despite prompt instructions. Nodes apply this narrow
normalization, then their schema/parser admission checks. It does not search
arbitrary prose for JSON or validate the body.
"""

from __future__ import annotations

__all__ = ["strip_outer_json_fence"]


def strip_outer_json_fence(text: str) -> str:
    """Remove one outer fence with a newline-separated, non-empty body.

    The opener must be the first non-whitespace content and the closer the last.
    Any opener tag is accepted. Outside whitespace and body trailing whitespace
    are stripped only when a wrapper is recognized; otherwise return the input.
    Nested or multiple fences remain in the returned body for downstream
    validation to reject. This helper does not parse JSON.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    if not stripped.endswith("```"):
        return text
    first_newline = stripped.find("\n")
    if first_newline == -1:
        # Single-line opener with no body separator — malformed; fall
        # through. Examples: "```" (just the opener), "```json}```"
        # (opener immediately followed by content on the same line).
        return text
    body = stripped[first_newline + 1 : -3].rstrip()
    if not body:
        # Empty body (e.g., "```\n```" — opener, newline, closer with
        # nothing in between) is malformed for a JSON-bearing wrapper.
        # Fall through so Pydantic raises on the original input rather
        # than on an empty string.
        return text
    return body
