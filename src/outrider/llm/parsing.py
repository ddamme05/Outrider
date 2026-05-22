"""Response-envelope normalization for LLM-returned structured output.

Defends against the well-known LLM-output-discipline gap: prompt
instructions like "do not wrap the JSON in markdown fences" are
advisory, not enforcement. Anthropic Haiku in particular sometimes
wraps responses in ```json ... ``` despite the system prompt. Without a
runtime defense, `Model.model_validate_json(response.text)` raises a
`ValidationError` on the first backtick, crashing the consuming node.

This module owns the narrow normalization step that nodes apply
between the raw LLM text and `model_validate_json`. The policy
intentionally accepts ONE outer well-formed wrapper only — malformed
or nested wrappers fall through to Pydantic for a clear schema
error. This module does not extract JSON from arbitrary prose; that
would couple the LLM-output trust boundary to prose-parsing
heuristics, which is exactly the relaxation the proof boundary is
designed to prevent.
"""

from __future__ import annotations

__all__ = ["strip_outer_json_fence"]


def strip_outer_json_fence(text: str) -> str:
    """Strip one outer ```json...``` or ```...``` wrapper from LLM output.

    Tolerates a single, well-formed outer markdown fence — the common
    "model wrapped output despite the prompt instruction" case. Any
    other shape returns the input unchanged so Pydantic raises a clean
    schema error downstream.

    Policy (per Codex review):

    - Exactly one outer fence, optionally with a language tag
      (``` or ```json or ```any-lang).
    - The opener is the first non-whitespace content and the closer is
      the last non-whitespace content.
    - Whitespace OUTSIDE the wrapper is tolerated; the inner body is
      returned with trailing whitespace stripped.
    - No prose extraction: if the input begins with `prose\\n```...```,
      the wrapper is NOT recognized and the input is returned unchanged.
    - No nested or multiple wrappers: only the outermost is handled.
    - Malformed wrappers (opener but no closer, or closer but no opener,
      or opener with no newline before body) fall through unchanged so
      Pydantic produces a clear ValidationError instead of this helper
      masking the failure.

    The helper imports nothing from a vendor SDK — the LLM provider
    boundary stays intact.
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
