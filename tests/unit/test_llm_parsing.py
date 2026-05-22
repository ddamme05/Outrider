"""Tests for `outrider.llm.parsing.strip_outer_json_fence`.

Defends the policy boundary: tolerate exactly one well-formed outer
markdown fence, leave everything else for Pydantic to reject cleanly.
The helper is the runtime defense behind the "Do NOT wrap the JSON in
markdown code fences" prompt instructions that Haiku in particular
sometimes ignores.
"""

from __future__ import annotations

import pytest

from outrider.llm.parsing import strip_outer_json_fence

# ---------------------------------------------------------------------------
# Pass-through cases (no fence to strip)
# ---------------------------------------------------------------------------


def test_unfenced_json_returns_input_unchanged() -> None:
    """Bare JSON object is the common case — must round-trip identically
    so the helper is a no-op for compliant LLM output."""
    raw = '{"file_tier_decisions": [], "reasoning": "Empty PR."}'
    assert strip_outer_json_fence(raw) == raw


def test_unfenced_json_with_surrounding_whitespace_returns_input_unchanged() -> None:
    """The helper does not strip whitespace from compliant output — that's
    Pydantic's job (it tolerates leading/trailing whitespace around JSON).
    Returning the original input keeps the non-fenced path a true no-op."""
    raw = '\n  {"a": 1}  \n'
    assert strip_outer_json_fence(raw) == raw


# ---------------------------------------------------------------------------
# Strip cases (single outer fence, well-formed)
# ---------------------------------------------------------------------------


def test_strips_json_tagged_fence() -> None:
    """The common Haiku misbehavior: ```json\\n{...}\\n```."""
    raw = '```json\n{"file_tier_decisions": [], "reasoning": "Test"}\n```'
    expected = '{"file_tier_decisions": [], "reasoning": "Test"}'
    assert strip_outer_json_fence(raw) == expected


def test_strips_bare_fence_no_language_tag() -> None:
    """Some models emit ```\\n{...}\\n``` without the language hint."""
    raw = '```\n{"a": 1}\n```'
    assert strip_outer_json_fence(raw) == '{"a": 1}'


def test_strips_fence_with_outer_whitespace() -> None:
    """Leading/trailing whitespace around the wrapper is tolerated; the
    inner body is returned with trailing whitespace stripped."""
    raw = '\n  ```json\n{"a": 1}\n```  \n'
    assert strip_outer_json_fence(raw) == '{"a": 1}'


def test_strips_fence_with_arbitrary_language_tag() -> None:
    """A non-json language tag (model confusion) should still be tolerated —
    the body is still the JSON we want."""
    raw = '```python\n{"a": 1}\n```'
    assert strip_outer_json_fence(raw) == '{"a": 1}'


def test_strips_fence_preserves_multiline_json_body() -> None:
    """The body may itself be multi-line JSON; the helper preserves internal
    newlines and only rstrips the trailing newline before the closing fence."""
    raw = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
    expected = '{\n  "a": 1,\n  "b": 2\n}'
    assert strip_outer_json_fence(raw) == expected


# ---------------------------------------------------------------------------
# Fall-through cases (malformed wrappers — return input unchanged so
# Pydantic raises a clear schema error rather than this helper silently
# masking the malformation)
# ---------------------------------------------------------------------------


def test_fence_with_no_closer_falls_through() -> None:
    """Opener but no closer is malformed; helper does NOT attempt recovery."""
    raw = '```json\n{"a": 1}'
    assert strip_outer_json_fence(raw) == raw


def test_fence_with_no_opener_falls_through() -> None:
    """Closer but no opener is malformed; helper returns input unchanged."""
    raw = '{"a": 1}\n```'
    assert strip_outer_json_fence(raw) == raw


def test_prose_before_fence_falls_through() -> None:
    """If the input begins with prose, the fence is not the outermost shape
    and the helper does NOT extract JSON from arbitrary prose. The proof
    boundary forbids that kind of relaxation."""
    raw = 'Sure, here is your JSON:\n```json\n{"a": 1}\n```'
    assert strip_outer_json_fence(raw) == raw


def test_prose_after_fence_falls_through() -> None:
    """Trailing prose after the closing fence — same as leading prose, the
    helper does not try to extract from a non-fence outer shape."""
    raw = '```json\n{"a": 1}\n```\nHope this helps!'
    assert strip_outer_json_fence(raw) == raw


def test_single_line_fence_no_newline_falls_through() -> None:
    """``` on its own line with body on the same line is malformed; the
    helper requires a newline between opener and body."""
    raw = '```{"a": 1}```'
    assert strip_outer_json_fence(raw) == raw


def test_empty_fence_body_falls_through() -> None:
    """`\\`\\`\\`\\n\\`\\`\\`` — opener immediately followed by closer with no
    body separator — is malformed. Falls through unchanged."""
    raw = "```\n```"
    assert strip_outer_json_fence(raw) == raw


# ---------------------------------------------------------------------------
# Boundary: nested or multiple fences (helper handles outer only; the
# inner content is forwarded to Pydantic as-is, which will reject if the
# remaining shape isn't valid JSON — that's the intended behavior)
# ---------------------------------------------------------------------------


def test_nested_fence_inside_body_preserved() -> None:
    """If the BODY contains another fence (e.g., the LLM wrapped JSON whose
    string values contain markdown), strip the OUTER fence only and forward
    the body. The body's inner content is Pydantic's problem."""
    raw = '```json\n{"explanation": "Use ```python\\ndef f(): pass\\n``` here"}\n```'
    expected = '{"explanation": "Use ```python\\ndef f(): pass\\n``` here"}'
    assert strip_outer_json_fence(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",  # empty string
        "   ",  # whitespace only
        "not even close to JSON",  # arbitrary prose with no fence
        "{",  # invalid JSON fragment (no fence)
    ],
)
def test_non_fence_inputs_return_unchanged(raw: str) -> None:
    """Any input that doesn't start with a fence after stripping leading
    whitespace returns unchanged. Pydantic handles the schema error."""
    assert strip_outer_json_fence(raw) == raw
