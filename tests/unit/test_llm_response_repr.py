"""__repr_args__ elision across LLMRequest/LLMResponse/LLMMessage.

Pydantic V2 builds `__repr__` AND `__str__` from `__repr_args__`, so
overriding it covers `repr()`, `str()`, and f-strings in one place.
Per AC#16. Round-14 verified empirically: overriding `__repr__` directly
does NOT cover `str()`/f-strings.
"""

from __future__ import annotations

from uuid import uuid4

from outrider.audit.events import ContextManifestEntry
from outrider.llm.base import LLMMessage, LLMRequest, LLMResponse

SECRET = "SECRET COMPLETION CONTENT"  # noqa: S105 — test fixture, not a credential


def _entry() -> ContextManifestEntry:
    return ContextManifestEntry(
        file_path="src/foo.py",
        scope_unit_name="Foo.bar",
        line_start=1,
        line_end=10,
        inclusion_reason="changed_scope",
    )


def _build_response() -> LLMResponse:
    return LLMResponse(
        text=SECRET,
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=200,
        cache_read_tokens=0,
        cache_write_tokens=0,
        finish_reason="end_turn",
        latency_ms=500,
    )


def _build_request() -> LLMRequest:
    return LLMRequest(
        system_prompt=SECRET,
        user_prompt=SECRET,
        model="claude-sonnet-4-6",
        max_tokens=1000,
        temperature=0.0,
        review_id=uuid4(),
        node_id="analyze",
        prompt_template_version="analyze@1.0.0",
        degraded_mode=False,
        context_summary=(_entry(),),
    )


# ---------------------------------------------------------------------------
# LLMResponse — repr/str/f-string all redacted.
# ---------------------------------------------------------------------------


def test_response_repr_does_not_contain_secret() -> None:
    response = _build_response()
    assert SECRET not in repr(response)


def test_response_str_does_not_contain_secret() -> None:
    """Pydantic V2's default __str__ is built from __repr_args__; without
    the override it would print `text='SECRET ...'` and leak."""
    response = _build_response()
    assert SECRET not in str(response)


def test_response_f_string_does_not_contain_secret() -> None:
    """f-strings call __format__ which falls through to __str__."""
    response = _build_response()
    formatted = f"got: {response}"
    assert SECRET not in formatted


def test_response_redacted_marker_present() -> None:
    response = _build_response()
    expected_marker = f"<redacted, {len(SECRET)} chars>"
    assert expected_marker in repr(response)
    assert expected_marker in str(response)
    assert expected_marker in f"{response}"


# ---------------------------------------------------------------------------
# LLMRequest — repr/str/f-string all redacted across both content fields.
# ---------------------------------------------------------------------------


def test_request_repr_does_not_contain_secret() -> None:
    request = _build_request()
    assert SECRET not in repr(request)


def test_request_str_does_not_contain_secret() -> None:
    request = _build_request()
    assert SECRET not in str(request)


def test_request_f_string_does_not_contain_secret() -> None:
    request = _build_request()
    formatted = f"sending: {request}"
    assert SECRET not in formatted


def test_request_repr_marks_both_prompts_redacted() -> None:
    request = _build_request()
    rendered = repr(request)
    expected = f"<redacted, {len(SECRET)} chars>"
    # Both system_prompt and user_prompt should appear redacted.
    assert rendered.count(expected) >= 2


# ---------------------------------------------------------------------------
# LLMMessage — covers the standalone-instance case.
# ---------------------------------------------------------------------------


def test_message_repr_does_not_contain_secret() -> None:
    msg = LLMMessage(role="user", content=SECRET)
    assert SECRET not in repr(msg)


def test_message_str_does_not_contain_secret() -> None:
    msg = LLMMessage(role="user", content=SECRET)
    assert SECRET not in str(msg)


def test_message_f_string_does_not_contain_secret() -> None:
    msg = LLMMessage(role="user", content=SECRET)
    assert SECRET not in f"msg: {msg}"


def test_message_redacted_marker_present() -> None:
    msg = LLMMessage(role="user", content=SECRET)
    assert f"<redacted, {len(SECRET)} chars>" in repr(msg)


# ---------------------------------------------------------------------------
# All three string paths produce same redaction (no leak path).
# ---------------------------------------------------------------------------


def test_response_all_three_paths_redact() -> None:
    """Defense-in-depth assertion: NO string-formatting path leaks."""
    response = _build_response()
    for rendered in (repr(response), str(response), f"{response}"):
        assert SECRET not in rendered, f"leak in: {rendered!r}"


def test_request_all_three_paths_redact() -> None:
    request = _build_request()
    for rendered in (repr(request), str(request), f"{request}"):
        assert SECRET not in rendered, f"leak in: {rendered!r}"


def test_message_all_three_paths_redact() -> None:
    msg = LLMMessage(role="user", content=SECRET)
    for rendered in (repr(msg), str(msg), f"{msg}"):
        assert SECRET not in rendered, f"leak in: {rendered!r}"
