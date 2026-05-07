"""Schema-level default-redaction on `model_dump()` (AC#22).

Five content-bearing fields total:
  - LLMRequest.system_prompt
  - LLMRequest.user_prompt
  - LLMRequest.messages (V1.5+, but the redaction lives on the field anyway)
  - LLMMessage.content (covers standalone + nested-via-LLMRequest cases)
  - LLMResponse.text

`model_dump()` redacts by default. Persister opts in via the typed
`INCLUDE_TEXT_OPT_IN` sentinel — identity check, not dict-key lookup.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from outrider.audit.events import ContextManifestEntry
from outrider.llm.base import (
    INCLUDE_TEXT_OPT_IN,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    _IncludeTextOptIn,
)

SECRET = "SECRET"  # noqa: S105 — test fixture, not a credential
SECRET_LEN = len(SECRET)
REDACTED = f"<redacted, {SECRET_LEN} chars>"


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
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        finish_reason="end_turn",
        latency_ms=100,
    )


def _build_request() -> LLMRequest:
    return LLMRequest(
        system_prompt=SECRET,
        user_prompt=SECRET,
        model="claude-sonnet-4-6",
        max_tokens=100,
        temperature=0.0,
        review_id=uuid4(),
        node_id="analyze",
        prompt_template_version="analyze@1.0.0",
        degraded_mode=False,
        context_summary=(_entry(),),
    )


# ---------------------------------------------------------------------------
# LLMResponse default-redact + opt-in.
# ---------------------------------------------------------------------------


def test_response_model_dump_default_redacts_text() -> None:
    response = _build_response()
    dumped = response.model_dump()
    assert dumped["text"] == REDACTED


def test_response_model_dump_with_sentinel_returns_text() -> None:
    response = _build_response()
    dumped = response.model_dump(context=INCLUDE_TEXT_OPT_IN)
    assert dumped["text"] == SECRET


def test_response_model_dump_json_default_redacts() -> None:
    response = _build_response()
    payload = json.loads(response.model_dump_json())
    assert payload["text"] == REDACTED


def test_response_model_dump_json_with_sentinel_returns_text() -> None:
    response = _build_response()
    payload = json.loads(response.model_dump_json(context=INCLUDE_TEXT_OPT_IN))
    assert payload["text"] == SECRET


# ---------------------------------------------------------------------------
# LLMRequest default-redact + opt-in.
# ---------------------------------------------------------------------------


def test_request_model_dump_default_redacts_system_prompt() -> None:
    request = _build_request()
    dumped = request.model_dump()
    assert dumped["system_prompt"] == REDACTED


def test_request_model_dump_default_redacts_user_prompt() -> None:
    request = _build_request()
    dumped = request.model_dump()
    assert dumped["user_prompt"] == REDACTED


def test_request_model_dump_with_sentinel_returns_prompts() -> None:
    request = _build_request()
    dumped = request.model_dump(context=INCLUDE_TEXT_OPT_IN)
    assert dumped["system_prompt"] == SECRET
    assert dumped["user_prompt"] == SECRET


# ---------------------------------------------------------------------------
# LLMMessage default-redact + opt-in (standalone case).
# ---------------------------------------------------------------------------


def test_message_standalone_model_dump_default_redacts() -> None:
    """Round-9 fix: LLMMessage is a public schema; standalone construction
    + dump must redact regardless of any parent."""
    msg = LLMMessage(role="user", content=SECRET)
    dumped = msg.model_dump()
    assert dumped["content"] == REDACTED


def test_message_standalone_model_dump_with_sentinel_returns_content() -> None:
    msg = LLMMessage(role="user", content=SECRET)
    dumped = msg.model_dump(context=INCLUDE_TEXT_OPT_IN)
    assert dumped["content"] == SECRET


# ---------------------------------------------------------------------------
# Typed-sentinel identity check — typos do NOT pass.
# ---------------------------------------------------------------------------


def test_string_keyed_dict_does_not_pass() -> None:
    """The previous round-8 design used `context={"include_text": True}`
    (string key); round 11 replaced with the typed sentinel for typo
    safety. A string-keyed dict context REDACTS now."""
    response = _build_response()
    dumped = response.model_dump(context={"include_text": True})
    assert dumped["text"] == REDACTED


def test_arbitrary_object_context_does_not_pass() -> None:
    response = _build_response()
    dumped = response.model_dump(context=object())
    assert dumped["text"] == REDACTED


def test_none_context_does_not_pass() -> None:
    response = _build_response()
    dumped = response.model_dump(context=None)
    assert dumped["text"] == REDACTED


def test_uppercase_typo_does_not_pass() -> None:
    """A common typo: `INCLUDE_TEXT` (caps) is a string, not the sentinel."""
    response = _build_response()
    dumped = response.model_dump(context={"INCLUDE_TEXT": True})
    assert dumped["text"] == REDACTED


def test_string_value_typo_does_not_pass() -> None:
    response = _build_response()
    dumped = response.model_dump(context={"include_text": "true"})
    assert dumped["text"] == REDACTED


def test_int_value_typo_does_not_pass() -> None:
    response = _build_response()
    dumped = response.model_dump(context={"include_text": 1})
    assert dumped["text"] == REDACTED


# ---------------------------------------------------------------------------
# Sentinel construction guard.
# ---------------------------------------------------------------------------


def test_sentinel_direct_construction_raises() -> None:
    """`_IncludeTextOptIn` rejects direct construction — only the
    module-level singleton is valid."""
    with pytest.raises(TypeError, match="INCLUDE_TEXT_OPT_IN"):
        _IncludeTextOptIn("anything else")


def test_sentinel_singleton_is_unique() -> None:
    """The module exposes exactly one INCLUDE_TEXT_OPT_IN object; identity
    check (`is`) on it is what the serializer uses. Re-importing must
    return the same instance — Python module-cache guarantee."""
    import outrider.llm.base as llm_base_a
    import outrider.llm.base as llm_base_b

    assert llm_base_a.INCLUDE_TEXT_OPT_IN is llm_base_b.INCLUDE_TEXT_OPT_IN
    assert llm_base_a.INCLUDE_TEXT_OPT_IN is INCLUDE_TEXT_OPT_IN


# ---------------------------------------------------------------------------
# Persister flow — context propagation through nested model_dump().
# ---------------------------------------------------------------------------


def test_persister_can_recover_text_via_attribute_access() -> None:
    """Even though model_dump redacts, attribute access works (the
    persister could legitimately use this path — though spec recommends
    the model_dump route for auditability)."""
    response = _build_response()
    assert response.text == SECRET


def test_response_attribute_access_unchanged() -> None:
    """The redaction is serialization-only; attribute access is unaffected."""
    response = _build_response()
    assert response.text == SECRET
    assert response.model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# AC#22 paired source-scan: INCLUDE_TEXT_OPT_IN imported only by approved
# modules. Any other import site is a bug — typed sentinel grep is the
# audit hook (round-11 fold).
# ---------------------------------------------------------------------------


def test_include_text_opt_in_appears_only_in_approved_paths() -> None:
    """Round-15+ AC#22 contract: the typed sentinel must be imported only
    by `outrider.llm` (definition + first-party callers) and
    `outrider.audit` (the persister). Any other import site is a bug."""
    import pathlib

    src_root = pathlib.Path(__file__).resolve().parents[2] / "src" / "outrider"
    allowed_subpaths = ("llm", "audit")
    leaks: list[str] = []
    for py in src_root.rglob("*.py"):
        rel = py.relative_to(src_root)
        if rel.parts and rel.parts[0] in allowed_subpaths:
            continue
        if "INCLUDE_TEXT_OPT_IN" in py.read_text(encoding="utf-8"):
            leaks.append(str(rel))
    assert not leaks, (
        f"INCLUDE_TEXT_OPT_IN leaked outside approved paths {allowed_subpaths!r}: "
        f"{leaks}. Only the persister (in audit/) and llm/ itself should "
        f"reference the sentinel."
    )
