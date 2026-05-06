"""LLMRequest constraint + validator tests.

Covers all field-level constraints + the two cross-field validators (the
V1-messages-unset rejection and the analyze/synthesize empty-context
rejection). Per AC#20.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import ContextManifestEntry
from outrider.llm.base import LLMMessage, LLMRequest


def _entry() -> ContextManifestEntry:
    return ContextManifestEntry(
        file_path="src/foo.py",
        scope_unit_name="Foo.bar",
        line_start=1,
        line_end=10,
        inclusion_reason="changed_scope",
    )


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "system_prompt": "You are a triage classifier.",
        "user_prompt": "Classify this PR.",
        "model": "claude-haiku-4-5",
        "max_tokens": 100,
        "temperature": 0.0,
        "review_id": uuid4(),
        "node_id": "triage",
        "prompt_template_version": "triage@1.0.0",
        "degraded_mode": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Transport-field constraints.
# ---------------------------------------------------------------------------


def test_system_prompt_min_length_one() -> None:
    """Empty system_prompt is a node-side bug (template render returned ''):
    fail at construction, not at the SDK billing surface."""
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(system_prompt=""))


def test_user_prompt_min_length_one() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(user_prompt=""))


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(max_tokens=0))
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(max_tokens=-1))


def test_max_tokens_upper_bound() -> None:
    """8192 is the V1 cap (cost-cliff guard)."""
    LLMRequest(**_kwargs(max_tokens=8192))  # admits at the boundary
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(max_tokens=8193))


def test_temperature_bounds() -> None:
    LLMRequest(**_kwargs(temperature=0.0))
    LLMRequest(**_kwargs(temperature=1.0))
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(temperature=-0.01))
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(temperature=1.01))


def test_temperature_required_no_default() -> None:
    """temperature has no default; required per spec (round 11 sharp-edges H3
    — None / SDK default would be 1.0, defeating replay determinism)."""
    kwargs = _kwargs()
    del kwargs["temperature"]
    with pytest.raises(ValidationError):
        LLMRequest(**kwargs)


# ---------------------------------------------------------------------------
# Audit-context fields.
# ---------------------------------------------------------------------------


def test_review_id_required() -> None:
    kwargs = _kwargs()
    del kwargs["review_id"]
    with pytest.raises(ValidationError):
        LLMRequest(**kwargs)


def test_review_id_must_be_uuid() -> None:
    """Pydantic auto-coerces UUID strings; arbitrary strings rejected."""
    LLMRequest(**_kwargs(review_id=str(uuid4())))  # admits a UUID string
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(review_id="not-a-uuid"))


@pytest.mark.parametrize("node_id", ["triage", "analyze", "synthesize", "trace"])
def test_node_id_admits_canonical_values(node_id: str) -> None:
    # analyze/synthesize need non-empty context_summary; provide one for those
    extra: dict[str, object] = {"node_id": node_id}
    if node_id in {"analyze", "synthesize"}:
        extra["context_summary"] = (_entry(),)
    req = LLMRequest(**_kwargs(**extra))
    assert req.node_id == node_id


def test_node_id_rejects_typo() -> None:
    """`"analyse"` (British spelling) is a real-world typo that
    `node_id: str` would have admitted; `Literal[...]` rejects."""
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(node_id="analyse"))


def test_node_id_rejects_arbitrary_string() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(node_id="custom_node"))


def test_prompt_template_version_min_length_one() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(**_kwargs(prompt_template_version=""))


def test_prompt_template_version_admits_canonical_shapes() -> None:
    """Spec §8.3 allows `analyze@1.0.0` (Git-SHA-style) and
    `analyze_v2.1.3` (semver-style) — strict shape validation lives in
    the prompts/ registry, not at the wrapper schema layer."""
    LLMRequest(**_kwargs(prompt_template_version="analyze@1.0.0"))
    LLMRequest(**_kwargs(prompt_template_version="analyze_v2.1.3"))
    LLMRequest(**_kwargs(prompt_template_version="v1"))
    LLMRequest(**_kwargs(prompt_template_version="v1.2.3"))
    LLMRequest(**_kwargs(prompt_template_version="abc123def456"))  # Git SHA shape


def test_degraded_mode_required_no_default() -> None:
    """`degraded_mode` has no default — caller must explicitly pass.
    Silent default `False` would poison the audit row when ast_facts
    actually returned a degraded parse (round 11 sharp-edges C3)."""
    kwargs = _kwargs()
    del kwargs["degraded_mode"]
    with pytest.raises(ValidationError):
        LLMRequest(**kwargs)


def test_is_eval_defaults_false() -> None:
    req = LLMRequest(**_kwargs())
    assert req.is_eval is False


def test_context_summary_defaults_empty_tuple() -> None:
    req = LLMRequest(**_kwargs())
    assert req.context_summary == ()
    assert isinstance(req.context_summary, tuple)


def test_context_summary_admits_tuple_of_entries() -> None:
    entries = (_entry(), _entry())
    req = LLMRequest(**_kwargs(context_summary=entries, node_id="analyze"))
    assert req.context_summary == entries


def test_context_summary_coerces_list_to_tuple() -> None:
    """Pydantic V2 auto-coerces list → tuple for `tuple[X, ...]`
    annotations. The frozen-tuple immutability guarantee is preserved on
    the stored field regardless of input shape — what matters is that
    `request.context_summary` is a tuple after construction (not a live
    list reference)."""
    req = LLMRequest(**_kwargs(context_summary=[_entry()], node_id="analyze"))
    assert isinstance(req.context_summary, tuple)
    assert len(req.context_summary) == 1


# ---------------------------------------------------------------------------
# Cross-field validators.
# ---------------------------------------------------------------------------


def test_messages_must_be_none_in_v1() -> None:
    """V1 reserves `messages` for V1.5+; non-None rejected at construction."""
    with pytest.raises(ValidationError, match="reserved for V1.5"):
        LLMRequest(**_kwargs(messages=[LLMMessage(role="user", content="hi")]))


def test_messages_default_none_admits() -> None:
    req = LLMRequest(**_kwargs())
    assert req.messages is None


def test_analyze_with_empty_context_summary_raises() -> None:
    with pytest.raises(ValidationError, match="non-empty context_summary"):
        LLMRequest(**_kwargs(node_id="analyze", context_summary=()))


def test_synthesize_with_empty_context_summary_raises() -> None:
    with pytest.raises(ValidationError, match="non-empty context_summary"):
        LLMRequest(**_kwargs(node_id="synthesize", context_summary=()))


def test_triage_admits_empty_context_summary() -> None:
    """Triage legitimately calls without a scope manifest — the cross-field
    rule only fires for analyze/synthesize."""
    req = LLMRequest(**_kwargs(node_id="triage", context_summary=()))
    assert req.context_summary == ()


def test_trace_admits_empty_context_summary() -> None:
    """Trace also calls without a scope manifest in some flows."""
    req = LLMRequest(**_kwargs(node_id="trace", context_summary=()))
    assert req.context_summary == ()


def test_analyze_with_non_empty_context_summary_admits() -> None:
    req = LLMRequest(**_kwargs(node_id="analyze", context_summary=(_entry(),)))
    assert req.node_id == "analyze"
    assert len(req.context_summary) == 1


# ---------------------------------------------------------------------------
# Smoke test: well-formed construction across all four node_ids.
# ---------------------------------------------------------------------------


def test_all_node_ids_construct_with_appropriate_context() -> None:
    for node_id in ("triage", "trace"):
        req = LLMRequest(**_kwargs(node_id=node_id))
        assert req.node_id == node_id
    for node_id in ("analyze", "synthesize"):
        req = LLMRequest(**_kwargs(node_id=node_id, context_summary=(_entry(),)))
        assert req.node_id == node_id
        assert len(req.context_summary) == 1
