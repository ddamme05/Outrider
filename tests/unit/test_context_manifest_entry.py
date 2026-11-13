"""Check immutable context-manifest entries and their allowed vocabulary.

Nested models need their own frozen/extra-forbid configuration; freezing the
outer LLM event alone does not protect them (``audit-events-frozen-extra-forbid``)."""

import pytest
from pydantic import ValidationError

from outrider.audit.events import ContextManifestEntry


def _build_entry(**overrides: object) -> ContextManifestEntry:
    fields: dict[str, object] = {
        "file_path": "src/foo.py",
        "scope_unit_name": "Foo.bar",
        "line_start": 1,
        "line_end": 10,
        "inclusion_reason": "changed_scope",
    }
    fields.update(overrides)
    return ContextManifestEntry(**fields)  # type: ignore[arg-type]


def test_context_manifest_entry_is_frozen() -> None:
    """Assigning to a field after construction raises.

    The nested-tuple in LLMCallEvent.context_summary doesn't help if
    the entry itself isn't frozen.
    """
    entry = _build_entry()
    with pytest.raises(ValidationError):
        entry.file_path = "evil.py"  # type: ignore[misc]


def test_context_manifest_entry_extra_forbid() -> None:
    """Unknown fields raise per audit-events-frozen-extra-forbid (nested too)."""
    with pytest.raises(ValidationError, match="extra"):
        _build_entry(unknown_field="oops")


def test_context_manifest_entry_inclusion_reason_admits_canonical_values() -> None:
    """Production context reasons and the evaluation reference reason all admit."""
    entry_changed = _build_entry(inclusion_reason="changed_scope")
    entry_same = _build_entry(inclusion_reason="same_file_context")
    entry_trace = _build_entry(inclusion_reason="trace_expansion")
    entry_reference = _build_entry(inclusion_reason="captured_reference")
    assert entry_changed.inclusion_reason == "changed_scope"
    assert entry_same.inclusion_reason == "same_file_context"
    assert entry_trace.inclusion_reason == "trace_expansion"
    assert entry_reference.inclusion_reason == "captured_reference"


def test_context_manifest_entry_inclusion_reason_rejects_other_values() -> None:
    """Anything outside the deterministic vocabulary raises."""
    with pytest.raises(ValidationError):
        _build_entry(inclusion_reason="manual")


def test_context_manifest_entry_line_start_ge_1() -> None:
    """line_start = 0 raises (1-indexed per coordinates/)."""
    with pytest.raises(ValidationError):
        _build_entry(line_start=0, line_end=10)


def test_context_manifest_entry_line_end_ge_line_start() -> None:
    """line_end < line_start raises via the model_validator."""
    with pytest.raises(ValidationError, match="line_end"):
        _build_entry(line_start=10, line_end=5)
