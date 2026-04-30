"""TraceDecisionEvent: resolution_status Literal + nullable target_file + three-rule validator.

Backs DECISIONS.md#017 (Amended same-day, two clauses):
(a) resolved ↔ non-None target_file
(b) unresolved / ambiguous ↔ target_file is None
(c) when resolved, target_file in candidates_considered
"""

import json
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import AuditEventAdapter, TraceDecisionEvent


def _build_event(**overrides: Any) -> TraceDecisionEvent:
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "source_finding_id": uuid4(),
        "target_file": "src/bar.py",
        "reason": "called from middleware/auth.py:42",
        "resolution_status": "resolved",
        "candidates_considered": ("src/bar.py", "src/baz.py"),
    }
    fields.update(overrides)
    return TraceDecisionEvent(**fields)


def test_resolution_status_admits_three_canonical_values() -> None:
    """resolved + unresolved + ambiguous all admit (resolved with target; others without)."""
    event_resolved = _build_event(resolution_status="resolved")
    event_unresolved = _build_event(
        resolution_status="unresolved",
        target_file=None,
    )
    event_ambiguous = _build_event(
        resolution_status="ambiguous",
        target_file=None,
    )
    assert event_resolved.resolution_status == "resolved"
    assert event_unresolved.resolution_status == "unresolved"
    assert event_ambiguous.resolution_status == "ambiguous"


def test_resolution_status_rejects_other_values() -> None:
    """The deterministic vocabulary is the gate; anything else raises."""
    with pytest.raises(ValidationError):
        _build_event(resolution_status="pending")


def test_round_trips_resolution_status() -> None:
    """JSON round-trip preserves the Literal value through the discriminated union."""
    original = _build_event()
    json_payload = original.model_dump_json(exclude={"sequence_number"})
    reconstructed = AuditEventAdapter.validate_json(json_payload)
    assert isinstance(reconstructed, TraceDecisionEvent)
    assert reconstructed.resolution_status == original.resolution_status


def test_resolved_admits_with_target_file() -> None:
    """Happy path: resolved + non-None target_file in candidates_considered."""
    event = _build_event(
        resolution_status="resolved",
        target_file="src/bar.py",
        candidates_considered=("src/bar.py", "src/baz.py"),
    )
    assert event.target_file == "src/bar.py"


def test_unresolved_admits_with_none_target_file() -> None:
    """unresolved (and ambiguous) + target_file=None admits."""
    event = _build_event(
        resolution_status="unresolved",
        target_file=None,
    )
    assert event.target_file is None


def test_resolved_without_target_file_raises() -> None:
    """Cross-field rule (a): resolved + target_file=None raises."""
    with pytest.raises(ValidationError, match="non-None target_file"):
        _build_event(resolution_status="resolved", target_file=None)


def test_unresolved_with_target_file_raises() -> None:
    """Cross-field rule (b): unresolved + non-None target_file raises."""
    with pytest.raises(ValidationError, match="target_file is None"):
        _build_event(
            resolution_status="unresolved",
            target_file="src/bar.py",
        )


def test_ambiguous_with_target_file_raises() -> None:
    """Cross-field rule (b), ambiguous side: ambiguous + non-None target_file raises."""
    with pytest.raises(ValidationError, match="target_file is None"):
        _build_event(
            resolution_status="ambiguous",
            target_file="src/bar.py",
        )


def test_resolved_target_file_must_be_in_candidates_considered() -> None:
    """Cross-field rule (c) per #017 clause (b): resolved target_file ∈ candidates_considered."""
    in_list = _build_event(
        resolution_status="resolved",
        target_file="src/bar.py",
        candidates_considered=("src/bar.py", "src/baz.py"),
    )
    assert in_list.target_file == "src/bar.py"

    with pytest.raises(ValidationError, match="member of candidates_considered"):
        _build_event(
            resolution_status="resolved",
            target_file="src/qux.py",
            candidates_considered=("src/bar.py", "src/baz.py"),
        )


def test_unresolved_with_nonempty_candidates_admits() -> None:
    """Canonical unresolved-with-non-empty-candidates case per #017 clause (b).

    LLM proposed candidates; ast_facts resolved zero of them. The
    candidates_considered list is the LLM-proposed list (any cardinality);
    resolution_status describes ast_facts-resolution count.
    """
    event = _build_event(
        resolution_status="unresolved",
        target_file=None,
        candidates_considered=("src/foo.py", "src/bar.py"),
    )
    assert event.candidates_considered == ("src/foo.py", "src/bar.py")
    assert event.resolution_status == "unresolved"


def test_missing_candidates_considered_raises() -> None:
    """candidates_considered is REQUIRED (no default) per #017 + replay equivalence.

    Defaulted field would silently absorb emitter bugs. Callers pass ()
    explicitly for the zero-candidate case.
    """
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "source_finding_id": uuid4(),
        "target_file": None,
        "reason": "x",
        "resolution_status": "unresolved",
    }
    with pytest.raises(ValidationError):
        TraceDecisionEvent(**fields)


def test_candidates_considered_admits_empty_tuple() -> None:
    """Explicit candidates_considered=() admits — well-typed zero-candidate case."""
    event = _build_event(
        resolution_status="unresolved",
        target_file=None,
        candidates_considered=(),
    )
    assert event.candidates_considered == ()


def test_candidates_considered_round_trips_as_tuple() -> None:
    """JSON round-trip preserves tuple shape (decodes from JSON array back to tuple)."""
    original = _build_event(
        candidates_considered=("src/bar.py", "src/baz.py"),
    )
    json_payload = original.model_dump_json(exclude={"sequence_number"})
    decoded_dict = json.loads(json_payload)
    assert isinstance(decoded_dict["candidates_considered"], list)

    reconstructed = AuditEventAdapter.validate_json(json_payload)
    assert isinstance(reconstructed, TraceDecisionEvent)
    assert isinstance(reconstructed.candidates_considered, tuple)
    assert reconstructed.candidates_considered == ("src/bar.py", "src/baz.py")
