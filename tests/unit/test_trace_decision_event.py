"""TraceDecisionEvent: resolution_status Literal + nullable target_file + validator rules.

Backs DECISIONS.md#017 (Amended 2026-04-29 same-day + 2026-05-24 by #024).
Per the #024 amendment, field shape:
- `candidates_considered` → renamed to `proposed_import_strings` (admitted
  import strings, two forms per #024 Amended 2026-07-03 — canonicalized
  model candidates plus corrected from-import siblings per the #024
  from-import amendment).
- New `resolved_candidate_paths` carries resolver outputs (file paths).
- Cross-field validator rules rewritten to consult `resolved_candidate_paths`
  cardinality (not `proposed_import_strings` membership):
  - resolved: len(resolved_candidate_paths) == 1 AND target_file ==
    resolved_candidate_paths[0]
  - unresolved: len(resolved_candidate_paths) == 0 AND target_file is None
  - ambiguous: len(resolved_candidate_paths) > 1 AND target_file is None
- Uniqueness validator split into two — one per tuple.
- target_file + every resolved_candidate_paths element pass through
  validate_diff_path at the audit-shadow boundary (#024 point 6).
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
        "proposed_import_strings": ("bar", "baz"),
        "resolved_candidate_paths": ("src/bar.py",),
    }
    fields.update(overrides)
    return TraceDecisionEvent(**fields)


def test_resolution_status_admits_three_canonical_values() -> None:
    """resolved + unresolved + ambiguous all admit (with appropriate
    target_file + resolved_candidate_paths cardinality per the validator)."""
    event_resolved = _build_event(resolution_status="resolved")
    event_unresolved = _build_event(
        resolution_status="unresolved",
        target_file=None,
        resolved_candidate_paths=(),
    )
    event_ambiguous = _build_event(
        resolution_status="ambiguous",
        target_file=None,
        resolved_candidate_paths=("src/foo.py", "src/bar.py"),
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


def test_resolved_admits_with_target_file_matching_resolved_path() -> None:
    """Happy path: resolved + target_file == resolved_candidate_paths[0]."""
    event = _build_event(
        resolution_status="resolved",
        target_file="src/bar.py",
        proposed_import_strings=("bar", "baz"),
        resolved_candidate_paths=("src/bar.py",),
    )
    assert event.target_file == "src/bar.py"


def test_unresolved_admits_with_none_target_file_and_empty_resolved() -> None:
    """unresolved + target_file=None + empty resolved_candidate_paths admits."""
    event = _build_event(
        resolution_status="unresolved",
        target_file=None,
        resolved_candidate_paths=(),
    )
    assert event.target_file is None
    assert event.resolved_candidate_paths == ()


def test_ambiguous_admits_with_none_target_file_and_multiple_resolved() -> None:
    """ambiguous + target_file=None + len(resolved_candidate_paths) > 1 admits."""
    event = _build_event(
        resolution_status="ambiguous",
        target_file=None,
        resolved_candidate_paths=("src/bar.py", "src/baz.py"),
    )
    assert event.target_file is None
    assert len(event.resolved_candidate_paths) == 2


def test_resolved_without_target_file_raises() -> None:
    """Cross-field rule: resolved + target_file=None raises."""
    with pytest.raises(ValidationError, match="non-None target_file"):
        _build_event(
            resolution_status="resolved",
            target_file=None,
            resolved_candidate_paths=("src/bar.py",),
        )


def test_resolved_with_wrong_resolved_count_raises() -> None:
    """Cross-field rule: resolved requires exactly one resolved_candidate_paths."""
    with pytest.raises(ValidationError, match="exactly one"):
        _build_event(
            resolution_status="resolved",
            target_file="src/bar.py",
            resolved_candidate_paths=(),
        )
    with pytest.raises(ValidationError, match="exactly one"):
        _build_event(
            resolution_status="resolved",
            target_file="src/bar.py",
            resolved_candidate_paths=("src/bar.py", "src/baz.py"),
        )


def test_resolved_target_file_must_equal_single_resolved_path() -> None:
    """Cross-field rule per #024 amendment: resolved target_file must
    EQUAL the single resolved_candidate_paths entry (no longer
    membership in proposed_import_strings)."""
    with pytest.raises(ValidationError, match="must equal the single"):
        _build_event(
            resolution_status="resolved",
            target_file="src/qux.py",  # ≠ resolved_candidate_paths[0]
            resolved_candidate_paths=("src/bar.py",),
        )


def test_unresolved_with_target_file_raises() -> None:
    """Cross-field rule: unresolved + non-None target_file raises."""
    with pytest.raises(ValidationError, match="target_file is None"):
        _build_event(
            resolution_status="unresolved",
            target_file="src/bar.py",
            resolved_candidate_paths=(),
        )


def test_unresolved_with_nonempty_resolved_raises() -> None:
    """Cross-field rule: unresolved + non-empty resolved_candidate_paths raises."""
    with pytest.raises(ValidationError, match="zero resolved_candidate_paths"):
        _build_event(
            resolution_status="unresolved",
            target_file=None,
            resolved_candidate_paths=("src/bar.py",),
        )


def test_ambiguous_with_target_file_raises() -> None:
    """Cross-field rule: ambiguous + non-None target_file raises."""
    with pytest.raises(ValidationError, match="target_file is None"):
        _build_event(
            resolution_status="ambiguous",
            target_file="src/bar.py",
            resolved_candidate_paths=("src/bar.py", "src/baz.py"),
        )


def test_ambiguous_with_single_resolved_raises() -> None:
    """Cross-field rule: ambiguous requires more than one resolved_candidate_paths."""
    with pytest.raises(ValidationError, match="more than one"):
        _build_event(
            resolution_status="ambiguous",
            target_file=None,
            resolved_candidate_paths=("src/bar.py",),
        )


def test_unresolved_with_nonempty_proposed_import_strings_admits() -> None:
    """LLM proposed import strings but resolver yielded zero: unresolved
    case per #017 amended clause (b). proposed_import_strings carries
    the LLM's proposals; resolved_candidate_paths is empty."""
    event = _build_event(
        resolution_status="unresolved",
        target_file=None,
        proposed_import_strings=("foo.bar", "foo.baz"),
        resolved_candidate_paths=(),
    )
    assert event.proposed_import_strings == ("foo.bar", "foo.baz")
    assert event.resolved_candidate_paths == ()


def test_missing_proposed_import_strings_raises() -> None:
    """proposed_import_strings is REQUIRED (no default) per #017 +
    replay equivalence. Defaulted field would silently absorb emitter
    bugs. Callers pass () explicitly for the zero-proposal case."""
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "source_finding_id": uuid4(),
        "target_file": None,
        "reason": "x",
        "resolution_status": "unresolved",
        "resolved_candidate_paths": (),
        # proposed_import_strings deliberately omitted
    }
    with pytest.raises(ValidationError):
        TraceDecisionEvent(**fields)


def test_missing_resolved_candidate_paths_raises() -> None:
    """resolved_candidate_paths is REQUIRED (no default) per #024 + #017."""
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "source_finding_id": uuid4(),
        "target_file": None,
        "reason": "x",
        "resolution_status": "unresolved",
        "proposed_import_strings": (),
        # resolved_candidate_paths deliberately omitted
    }
    with pytest.raises(ValidationError):
        TraceDecisionEvent(**fields)


def test_proposed_import_strings_admits_empty_tuple() -> None:
    """Explicit proposed_import_strings=() admits — well-typed
    zero-proposal case."""
    event = _build_event(
        resolution_status="unresolved",
        target_file=None,
        proposed_import_strings=(),
        resolved_candidate_paths=(),
    )
    assert event.proposed_import_strings == ()


def test_round_trips_as_tuples() -> None:
    """JSON round-trip preserves both tuple shapes."""
    original = _build_event(
        proposed_import_strings=("foo.bar", "foo.baz"),
        resolved_candidate_paths=("src/bar.py",),
    )
    json_payload = original.model_dump_json(exclude={"sequence_number"})
    decoded_dict = json.loads(json_payload)
    assert isinstance(decoded_dict["proposed_import_strings"], list)
    assert isinstance(decoded_dict["resolved_candidate_paths"], list)

    reconstructed = AuditEventAdapter.validate_json(json_payload)
    assert isinstance(reconstructed, TraceDecisionEvent)
    assert isinstance(reconstructed.proposed_import_strings, tuple)
    assert isinstance(reconstructed.resolved_candidate_paths, tuple)
    assert reconstructed.proposed_import_strings == ("foo.bar", "foo.baz")
    assert reconstructed.resolved_candidate_paths == ("src/bar.py",)


def test_proposed_import_strings_uniqueness_validator() -> None:
    """proposed_import_strings is set-semantic; duplicates raise. Per
    #024 amendment to #017's uniqueness validator (split into two).
    Matches on field name only — wording of the error message is not
    contract, only the validator-attached-to-this-field is."""
    with pytest.raises(ValidationError, match="proposed_import_strings"):
        _build_event(
            proposed_import_strings=("foo.bar", "foo.bar"),
        )


def test_resolved_candidate_paths_uniqueness_validator() -> None:
    """resolved_candidate_paths is set-semantic; duplicates raise.
    Matches on field name only (see sibling test rationale)."""
    with pytest.raises(ValidationError, match="resolved_candidate_paths"):
        _build_event(
            resolution_status="ambiguous",
            target_file=None,
            resolved_candidate_paths=("src/bar.py", "src/bar.py"),
        )


def test_target_file_audit_shadow_validate_diff_path() -> None:
    """Per #024 point 6: target_file passes through validate_diff_path
    at the audit-event boundary. Traversal-bearing target raises.
    Pydantic V2 re-raises non-ValueError exceptions from field_validators
    directly, so CoordinateError propagates as itself, not wrapped.

    Isolates the failure to `target_file` — keeps `resolved_candidate_paths`
    valid so only the target_file field_validator can fire, ensuring a
    future regression in `_enforce_canonical_target_file` can't be
    masked by `_enforce_canonical_resolved_paths` firing on the same
    payload. The cross-field validator (`target_file ==
    resolved_candidate_paths[0]`) does not get reached because
    Pydantic stops at field-level errors before running model
    validators."""
    from outrider.coordinates import CoordinateError

    # Assert specifically on the `target_file` field validator firing —
    # accepting any ValidationError here would silently pass even if a
    # future refactor deletes the field-level audit shadow, because the
    # model-level `target_file == resolved_candidate_paths[0]` check
    # also raises on this same payload (mismatched strings). Targeting
    # `loc == ("target_file",)` pins the field-level path validator.
    try:
        _build_event(
            target_file="../../etc/passwd",
            resolved_candidate_paths=("src/some/path.py",),
        )
    except CoordinateError:
        pass
    except ValidationError as exc:
        assert any(err["loc"] == ("target_file",) for err in exc.errors()), (
            f"expected target_file field_validator to fire; got errors: {exc.errors()}"
        )
    else:
        pytest.fail("expected target_file validation to fail")


def test_proposed_import_strings_per_element_shape_validation() -> None:
    """Per-element `is_valid_trace_import_string` runs on every entry at the
    audit-event boundary. Defense in depth against a direct emitter
    (replay path, test fixture) that bypasses
    `TraceCandidate.import_string`'s field validator: a path-shaped
    element (forward slash) raises here even though the upstream
    singleton validator would have rejected it earlier."""
    with pytest.raises((ValidationError, ValueError)):
        _build_event(
            proposed_import_strings=("foo.bar", "src/baz.py"),
        )


def test_resolved_candidate_paths_audit_shadow_per_element() -> None:
    """Per #024 point 6: every resolved_candidate_paths element passes
    through validate_diff_path. Even one bad path in the tuple raises.
    Load-bearing for the ambiguous branch where target_file is None.
    Pydantic V2 re-raises non-ValueError exceptions from field_validators
    directly, so CoordinateError propagates as itself, not wrapped."""
    from outrider.coordinates import CoordinateError

    with pytest.raises((ValidationError, CoordinateError)):
        _build_event(
            resolution_status="ambiguous",
            target_file=None,
            resolved_candidate_paths=("src/bar.py", "../../etc/passwd"),
        )


def test_proposed_import_strings_admits_relative_specifier_form() -> None:
    """Two-form contract per DECISIONS.md#024 (Amended 2026-07-03): the
    audit-shadow validator admits specifier-form entries alongside
    module form, sorted canonical — state <-> audit lockstep for both
    forms."""
    e = _build_event(proposed_import_strings=("foo.bar", "../db"))
    assert e.proposed_import_strings == ("../db", "foo.bar")


def test_proposed_import_strings_rejects_malformed_specifier() -> None:
    """Interior `..` rejects on the specifier branch at the audit
    boundary too — no malformed specifier persists into audit_events."""
    with pytest.raises(ValidationError):
        _build_event(proposed_import_strings=("./a/../b",))
