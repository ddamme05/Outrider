# See specs/2026-05-23-trace-node.md (Q3) and DECISIONS.md#017 × #024.
"""`TraceDecision` schema (state-layer mirror of `TraceDecisionEvent`).

Pins the schema discipline: frozen + extra="forbid", parallel
proposed_import_strings + resolved_candidate_paths tuples per #024
amendment, three-rule cross-field validator, split uniqueness
validators, audit-shadow validate_diff_path on target_file +
resolved_candidate_paths per-element per #024 point 6.

The state-layer mirror's tests run independently of the audit-event
tests — same shape, different consumer (state-side reducer vs.
append-only audit log) — so a producer that constructs one but not the
other is caught at fixture-construction time, not at lift-to-audit time.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.coordinates import CoordinateError
from outrider.schemas import TraceDecision


def _build(**overrides: Any) -> TraceDecision:
    fields: dict[str, Any] = {
        "source_finding_id": uuid4(),
        "target_file": "src/bar.py",
        "reason": "called from middleware/auth.py:42",
        "resolution_status": "resolved",
        "proposed_import_strings": ("bar", "baz"),
        "resolved_candidate_paths": ("src/bar.py",),
    }
    fields.update(overrides)
    return TraceDecision(**fields)


# ----------------------------------------------------------------------------
# Happy paths — resolved / unresolved / ambiguous each admit with the
# correct field shape.
# ----------------------------------------------------------------------------


def test_resolved_admits_with_matching_target_file() -> None:
    d = _build()
    assert d.resolution_status == "resolved"
    assert d.target_file == "src/bar.py"
    assert d.resolved_candidate_paths == ("src/bar.py",)


def test_unresolved_admits_with_none_target_file_and_empty_resolved() -> None:
    d = _build(
        resolution_status="unresolved",
        target_file=None,
        proposed_import_strings=("foo.missing",),
        resolved_candidate_paths=(),
    )
    assert d.resolution_status == "unresolved"
    assert d.target_file is None
    assert d.resolved_candidate_paths == ()


def test_ambiguous_admits_with_none_target_file_and_multiple_resolved() -> None:
    d = _build(
        resolution_status="ambiguous",
        target_file=None,
        proposed_import_strings=("foo",),
        resolved_candidate_paths=("src/foo.py", "src/foo/__init__.py"),
    )
    assert d.resolution_status == "ambiguous"
    assert d.target_file is None
    assert len(d.resolved_candidate_paths) == 2


# ----------------------------------------------------------------------------
# Cross-field validator rules per #017 × #024 amendment.
# ----------------------------------------------------------------------------


def test_resolved_requires_exactly_one_resolved_path() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        _build(
            resolution_status="resolved",
            target_file="src/bar.py",
            resolved_candidate_paths=(),
        )
    with pytest.raises(ValidationError, match="exactly one"):
        _build(
            resolution_status="resolved",
            target_file="src/bar.py",
            resolved_candidate_paths=("src/bar.py", "src/baz.py"),
        )


def test_resolved_target_file_must_equal_single_resolved_path() -> None:
    """Per #024 point 5 (rule a): target_file == resolved_candidate_paths[0]."""
    with pytest.raises(ValidationError, match="must equal the single"):
        _build(
            resolution_status="resolved",
            target_file="src/qux.py",
            resolved_candidate_paths=("src/bar.py",),
        )


def test_resolved_with_none_target_file_raises() -> None:
    with pytest.raises(ValidationError, match="non-None target_file"):
        _build(
            resolution_status="resolved",
            target_file=None,
            resolved_candidate_paths=("src/bar.py",),
        )


def test_unresolved_with_nonempty_resolved_raises() -> None:
    with pytest.raises(ValidationError, match="zero resolved_candidate_paths"):
        _build(
            resolution_status="unresolved",
            target_file=None,
            resolved_candidate_paths=("src/bar.py",),
        )


def test_unresolved_with_target_file_raises() -> None:
    with pytest.raises(ValidationError, match="target_file is None"):
        _build(
            resolution_status="unresolved",
            target_file="src/bar.py",
            resolved_candidate_paths=(),
        )


def test_ambiguous_with_single_resolved_raises() -> None:
    with pytest.raises(ValidationError, match="more than one"):
        _build(
            resolution_status="ambiguous",
            target_file=None,
            resolved_candidate_paths=("src/bar.py",),
        )


def test_ambiguous_with_target_file_raises() -> None:
    with pytest.raises(ValidationError, match="target_file is None"):
        _build(
            resolution_status="ambiguous",
            target_file="src/bar.py",
            resolved_candidate_paths=("src/bar.py", "src/baz.py"),
        )


# ----------------------------------------------------------------------------
# Required fields — no default per #017 + replay equivalence.
# ----------------------------------------------------------------------------


def test_missing_proposed_import_strings_raises() -> None:
    fields: dict[str, Any] = {
        "source_finding_id": uuid4(),
        "target_file": None,
        "reason": "x",
        "resolution_status": "unresolved",
        "resolved_candidate_paths": (),
        # proposed_import_strings deliberately omitted
    }
    with pytest.raises(ValidationError):
        TraceDecision(**fields)


def test_missing_resolved_candidate_paths_raises() -> None:
    fields: dict[str, Any] = {
        "source_finding_id": uuid4(),
        "target_file": None,
        "reason": "x",
        "resolution_status": "unresolved",
        "proposed_import_strings": (),
        # resolved_candidate_paths deliberately omitted
    }
    with pytest.raises(ValidationError):
        TraceDecision(**fields)


# ----------------------------------------------------------------------------
# Split uniqueness validators per #024 amendment.
# ----------------------------------------------------------------------------


def test_proposed_import_strings_uniqueness_validator() -> None:
    with pytest.raises(ValidationError, match="proposed_import_strings contains duplicates"):
        _build(proposed_import_strings=("foo.bar", "foo.bar"))


def test_resolved_candidate_paths_uniqueness_validator() -> None:
    with pytest.raises(ValidationError, match="resolved_candidate_paths contains duplicates"):
        _build(
            resolution_status="ambiguous",
            target_file=None,
            resolved_candidate_paths=("src/bar.py", "src/bar.py"),
        )


# ----------------------------------------------------------------------------
# Audit-shadow validate_diff_path per #024 point 6.
# ----------------------------------------------------------------------------


def test_target_file_audit_shadow_rejects_traversal() -> None:
    """Pydantic V2 re-raises non-ValueError exceptions from field_validators
    directly, so CoordinateError propagates as itself, not wrapped."""
    with pytest.raises((ValidationError, CoordinateError)):
        _build(
            target_file="../../etc/passwd",
            resolved_candidate_paths=("../../etc/passwd",),
        )


def test_resolved_candidate_paths_audit_shadow_per_element() -> None:
    """Even one bad path in the tuple raises. Load-bearing for the
    ambiguous branch where target_file is None but the tuple still
    carries multiple paths."""
    with pytest.raises((ValidationError, CoordinateError)):
        _build(
            resolution_status="ambiguous",
            target_file=None,
            resolved_candidate_paths=("src/bar.py", "../../etc/passwd"),
        )


def test_proposed_import_strings_per_element_is_valid_import_string() -> None:
    """Per-element `is_valid_import_string` runs on every entry. A
    path-shaped element (forward slash) bypasses the
    upstream `TraceCandidate.import_string` validator only when a
    direct emitter constructs `TraceDecision` without flowing through
    the candidate path. The schema-layer per-element validator catches
    it — defense in depth at the collection boundary.
    """
    with pytest.raises((ValidationError, ValueError)):
        _build(proposed_import_strings=("foo.bar", "src/baz.py"))


# ----------------------------------------------------------------------------
# Frozen + extra="forbid" per cross-boundary schema discipline.
# ----------------------------------------------------------------------------


def test_frozen_rejects_mutation() -> None:
    d = _build()
    with pytest.raises(ValidationError):
        d.target_file = "src/other.py"  # type: ignore[misc]


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _build(unexpected="bad")


def test_reason_max_length_500() -> None:
    with pytest.raises(ValidationError):
        _build(reason="x" * 501)
