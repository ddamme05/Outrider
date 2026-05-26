# See specs/2026-05-23-trace-node.md M7 (b) + DECISIONS.md#026.
"""`AuditPersisterNaturalKeyConflict` + `AuditPersisterTraceIdempotencyLookupError`
constructor + metadata-only contract tests.

Sibling of `test_audit_persister.py` but scoped to the natural-key
exception types added per M7 (b). The metadata-only allowlist
inventory test in `test_audit_persister.py` already pins membership;
this file pins constructor shape + str() content.

DB-touching behavior tests (`_persist_keyed_by_natural_key` happy path
/ no-op path / conflict path) live in
`tests/integration/test_audit_persister_natural_key.py` per spec
implementation-sketch group 4.
"""

from __future__ import annotations

import inspect
from uuid import uuid4

import pytest

from outrider.audit.persister import (
    AuditPersisterNaturalKeyConflict,
    AuditPersisterTraceIdempotencyLookupError,
)

# ---------------------------------------------------------------------------
# AuditPersisterNaturalKeyConflict
# ---------------------------------------------------------------------------


def test_natural_key_conflict_keyword_only_construction() -> None:
    """Constructor is `*`-keyword-only; positional args raise TypeError.
    Mirrors AuditPersisterIdempotencyConflict's keyword-only convention."""
    existing = uuid4()
    incoming = uuid4()
    review = uuid4()
    finding = uuid4()
    with pytest.raises(TypeError):
        AuditPersisterNaturalKeyConflict(  # type: ignore[misc]
            existing, incoming, review, (("source_finding_id", str(finding)),), ("target_file",)
        )

    # Keyword form succeeds — trace's natural-key shape with the JSONB
    # component named via the generalized `natural_key` tuple.
    AuditPersisterNaturalKeyConflict(
        existing_event_id=existing,
        incoming_event_id=incoming,
        review_id=review,
        natural_key=(("source_finding_id", str(finding)),),
        mismatched_fields=("target_file",),
    )


def test_natural_key_conflict_str_carries_only_metadata() -> None:
    """`str()` contains schema identifiers (UUIDs, mismatched-field names)
    only — never raw payload values. The mismatched_fields names ARE
    class-level identifiers (the M7 (c) enumeration), not content."""
    secret = "OUTRIDER_SECRET_VALUE_DO_NOT_LEAK_xyz789"  # noqa: S105 — test sentinel
    existing = uuid4()
    incoming = uuid4()
    review = uuid4()
    finding = uuid4()

    exc = AuditPersisterNaturalKeyConflict(
        existing_event_id=existing,
        incoming_event_id=incoming,
        review_id=review,
        natural_key=(("source_finding_id", str(finding)),),
        mismatched_fields=("target_file", "resolution_status"),
    )
    rendered = str(exc)
    assert str(existing) in rendered
    assert str(incoming) in rendered
    assert str(review) in rendered
    assert str(finding) in rendered
    assert "target_file" in rendered
    assert "resolution_status" in rendered
    # No value injection surface — the constructor has no kwarg through
    # which a caller could pass a payload string.
    sig = inspect.signature(AuditPersisterNaturalKeyConflict.__init__)
    assert set(sig.parameters) - {"self"} == {
        "existing_event_id",
        "incoming_event_id",
        "review_id",
        "natural_key",
        "mismatched_fields",
    }
    assert secret not in rendered
    assert secret not in repr(exc)


def test_natural_key_conflict_attributes_round_trip() -> None:
    """Constructor stores every kwarg as an attribute under the same
    name — operators query attributes directly from logs, not from
    str() parsing."""
    existing = uuid4()
    incoming = uuid4()
    review = uuid4()
    finding = uuid4()
    fields = ("target_file",)

    exc = AuditPersisterNaturalKeyConflict(
        existing_event_id=existing,
        incoming_event_id=incoming,
        review_id=review,
        natural_key=(("source_finding_id", str(finding)),),
        mismatched_fields=fields,
    )
    assert exc.existing_event_id == existing
    assert exc.incoming_event_id == incoming
    assert exc.review_id == review
    # Backward-compat accessor: when `natural_key` carries `source_finding_id`
    # the property reconstitutes the UUID for trace's existing consumers.
    assert exc.source_finding_id == finding
    assert exc.natural_key == (("source_finding_id", str(finding)),)
    assert exc.mismatched_fields == fields


def test_natural_key_conflict_rejects_empty_mismatched_fields() -> None:
    """Construction with empty mismatched_fields would mean the caller
    raised the exception on identity-subset EQUALITY, which is the
    no-op recovery path (return existing event), not a conflict.
    Fail loud on the caller bug."""
    with pytest.raises(ValueError, match="non-empty mismatched_fields"):
        AuditPersisterNaturalKeyConflict(
            existing_event_id=uuid4(),
            incoming_event_id=uuid4(),
            review_id=uuid4(),
            natural_key=(("source_finding_id", str(uuid4())),),
            mismatched_fields=(),
        )


def test_natural_key_conflict_is_value_error_subclass() -> None:
    """Subclasses `ValueError` like its sibling
    `AuditPersisterIdempotencyConflict`. Callers `except ValueError`
    catch both."""
    assert issubclass(AuditPersisterNaturalKeyConflict, ValueError)


# ---------------------------------------------------------------------------
# AuditPersisterTraceIdempotencyLookupError
# ---------------------------------------------------------------------------


def test_trace_idempotency_lookup_error_keyword_only_construction() -> None:
    """Keyword-only constructor — same shape as
    `AuditPersisterReviewNotFoundError`."""
    review = uuid4()
    finding = uuid4()

    with pytest.raises(TypeError):
        AuditPersisterTraceIdempotencyLookupError(review, finding)  # type: ignore[misc]

    AuditPersisterTraceIdempotencyLookupError(review_id=review, source_finding_id=finding)


def test_trace_idempotency_lookup_error_str_carries_only_metadata() -> None:
    """`str()` carries the natural-key UUIDs only — no payload content."""
    secret = "OUTRIDER_SECRET_VALUE_DO_NOT_LEAK_xyz789"  # noqa: S105 — test sentinel
    review = uuid4()
    finding = uuid4()

    exc = AuditPersisterTraceIdempotencyLookupError(review_id=review, source_finding_id=finding)
    rendered = str(exc)
    assert str(review) in rendered
    assert str(finding) in rendered
    sig = inspect.signature(AuditPersisterTraceIdempotencyLookupError.__init__)
    assert set(sig.parameters) - {"self"} == {"review_id", "source_finding_id"}
    assert secret not in rendered
    assert secret not in repr(exc)


def test_trace_idempotency_lookup_error_is_lookup_error_subclass() -> None:
    """Subclasses `LookupError` — distinguishes lookup failure from a
    generic value-error class (`AuditPersisterNaturalKeyConflict` is
    the conflict, this is the lookup-miss)."""
    assert issubclass(AuditPersisterTraceIdempotencyLookupError, LookupError)
