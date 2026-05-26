# See specs/2026-05-23-trace-node.md M7 (c).
"""`_payload_identity_subset` golden-pin tests.

Per spec M7 (c) + DECISIONS.md#026, the natural-key idempotency mode's
identity-subset comparison MUST match a specific enumeration so retries
collapse cleanly to no-ops while real producer divergence raises.

This file pins the canonical subset for `trace_decision` and the
cross-check against `TraceDecisionEvent`'s actual payload fields.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from outrider.audit.events import TraceDecisionEvent
from outrider.audit.persister import (
    _payload_identity_subset,
    _serialize_event_payload,
)


def test_trace_decision_identity_subset_matches_spec_enumeration() -> None:
    """Golden-pin: the M7 (c) enumeration is exactly
    `{source_finding_id, target_file, resolution_status,
    resolved_candidate_paths, is_eval}`. Drift here is a spec-versus-
    impl divergence and should be a deliberate spec update with a
    `DECISIONS.md` paired change, not a silent rename / addition /
    removal.

    `resolved_candidate_paths` is in the subset because `target_file`
    is None for `resolution_status='ambiguous'` outcomes and can't
    carry the divergence signal alone; set-canonicalized via
    `_canonicalize_for_identity_compare` so probe-order shuffle on
    legitimate retries doesn't trip identity-equality."""
    subset = _payload_identity_subset("trace_decision")
    assert subset == frozenset(
        {
            "source_finding_id",
            "target_file",
            "resolution_status",
            "resolved_candidate_paths",
            "is_eval",
        }
    )


def test_trace_decision_identity_subset_is_frozen() -> None:
    """`frozenset` return type pins the no-mutation contract — a caller
    cannot widen the subset by `.add(...)` and pollute the persister's
    comparison set."""
    subset = _payload_identity_subset("trace_decision")
    assert isinstance(subset, frozenset)


def test_identity_subset_unsupported_event_type_raises() -> None:
    """V1 supports `trace_decision` only — natural-key mode per #026
    has no other instance. Unsupported event types are a producer-side
    routing bug; fail loud at the helper to surface the bug at the
    persister boundary rather than silently admit a wrong-mode write."""
    with pytest.raises(ValueError, match="unsupported event_type"):
        _payload_identity_subset("finding")
    with pytest.raises(ValueError, match="unsupported event_type"):
        _payload_identity_subset("publish_routing")
    with pytest.raises(ValueError, match="unsupported event_type"):
        _payload_identity_subset("")


def test_trace_decision_identity_subset_is_subset_of_actual_payload_fields() -> None:
    """Cross-check: every name in the subset MUST appear in the actual
    `TraceDecisionEvent` serialized payload. A typo / rename in the
    enumeration would silently bypass the comparison (the persister's
    `.get(field)` would return `None` on both sides and falsely declare
    equality)."""
    event = TraceDecisionEvent(
        review_id=uuid4(),
        timestamp=datetime.now(UTC),
        source_finding_id=uuid4(),
        target_file="src/foo.py",
        reason="x",
        resolution_status="resolved",
        proposed_import_strings=("foo",),
        resolved_candidate_paths=("src/foo.py",),
    )
    payload = _serialize_event_payload(event)
    subset = _payload_identity_subset("trace_decision")
    missing = subset - set(payload)
    assert not missing, (
        f"identity-subset names absent from serialized payload: {sorted(missing)}; "
        "either the schema dropped a field or the subset names drifted"
    )


def test_trace_decision_identity_subset_explicitly_excludes_per_emission_fields() -> None:
    """Per M7 (b)+(c) rationale, per-emission fields (`event_id`,
    `timestamp`, `reason`, `proposed_import_strings`, `trace_path`)
    are EXCLUDED so legitimate retries (which produce fresh values
    for these) collapse to no-ops rather than firing
    `AuditPersisterNaturalKeyConflict`. `resolved_candidate_paths`
    was added BACK to the identity subset (set-canonicalized) because
    it's the only divergence signal for ambiguous outcomes —
    `target_file=None` for ambiguous, so a tree-mutation mid-PR would
    silently no-op without this field. See the spec-enumeration test
    above for the full subset."""
    subset = _payload_identity_subset("trace_decision")
    excluded = {
        "event_id",
        "timestamp",
        "reason",
        "proposed_import_strings",
        "trace_path",
        "review_id",  # tautological (index lookup pins it)
        "event_type",  # tautological (partial-index WHERE pins it)
        "sequence_number",  # already excluded by _serialize_event_payload
    }
    overlap = subset & excluded
    assert not overlap, (
        f"identity-subset MUST NOT include per-emission / tautological "
        f"fields; got overlap: {sorted(overlap)}"
    )
