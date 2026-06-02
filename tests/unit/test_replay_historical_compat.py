"""Read-side replay tolerance for historical events that predate a now-required
provenance field, plus the write-time reserved-sentinel guard. See
DECISIONS.md#032 / FUP-136 / specs/2026-06-01-replay-event-schema-compat.md.
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import (
    REPLAY_HISTORICAL_CONTEXT_KEY,
    REPLAY_TOLERABLE_SENTINEL_FIELDS,
    RESERVED_HISTORICAL_PROPOSAL_HASH,
    AuditEventAdapter,
    FindingEvent,
    FindingProposalRejectedEvent,
    compute_finding_content_hash,
)
from outrider.audit.replay import _HISTORICAL_FIELD_DEFAULTS, _normalize_historical_payload
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.schemas import ReviewDimension

_FILE = "src/foo.py"


def _valid_finding_event(**overrides: Any) -> FindingEvent:
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "finding_id": uuid4(),
        "finding_type": FindingType.SQL_INJECTION,
        "severity": FindingSeverity.CRITICAL,
        "file_path": _FILE,
        "line_start": 10,
        "line_end": 12,
        "dimension": ReviewDimension.SECURITY,
        "finding_content_hash": compute_finding_content_hash(
            file_path=_FILE, line_start=10, line_end=12, finding_type=FindingType.SQL_INJECTION
        ),
        "evidence_tier": EvidenceTier.JUDGED,
        "policy_version": "1.0.0",
        "proposal_hash": "a" * 64,
    }
    fields.update(overrides)
    return FindingEvent(**fields)


def _rejected_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "review_id": uuid4(),
        "file_path": _FILE,
        "proposal_hash": "c" * 64,
        "claimed_evidence_tier": EvidenceTier.JUDGED,
        "claimed_finding_type_hash": "abcdef0123456789",
        "claimed_finding_type_len": 12,
        "rejection_reason": "span_outside_scope_unit",
        "rejection_detail": "(100,200)",
    }
    base.update(overrides)
    return base


# --- Registry allowlist (the two-part gate from the #032 review) ---


def test_registry_is_exactly_the_v1_approved_pair() -> None:
    """Fails if `_HISTORICAL_FIELD_DEFAULTS` contains anything other than the
    one approved pair — adding a field is a conscious change that must update
    this test (and route the provenance-only justification through review).
    """
    assert _HISTORICAL_FIELD_DEFAULTS == {
        "finding": {"proposal_hash": RESERVED_HISTORICAL_PROPOSAL_HASH}
    }


def test_registry_never_defaults_a_proof_or_content_field() -> None:
    """Independent of the exact set: no registry entry may ever default a
    proof-boundary field or a content/equivalence (finding_content_hash recipe)
    field. Survives legitimate future growth of the registry.
    """
    forbidden = {
        # proof boundary
        "evidence_tier",
        "query_match_id",
        "trace_path",
        # content / equivalence (finding_content_hash + its recipe inputs)
        "finding_content_hash",
        "file_path",
        "line_start",
        "line_end",
        "finding_type",
    }
    for event_type, fields in _HISTORICAL_FIELD_DEFAULTS.items():
        overlap = set(fields) & forbidden
        assert not overlap, f"{event_type}: registry must never default {overlap}"


def test_registry_keys_match_the_guard_tolerable_pairs() -> None:
    """The replay registry (what the normalizer injects) and the write-time
    guards' tolerable-pair set must be identical — otherwise the guard could
    permit a sentinel the normalizer never injects, or vice versa.
    """
    registry_pairs = {
        (event_type, field)
        for event_type, fields in _HISTORICAL_FIELD_DEFAULTS.items()
        for field in fields
    }
    assert registry_pairs == set(REPLAY_TOLERABLE_SENTINEL_FIELDS)


# --- Write-time reserve guard (sentinel rejected at construction) ---


def test_finding_event_rejects_reserved_sentinel() -> None:
    with pytest.raises(ValidationError, match="reserved all-zero sentinel"):
        _valid_finding_event(proposal_hash=RESERVED_HISTORICAL_PROPOSAL_HASH)


def test_finding_proposal_rejected_event_rejects_reserved_sentinel() -> None:
    with pytest.raises(ValidationError, match="reserved all-zero sentinel"):
        FindingProposalRejectedEvent(
            **_rejected_kwargs(proposal_hash=RESERVED_HISTORICAL_PROPOSAL_HASH)
        )


def test_finding_event_still_requires_proposal_hash() -> None:
    """Missing (not just reserved) proposal_hash still raises at write time —
    the tolerance did not leak into the canonical model.
    """
    fields = {k: v for k, v in _valid_finding_event().model_dump().items() if k != "proposal_hash"}
    with pytest.raises(ValidationError, match="proposal_hash"):
        FindingEvent(**fields)


# --- Normalizer (read-side, in-memory, non-mutating) ---


def test_normalizer_injects_sentinel_when_absent() -> None:
    payload = _valid_finding_event().model_dump(mode="json")
    del payload["proposal_hash"]
    out = _normalize_historical_payload(payload)
    assert out["proposal_hash"] == RESERVED_HISTORICAL_PROPOSAL_HASH
    assert "proposal_hash" not in payload  # original dict untouched (shallow copy)


def test_normalizer_leaves_present_value_untouched() -> None:
    payload = _valid_finding_event(proposal_hash="b" * 64).model_dump(mode="json")
    assert _normalize_historical_payload(payload)["proposal_hash"] == "b" * 64


def test_normalizer_ignores_non_registered_event_types() -> None:
    payload = {"event_type": "llm_call", "foo": 1}
    assert _normalize_historical_payload(payload) is payload


# --- Round-trip through the adapter (the actual replay path) ---


def test_historical_finding_reconstructs_under_replay_context() -> None:
    """A pre-#025 finding payload (no proposal_hash) reconstructs under the
    replay context, carrying the sentinel; the equivalence-bearing content hash
    is unchanged by the provenance default.
    """
    payload = _valid_finding_event().model_dump(mode="json")
    original_content_hash = payload["finding_content_hash"]
    del payload["proposal_hash"]
    rebuilt = AuditEventAdapter.validate_python(
        _normalize_historical_payload(payload),
        context={REPLAY_HISTORICAL_CONTEXT_KEY: True},
    )
    assert rebuilt.proposal_hash == RESERVED_HISTORICAL_PROPOSAL_HASH
    assert rebuilt.finding_content_hash == original_content_hash


def test_sentinel_rejected_without_replay_context() -> None:
    """The sentinel is permitted ONLY under the replay context — a normal
    validate_python (write/other read paths) still rejects it.
    """
    payload = _valid_finding_event().model_dump(mode="json")
    payload["proposal_hash"] = RESERVED_HISTORICAL_PROPOSAL_HASH
    with pytest.raises(ValidationError, match="reserved all-zero sentinel"):
        AuditEventAdapter.validate_python(payload)


def test_unregistered_pair_rejects_sentinel_even_under_replay_context() -> None:
    """The replay permission is PAIR-scoped, not context-wide (DECISIONS.md#032).
    `finding_proposal_rejected`/`proposal_hash` is NOT a registered tolerable
    pair, so a persisted row carrying the reserved sentinel must still fail —
    even under the replay context — because the normalizer never injects it
    there and its presence would be corruption.
    """
    payload = FindingProposalRejectedEvent(**_rejected_kwargs()).model_dump(mode="json")
    payload["proposal_hash"] = RESERVED_HISTORICAL_PROPOSAL_HASH
    with pytest.raises(ValidationError, match="reserved all-zero sentinel"):
        AuditEventAdapter.validate_python(
            _normalize_historical_payload(payload),
            context={REPLAY_HISTORICAL_CONTEXT_KEY: True},
        )


def test_proof_field_absence_still_raises_under_replay_context() -> None:
    """Tolerance is whitelisted to provenance: an OBSERVED finding missing its
    query_match_id is genuine corruption and must still fail, even at replay.
    """
    payload = _valid_finding_event(
        evidence_tier=EvidenceTier.OBSERVED, query_match_id="sql_string_format.scm"
    ).model_dump(mode="json")
    del payload["query_match_id"]  # corruption the normalizer must NOT paper over
    del payload["proposal_hash"]  # also historical
    with pytest.raises(ValidationError):
        AuditEventAdapter.validate_python(
            _normalize_historical_payload(payload),
            context={REPLAY_HISTORICAL_CONTEXT_KEY: True},
        )
