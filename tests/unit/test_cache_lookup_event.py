# Per specs/2026-06-11-file-hash-analyze-cache.md — shadow-event contract pins.
"""CacheLookupEvent: discriminator round-trip, the path audit-shadow,
and the bounded outcome/key shapes."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import AuditEventAdapter, CacheLookupEvent
from outrider.coordinates import CoordinateError


def _event(**overrides: object) -> CacheLookupEvent:
    kwargs: dict = {
        "review_id": uuid4(),
        "timestamp": datetime.now(UTC),
        "sequence_number": 1,
        "is_eval": True,
        "file_path": "src/example.py",
        "outcome": "miss",
        "cache_key": "a" * 64,
    }
    kwargs.update(overrides)
    return CacheLookupEvent(**kwargs)


def test_discriminator_round_trip() -> None:
    event = _event(outcome="would_hit")
    rebuilt = AuditEventAdapter.validate_python(event.model_dump(mode="json"))
    assert type(rebuilt) is CacheLookupEvent
    assert rebuilt.node_id == "analyze"
    assert rebuilt.outcome == "would_hit"


def test_file_path_audit_shadow_rejects_traversal() -> None:
    with pytest.raises(CoordinateError):
        _event(file_path="../evil.py")


def test_outcome_is_bounded() -> None:
    with pytest.raises(ValidationError):
        _event(outcome="hit")  # serve-stage vocabulary doesn't exist in shadow


def test_cache_key_must_be_sha256_hex() -> None:
    with pytest.raises(ValidationError):
        _event(cache_key="not-a-digest")
