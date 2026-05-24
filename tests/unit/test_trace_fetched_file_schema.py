# See specs/2026-05-23-trace-node.md (Q3 resolution).
"""`TraceFetchedFile` schema tests.

Pins frozen + extra="forbid", path field validator (re-runs
validate_diff_path), required fields (no defaults), three-field V1
shape (path / content_head / source_finding_id) per spec Q3 revision
post-Codex round-7 (no source_import_string / source_proposal_hash —
those would diverge under retry; provenance recovers via
state.trace_decisions cross-reference).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.coordinates import CoordinateError
from outrider.schemas import TraceFetchedFile


def _build(**overrides: Any) -> TraceFetchedFile:
    fields: dict[str, Any] = {
        "path": "src/middleware/auth.py",
        "content_head": "def authenticate(token: str) -> bool:\n    return True\n",
        "source_finding_id": uuid4(),
    }
    fields.update(overrides)
    return TraceFetchedFile(**fields)


def test_admits_well_formed() -> None:
    f = _build()
    assert f.path == "src/middleware/auth.py"
    assert "authenticate" in f.content_head


def test_frozen_rejects_mutation() -> None:
    f = _build()
    with pytest.raises(ValidationError):
        f.path = "src/other.py"  # type: ignore[misc]


def test_rejects_extra_fields() -> None:
    """Per Codex round-7: no source_import_string / source_proposal_hash
    on this schema — those would diverge under retry. Schema is
    extra='forbid' so passing them raises (pit-of-success: a future
    contributor copy-pasting the old field set fails loud)."""
    with pytest.raises(ValidationError):
        _build(source_import_string="middleware.auth")
    with pytest.raises(ValidationError):
        _build(source_proposal_hash="a" * 64)


def test_path_max_length_1024() -> None:
    """1024-char cap matches sibling path-bearing fields
    (ReviewFinding.file_path, TraceCandidate.import_string)."""
    with pytest.raises(ValidationError):
        _build(path="src/" + ("a" * 1100) + ".py")


def test_path_audit_shadow_rejects_traversal() -> None:
    """The path field validator re-runs validate_diff_path so a
    traversal-bearing path is refused at the schema boundary. Pydantic V2
    re-raises non-ValueError exceptions from field_validators directly,
    so CoordinateError propagates as itself, not wrapped."""
    with pytest.raises((ValidationError, CoordinateError)):
        _build(path="../../etc/passwd")


def test_missing_path_raises() -> None:
    with pytest.raises(ValidationError):
        TraceFetchedFile(  # type: ignore[call-arg]
            content_head="x",
            source_finding_id=uuid4(),
        )


def test_missing_content_head_raises() -> None:
    with pytest.raises(ValidationError):
        TraceFetchedFile(  # type: ignore[call-arg]
            path="src/foo.py",
            source_finding_id=uuid4(),
        )


def test_missing_source_finding_id_raises() -> None:
    with pytest.raises(ValidationError):
        TraceFetchedFile(  # type: ignore[call-arg]
            path="src/foo.py",
            content_head="x",
        )


def test_path_normalizes_via_validate_diff_path() -> None:
    """Alias paths like `./src/foo.py` canonicalize to `src/foo.py` via
    the field validator (validate_diff_path normalization)."""
    f = _build(path="./src/middleware/auth.py")
    assert f.path == "src/middleware/auth.py"
