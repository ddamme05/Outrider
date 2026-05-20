# See specs/2026-05-19-analyze-foundation.md §1.
"""`AnalysisRound` per-pass results record tests.

Pins the §1 schema discipline: frozen + `extra="forbid"`, SHA-256 hex
`round_id`, `pass_index >= 0`, AwareDatetime timestamps, tuple-typed
`findings` / `files_examined` / `files_skipped`. Construction via the
canonical identity-hash recipe so `round_id` is content-derived.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import compute_identity_hash
from outrider.schemas import AnalysisRound, ReviewDimension, ReviewFinding


def _finding() -> ReviewFinding:
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=12345,
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        file_path="src/foo.py",
        line_start=10,
        line_end=12,
        title="SQL injection in raw query",
        description="User input is concatenated into the SQL string.",
        evidence="raw SQL string concatenation at src/foo.py:11",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version="1.0.0",
        content_hash=compute_identity_hash(
            {
                "file_path": "src/foo.py",
                "line_start": 10,
                "line_end": 12,
                "finding_type": "sql_injection",
            }
        ),
    )


def _round_kwargs(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    base: dict[str, object] = {
        "round_id": compute_identity_hash({"pass_index": 0, "fixture": "test"}),
        "pass_index": 0,
        "findings": (_finding(),),
        "files_examined": ("src/foo.py",),
        "files_skipped": (),
        "started_at": now,
        "ended_at": now,
    }
    base.update(overrides)
    return base


def test_analysis_round_admits_well_formed() -> None:
    r = AnalysisRound(**_round_kwargs())  # type: ignore[arg-type]
    assert r.pass_index == 0
    assert len(r.findings) == 1
    assert r.files_examined == ("src/foo.py",)


def test_analysis_round_round_id_rejects_non_hex() -> None:
    """`round_id` must match the canonical SHA-256 hex pattern."""
    with pytest.raises(ValidationError):
        AnalysisRound(**_round_kwargs(round_id="not-a-hash"))  # type: ignore[arg-type]


def test_analysis_round_round_id_rejects_short_hex() -> None:
    with pytest.raises(ValidationError):
        AnalysisRound(**_round_kwargs(round_id="a" * 63))  # type: ignore[arg-type]


def test_analysis_round_round_id_admits_canonical_hash() -> None:
    """The canonical recipe produces a valid round_id."""
    rid = compute_identity_hash({"any": "payload"})
    r = AnalysisRound(**_round_kwargs(round_id=rid))  # type: ignore[arg-type]
    assert r.round_id == rid


def test_analysis_round_pass_index_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        AnalysisRound(**_round_kwargs(pass_index=-1))  # type: ignore[arg-type]


def test_analysis_round_rejects_naive_started_at() -> None:
    """`AwareDatetime` rejects naive datetimes per the schema discipline."""
    with pytest.raises(ValidationError):
        AnalysisRound(**_round_kwargs(started_at=datetime(2026, 5, 20)))  # type: ignore[arg-type]


def test_analysis_round_frozen_rejects_mutation() -> None:
    r = AnalysisRound(**_round_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        r.pass_index = 99  # type: ignore[misc]


def test_analysis_round_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AnalysisRound(**_round_kwargs(unexpected_field="bad"))  # type: ignore[arg-type]


def test_analysis_round_findings_coerces_list_to_tuple() -> None:
    """Pydantic V2 list→tuple coercion preserves the frozen-tuple
    immutability guarantee on the stored field."""
    r = AnalysisRound(**_round_kwargs(findings=[_finding()]))  # type: ignore[arg-type]
    assert isinstance(r.findings, tuple)


def test_analysis_round_files_skipped_admits_empty() -> None:
    """A clean pass with no skips is the common case — pin the empty tuple."""
    r = AnalysisRound(**_round_kwargs(files_skipped=()))  # type: ignore[arg-type]
    assert r.files_skipped == ()


def test_analysis_round_files_examined_path_length_bounded() -> None:
    """Per-element `max_length=1024` defends against pathologically long paths."""
    long_path = "src/" + ("a" * 1025) + ".py"
    with pytest.raises(ValidationError):
        AnalysisRound(**_round_kwargs(files_examined=(long_path,)))  # type: ignore[arg-type]
