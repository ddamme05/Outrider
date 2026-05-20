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

from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import compute_identity_hash, compute_round_id
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
        content_hash=compute_finding_content_hash(
            file_path="src/foo.py",
            line_start=10,
            line_end=12,
            finding_type=FindingType.SQL_INJECTION,
        ),
    )


def _round_kwargs(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    finding = _finding()
    # Derive round_id from the canonical recipe so the model validator
    # passes. Tests that DELIBERATELY exercise drift override round_id.
    base: dict[str, object] = {
        "round_id": compute_round_id(
            pass_index=0,
            files_examined=("src/foo.py",),
            files_skipped=(),
            finding_content_hashes=(finding.content_hash,),
        ),
        "pass_index": 0,
        "findings": (finding,),
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


def test_analysis_round_round_id_must_match_payload() -> None:
    """Post-foundation audit: round_id is bound to the payload by
    model_validator. An arbitrary 64-hex string fails even though it
    matches `SHA256_HEX_PATTERN`; the only admissible round_id is the
    canonical `compute_round_id(...)` output for this round's actual
    content."""
    arbitrary_hash = compute_identity_hash({"any": "payload"})
    with pytest.raises(ValidationError, match="does not match the canonical id"):
        AnalysisRound(**_round_kwargs(round_id=arbitrary_hash))  # type: ignore[arg-type]


def test_analysis_round_admits_canonical_round_id() -> None:
    """The canonical recipe produces a valid round_id."""
    # _round_kwargs() already builds with canonical round_id; verify.
    kwargs = _round_kwargs()
    r = AnalysisRound(**kwargs)  # type: ignore[arg-type]
    assert isinstance(r.round_id, str)
    assert len(r.round_id) == 64


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


def test_analysis_round_rejects_file_in_both_examined_and_skipped() -> None:
    """A file is either examined or skipped per pass, never both."""
    finding = _finding()
    bad_round_id = compute_round_id(
        pass_index=0,
        files_examined=("src/foo.py",),
        files_skipped=("src/foo.py",),
        finding_content_hashes=(finding.content_hash,),
    )
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="files_examined and files_skipped"):
        AnalysisRound(
            round_id=bad_round_id,
            pass_index=0,
            findings=(finding,),
            files_examined=("src/foo.py",),
            files_skipped=("src/foo.py",),
            started_at=now,
            ended_at=now,
        )


def test_analysis_round_rejects_duplicate_in_files_examined() -> None:
    """Set-semantic field: duplicates let logically identical rounds hash
    to different `round_id` values, defeating dedup."""
    finding = _finding()
    bad_round_id = compute_round_id(
        pass_index=0,
        files_examined=("src/foo.py", "src/foo.py"),
        files_skipped=(),
        finding_content_hashes=(finding.content_hash,),
    )
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="files_examined contains duplicates"):
        AnalysisRound(
            round_id=bad_round_id,
            pass_index=0,
            findings=(finding,),
            files_examined=("src/foo.py", "src/foo.py"),
            files_skipped=(),
            started_at=now,
            ended_at=now,
        )


def test_analysis_round_rejects_duplicate_in_files_skipped() -> None:
    """Symmetric check on the skipped tuple."""
    finding = _finding()
    bad_round_id = compute_round_id(
        pass_index=0,
        files_examined=(),
        files_skipped=("src/bar.py", "src/bar.py"),
        finding_content_hashes=(finding.content_hash,),
    )
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="files_skipped contains duplicates"):
        AnalysisRound(
            round_id=bad_round_id,
            pass_index=0,
            findings=(finding,),
            files_examined=(),
            files_skipped=("src/bar.py", "src/bar.py"),
            started_at=now,
            ended_at=now,
        )
