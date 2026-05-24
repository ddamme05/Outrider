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


def _finding(proposal_hash: str = "a" * 64) -> ReviewFinding:
    """Build a fixture finding. `proposal_hash` is parametrized so tests
    that put multiple findings in one AnalysisRound (exercising the new
    `_enforce_findings_proposal_hash_unique` validator per #025) can
    distinguish each finding without overriding all other fields."""
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
        proposal_hash=proposal_hash,
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


def test_analysis_round_rejects_duplicate_finding_ids() -> None:
    """Two ReviewFindings with the same finding_id is a producer bug —
    `findings` is set-semantic by finding_id."""
    f1 = _finding()  # proposal_hash default "a" * 64
    # Same finding_id, otherwise valid (different content_hash via file_path,
    # distinct proposal_hash so the #025 within-round-uniqueness validator
    # doesn't fire FIRST — this test targets finding_id collision).
    duplicate = ReviewFinding(
        finding_id=f1.finding_id,  # same id — bug
        review_id=uuid4(),
        installation_id=12345,
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        file_path="src/other.py",
        line_start=1,
        line_end=2,
        title="x",
        description="y",
        evidence="z",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version="1.0.0",
        content_hash=compute_finding_content_hash(
            file_path="src/other.py",
            line_start=1,
            line_end=2,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash="b" * 64,  # Distinct from f1's default; targets finding_id validator.
    )
    now = datetime.now(UTC)
    bad_round_id = compute_round_id(
        pass_index=0,
        files_examined=("src/foo.py", "src/other.py"),
        files_skipped=(),
        finding_content_hashes=(f1.content_hash, duplicate.content_hash),
    )
    with pytest.raises(ValidationError, match="duplicate finding_ids"):
        AnalysisRound(
            round_id=bad_round_id,
            pass_index=0,
            findings=(f1, duplicate),
            files_examined=("src/foo.py", "src/other.py"),
            files_skipped=(),
            started_at=now,
            ended_at=now,
        )


def test_analysis_round_rejects_duplicate_content_hashes() -> None:
    """Two findings with the same content_hash collapse the finding-
    content-hash tuple's sorted form, changing the round_id digest."""
    f1 = _finding()  # proposal_hash default "a" * 64
    # Same content_hash inputs => same hash, but different finding_id +
    # distinct proposal_hash so the #025 within-round-uniqueness validator
    # doesn't fire FIRST — this test targets content_hash collision.
    duplicate = ReviewFinding(
        finding_id=uuid4(),  # different id...
        review_id=uuid4(),
        installation_id=12345,
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        file_path="src/foo.py",  # ...but same file_path, line range, finding_type
        line_start=10,
        line_end=12,
        title="x",
        description="y",
        evidence="z",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version="1.0.0",
        content_hash=f1.content_hash,
        proposal_hash="b" * 64,  # Distinct from f1's default; targets content_hash validator.
    )
    now = datetime.now(UTC)
    bad_round_id = compute_round_id(
        pass_index=0,
        files_examined=("src/foo.py",),
        files_skipped=(),
        finding_content_hashes=(f1.content_hash, duplicate.content_hash),
    )
    with pytest.raises(ValidationError, match="duplicate content_hashes"):
        AnalysisRound(
            round_id=bad_round_id,
            pass_index=0,
            findings=(f1, duplicate),
            files_examined=("src/foo.py",),
            files_skipped=(),
            started_at=now,
            ended_at=now,
        )


def test_analysis_round_rejects_duplicate_proposal_hashes() -> None:
    """Per DECISIONS.md#025 point 4: admitted findings within a round
    have unique proposal_hashes. Two findings sharing a proposal_hash
    is a producer bug (compute_proposal_hash is content-derived from
    the raw proposal payload; collision means analyze emitted two
    findings from THE SAME logical proposal). Load-bearing for trace's
    join contract — catching the collision at construction time
    prevents the upstream producer bug from reaching trace's
    collision-detecting lookup."""
    # Two findings with DISTINCT finding_id + DISTINCT content_hash but
    # SAME proposal_hash. The other validators ( _enforce_findings_unique +
    # _enforce_findings_proposal_hash_unique) split responsibility cleanly:
    # this test isolates the proposal_hash validator by varying
    # finding_id + content_hash inputs but pinning proposal_hash.
    f1 = _finding(proposal_hash="a" * 64)
    duplicate = ReviewFinding(
        finding_id=uuid4(),  # distinct from f1.finding_id
        review_id=uuid4(),
        installation_id=12345,
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        file_path="src/other.py",  # distinct file → distinct content_hash
        line_start=20,
        line_end=22,
        title="x",
        description="y",
        evidence="z",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version="1.0.0",
        content_hash=compute_finding_content_hash(
            file_path="src/other.py",
            line_start=20,
            line_end=22,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash="a" * 64,  # SAME as f1 — triggers the validator
    )
    now = datetime.now(UTC)
    bad_round_id = compute_round_id(
        pass_index=0,
        files_examined=("src/foo.py", "src/other.py"),
        files_skipped=(),
        finding_content_hashes=(f1.content_hash, duplicate.content_hash),
    )
    with pytest.raises(ValidationError, match="duplicate proposal_hashes"):
        AnalysisRound(
            round_id=bad_round_id,
            pass_index=0,
            findings=(f1, duplicate),
            files_examined=("src/foo.py", "src/other.py"),
            files_skipped=(),
            started_at=now,
            ended_at=now,
        )
