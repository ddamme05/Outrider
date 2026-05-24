# See specs/2026-05-23-trace-node.md M5 + M7 + M8.
"""Trace node unit tests — load-bearing contracts only.

Covers:
  - `TraceJoinIntegrityError` raises on duplicate proposal_hash across
    findings in `state.analysis_rounds` (M5 last-resort guard).
  - `_candidate_paths_for(import_string)` constructs module + package
    paths deterministically (Phase 1 probe-path construction per M8).
  - Bucket dropping for already-traced findings (M1 + #025 point 5
    within-graph re-entry idempotency).

DB-touching integration tests (Phase 1 probes + Phase 2 fetch end-to-
end with mock GitHub) are deferred to a follow-up integration test
file; the unit tests pin the producer-deterministic invariants the
spec calls out explicitly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from outrider.agent.nodes.trace import (
    TraceJoinIntegrityError,
    _bucket_candidates_by_finding,
    _build_proposal_hash_join,
    _candidate_paths_for,
)
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import compute_candidate_id, compute_round_id
from outrider.schemas import (
    AnalysisRound,
    ReviewDimension,
    ReviewFinding,
    ReviewState,
    TraceCandidate,
)
from outrider.schemas.pr_context import PRContext

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _build_finding(
    *, proposal_hash: str | None = None, file_path: str = "src/foo.py"
) -> ReviewFinding:
    """Build a ReviewFinding fixture; defaults to a fresh proposal_hash.
    `file_path` is parameterized so callers can vary it across siblings
    in one round (the AnalysisRound validator rejects duplicate
    content_hashes; varying file_path produces distinct hashes)."""
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=12345,
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        file_path=file_path,
        line_start=10,
        line_end=12,
        title="SQL injection",
        description="raw concat",
        evidence=f"concat at {file_path}:11",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version="1.0.0",
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=10,
            line_end=12,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash=proposal_hash if proposal_hash is not None else "a" * 64,
    )


def _build_round(findings: tuple[ReviewFinding, ...], *, pass_index: int = 0) -> AnalysisRound:
    """Build an AnalysisRound with a canonical round_id derived from content."""
    now = datetime.now(UTC)
    files_examined = tuple(sorted({f.file_path for f in findings})) or ("src/foo.py",)
    return AnalysisRound(
        round_id=compute_round_id(
            pass_index=pass_index,
            files_examined=files_examined,
            files_skipped=(),
            finding_content_hashes=tuple(f.content_hash for f in findings),
        ),
        pass_index=pass_index,
        findings=findings,
        files_examined=files_examined,
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )


def _build_state(rounds: tuple[AnalysisRound, ...]) -> ReviewState:
    return ReviewState(
        review_id=uuid4(),
        pr_context=PRContext(
            installation_id=1,
            owner="o",
            repo="r",
            pr_number=1,
            pr_title="x",
            head_sha="a" * 40,
            base_sha="b" * 40,
            author="dev",
            total_additions=5,
            total_deletions=2,
            changed_files=(),
        ),
        received_at=datetime.now(UTC),
        analysis_rounds=list(rounds),
    )


def _build_candidate(
    *,
    source_proposal_hash: str,
    import_string: str = "pkg.mod",
) -> TraceCandidate:
    reason = "x"
    return TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=source_proposal_hash,
            import_string=import_string,
            reason=reason,
        ),
        source_proposal_hash=source_proposal_hash,
        reason=reason,
        import_string=import_string,
    )


# ---------------------------------------------------------------------------
# M5: TraceJoinIntegrityError raises on duplicate proposal_hash.
# ---------------------------------------------------------------------------


def test_join_integrity_error_raises_on_duplicate_proposal_hash_across_rounds() -> None:
    """Two findings in two different rounds sharing the same proposal_hash —
    `_build_proposal_hash_join` raises `TraceJoinIntegrityError` with both
    finding_ids. M5's last-resort guard: the analyze-side
    `AnalysisRound._enforce_findings_proposal_hash_unique` validator
    catches within-round collisions; this guard catches cross-round
    collisions that would only arise from a `compute_proposal_hash`
    recipe drift (or a producer bypassing the validator).

    Within-round collisions are already rejected by the AnalysisRound
    validator before reaching trace, so this test uses two SEPARATE
    rounds to exercise the cross-round path.
    """
    shared_hash = "c" * 64
    finding_a = _build_finding(proposal_hash=shared_hash, file_path="src/foo.py")
    finding_b = _build_finding(proposal_hash=shared_hash, file_path="src/bar.py")
    round_1 = _build_round((finding_a,), pass_index=0)
    round_2 = _build_round((finding_b,), pass_index=1)
    state = _build_state((round_1, round_2))

    with pytest.raises(TraceJoinIntegrityError) as exc_info:
        _build_proposal_hash_join(state)

    assert exc_info.value.proposal_hash == shared_hash
    assert exc_info.value.first_finding_id == finding_a.finding_id
    assert exc_info.value.second_finding_id == finding_b.finding_id


def test_join_lookup_succeeds_on_distinct_proposal_hashes() -> None:
    """Distinct hashes across findings → join succeeds with one entry per."""
    finding_a = _build_finding(proposal_hash="e" * 64, file_path="src/alpha.py")
    finding_b = _build_finding(proposal_hash="f" * 64, file_path="src/beta.py")
    state = _build_state((_build_round((finding_a, finding_b)),))

    join = _build_proposal_hash_join(state)
    assert join == {
        "e" * 64: finding_a.finding_id,
        "f" * 64: finding_b.finding_id,
    }


# ---------------------------------------------------------------------------
# Bucket-build: unjoinable candidates drop silently.
# ---------------------------------------------------------------------------


def test_bucket_drops_candidates_whose_proposal_hash_has_no_finding() -> None:
    """Candidate whose source_proposal_hash isn't in the join is dropped
    (logged at DEBUG, not raised). Other candidates land in their
    proper bucket. The join contract is the producer-side responsibility;
    trace consumes state defensively."""
    finding = _build_finding(proposal_hash="1" * 64)
    candidate_in_join = _build_candidate(source_proposal_hash="1" * 64)
    candidate_unjoinable = _build_candidate(
        source_proposal_hash="2" * 64,
        import_string="pkg.unjoined",
    )
    join = {"1" * 64: finding.finding_id}

    buckets = _bucket_candidates_by_finding(
        (candidate_in_join, candidate_unjoinable),
        join,
    )
    assert set(buckets.keys()) == {finding.finding_id}
    assert buckets[finding.finding_id] == [candidate_in_join]


# ---------------------------------------------------------------------------
# M8: probe-path construction is deterministic.
# ---------------------------------------------------------------------------


def test_candidate_paths_for_emits_module_and_package_forms() -> None:
    """`foo.bar` → exactly two candidate paths: `foo/bar.py` and
    `foo/bar/__init__.py`. Pinned per M8 — this is Phase 1's
    probe-path construction; downstream probes fetch-test each."""
    assert _candidate_paths_for("foo.bar") == ("foo/bar.py", "foo/bar/__init__.py")
    assert _candidate_paths_for("single") == ("single.py", "single/__init__.py")
    assert _candidate_paths_for("a.b.c") == ("a/b/c.py", "a/b/c/__init__.py")
