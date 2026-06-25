# Tests for agent/nodes/finding_cap.py (FUP-180).
"""Pin the severity-ordered finding-cap helper.

The cap keeps the N highest-severity findings before any FindingEvent fires.
The load-bearing property is HITL-safety: a drop can only ever remove a
finding no higher in severity than every kept one, so a CRITICAL/HIGH is
never dropped to keep a lower-severity finding (`hitl-gates-high-severity`).
Selection is content-deterministic (replay-stable) — `content_hash` tiebreak.
"""

from __future__ import annotations

from uuid import uuid4

from outrider.agent.nodes.finding_cap import cap_findings_by_severity
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewFinding

_TYPE_BY_SEVERITY = {
    FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
    FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
    FindingSeverity.MEDIUM: FindingType.MISSING_INPUT_VALIDATION,
    FindingSeverity.LOW: FindingType.MISSING_ERROR_HANDLING,
    FindingSeverity.INFO: FindingType.UNUSED_IMPORT,
}


def _finding(severity: FindingSeverity, line: int) -> ReviewFinding:
    """A minimal ReviewFinding of the given severity. `line` varies the span so
    each finding gets a DISTINCT content_hash (the cap's tiebreak + the round's
    uniqueness invariant both key on it)."""
    finding_type = _TYPE_BY_SEVERITY[severity]
    file_path = "src/foo.py"
    ls = line + 1  # line_start is 1-indexed (Field ge=1); keep spans distinct.
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=finding_type,
        severity=severity,
        file_path=file_path,
        line_start=ls,
        line_end=ls,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path, line_start=ls, line_end=ls, finding_type=finding_type
        ),
        proposal_hash="a" * 64,
    )


def test_no_op_when_within_cap() -> None:
    """<= cap: nothing dropped, kept preserves input order (no truncation)."""
    findings = [_finding(FindingSeverity.MEDIUM, line) for line in range(5)]
    kept, dropped = cap_findings_by_severity(findings, cap=5)
    assert dropped == []
    assert kept == findings  # same objects, same order


def test_empty_input() -> None:
    kept, dropped = cap_findings_by_severity([], cap=10)
    assert kept == []
    assert dropped == []


def test_over_cap_keeps_highest_severity() -> None:
    """Over cap: the kept set is exactly the `cap` highest-severity findings;
    the dropped set is the lowest-severity remainder."""
    crit = [_finding(FindingSeverity.CRITICAL, line) for line in range(2)]
    high = [_finding(FindingSeverity.HIGH, line) for line in range(10, 12)]
    low = [_finding(FindingSeverity.LOW, line) for line in range(20, 23)]
    kept, dropped = cap_findings_by_severity([*low, *crit, *high], cap=4)
    assert len(kept) == 4
    assert len(dropped) == 3
    # The 4 kept are the 2 CRITICAL + 2 HIGH; all 3 LOW dropped.
    kept_sev = {f.severity for f in kept}
    assert kept_sev == {FindingSeverity.CRITICAL, FindingSeverity.HIGH}
    assert all(f.severity is FindingSeverity.LOW for f in dropped)


def test_hitl_safety_no_high_dropped_for_lower() -> None:
    """The HITL-safety invariant: every dropped finding is no higher in severity
    than every kept finding — so a CRITICAL/HIGH is never dropped while a
    lower-severity finding survives."""
    rank = {s: i for i, s in enumerate(FindingSeverity)}
    findings = (
        [_finding(FindingSeverity.CRITICAL, line) for line in range(3)]
        + [_finding(FindingSeverity.MEDIUM, line) for line in range(10, 13)]
        + [_finding(FindingSeverity.INFO, line) for line in range(20, 23)]
    )
    kept, dropped = cap_findings_by_severity(findings, cap=5)
    worst_kept_rank = max(rank[f.severity] for f in kept)
    best_dropped_rank = min(rank[f.severity] for f in dropped)
    # Lower rank = higher severity. Every kept finding is at least as severe as
    # every dropped one.
    assert worst_kept_rank <= best_dropped_rank


def test_deterministic_under_shuffle() -> None:
    """Selection is content-deterministic: the kept SET (by content_hash) does not
    depend on input order — the `content_hash` tiebreak makes it a pure function."""
    findings = [_finding(FindingSeverity.MEDIUM, line) for line in range(10)]
    kept_a, _ = cap_findings_by_severity(findings, cap=4)
    kept_b, _ = cap_findings_by_severity(list(reversed(findings)), cap=4)
    assert {f.content_hash for f in kept_a} == {f.content_hash for f in kept_b}
