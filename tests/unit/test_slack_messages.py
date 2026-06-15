"""Block Kit message builders — ordering, overflow, metadata-first, review-posted.

Pins: HITL card sorts by severity + caps at top_n + collapses the rest to one
overflow line; the deep-link button carries the URL; metadata-first (no
`description`/`evidence` leaks into Slack); review-posted phrasing for the
posted/dashboard-only/empty cases. See specs/2026-06-15-slack-dashboard-in-slack.md.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from outrider.audit.events import compute_finding_content_hash
from outrider.notify.messages import build_hitl_pending_message, build_review_posted_message
from outrider.policy import EvidenceTier
from outrider.policy.severity import ACTIVE_POLICY_VERSION, FindingSeverity, FindingType
from outrider.schemas import ReviewDimension
from outrider.schemas.review_finding import ReviewFinding

# (finding_type, severity) triples that satisfy SEVERITY_POLICY + the SECURITY dimension lockstep.
_BY_SEVERITY = {
    FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
    FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
    FindingSeverity.MEDIUM: FindingType.MISSING_INPUT_VALIDATION,
}


def _finding(severity: FindingSeverity, *, title: str, line: int = 10) -> ReviewFinding:
    ft = _BY_SEVERITY[severity]
    fp = "src/x.py"
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=1,
        finding_type=ft,
        dimension=ReviewDimension.SECURITY,
        severity=severity,
        file_path=fp,
        line_start=line,
        line_end=line,
        title=title,
        description="DESCRIPTION THAT MUST NOT APPEAR IN SLACK",
        evidence="EVIDENCE THAT MUST NOT APPEAR IN SLACK",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=fp, line_start=line, line_end=line, finding_type=ft
        ),
        proposal_hash=hashlib.sha256(f"{ft}{line}{title}".encode()).hexdigest(),
    )


def _actions_button_url(blocks: list[dict[str, Any]]) -> str:
    actions = next(b for b in blocks if b["type"] == "actions")
    url: str = actions["elements"][0]["url"]
    return url


def test_hitl_card_orders_by_severity_caps_and_overflows() -> None:
    findings = [
        _finding(FindingSeverity.MEDIUM, title="med-finding-a"),
        _finding(FindingSeverity.CRITICAL, title="crit-finding-a"),
        _finding(FindingSeverity.MEDIUM, title="med-finding-b"),
        _finding(FindingSeverity.CRITICAL, title="crit-finding-b"),
        _finding(FindingSeverity.HIGH, title="high-finding"),
    ]
    msg = build_hitl_pending_message(
        repo="acme/api",
        pr_number=1287,
        pr_title="Add Stripe webhook",
        findings=findings,
        deep_link="https://dash.example.com/reviews/abc?finding=def",
        top_n=3,
    )
    dumped = json.dumps(msg.blocks, ensure_ascii=False)

    # Lead + counts (severity order).
    assert "Review needs approval" in dumped
    assert "acme/api" in dumped and "#1287" in dumped and "Add Stripe webhook" in dumped
    assert "2 Critical · 1 High · 2 Medium" in dumped
    # Top-3 by severity = the 2 criticals + 1 high; the 2 mediums collapse to the overflow.
    assert "crit-finding-a" in dumped and "crit-finding-b" in dumped and "high-finding" in dumped
    assert "med-finding-a" not in dumped and "med-finding-b" not in dumped
    assert dumped.index("*CRITICAL*") < dumped.index("*HIGH*")  # ordering
    # Overflow line.
    assert "+2 more (2 Medium)" in dumped
    assert "view all 5 in the dashboard" in dumped
    # Button carries the deep-link.
    assert _actions_button_url(msg.blocks) == "https://dash.example.com/reviews/abc?finding=def"


def test_hitl_card_no_overflow_within_top_n() -> None:
    findings = [
        _finding(FindingSeverity.CRITICAL, title="c1"),
        _finding(FindingSeverity.HIGH, title="h1"),
    ]
    msg = build_hitl_pending_message(
        repo="r", pr_number=1, pr_title="t", findings=findings, deep_link="https://d/x", top_n=3
    )
    dumped = json.dumps(msg.blocks, ensure_ascii=False)
    assert "more" not in dumped  # no overflow line
    assert "c1" in dumped and "h1" in dumped


def test_hitl_card_is_metadata_first() -> None:
    findings = [_finding(FindingSeverity.CRITICAL, title="crit")]
    msg = build_hitl_pending_message(
        repo="r", pr_number=1, pr_title="t", findings=findings, deep_link="https://d/x"
    )
    blob = json.dumps(msg.blocks) + msg.text
    assert "MUST NOT APPEAR" not in blob  # no description / evidence leaks to Slack


def test_review_posted_with_dashboard_only() -> None:
    msg = build_review_posted_message(
        repo="acme/api",
        pr_number=1293,
        posted_count=3,
        dashboard_only_count=1,
        deep_link="https://d/x",
    )
    blob = json.dumps(msg.blocks) + msg.text
    assert "4 findings (3 posted · 1 dashboard-only)" in blob
    assert "no approval needed" in blob


def test_review_posted_all_posted() -> None:
    msg = build_review_posted_message(
        repo="r", pr_number=1, posted_count=4, dashboard_only_count=0, deep_link="https://d/x"
    )
    assert "4 findings posted" in (json.dumps(msg.blocks) + msg.text)


def test_review_posted_no_findings() -> None:
    msg = build_review_posted_message(
        repo="r", pr_number=1, posted_count=0, dashboard_only_count=0, deep_link="https://d/x"
    )
    assert "no findings" in (json.dumps(msg.blocks) + msg.text)
