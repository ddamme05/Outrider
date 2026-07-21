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
from outrider.notify.messages import (
    build_hitl_pending_message,
    build_review_posted_message,
    build_status_mirror_message,
)
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


def _finding(
    severity: FindingSeverity,
    *,
    title: str,
    line: int = 10,
    line_end: int | None = None,
    file_path: str = "src/x.py",
) -> ReviewFinding:
    ft = _BY_SEVERITY[severity]
    fp = file_path
    le = line if line_end is None else line_end
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=1,
        finding_type=ft,
        dimension=ReviewDimension.SECURITY,
        severity=severity,
        file_path=fp,
        line_start=line,
        line_end=le,
        title=title,
        description="DESCRIPTION THAT MUST NOT APPEAR IN SLACK",
        evidence="EVIDENCE THAT MUST NOT APPEAR IN SLACK",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=fp, line_start=line, line_end=le, finding_type=ft
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
    assert dumped.index("*Critical*") < dumped.index("*High*")  # ordering (humanized labels)
    # Humanized type label, NO raw enum, NO severity emoji (a11y: Slack announces emoji aloud).
    assert "SQL injection" in dumped and "sql_injection" not in dumped
    assert ":red_circle:" not in dumped and ":large_orange_circle:" not in dumped
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


def test_finding_line_humanized_no_emoji_full_range() -> None:
    """One finding renders humanized (severity + type label, no raw enum, no emoji) and carries
    the full line range when line_start != line_end (was line_start only)."""
    from outrider.notify.messages import _finding_line

    line = _finding_line(_finding(FindingSeverity.CRITICAL, title="t", line=10, line_end=14))
    assert line.startswith("*Critical* · SQL injection — ")
    assert "`src/x.py:10-14`" in line  # full range, not just line_start
    assert "sql_injection" not in line  # humanized, not raw enum
    assert ":red_circle:" not in line  # no severity emoji (a11y)


def _utf16_units(s: str) -> int:
    """UTF-16 code-unit count — the metric Slack measures its length limits in (an astral emoji
    is 2 units)."""
    return len(s.encode("utf-16-le")) // 2


def test_pathological_pr_title_bounded_everywhere_and_action_block_survives() -> None:
    """A pathological pr_title is clipped at the source, so BOTH the section blocks AND the `text`
    notification fallback stay under Slack's limits, and the deep-link action block is NOT dropped.
    Regression guard for the previously-uncapped `text` field (only section blocks were bounded)."""
    findings = [_finding(FindingSeverity.CRITICAL, title="t")]
    msg = build_hitl_pending_message(
        repo="r", pr_number=1, pr_title="A" * 10_000, findings=findings, deep_link="https://d/x"
    )
    for b in msg.blocks:
        if b["type"] == "section":
            assert len(b["text"]["text"]) <= 2900  # every section within the cap
    assert len(msg.text) < 3000  # the `text` fallback is bounded too (was fully uncapped)
    assert _actions_button_url(msg.blocks) == "https://d/x"  # action block survived


def test_astral_pr_title_stays_within_utf16_budget() -> None:
    """Astral emoji count as 2 UTF-16 units (Slack's metric), so a code-point cap would under-count;
    source-clipping keeps both the header section and the `text` fallback under the UTF-16 limit."""
    findings = [_finding(FindingSeverity.CRITICAL, title="t")]
    msg = build_hitl_pending_message(
        repo="r", pr_number=1, pr_title="😀" * 5000, findings=findings, deep_link="https://d/x"
    )
    for b in msg.blocks:
        if b["type"] == "section":
            assert _utf16_units(b["text"]["text"]) <= 3000
    assert _utf16_units(msg.text) <= 3000


def _no_split_entity(s: str) -> bool:
    """True iff `s` contains no severed `&amp;`/`&lt;`/`&gt;`: strip complete entities, then no bare
    `&` may remain (every `&` in escaped mrkdwn opens one of the three entities)."""
    return "&" not in s.replace("&amp;", "").replace("&lt;", "").replace("&gt;", "")


def test_cap_text_never_severs_an_escape_entity() -> None:
    """Directly: a cut that lands INSIDE a `&amp;` trims back to before the `&`, never leaving a
    dangling `&a…`. Offset-crafted so it fails if the escape-aware trim is reverted."""
    from outrider.notify.messages import _SLACK_SECTION_MAX, _cap_text

    # Boundary at index _SLACK_SECTION_MAX-1 lands 2 chars into the first `&amp;` ("&a|mp;").
    text = "x" * (_SLACK_SECTION_MAX - 3) + "&amp;" * 10
    out = _cap_text(text)
    assert len(out) <= _SLACK_SECTION_MAX
    assert out.endswith("…")
    assert _no_split_entity(out)  # the "&a" fragment was trimmed, not shipped


def test_escape_expanding_inputs_stay_bounded_without_split_entities() -> None:
    """The worst case the cap must PROVE: repo + pr_title full of `&`, which `_escape_mrkdwn`
    expands ~5× (`&`→`&amp;`) so a `_clip`-ped 300-cp input becomes ~1500 chars and the compound
    header section exceeds the limit — forcing `_cap_text` to fire. Every section AND the `text`
    fallback must land within the limit with NO split entity, and the action block must survive."""
    from outrider.notify.messages import _SLACK_SECTION_MAX

    # repo + pr_title are raw webhook strings (not schema-validated like file_path); both full of
    # `&` push the compound header section past the limit once escaped, forcing _cap_text to fire.
    findings = [_finding(FindingSeverity.CRITICAL, title="t")]
    msg = build_hitl_pending_message(
        repo="&" * 400, pr_number=1, pr_title="&" * 400, findings=findings, deep_link="https://d/x"
    )
    for b in msg.blocks:
        if b["type"] == "section":
            t = b["text"]["text"]
            assert len(t) <= _SLACK_SECTION_MAX  # final serialized length proven, post-escape
            assert _no_split_entity(t)  # truncation never severed a &amp;/&lt;/&gt;
    # The trusted deep-link (no `&`) is appended AFTER the cap, so the fallback is bounded + clean.
    assert len(msg.text) <= _SLACK_SECTION_MAX + len(" https://d/x")
    assert _no_split_entity(msg.text)
    assert _actions_button_url(msg.blocks) == "https://d/x"


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


def test_escape_mrkdwn_neutralizes_control_chars() -> None:
    from outrider.notify.messages import _escape_mrkdwn

    assert _escape_mrkdwn("a & b < c > d") == "a &amp; b &lt; c &gt; d"
    assert _escape_mrkdwn("<!here> <@U1>") == "&lt;!here&gt; &lt;@U1&gt;"
    # `&` escaped first, so the entity ampersands are not double-escaped.
    assert _escape_mrkdwn("<x>") == "&lt;x&gt;"


def test_attacker_metadata_does_not_become_live_formatting() -> None:
    # repo / pr_title / finding title are free text (file_path is path-validated upstream,
    # so it can't carry control chars — escaping it is defense-in-depth, covered by the
    # _escape_mrkdwn unit test above).
    findings = [_finding(FindingSeverity.CRITICAL, title="pwn <@U123> & <!here>")]
    msg = build_hitl_pending_message(
        repo="evil/<!here>",
        pr_number=1,
        pr_title="title <@U999> & <https://x|x>",
        findings=findings,
        deep_link="https://dash/r/1?finding=2",
    )
    blob = json.dumps(msg.blocks, ensure_ascii=False) + msg.text
    # No raw control sequence survives (no live mentions / channel pings / links).
    for raw in ("<!here>", "<@U123>", "<@U999>", "<https://x|x>"):
        assert raw not in blob
    # They are present only as inert HTML entities.
    assert "&lt;!here&gt;" in blob and "&amp;" in blob
    # The builder-owned deep-link button is still a real, unescaped URL.
    assert _actions_button_url(msg.blocks) == "https://dash/r/1?finding=2"


# ---------------------------------------------------------------------------
# No-link fallback — deep_link=None (malformed/unconfigured base URL; FUP-190)
# ---------------------------------------------------------------------------


def test_hitl_card_no_link_fallback_when_deep_link_none() -> None:
    """deep_link=None (build_review_deeplink rejected a malformed base URL) → no
    deep-link button and no URL anywhere, rather than a broken mrkdwn link."""
    findings = [_finding(FindingSeverity.HIGH, title="t")]
    msg = build_hitl_pending_message(
        repo="o/r", pr_number=1, pr_title="t", findings=findings, deep_link=None
    )
    assert not any(b["type"] == "actions" for b in msg.blocks)  # no deep-link button
    assert "http" not in json.dumps(msg.blocks)
    assert "http" not in msg.text


def test_review_posted_no_link_fallback_when_deep_link_none() -> None:
    """deep_link=None → the compact FYI renders without the `<url|view>` mrkdwn link
    and without a URL in the fallback text."""
    msg = build_review_posted_message(
        repo="o/r", pr_number=1, posted_count=2, dashboard_only_count=1, deep_link=None
    )
    assert "http" not in json.dumps(msg.blocks)
    assert "http" not in msg.text
    assert "|view>" not in json.dumps(msg.blocks)  # no mrkdwn link


# ---------------------------------------------------------------------------
# Status mirror card
# ---------------------------------------------------------------------------


def _mirror(**kw: Any) -> Any:
    defaults: dict[str, Any] = {
        "repo": "acme/api",
        "pr_number": 7,
        "pr_title": "Add webhook",
        "findings": [_finding(FindingSeverity.CRITICAL, title="SQLi in lookup")],
        "deep_link": "https://dash.example.com/reviews/abc",
    }
    defaults.update(kw)
    return build_status_mirror_message(**defaults)


def _all_text(msg: Any) -> str:
    return json.dumps(msg.blocks)


def test_mirror_preserves_findings_and_button() -> None:
    """The mirror edits the HITL card in place: the finding record and the dashboard
    button survive every status change (that record is the channel's value)."""
    msg = _mirror(posted_count=3, dashboard_only_count=1)
    assert "SQLi in lookup" in _all_text(msg)
    assert _actions_button_url(msg.blocks) == "https://dash.example.com/reviews/abc"


def test_mirror_published_reads_approved_when_all_gated_approved() -> None:
    msg = _mirror(
        reviewer_id="alice",
        approved_count=2,
        gated_count=2,
        posted_count=3,
        dashboard_only_count=1,
    )
    body = _all_text(msg)
    assert "Approved" in body
    assert "2 of 2 gated approved" in body
    assert "3 posted" in body
    assert "1 dashboard-only" in body


def test_mirror_verdict_is_not_inferred_from_posted_count() -> None:
    """Routing counts CANNOT express the reviewer's verdict, and inferring it from
    them produced two false readings. Both are pinned here.

    (a) Every gated finding rejected, but a non-gated MEDIUM auto-posts →
        `posted_count > 0`. Must not read "Approved".
    (b) A gated finding approved but routed DASHBOARD_ONLY → `posted_count == 0`.
        Must not read "Dismissed".
    """
    rejected_but_autopost = _all_text(
        _mirror(
            reviewer_id="alice",
            approved_count=0,
            gated_count=1,
            posted_count=1,  # the auto-post MEDIUM, not the gated finding
            dashboard_only_count=0,
        )
    )
    assert "Dismissed" in rejected_but_autopost
    assert "Approved" not in rejected_but_autopost
    assert "0 of 1 gated approved" in rejected_but_autopost

    approved_but_dashboard_only = _all_text(
        _mirror(
            reviewer_id="alice",
            approved_count=1,
            gated_count=1,
            posted_count=0,
            dashboard_only_count=1,
        )
    )
    assert "Approved" in approved_but_dashboard_only
    assert "Dismissed" not in approved_but_dashboard_only
    assert "1 of 1 gated approved" in approved_but_dashboard_only


def test_mirror_published_reads_dismissed_when_no_gated_finding_approved() -> None:
    msg = _mirror(
        reviewer_id="alice",
        approved_count=0,
        gated_count=2,
        posted_count=0,
        dashboard_only_count=0,
    )
    body = _all_text(msg)
    assert "Dismissed" in body
    assert "Approved" not in body
    assert "nothing posted" in body


def test_mirror_published_reads_partially_approved_on_a_split_decision() -> None:
    """A mixed decision is neither "Approved" nor "Dismissed" — claiming either
    overstates what the reviewer did."""
    msg = _mirror(
        reviewer_id="alice",
        approved_count=1,
        gated_count=3,
        posted_count=1,
        dashboard_only_count=0,
    )
    body = _all_text(msg)
    assert "Partially approved" in body
    assert "1 of 3 gated approved" in body


def test_mirror_escapes_reviewer_and_never_leaks_finding_detail() -> None:
    """`reviewer_id` is escaped like every other interpolated value, and the mirror
    inherits the metadata-first rule — no description/evidence reaches Slack."""
    msg = _mirror(reviewer_id="<!here>", posted_count=1)
    body = _all_text(msg)
    assert "&lt;!here&gt;" in body
    assert "<!here>" not in body
    assert "DESCRIPTION THAT MUST NOT APPEAR IN SLACK" not in body
    assert "EVIDENCE THAT MUST NOT APPEAR IN SLACK" not in body


def test_mirror_without_deep_link_drops_the_button() -> None:
    msg = _mirror(deep_link=None, posted_count=1)
    assert all(b["type"] != "actions" for b in msg.blocks)
    assert "https://" not in msg.text
