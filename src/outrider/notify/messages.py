"""Block Kit message builders for Slack notifications — metadata-first.

Pure rendering: domain objects + a prebuilt deep-link in, Slack `(text, blocks)`
out (no IO, no config, no SDK). **Metadata-first** per the spec: a finding renders
only severity / type / `file:line` / title — never `description` or `evidence`
(no raw code leaves to Slack; the dashboard link carries the detail). The
HITL-pending card caps inline findings at `top_n` by severity and collapses the
rest to one overflow line; there is no interactive "show more" (that needs V2
inbound). See specs/2026-06-15-slack-dashboard-in-slack.md (Output boundary).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Final, NamedTuple

from outrider.policy.severity import FindingSeverity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.schemas.review_finding import ReviewFinding

__all__ = ["RenderedSlackMessage", "build_hitl_pending_message", "build_review_posted_message"]

# Severity order (most → least severe) for sorting + count rendering.
_SEVERITY_ORDER: Final[tuple[FindingSeverity, ...]] = (
    FindingSeverity.CRITICAL,
    FindingSeverity.HIGH,
    FindingSeverity.MEDIUM,
    FindingSeverity.LOW,
    FindingSeverity.INFO,
)
_SEVERITY_RANK: Final[dict[FindingSeverity, int]] = {s: i for i, s in enumerate(_SEVERITY_ORDER)}
_SEVERITY_EMOJI: Final[dict[FindingSeverity, str]] = {
    FindingSeverity.CRITICAL: ":red_circle:",
    FindingSeverity.HIGH: ":large_orange_circle:",
    FindingSeverity.MEDIUM: ":large_yellow_circle:",
    FindingSeverity.LOW: ":large_blue_circle:",
    FindingSeverity.INFO: ":white_circle:",
}


class RenderedSlackMessage(NamedTuple):
    """A ready-to-post message: `text` (notification fallback) + Block Kit `blocks`."""

    text: str
    blocks: list[dict[str, Any]]


def _section(markdown: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": markdown}}


def _context(markdown: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": markdown}]}


def _severity_counts(findings: Sequence[ReviewFinding]) -> list[tuple[FindingSeverity, int]]:
    counts = Counter(f.severity for f in findings)
    return [(s, counts[s]) for s in _SEVERITY_ORDER if counts[s]]


def _counts_phrase(counts: list[tuple[FindingSeverity, int]]) -> str:
    return " · ".join(f"{n} {s.value.capitalize()}" for s, n in counts)


def _finding_line(f: ReviewFinding) -> str:
    """One finding, metadata-first: severity · type — file:line + title. No code/evidence."""
    emoji = _SEVERITY_EMOJI[f.severity]
    type_label = f.finding_type.value.replace("_", " ")
    return (
        f"{emoji} *{f.severity.value.upper()}* · {type_label} — "
        f"`{f.file_path}:{f.line_start}`\n{f.title}"
    )


def build_hitl_pending_message(
    *,
    repo: str,
    pr_number: int,
    pr_title: str,
    findings: Sequence[ReviewFinding],
    deep_link: str,
    top_n: int = 3,
) -> RenderedSlackMessage:
    """The rich HITL-pending card: PR identity, severity counts, top-N findings by
    severity, an overflow line for the rest, and a deep-link button."""
    ordered = sorted(findings, key=lambda f: _SEVERITY_RANK[f.severity])
    counts = _severity_counts(ordered)
    total = len(ordered)
    shown = ordered[:top_n]

    blocks: list[dict[str, Any]] = [
        _section(f":warning: *Review needs approval* — `{repo}` #{pr_number}\n{pr_title}"),
    ]
    if counts:
        blocks.append(_context(_counts_phrase(counts)))
    for f in shown:
        blocks.append(_section(_finding_line(f)))
    if total > len(shown):
        remaining = _severity_counts(ordered[top_n:])
        blocks.append(
            _context(
                f"+{total - len(shown)} more ({_counts_phrase(remaining)}) · "
                f"<{deep_link}|view all {total} in the dashboard>"
            )
        )
    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Review in dashboard"},
                    "url": deep_link,
                }
            ],
        }
    )

    counts_text = _counts_phrase(counts) if counts else "no findings"
    text = f"Review needs approval: {repo} #{pr_number} — {pr_title} ({counts_text}). {deep_link}"
    return RenderedSlackMessage(text=text, blocks=blocks)


def build_review_posted_message(
    *,
    repo: str,
    pr_number: int,
    posted_count: int,
    dashboard_only_count: int,
    deep_link: str,
) -> RenderedSlackMessage:
    """The compact one-line review-posted FYI (non-gated reviews). Counts come from
    publish routing: `posted_count` = inline + review-body (on the PR);
    `dashboard_only_count` = findings not posted to GitHub."""
    total = posted_count + dashboard_only_count
    if total == 0:
        summary = "no findings"
    elif dashboard_only_count == 0:
        summary = f"{total} findings posted"
    else:
        summary = (
            f"{total} findings ({posted_count} posted · {dashboard_only_count} dashboard-only)"
        )

    text = f"Reviewed {repo} #{pr_number} — {summary}, no approval needed. {deep_link}"
    blocks = [
        _section(
            f":white_check_mark: Reviewed `{repo}` #{pr_number} — {summary}, "
            f"no approval needed · <{deep_link}|view>"
        )
    ]
    return RenderedSlackMessage(text=text, blocks=blocks)
