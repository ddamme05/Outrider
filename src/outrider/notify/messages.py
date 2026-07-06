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
from outrider.presentation.finding_sections import build_finding_sections

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

# Slack length limits. Primary defense: clip each raw attacker-controlled string (pr_title, repo,
# file_path) to `_SLACK_INPUT_MAX` code points BEFORE escaping + composition — this bounds both the
# section blocks AND the top-level `text` notification fallback, keeps the UTF-16 length Slack
# actually counts well under its cap (even all-astral input stays ~2× the clip), and means later
# truncation never lands mid-escape-entity. `_cap_text` is the per-section backstop (≤3000-char
# mrkdwn limit); it truncates WITHOUT dropping blocks, so the overflow + deep-link/action blocks
# (appended after, no attacker text) always survive. Total block count is bounded under Slack's
# 50-block message limit by `top_n`.
_SLACK_SECTION_MAX: Final[int] = 2900
_SLACK_INPUT_MAX: Final[int] = 300


class RenderedSlackMessage(NamedTuple):
    """A ready-to-post message: `text` (notification fallback) + Block Kit `blocks`."""

    text: str
    blocks: list[dict[str, Any]]


def _escape_mrkdwn(text: str) -> str:
    """Neutralize Slack mrkdwn control chars in attacker-controlled values (PR/finding
    titles, repo, file paths) so `<!here>` / `<@U…>` / `<url|text>` / `&<>` render as inert
    text — not live mentions, channel pings, or links. Slack decodes only these three;
    `&` is escaped first (per slack-sdk messaging/formatting-message-text). Builder-owned
    link syntax (`<{deep_link}|…>`) is composed from trusted values and is NOT escaped.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clip(raw: str) -> str:
    """Clip a raw attacker-controlled string to a display bound BEFORE escaping/composition.
    The primary length defense: bounds the section blocks AND the `text` fallback, keeps the
    UTF-16 length under Slack's cap, and (by clipping the raw value) guarantees later escaping +
    truncation never split a multi-char `&amp;`/`&lt;` entity."""
    return raw if len(raw) <= _SLACK_INPUT_MAX else raw[: _SLACK_INPUT_MAX - 1] + "…"


def _cap_text(text: str) -> str:
    """Per-section backstop: truncate composed mrkdwn to Slack's ≤3000-char section limit. With
    raw inputs already `_clip`-ped this rarely fires; kept as defense in depth for any future
    section that composes many inputs."""
    return text if len(text) <= _SLACK_SECTION_MAX else text[: _SLACK_SECTION_MAX - 1] + "…"


def _section(markdown: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": _cap_text(markdown)}}


def _context(markdown: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": markdown}]}


def _severity_counts(findings: Sequence[ReviewFinding]) -> list[tuple[FindingSeverity, int]]:
    counts = Counter(f.severity for f in findings)
    return [(s, counts[s]) for s in _SEVERITY_ORDER if counts[s]]


def _counts_phrase(counts: list[tuple[FindingSeverity, int]]) -> str:
    return " · ".join(f"{n} {s.value.capitalize()}" for s, n in counts)


def _finding_line(f: ReviewFinding) -> str:
    """One finding, metadata-first: severity · type — file:line-range + title. No code/evidence.
    Humanized labels from the shared presentation layer; full line range (was line_start only). NO
    severity emoji — Slack announces it to screen readers (a11y); severity is the text label."""
    sections = build_finding_sections(f, effective_severity=f.severity)
    line_range = str(f.line_start) if f.line_start == f.line_end else f"{f.line_start}-{f.line_end}"
    return (
        f"*{sections.severity_label}* · {sections.type_label} — "
        f"`{_escape_mrkdwn(_clip(f.file_path))}:{line_range}`\n{_escape_mrkdwn(_clip(f.title))}"
    )


def build_hitl_pending_message(
    *,
    repo: str,
    pr_number: int,
    pr_title: str,
    findings: Sequence[ReviewFinding],
    deep_link: str | None,
    top_n: int = 3,
) -> RenderedSlackMessage:
    """The rich HITL-pending card: PR identity, severity counts, top-N findings by
    severity, an overflow line for the rest, and a deep-link button."""
    repo_s, pr_title_s = _escape_mrkdwn(_clip(repo)), _escape_mrkdwn(_clip(pr_title))
    ordered = sorted(findings, key=lambda f: _SEVERITY_RANK[f.severity])
    counts = _severity_counts(ordered)
    total = len(ordered)
    shown = ordered[:top_n]

    blocks: list[dict[str, Any]] = [
        _section(f":warning: *Review needs approval* — `{repo_s}` #{pr_number}\n{pr_title_s}"),
    ]
    if counts:
        blocks.append(_context(_counts_phrase(counts)))
    for f in shown:
        blocks.append(_section(_finding_line(f)))
    if total > len(shown):
        remaining = _severity_counts(ordered[top_n:])
        more = f"+{total - len(shown)} more ({_counts_phrase(remaining)})"
        more += (
            f" · <{deep_link}|view all {total} in the dashboard>"
            if deep_link is not None
            else f" · view all {total} in the Outrider dashboard"
        )
        blocks.append(_context(more))
    # No-link fallback: when no valid base URL is configured
    # (build_review_deeplink returned None), drop the divider + deep-link button
    # rather than emit a broken-link button.
    if deep_link is not None:
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
    text = f"Review needs approval: {repo_s} #{pr_number} — {pr_title_s} ({counts_text})."
    if deep_link is not None:
        text += f" {deep_link}"
    return RenderedSlackMessage(text=text, blocks=blocks)


def build_review_posted_message(
    *,
    repo: str,
    pr_number: int,
    posted_count: int,
    dashboard_only_count: int,
    deep_link: str | None,
) -> RenderedSlackMessage:
    """The compact one-line review-posted FYI (non-gated reviews). Counts come from
    publish routing: `posted_count` = inline + review-body (on the PR);
    `dashboard_only_count` = findings not posted to GitHub."""
    repo_s = _escape_mrkdwn(_clip(repo))
    total = posted_count + dashboard_only_count
    if total == 0:
        summary = "no findings"
    elif dashboard_only_count == 0:
        summary = f"{total} findings posted"
    else:
        summary = (
            f"{total} findings ({posted_count} posted · {dashboard_only_count} dashboard-only)"
        )

    section_text = (
        f":white_check_mark: Reviewed `{repo_s}` #{pr_number} — {summary}, no approval needed"
    )
    text = f"Reviewed {repo_s} #{pr_number} — {summary}, no approval needed."
    if deep_link is not None:
        section_text += f" · <{deep_link}|view>"
        text += f" {deep_link}"
    blocks = [_section(section_text)]
    return RenderedSlackMessage(text=text, blocks=blocks)
