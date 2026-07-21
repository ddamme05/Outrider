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

__all__ = [
    "RenderedSlackMessage",
    "build_hitl_pending_message",
    "build_review_posted_message",
    "build_status_mirror_message",
]


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
    """Truncate composed mrkdwn to Slack's ≤3000-char section limit, ESCAPE-AWARE so a cut never
    severs a `&amp;`/`&lt;`/`&gt;` entity into a dangling `&…`. Backstops the section blocks AND
    the `text` fallback: `_escape_mrkdwn` expands each `&`/`<`/`>` (to ≤5 chars), so a `_clip`-ped
    (≤300-cp) input can still exceed the limit once escaped — this proves the FINAL serialized
    length regardless of that expansion."""
    if len(text) <= _SLACK_SECTION_MAX:
        return text
    cut = text[: _SLACK_SECTION_MAX - 1]
    # `_escape_mrkdwn` emits only `&amp;`/`&lt;`/`&gt;` — every `&` opens an entity. A trailing `&`
    # with no closing `;` after it is a split entity; trim back to before it (window is ≤5 chars).
    amp = cut.rfind("&")
    if amp != -1 and ";" not in cut[amp:]:
        cut = cut[:amp]
    return cut + "…"


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


def _finding_blocks(
    ordered: Sequence[ReviewFinding], *, top_n: int, deep_link: str | None
) -> list[dict[str, Any]]:
    """Top-N finding sections plus the `+N more` overflow line. Shared by the
    HITL-pending card and the status mirror, which edits that same card in place —
    the finding list is the channel's record and survives every mirror edit."""
    total = len(ordered)
    shown = ordered[:top_n]
    blocks: list[dict[str, Any]] = [_section(_finding_line(f)) for f in shown]
    if total > len(shown):
        remaining = _severity_counts(ordered[top_n:])
        more = f"+{total - len(shown)} more ({_counts_phrase(remaining)})"
        more += (
            f" · <{deep_link}|view all {total} in the dashboard>"
            if deep_link is not None
            else f" · view all {total} in the Outrider dashboard"
        )
        blocks.append(_context(more))
    return blocks


def _deeplink_blocks(deep_link: str | None) -> list[dict[str, Any]]:
    """Divider + deep-link button. No-link fallback: when no valid base URL is
    configured (build_review_deeplink returned None), emit nothing rather than a
    broken-link button."""
    if deep_link is None:
        return []
    return [
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Review in dashboard"},
                    "url": deep_link,
                }
            ],
        },
    ]


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

    blocks: list[dict[str, Any]] = [
        _section(f":warning: *Review needs approval* — `{repo_s}` #{pr_number}\n{pr_title_s}"),
    ]
    if counts:
        blocks.append(_context(_counts_phrase(counts)))
    blocks.extend(_finding_blocks(ordered, top_n=top_n, deep_link=deep_link))
    blocks.extend(_deeplink_blocks(deep_link))

    counts_text = _counts_phrase(counts) if counts else "no findings"
    # Cap the attacker-bearing fallback BEFORE appending the trusted deep-link (escape-aware, so
    # the escaped repo/pr_title can't push `text` past the limit or strand a split entity).
    text = _cap_text(
        f"Review needs approval: {repo_s} #{pr_number} — {pr_title_s} ({counts_text})."
    )
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
    text = _cap_text(f"Reviewed {repo_s} #{pr_number} — {summary}, no approval needed.")
    if deep_link is not None:
        section_text += f" · <{deep_link}|view>"
        text += f" {deep_link}"
    blocks = [_section(section_text)]
    return RenderedSlackMessage(text=text, blocks=blocks)


def _mirror_headline(*, approved_count: int, gated_count: int) -> tuple[str, str]:
    """(icon, headline) for a mirror edit.

    The verdict comes from the reviewer's decisions over the GATED findings, never
    from publish routing counts. Routing cannot express the verdict: `posted_count`
    also counts auto-post (non-gated) findings, so a review that rejected everything
    can still post; and an APPROVED finding routed DASHBOARD_ONLY posts nothing.
    Counts are reported separately in the detail line.

    There is deliberately NO status dimension. Every rendering the mirror can produce
    must be IDENTICAL for a given review, because `chat.update` replaces the whole
    message with no compare-and-set: it is idempotent across concurrent writers only
    while they all render the same thing. The remaining publish outcomes (success,
    both idempotent skips, empty) all derive their counts from the same durable state,
    so they converge. A status-varying rendering reintroduces last-writer-wins and
    needs a monotonic lifecycle mechanism first — see FUP-252.
    """
    if gated_count and approved_count == gated_count:
        return ":white_check_mark:", "Approved"
    if approved_count == 0:
        return ":no_entry_sign:", "Dismissed"
    return ":white_check_mark:", "Partially approved"


def _routing_phrase(*, posted_count: int, dashboard_only_count: int) -> str:
    """What reached the PR, stated separately from the verdict."""
    if posted_count == 0 and dashboard_only_count == 0:
        return "nothing posted"
    parts = [f"{posted_count} posted"]
    if dashboard_only_count:
        parts.append(f"{dashboard_only_count} dashboard-only")
    return " · ".join(parts)


def _mirror_detail(
    *,
    reviewer_s: str | None,
    approved_count: int,
    gated_count: int,
    posted_count: int,
    dashboard_only_count: int,
) -> str:
    """The context line: who decided, what they decided, and what reached the PR —
    kept as three separate clauses so the routing counts can never be read as the
    reviewer's verdict."""
    by = f"Decided by {reviewer_s}" if reviewer_s else "Decided"
    verdict = f"{approved_count} of {gated_count} gated approved"
    routing = _routing_phrase(posted_count=posted_count, dashboard_only_count=dashboard_only_count)
    return f"{by} · {verdict} · {routing}"


def build_status_mirror_message(
    *,
    repo: str,
    pr_number: int,
    pr_title: str,
    findings: Sequence[ReviewFinding],
    deep_link: str | None,
    reviewer_id: str | None = None,
    approved_count: int = 0,
    gated_count: int = 0,
    posted_count: int = 0,
    dashboard_only_count: int = 0,
    top_n: int = 3,
) -> RenderedSlackMessage:
    """Re-render the HITL card for a `chat.update` status mirror.

    Same card, terminal state: the header and context lines carry the outcome while
    the finding sections and dashboard button are preserved, so the channel keeps its
    record of what was flagged. `findings` is the ORIGINAL gated set (the mirror
    reflects the decision, it does not re-triage). There is one rendering per review
    by construction — see `_mirror_headline` for why that is load-bearing.

    `approved_count` / `gated_count` carry the reviewer's verdict over the gated set
    (from `policy.publish_eligibility.count_gated_approvals`); `posted_count` /
    `dashboard_only_count` carry publish routing. The two are rendered as separate
    clauses and MUST NOT be conflated — routing spans auto-post findings, so it
    cannot express what the reviewer decided.

    Emits no audit event — the mirror is a side-effecting reflection of facts already
    captured by `HITLDecisionEvent` / `PublishEvent` (spec: Audit events emitted).
    """
    repo_s, pr_title_s = _escape_mrkdwn(_clip(repo)), _escape_mrkdwn(_clip(pr_title))
    reviewer_s = _escape_mrkdwn(_clip(reviewer_id)) if reviewer_id else None
    ordered = sorted(findings, key=lambda f: _SEVERITY_RANK[f.severity])
    icon, headline = _mirror_headline(approved_count=approved_count, gated_count=gated_count)
    detail = _mirror_detail(
        reviewer_s=reviewer_s,
        approved_count=approved_count,
        gated_count=gated_count,
        posted_count=posted_count,
        dashboard_only_count=dashboard_only_count,
    )

    blocks: list[dict[str, Any]] = [
        _section(f"{icon} *{headline}* — `{repo_s}` #{pr_number}\n{pr_title_s}"),
        _context(detail),
    ]
    blocks.extend(_finding_blocks(ordered, top_n=top_n, deep_link=deep_link))
    blocks.extend(_deeplink_blocks(deep_link))

    # Cap the attacker-bearing fallback BEFORE appending the trusted deep-link, same
    # escape-aware discipline as the other builders.
    text = _cap_text(f"{headline}: {repo_s} #{pr_number} — {pr_title_s} ({detail}).")
    if deep_link is not None:
        text += f" {deep_link}"
    return RenderedSlackMessage(text=text, blocks=blocks)
