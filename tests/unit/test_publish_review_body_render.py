# Tests for the publish-node review-body renderers + create_review body param
# (DECISIONS.md#050, commit 2b). The renderers are PURE + PRE-GATED — they
# render exactly the (eligible) findings handed to them; the eligibility gate
# lives in the routing loop. The create_review body param is inert by default
# (body=None falls back to body_marker) and guards the marker-first
# crash-recovery invariant.
"""Pin the review-body renderers and the create_review body pass-through.

Covers:
- `_review_deep_link`: base-url fallback, finding anchor, trailing-slash strip.
- `_render_related_concern_entry`: effective_severity drives the displayed
  severity (never finding.severity / model output), file:line + title both pass
  through `sanitize_display_string` (the path is rendered as TEXT here, so an
  `@`/`#`/backtick would otherwise spawn a mention/ref/code-span), no-link
  fallback prose.
- `_render_review_body`: marker at offset 0 (crash-recovery startswith), the
  "Related concerns" section, the aggregate dashboard-only note (N distinct
  files / M findings + pluralization + no-link fallback), marker-survives-cap.
- `GitHubKitPublisher.create_review`: body=None -> request payload uses
  body_marker; body=valid -> used verbatim; body not starting with the marker ->
  ValueError before any HTTP call.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from outrider.agent.nodes.publish import (
    _is_markdown_link_safe_url,
    _render_related_concern_entry,
    _render_review_body,
    _review_deep_link,
)
from outrider.audit.events import compute_finding_content_hash
from outrider.github.publisher import GitHubKitPublisher
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.output_sanitizer import GITHUB_REVIEW_BODY_MAX
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewFinding

_FINDING_TYPE_BY_SEVERITY = {
    FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
    FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
    FindingSeverity.MEDIUM: FindingType.MISSING_INPUT_VALIDATION,
    FindingSeverity.LOW: FindingType.MISSING_ERROR_HANDLING,
    FindingSeverity.INFO: FindingType.UNUSED_IMPORT,
}


@pytest.fixture(autouse=True)
def _set_truncation_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # apply_size_cap HMAC-signs its truncation marker; test_body_marker_survives_size_cap
    # forces truncation (>64KB body), which reads the secret. The under-cap tests don't
    # (apply_size_cap returns early before the HMAC) — autouse just beats per-test setup.
    monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", "test-secret-for-unit-tests")


def _make_finding(
    *,
    severity: FindingSeverity = FindingSeverity.MEDIUM,
    file_path: str = "src/foo.py",
    line_start: int = 1,
    line_end: int | None = None,
    title: str = "t",
    finding_id: UUID | None = None,
) -> ReviewFinding:
    finding_type = _FINDING_TYPE_BY_SEVERITY[severity]
    line_end = line_end if line_end is not None else line_start
    return ReviewFinding(
        finding_id=finding_id or uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=finding_type,
        severity=severity,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        title=title,
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        proposal_hash=uuid4().hex + uuid4().hex,
    )


# ---------------------------------------------------------------------------
# _review_deep_link
# ---------------------------------------------------------------------------


def test_deep_link_none_base_url_returns_none() -> None:
    assert _review_deep_link(None, uuid4(), uuid4()) is None
    assert _review_deep_link("", uuid4(), uuid4()) is None


def test_deep_link_with_finding_anchor() -> None:
    rid = uuid4()
    fid = uuid4()
    link = _review_deep_link("https://dash.example", rid, fid)
    assert link == f"https://dash.example/reviews/{rid}?finding={fid}"


def test_deep_link_without_finding_anchor() -> None:
    rid = uuid4()
    link = _review_deep_link("https://dash.example", rid, None)
    assert link == f"https://dash.example/reviews/{rid}"


def test_deep_link_strips_trailing_slash() -> None:
    rid = uuid4()
    link = _review_deep_link("https://dash.example/", rid, None)
    assert link == f"https://dash.example/reviews/{rid}"


# ---------------------------------------------------------------------------
# _render_related_concern_entry
# ---------------------------------------------------------------------------


def test_entry_uses_effective_severity_not_finding_severity() -> None:
    # effective_severity (policy/HITL-resolved) drives the displayed severity,
    # never the finding's own / model-set value (boundary #2).
    finding = _make_finding(severity=FindingSeverity.CRITICAL)
    entry = _render_related_concern_entry(
        finding, effective_severity=FindingSeverity.INFO, deep_link=None
    )
    assert entry.startswith("- **INFO**")
    assert "CRITICAL" not in entry


def test_entry_includes_type_location_title_and_link() -> None:
    finding = _make_finding(
        severity=FindingSeverity.MEDIUM,
        file_path="src/auth/login.py",
        line_start=42,
        title="weak check",
    )
    entry = _render_related_concern_entry(
        finding,
        effective_severity=FindingSeverity.MEDIUM,
        deep_link="https://dash.example/reviews/x?finding=y",
    )
    assert FindingType.MISSING_INPUT_VALIDATION.value in entry
    assert "src/auth/login.py:42" in entry
    assert "weak check" in entry
    assert "[view in dashboard](https://dash.example/reviews/x?finding=y)" in entry


def test_entry_no_link_fallback_prose() -> None:
    finding = _make_finding()
    entry = _render_related_concern_entry(
        finding, effective_severity=FindingSeverity.MEDIUM, deep_link=None
    )
    assert "(see the Outrider dashboard)" in entry
    assert "](http" not in entry  # no markdown link emitted


def test_entry_sanitizes_path_metachars_rendered_as_text() -> None:
    # The path is rendered as display TEXT here (unlike inline comments, where it
    # is the GitHub API anchor). `@`/`#` pass validate_diff_path (npm scopes,
    # fragments), so the renderer must escape them or they spawn a mention/ref.
    finding = _make_finding(file_path="src/@scope/mod#sec.py", line_start=7)
    entry = _render_related_concern_entry(
        finding, effective_severity=FindingSeverity.MEDIUM, deep_link=None
    )
    assert "\\@" in entry
    assert "\\#" in entry


def test_entry_sanitizes_title_metachars() -> None:
    finding = _make_finding(title="use `eval` for @admin #123")
    entry = _render_related_concern_entry(
        finding, effective_severity=FindingSeverity.MEDIUM, deep_link=None
    )
    assert "\\`" in entry
    assert "\\@" in entry
    assert "\\#" in entry


# ---------------------------------------------------------------------------
# _render_review_body
# ---------------------------------------------------------------------------

_MARKER = "<!-- outrider-review-id:abc -->"


def test_body_marker_at_offset_zero_when_empty() -> None:
    body = _render_review_body(
        body_marker=_MARKER,
        review_body_findings=(),
        dashboard_only_findings=(),
        review_id=uuid4(),
        dashboard_base_url=None,
    )
    assert body == _MARKER  # nothing else — no sections


def test_body_includes_related_concerns_section() -> None:
    f = _make_finding(severity=FindingSeverity.MEDIUM, title="concern one")
    body = _render_review_body(
        body_marker=_MARKER,
        review_body_findings=((f, FindingSeverity.MEDIUM),),
        dashboard_only_findings=(),
        review_id=uuid4(),
        dashboard_base_url="https://dash.example",
    )
    assert body.startswith(_MARKER)
    assert "## Related concerns" in body
    assert "concern one" in body
    assert "?finding=" in body  # deep link with finding anchor


def test_body_aggregate_note_plural_counts_distinct_files() -> None:
    # 3 dashboard-only findings across 2 distinct files -> "3 additional concerns
    # in 2 files".
    findings = (
        _make_finding(file_path="src/a.py", line_start=1),
        _make_finding(file_path="src/a.py", line_start=9),
        _make_finding(file_path="src/b.py", line_start=1),
    )
    rid = uuid4()
    body = _render_review_body(
        body_marker=_MARKER,
        review_body_findings=(),
        dashboard_only_findings=findings,
        review_id=rid,
        dashboard_base_url="https://dash.example",
    )
    assert "3 additional concerns in 2 files" in body
    assert "couldn't comment on inline" in body
    assert f"https://dash.example/reviews/{rid}" in body
    # aggregate note is count-only — no per-finding file paths
    assert "src/a.py" not in body
    assert "src/b.py" not in body


def test_body_aggregate_note_singular_and_no_link_fallback() -> None:
    findings = (_make_finding(file_path="src/only.py"),)
    body = _render_review_body(
        body_marker=_MARKER,
        review_body_findings=(),
        dashboard_only_findings=findings,
        review_id=uuid4(),
        dashboard_base_url=None,
    )
    assert "1 additional concern in 1 file " in body  # both singular (concern + file)
    assert "couldn't comment on inline" in body
    assert "View it in the Outrider dashboard." in body  # pronoun agrees with m==1
    assert "them" not in body  # no plural pronoun for the singular case
    assert "http" not in body


def test_body_includes_both_sections_in_order() -> None:
    # Both tiers present: related-concerns section (review-body findings) THEN the
    # aggregate dashboard-only note, marker first.
    rb = _make_finding(severity=FindingSeverity.MEDIUM, title="in-diff concern")
    do = (_make_finding(file_path="src/elsewhere.py"),)
    body = _render_review_body(
        body_marker=_MARKER,
        review_body_findings=((rb, FindingSeverity.MEDIUM),),
        dashboard_only_findings=do,
        review_id=uuid4(),
        dashboard_base_url="https://dash.example",
    )
    assert body.startswith(_MARKER)
    assert "## Related concerns" in body
    assert "in-diff concern" in body
    assert "in 1 file" in body  # the aggregate note is present too
    assert "1 additional concern in 1 file" in body
    # ordering: related-concerns section precedes the aggregate note
    assert body.index("## Related concerns") < body.index("Outrider found")


def test_body_marker_survives_size_cap() -> None:
    # ~800 entries blow past GITHUB_REVIEW_BODY_MAX; the cap tail-truncates, so
    # the offset-0 marker survives (preserving the startswith recovery contract).
    findings = tuple(
        (
            _make_finding(file_path=f"src/f{i}.py", line_start=i, title=f"finding {i}"),
            FindingSeverity.MEDIUM,
        )
        for i in range(1, 801)
    )
    body = _render_review_body(
        body_marker=_MARKER,
        review_body_findings=findings,
        dashboard_only_findings=(),
        review_id=uuid4(),
        dashboard_base_url="https://dash.example",
    )
    assert body.startswith(_MARKER)
    assert len(body.encode("utf-8")) <= GITHUB_REVIEW_BODY_MAX


# ---------------------------------------------------------------------------
# create_review — body param pass-through + marker-first guard
# ---------------------------------------------------------------------------


class _CapturingResponse:
    def __init__(self, text: str) -> None:
        self.status_code = 200
        self.text = text


class _CapturingGitHub:
    """Captures the `json=` payload of the create-review POST."""

    def __init__(self) -> None:
        self.captured_json: dict[str, Any] | None = None

    async def arequest(self, *args: Any, **kwargs: Any) -> _CapturingResponse:  # noqa: ARG002
        self.captured_json = kwargs["json"]
        return _CapturingResponse('{"id": 7}')


@pytest.mark.asyncio
async def test_create_review_body_none_falls_back_to_marker() -> None:
    gh = _CapturingGitHub()
    publisher = GitHubKitPublisher()
    await publisher.create_review(
        gh=gh,  # type: ignore[arg-type]
        owner="o",
        repo="r",
        pull_number=1,
        head_sha="0" * 40,
        review_status="COMMENT",
        body_marker="<!-- m -->",
        comments=(),
    )
    assert gh.captured_json is not None
    assert gh.captured_json["body"] == "<!-- m -->"


@pytest.mark.asyncio
async def test_create_review_body_used_verbatim_when_marker_first() -> None:
    gh = _CapturingGitHub()
    publisher = GitHubKitPublisher()
    composed = "<!-- m -->\n\n## Related concerns\n\n- **LOW** ..."
    await publisher.create_review(
        gh=gh,  # type: ignore[arg-type]
        owner="o",
        repo="r",
        pull_number=1,
        head_sha="0" * 40,
        review_status="COMMENT",
        body_marker="<!-- m -->",
        body=composed,
        comments=(),
    )
    assert gh.captured_json is not None
    assert gh.captured_json["body"] == composed


@pytest.mark.asyncio
async def test_create_review_body_not_marker_first_raises_before_post() -> None:
    gh = _CapturingGitHub()
    publisher = GitHubKitPublisher()
    with pytest.raises(ValueError, match="must start with body_marker"):
        await publisher.create_review(
            gh=gh,  # type: ignore[arg-type]
            owner="o",
            repo="r",
            pull_number=1,
            head_sha="0" * 40,
            review_status="COMMENT",
            body_marker="<!-- m -->",
            body="## Related concerns\n\n(no marker prefix)",
            comments=(),
        )
    assert gh.captured_json is None  # guard fired before any HTTP call


# ---------------------------------------------------------------------------
# _is_markdown_link_safe_url + _review_deep_link malformed-URL fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://dash.example",
        "http://localhost:5173",
        "https://dash.example/base/path",
        "https://dash.example:8443/x?a=b#frag",
        "HTTPS://dash.example",  # uppercase scheme (RFC-3986 case-insensitive)
    ],
)
def test_is_markdown_link_safe_url_accepts_well_formed(url: str) -> None:
    assert _is_markdown_link_safe_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "ftp://dash.example",  # non-http scheme
        "javascript:alert(1)",  # non-http scheme
        "dash.example",  # no scheme
        "https://dash.example/a)b",  # close paren breaks markdown target
        "https://dash.example/a(b",  # open paren
        "https://dash.example/<b>",  # angle brackets -> HTML injection in prose
        "https://dash.example/[x]",  # square brackets -> markdown link syntax
        "https://dash.example/a b",  # whitespace
        "https://dash.example/a\tb",  # tab
        "https://dash.example/a\nb",  # newline
        "https://dash.example/a\x00b",  # NUL control char
        "https://dash.example/a\x7fb",  # DEL control char
        "https://",  # scheme-only / host-less (rstrip would strip scheme slashes)
        "https:///",  # only slashes after the scheme
        "https:///foo",  # empty host with a path (urlparse netloc == "")
    ],
)
def test_is_markdown_link_safe_url_rejects_malformed(url: str) -> None:
    assert _is_markdown_link_safe_url(url) is False


def test_review_deep_link_malformed_base_url_falls_back_to_none() -> None:
    # A malformed (non-empty) base URL degrades to None — the renderer then uses
    # the no-link fallback prose rather than emitting a broken/unsafe link.
    assert _review_deep_link("https://dash.example/x)y", uuid4(), uuid4()) is None
    assert _review_deep_link("ftp://dash.example", uuid4(), None) is None
    assert _review_deep_link("not a url", uuid4(), None) is None


def test_entry_collapses_title_newlines() -> None:
    # Each entry is a single markdown list item (joined by "\n"); a newline in the
    # model-authored title must collapse to a space, not splinter the bullet.
    finding = _make_finding(title="line one\nline two\r\nline three")
    entry = _render_related_concern_entry(
        finding, effective_severity=FindingSeverity.MEDIUM, deep_link=None
    )
    assert "\n" not in entry
    assert "\r" not in entry
    assert "line one line two" in entry  # collapsed to spaces


def test_render_review_body_malformed_base_url_uses_no_link_fallback() -> None:
    # End-to-end: a malformed base URL → body renders with no-link prose, no URL.
    f = _make_finding(severity=FindingSeverity.MEDIUM, title="concern")
    body = _render_review_body(
        body_marker=_MARKER,
        review_body_findings=((f, FindingSeverity.MEDIUM),),
        dashboard_only_findings=(_make_finding(file_path="src/x.py"),),
        review_id=uuid4(),
        dashboard_base_url="https://dash.example/bad)url",
    )
    assert "http" not in body
    assert "see the Outrider dashboard" in body  # review-body entry fallback
    assert "in the Outrider dashboard" in body  # aggregate note fallback
