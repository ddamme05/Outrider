"""Check that a finding on an indented function's first body line is accepted.

Drive the real graph with a scripted medium-severity finding and check its
file line range and inline GitHub comment. Requires --is-eval and postgres-test;
the driver manages an isolated database and substitutes external services."""

from __future__ import annotations

from pathlib import Path

from outrider.agent import run_review

_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "mock_github" / "fup137_blocking_async.json"
)

# The blocking-call line in the fixture's content_head (1-indexed, file-frame):
# `        time.sleep(1.0)` is line 13, the first body line of the indented
# `async def poll` method (lines 12-14).
_BLOCKING_CALL_LINE = 13


def test_clip_head_finding_in_indented_scope_is_admitted_and_published() -> None:
    """FUP-137: the indented-scope clip-head finding is admitted post-fix,
    posts INLINE on its file-frame line, and does not trip the HITL gate."""
    result = run_review(str(_FIXTURE))

    # 1. The blocking_call_in_async finding is admitted (post-fix). Pre-fix it
    #    was dropped at the byte-containment gate as span_outside_scope_unit.
    blocking = [f for f in result.findings if f.finding_type == "blocking_call_in_async"]
    assert len(blocking) == 1, (
        "expected exactly one blocking_call_in_async finding admitted post-FUP-126-fix; "
        f"got findings: {[f.finding_type for f in result.findings]} "
        "(if empty, the line-space admission gate regressed and the finding was "
        "silently dropped — the FUP-126 bug class)"
    )
    finding = blocking[0]
    assert finding.line_start == _BLOCKING_CALL_LINE
    assert finding.line_end == _BLOCKING_CALL_LINE

    # 2. MEDIUM severity -> the HITL gate is a pass-through, publish ran.
    assert result.hitl_gated is False, (
        "blocking_call_in_async is MEDIUM; it must not trip the HITL gate "
        "(a gated run would leave published_comments empty for the wrong reason)"
    )

    # 3. Publish posted the finding INLINE: the captured comment lands on the
    #    changed-region file-frame line. Post-fix this is a correct line range;
    #    pre-fix there was nothing to post.
    assert result.published_comments, (
        "expected at least one inline comment posted; an empty set here means the "
        "finding was dropped before publish or routed off the diff"
    )
    posted_lines = {c.line for c in result.published_comments}
    assert _BLOCKING_CALL_LINE in posted_lines, (
        f"expected an inline comment on file-frame line {_BLOCKING_CALL_LINE}; "
        f"got comment lines {sorted(posted_lines)}"
    )
