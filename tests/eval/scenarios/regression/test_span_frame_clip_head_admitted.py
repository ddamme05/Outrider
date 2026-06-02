"""FUP-137 end-to-end: a clip-head finding inside an indented scope is admitted.

The end-to-end layer on top of the parser-level regression
(`tests/unit/test_analyze_parser.py::test_step5_clean_line_range_inside_indented_scope_admits`).
It drives the real 7-node graph via `run_review` against a JSON PR fixture
(`mock_github/fup137_blocking_async.json`) whose scripted analyze response
plants the exact FUP-126 shape: a `blocking_call_in_async` finding on the
FIRST body line of an indented `async def` method (`Worker.poll`, whose token
`byte_start` is far past column 0).

Pre-FUP-126-fix this finding was silently dropped — the model anchored at the
clip head (byte 0), and the byte-containment gate rejected it because
`scope_unit.byte_start (>0) <= 0` is false (`finding_proposal_rejected{reason=
span_outside_scope_unit}`). Post-fix the proposal is line-based and admission is
line-space scope containment, so the finding survives, hashes, routes, and posts
with a correct file-frame line range.

The finding is MEDIUM (`blocking_call_in_async` → MEDIUM per `SEVERITY_POLICY`),
so the HITL gate is a pass-through and publish runs to completion. The
blocking call lands on an added diff line, so publish routes it INLINE and the
capturing publisher records a comment.

Unlike the LLM-free structural scenarios (in `scenarios/structural/`, which
validate `ast_facts` directly), this one needs the eval graph driver: it runs a
real (scripted-LLM) analyze pass, which is what FUP-126 is a behavior of — hence
it lives in `regression/`, not `structural/`. It therefore requires `--is-eval` +
the
`postgres-test` container (`run_review` self-manages an ephemeral DB and refuses
to touch a non-test URL). Shipped UNSKIPPED — it is new with the driver.
"""

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
