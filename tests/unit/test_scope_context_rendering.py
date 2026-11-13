"""Keep selected scope coverage while avoiding repeated source and patch text."""

from __future__ import annotations

import json
from difflib import unified_diff
from types import SimpleNamespace

import pytest
import test_analyze_node as recorders

from outrider.agent.nodes import analyze as node
from outrider.ast_facts.models import ScopeUnit
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER

_PATH = "app/packet.py"
_HEAD = """class Packet:
    def first(self):
        return 'first-new-marker'

    def second(self):
        return 'second-new-marker'
"""


async def _capture():
    before = _HEAD.replace("new-marker", "old-marker")
    changed = recorders._build_changed_file(
        path=_PATH,
        content=_HEAD.encode(),
        content_base=before,
        patch="".join(
            unified_diff(
                before.splitlines(keepends=True),
                _HEAD.splitlines(keepends=True),
                fromfile="a/" + _PATH,
                tofile="b/" + _PATH,
                n=0,
            )
        ),
    )
    prepared = node._select_file_review(
        changed_file=changed, import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER
    )
    assert isinstance(prepared, node._PreparedFileReview)
    state = recorders._build_review_state(
        pr_context=recorders._build_pr_context(changed_files=(changed,)),
        triage_result=recorders._build_triage_result(file_tiers={_PATH: recorders.ReviewTier.DEEP}),
    )
    response = json.loads(
        recorders._build_finding_proposal_json(
            finding_type="correctness_bug",
            line_start=3,
            line_end=3,
        )
    )
    response["findings"][0]["evidence"] = "return 'first-new-marker'"
    deps = recorders.deps.__wrapped__()
    deps.update(
        provider=recorders._StubLLMProvider(json.dumps(response)),
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    command = await node.analyze(state, **deps)
    (send,) = command.goto
    worker = await node.analyze_file(
        send.arg, **{key: deps[key] for key in recorders._WORKER_DEP_KEYS if key in deps}
    )
    (request,) = deps["provider"].calls
    return SimpleNamespace(request=request, payload=send.arg, prepared=prepared, worker=worker)


@pytest.mark.asyncio
async def test_native_nested_source_is_rendered_once_with_all_scope_metadata() -> None:
    result = await _capture()
    scopes = result.prepared.decision.included_scope_units
    assert [scope.qualified_name for scope in scopes] == ["Packet", "Packet.first", "Packet.second"]
    source = result.request.user_prompt.split("## Scope-unit context", 1)[1].split(
        "## Pre-fired query matches", 1
    )[0]
    assert source.count("return 'first-new-marker'") == 1
    assert source.count("return 'second-new-marker'") == 1
    for scope in scopes:
        assert (
            f"### {scope.kind} `{scope.qualified_name}` (lines {scope.line_start}-{scope.line_end})"
            in source
        )
    assert source.count("Body included in selected class `Packet`") == 2
    assert [
        entry.scope_unit_name
        for entry in result.request.context_summary
        if entry.inclusion_reason == "changed_scope"
    ] == [scope.qualified_name for scope in scopes]
    outcome = result.worker["analyze_worker_outcomes"][0]
    assert len(outcome.admitted_findings) == 1
    assert outcome.admitted_findings[0].evidence == "return 'first-new-marker'"


@pytest.mark.asyncio
async def test_native_repeated_hunks_render_once_across_distinct_scope_blocks() -> None:
    result = await _capture()
    hunks = result.prepared.decision.included_clipped_hunks
    assert tuple(map(len, hunks)) == (2, 1, 1)
    assert hunks[0] == (hunks[1][0], hunks[2][0])
    diff = result.request.user_prompt.split("## Changed diff (scope-unit-clipped)", 1)[1].split(
        "## Additional bounded review context", 1
    )[0]
    assert diff.count("+        return 'first-new-marker'") == 1
    assert diff.count("+        return 'second-new-marker'") == 1
    assert diff.index(hunks[0][0]) < diff.index(hunks[0][1])


def _scope(index, start, end, *, path=_PATH):
    return ScopeUnit(
        unit_id=f"{index:064x}",
        kind="function",
        name=f"scope{index}",
        qualified_name=f"scope{index}",
        file_path=path,
        line_start=1,
        line_end=1,
        byte_start=start,
        byte_end=end,
    )


def test_body_ownership_handles_order_equal_spans_overlap_and_file_boundaries() -> None:
    from outrider.coordinates import scope_unit_body_owners

    child, middle, outer = _scope(1, 6, 10), _scope(2, 4, 12), _scope(3, 0, 16)
    assert scope_unit_body_owners((child, middle, outer)) == (2, 2, 2)
    assert scope_unit_body_owners((_scope(1, 0, 10), _scope(2, 5, 15), _scope(3, 6, 8))) == (
        0,
        1,
        1,
    )
    assert scope_unit_body_owners((_scope(1, 0, 10), _scope(2, 0, 10))) == (0, 0)
    assert scope_unit_body_owners((_scope(1, 0, 10), _scope(2, 0, 10, path="other.py"))) == (0, 1)
    assert scope_unit_body_owners((_scope(1, 0, 5), _scope(2, 5, 10))) == (0, 1)
    selected = (child, outer)
    rendered = node._assemble_scope_unit_context(
        included_scope_units=selected, source_bytes=b"abcdefghijklmnop"
    )
    assert rendered.index("### function `scope1`") < rendered.index("### function `scope3`")
    assert "Body included in selected function `scope3`" in rendered
    kept = node._assemble_scope_unit_context(
        included_scope_units=(child,), source_bytes=b"abcdefghijklmnop"
    )
    assert "ghij" in kept and "Body included" not in kept
    many = tuple(_scope(index, index * 2, index * 2 + 1) for index in range(4000))
    assert scope_unit_body_owners(many) == tuple(range(4000))


def test_nonnested_rendering_and_distinct_hunk_separator_bytes_are_preserved() -> None:
    scopes = (_scope(1, 0, 5), _scope(2, 5, 10))
    assert node._assemble_scope_unit_context(
        included_scope_units=scopes, source_bytes=b"abcdefghij"
    ) == (
        "### function `scope1` (lines 1-1)\n```python\nabcde\n```\n\n"
        "### function `scope2` (lines 1-1)\n```python\nfghij\n```"
    )
    unique = ((), ("@@ first\n+alpha\n", "@@ second\n+beta\n"), (), ("@@ third\n+gamma\n",))
    assert (
        node._concat_clipped_hunks(unique)
        == "\n\n@@ first\n+alpha\n\n@@ second\n+beta\n\n\n\n\n@@ third\n+gamma\n"
    )
    assert (
        node._concat_clipped_hunks((("@@ first\n+alpha\n",), ("@@ first\n+alpha\n+beta\n",)))
        == "@@ first\n+alpha\n\n\n@@ first\n+alpha\n+beta\n"
    )
