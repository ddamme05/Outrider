"""OBSERVED skip-routing telemetry — `compute_observed_skip_shadow`
(Cost Lever 3, DECISIONS.md#049).

Default-deny coverage: a file is `would_skip` iff every changed region lies in a
`skip_safe` envelope; `signal_only` matches never count; base/removed regions are
un-coverable by head-content matches → always blockers. V1 seeds zero `skip_safe`
queries, so real registry matches are all `signal_only` → `not_eligible`. The
function RECORDS the decision; it never skips the LLM (enforcement is the later
evidence-gated flip).
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from unidiff import PatchSet

from outrider.agent.nodes.analyze_observed import (
    ObservedMatch,
    compute_observed_skip_shadow,
    run_observed_matches,
)
from outrider.ast_facts import parse_python
from outrider.ast_facts.models import ScopeUnit
from outrider.policy.severity import FindingType
from outrider.queries.observed import QueryClass

# A one-added-line diff: head line 3 (`eval(x)`) is the only changed region.
_HEAD = "def foo():\n    return 1\n    eval(x)\n"
_BASE = "def foo():\n    return 1\n"
_PATCH_ADD = (
    "--- a/src/foo.py\n+++ b/src/foo.py\n"
    "@@ -1,2 +1,3 @@\n"
    " def foo():\n"
    "     return 1\n"
    "+    eval(x)\n"
)

# A ride-along deletion: scope [1,3] yields head_added=[3] AND base_removed=[1]
# (mirrors test_coordinates_spans' decorator-deletion fixture).
_DEL_HEAD = "def foo():\n    return 1\n    # note\n"
_DEL_BASE = "@require_auth\ndef foo():\n    return 1\n"
_PATCH_DEL = (
    "--- a/x.py\n+++ b/x.py\n"
    "@@ -1,3 +1,3 @@\n"
    "-@require_auth\n"
    " def foo():\n"
    "     return 1\n"
    "+    # note\n"
)


def _patched(patch: str = _PATCH_ADD):  # noqa: ANN202 — unidiff PatchedFile, test-local
    return PatchSet.from_string(patch)[0]


def _scope(line_start: int = 1, line_end: int = 3) -> ScopeUnit:
    return ScopeUnit(
        unit_id="a" * 64,
        kind="function",
        name="foo",
        qualified_name="m.foo",
        file_path="src/foo.py",
        line_start=line_start,
        line_end=line_end,
        byte_start=0,
        byte_end=len(_HEAD.encode()),
    )


def _match(
    query_class: QueryClass,
    *,
    line_start: int = 3,
    line_end: int = 3,
    query_match_id: str = "q-1",
) -> ObservedMatch:
    return ObservedMatch(
        query_match_id=query_match_id,
        query_class=query_class,
        finding_type=FindingType.COMMAND_INJECTION,
        title="t",
        description="d",
        evidence="eval(x)",
        line_start=line_start,
        line_end=line_end,
    )


def _compute(matches: tuple[ObservedMatch, ...], *, scopes: tuple[ScopeUnit, ...] | None = None):  # noqa: ANN202
    return compute_observed_skip_shadow(
        matches,
        file_path="src/foo.py",
        included_scope_units=scopes if scopes is not None else (_scope(),),
        patched_file=_patched(),
        head_source=_HEAD,
        base_source=_BASE,
        review_id=uuid4(),
        is_eval=True,
    )


def test_skip_safe_full_coverage_yields_would_skip() -> None:
    """A skip_safe match whose envelope contains the only changed line → the file
    is would_skip with that match as the covering envelope and no blockers."""
    ev = _compute((_match(QueryClass.SKIP_SAFE),))
    assert ev is not None
    assert ev.outcome == "would_skip"
    assert ev.blockers == ()
    assert [m.query_match_id for m in ev.covering_matches] == ["q-1"]
    assert ev.covering_matches[0].side == "head"
    assert {(r.side, r.line_start) for r in ev.changed_regions} == {("head", 3)}


def test_signal_only_match_does_not_count_for_coverage() -> None:
    """The guardrail: a signal_only hit covering the change does NOT make it
    would_skip — signal_only augments the LLM, it never justifies a skip."""
    ev = _compute((_match(QueryClass.SIGNAL_ONLY),))
    assert ev is not None
    assert ev.outcome == "not_eligible"
    assert ev.covering_matches == ()
    assert [(b.side, b.line_start) for b in ev.blockers] == [("head", 3)]


def test_no_matches_yields_not_eligible() -> None:
    ev = _compute(())
    assert ev is not None
    assert ev.outcome == "not_eligible"
    assert [(b.side, b.line_start) for b in ev.blockers] == [("head", 3)]
    assert ev.covering_matches == ()


def test_skip_safe_not_covering_change_yields_not_eligible() -> None:
    """A skip_safe match whose envelope misses the changed line covers nothing."""
    ev = _compute((_match(QueryClass.SKIP_SAFE, line_start=1, line_end=2),))
    assert ev is not None
    assert ev.outcome == "not_eligible"
    assert ev.covering_matches == ()


def test_base_removed_region_always_blocks_even_when_head_covered() -> None:
    """Base/removed lines are un-coverable by head-content matches → always
    blockers. A skip_safe match covering the head change does NOT make the file
    would_skip while a removed line is in play."""
    skip_safe = _match(QueryClass.SKIP_SAFE, line_start=3, line_end=3)  # covers head line 3
    ev = compute_observed_skip_shadow(
        (skip_safe,),
        file_path="x.py",
        included_scope_units=(_scope(line_start=1, line_end=3),),
        patched_file=_patched(_PATCH_DEL),
        head_source=_DEL_HEAD,
        base_source=_DEL_BASE,
        review_id=uuid4(),
        is_eval=True,
    )
    assert ev is not None
    assert ev.outcome == "not_eligible"
    # The removed base line is the blocker; the covered head line is not.
    assert [(b.side, b.line_start) for b in ev.blockers] == [("base", 1)]
    assert {(r.side, r.line_start) for r in ev.changed_regions} == {("head", 3), ("base", 1)}


def test_no_included_scopes_returns_none() -> None:
    assert _compute((_match(QueryClass.SKIP_SAFE),), scopes=()) is None


def test_v1_real_registry_matches_are_signal_only_so_not_eligible() -> None:
    """End-to-end with the REAL registry: run_observed_matches surfaces the eval()
    match (signal_only in V1), so the file is not_eligible — the V1 contract that
    no file is skip-eligible until a query is promoted to skip_safe."""
    scopes = parse_python(_HEAD.encode(), "src/foo.py", MagicMock()).scope_units
    matches = run_observed_matches(
        file_path="src/foo.py", head_content=_HEAD, included_scope_units=scopes
    )
    assert matches
    assert all(m.query_class == QueryClass.SIGNAL_ONLY for m in matches)
    ev = compute_observed_skip_shadow(
        matches,
        file_path="src/foo.py",
        included_scope_units=scopes,
        patched_file=_patched(),
        head_source=_HEAD,
        base_source=_BASE,
        review_id=uuid4(),
        is_eval=True,
    )
    assert ev is not None
    assert ev.outcome == "not_eligible"
