# See specs/2026-05-28-synthesize-node.md §Audit append-only.
# See DECISIONS.md#030-reviewreport-tuple-not-list-findings-field
# (the canonical-record anchor for the metric's union recipe semantic;
# this file is the executable contract pin in #030's Referenced from line).
"""Unit tests for `_compute_files_traced_beyond_diff`.

Pins the union semantic CodeRabbit 2026-05-28 catch + Codex narrowing
codified at `agent/nodes/synthesize.py:391-432`:

  files_traced_beyond_diff =
      | { decision.target_file for d in state.trace_decisions
                                if d.target_file is not None }
      | { p for d in state.trace_decisions for p in d.resolved_candidate_paths }
      | { f.path for f in state.trace_fetched_files }
      - { cf.path for cf in state.pr_context.changed_files }

"Beyond diff" = "outside the PR's changed-files set" — NOT
"Phase-2-fetched specifically." Per Pass-1 multi-lens audit
(adversarial F1): the rewritten union has THREE input sources but
the prior implementation only fed one, so the new branches need
direct regression-pin coverage. Otherwise a future refactor
that drops `resolved_candidate_paths` or `trace_fetched_files` from
the union would silently pass CI.
"""

from __future__ import annotations

from typing import Any

from outrider.agent.nodes.synthesize import _compute_files_traced_beyond_diff


def _make_decision_stub(
    *,
    target_file: str | None = None,
    resolved_candidate_paths: tuple[str, ...] = (),
) -> Any:
    """Duck-typed TraceDecision; only the two fields the metric reads.

    Avoids constructing a full TraceDecision (which would trip its
    own three-state validator: resolved ↔ target_file ↔
    resolved_candidate_paths shape — exactly the surfaces we want to
    test independent of)."""

    class _DecisionStub:
        def __init__(self) -> None:
            self.target_file = target_file
            self.resolved_candidate_paths = resolved_candidate_paths

    return _DecisionStub()


def _make_fetched_stub(*, path: str) -> Any:
    """Duck-typed TraceFetchedFile; only the path field."""

    class _FetchedStub:
        def __init__(self) -> None:
            self.path = path

    return _FetchedStub()


def _make_state_stub(
    *,
    diff_paths: tuple[str, ...] = (),
    decisions: tuple[Any, ...] = (),
    fetched: tuple[Any, ...] = (),
    analysis_rounds: tuple[Any, ...] = (),
) -> Any:
    """Duck-typed ReviewState exposing only the surfaces the metric
    helpers read. Sibling to `_make_state_stub` in
    `test_synthesize_node_defenses.py` (different field set — that
    helper is forge-detection-scoped)."""

    class _ChangedFileStub:
        def __init__(self, path: str) -> None:
            self.path = path

    class _PRContextStub:
        def __init__(self) -> None:
            self.changed_files = tuple(_ChangedFileStub(p) for p in diff_paths)

    class _StateStub:
        def __init__(self) -> None:
            self.pr_context = _PRContextStub()
            self.trace_decisions = decisions
            self.trace_fetched_files = fetched
            self.analysis_rounds = analysis_rounds

    return _StateStub()


def test_compute_metrics_populates_llm_aggregates() -> None:
    """FUP-093 exit pin: `_compute_metrics` carries the audit-stream LLM
    aggregates onto `ReviewMetrics` (the synthesize node queries them and
    passes them in). Guards against a regression that re-hardcodes `None`."""
    from outrider.agent.nodes.synthesize import _compute_metrics
    from outrider.audit.aggregates import ReviewLLMAggregates

    metrics = _compute_metrics(
        state=_make_state_stub(),  # empty state → files_examined / traced = 0
        wall_clock_seconds=1.5,
        llm_aggregates=ReviewLLMAggregates(
            llm_calls_made=3,
            total_input_tokens=300,
            total_output_tokens=150,
            total_cost_usd=0.05,
        ),
    )
    assert metrics.llm_calls_made == 3
    assert metrics.total_input_tokens == 300
    assert metrics.total_output_tokens == 150
    assert metrics.total_cost_usd == 0.05
    assert metrics.wall_clock_seconds == 1.5
    assert metrics.files_examined == 0
    assert metrics.files_traced_beyond_diff == 0


# ---------------------------------------------------------------------------
# Degenerate paths
# ---------------------------------------------------------------------------


def test_empty_state_returns_zero() -> None:
    """No decisions, no fetched files, no diff — the union is empty
    and the result is zero. Sanity floor; if this regresses, the
    function returns a positive int and something is very wrong."""
    state = _make_state_stub()
    assert _compute_files_traced_beyond_diff(state) == 0


def test_unresolved_decisions_contribute_nothing() -> None:
    """Per `schemas/trace_decision.py` tri-state contract: an
    unresolved decision has `target_file=None` AND
    `resolved_candidate_paths=()`. Neither branch of the union picks
    anything up, so a state with only unresolved decisions returns 0
    regardless of how many decisions land."""
    state = _make_state_stub(
        decisions=(
            _make_decision_stub(),
            _make_decision_stub(),
            _make_decision_stub(),
        )
    )
    assert _compute_files_traced_beyond_diff(state) == 0


# ---------------------------------------------------------------------------
# Single-source contributions
# ---------------------------------------------------------------------------


def test_resolved_decision_target_outside_diff_counts() -> None:
    """Resolved decision contributes `target_file` to the union when
    `target_file` is outside the diff."""
    state = _make_state_stub(
        diff_paths=("src/a.py",),
        decisions=(_make_decision_stub(target_file="src/utils.py"),),
    )
    assert _compute_files_traced_beyond_diff(state) == 1


def test_resolved_decision_target_inside_diff_does_not_count() -> None:
    """`target_file` IN `diff_paths` is subtracted; net contribution
    is zero. Pins that the subtraction step actually runs."""
    state = _make_state_stub(
        diff_paths=("src/a.py", "src/utils.py"),
        decisions=(_make_decision_stub(target_file="src/utils.py"),),
    )
    assert _compute_files_traced_beyond_diff(state) == 0


def test_ambiguous_decision_resolved_candidate_paths_count() -> None:
    """Per `schemas/trace_decision.py`: ambiguous decisions have
    `target_file=None` AND `len(resolved_candidate_paths) > 1`. The
    new union INCLUDES `resolved_candidate_paths` for any decision
    that has them — the prior impl missed this branch (CodeRabbit
    F2 catch). Pin the ambiguous-decision contribution."""
    state = _make_state_stub(
        diff_paths=(),
        decisions=(
            _make_decision_stub(
                target_file=None,
                resolved_candidate_paths=("src/a.py", "src/b.py", "src/c.py"),
            ),
        ),
    )
    assert _compute_files_traced_beyond_diff(state) == 3


def test_trace_fetched_files_outside_diff_count() -> None:
    """`state.trace_fetched_files[*].path` is the third source of the
    union. Pins that fetched-files contribute independently of
    whether a TraceDecision row carries the same path."""
    state = _make_state_stub(
        diff_paths=(),
        fetched=(_make_fetched_stub(path="src/fetched.py"),),
    )
    assert _compute_files_traced_beyond_diff(state) == 1


# ---------------------------------------------------------------------------
# Union semantics: deduplication + subtraction
# ---------------------------------------------------------------------------


def test_same_path_across_multiple_sources_counts_once() -> None:
    """The union dedup property: a path appearing in target_file AND
    resolved_candidate_paths AND trace_fetched_files counts ONE time,
    not three. Without set semantics this would be 3."""
    state = _make_state_stub(
        diff_paths=(),
        decisions=(
            _make_decision_stub(
                target_file="src/shared.py",
                resolved_candidate_paths=("src/shared.py",),
            ),
        ),
        fetched=(_make_fetched_stub(path="src/shared.py"),),
    )
    assert _compute_files_traced_beyond_diff(state) == 1


def test_combined_sources_with_partial_diff_overlap() -> None:
    """Full integration: resolved target + ambiguous candidates +
    fetched file; some inside the diff, some outside. Pins all three
    branches contribute AND the subtraction filters correctly.

    Setup:
      diff: src/a.py, src/b.py
      resolved decision target: src/c.py (outside)
      ambiguous candidates: src/a.py (inside), src/d.py (outside)
      fetched: src/b.py (inside), src/e.py (outside)
      expected union: {c, d, e} (a, b filtered by diff subtraction)
    """
    state = _make_state_stub(
        diff_paths=("src/a.py", "src/b.py"),
        decisions=(
            _make_decision_stub(target_file="src/c.py"),
            _make_decision_stub(
                target_file=None,
                resolved_candidate_paths=("src/a.py", "src/d.py"),
            ),
        ),
        fetched=(
            _make_fetched_stub(path="src/b.py"),
            _make_fetched_stub(path="src/e.py"),
        ),
    )
    assert _compute_files_traced_beyond_diff(state) == 3


# ---------------------------------------------------------------------------
# Revert-the-fold thought experiments
# ---------------------------------------------------------------------------


def test_dropping_resolved_candidate_paths_from_union_would_fail() -> None:
    """Negative pin: if a future refactor dropped
    `resolved_candidate_paths` from the union (regression to the
    pre-CodeRabbit `target_file`-only shape), this test would catch
    it because the only contribution here comes from an ambiguous
    decision whose `target_file` is None. The pre-rewrite impl
    returned 0 here; the new impl returns 3."""
    state = _make_state_stub(
        diff_paths=(),
        decisions=(
            _make_decision_stub(
                target_file=None,
                resolved_candidate_paths=("a", "b", "c"),
            ),
        ),
    )
    assert _compute_files_traced_beyond_diff(state) == 3


def test_dropping_trace_fetched_files_from_union_would_fail() -> None:
    """Negative pin: if a future refactor dropped
    `state.trace_fetched_files` from the union, this test would
    catch it because the only contribution comes from a fetched
    file whose path has no matching TraceDecision row."""
    state = _make_state_stub(
        diff_paths=(),
        fetched=(_make_fetched_stub(path="src/only_in_fetched.py"),),
    )
    assert _compute_files_traced_beyond_diff(state) == 1
