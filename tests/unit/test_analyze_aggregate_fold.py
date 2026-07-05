# Aggregate-fold pins per specs/2026-07-05-parallel-analyze.md + DECISIONS.md#063.
"""`fold_worker_outcomes` pins.

The load-bearing properties: completion-order determinism (same
`round_id` from any ordering), the sequential dedup/collapse/cap chain,
the post-cap recompute by ORIGIN IDENTITY feeding a real
`AnalyzeCompletedEvent` whose accounting equation validates, the
zero-worker empty pass, non-aliasing (cloned findings), and fail-loud
inputs (mixed passes, duplicate paths).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from outrider.agent.nodes.analyze_aggregate import (
    FoldInputError,
    fold_worker_outcomes,
)
from outrider.ast_facts.models import SkipReason
from outrider.audit.events import AnalyzeCompletedEvent, compute_finding_content_hash
from outrider.policy import EvidenceTier
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import FindingType, lookup_severity
from outrider.policy.versions import ACTIVE_POLICY_VERSION
from outrider.schemas.analyze_worker import AnalyzeWorkerOutcome
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.triage_result import ReviewTier

_REVIEW_ID = uuid4()
_T0 = datetime(2026, 7, 5, 12, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(seconds=30)


def _finding(
    path: str,
    *,
    line: int = 3,
    tier: EvidenceTier = EvidenceTier.JUDGED,
    proposal_hash: str | None = None,
) -> ReviewFinding:
    # Distinct per (path, line) by default: AnalysisRound enforces
    # per-round proposal_hash uniqueness (#025 point 4).
    if proposal_hash is None:
        proposal_hash = hashlib.sha256(f"{path}:{line}".encode()).hexdigest()
    finding_type = FindingType.HARDCODED_SECRET
    return ReviewFinding(
        review_id=_REVIEW_ID,
        installation_id=42,
        finding_type=finding_type,
        severity=lookup_severity(finding_type),
        file_path=path,
        line_start=line,
        line_end=line,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=tier,
        query_match_id=(
            "javascript.tls_env_verify_disabled" if tier is EvidenceTier.OBSERVED else None
        ),
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            path, line_start=line, line_end=line, finding_type=finding_type
        ),
        proposal_hash=proposal_hash,
    )


def _parser_outcome(path: str, **overrides: Any) -> AnalyzeWorkerOutcome:
    finding = _finding(path)
    base: dict[str, Any] = {
        "path": path,
        "pass_index": 0,
        "source": "parser",
        "parse_status": "clean",
        "review_tier": ReviewTier.DEEP,
        "n_proposals_seen": 1,
        "admitted_findings": (finding,),
        "cost": Decimal("0.01"),
        "input_tokens": 100,
        "output_tokens": 50,
    }
    base.update(overrides)
    return AnalyzeWorkerOutcome(**base)


def _skip_outcome(path: str, **overrides: Any) -> AnalyzeWorkerOutcome:
    base: dict[str, Any] = {
        "path": path,
        "pass_index": 0,
        "source": "plain_skip",
        "parse_status": "skipped",
        "skip_reason": SkipReason.COST_BUDGET_EXHAUSTED,
        "review_tier": ReviewTier.DEEP,
    }
    base.update(overrides)
    return AnalyzeWorkerOutcome(**base)


def _fold(outcomes: tuple[AnalyzeWorkerOutcome, ...], pass_index: int = 0) -> Any:
    return fold_worker_outcomes(outcomes, pass_index=pass_index, started_at=_T0, ended_at=_T1)


# ---------------------------------------------------------------------------
# Determinism + shape.
# ---------------------------------------------------------------------------


def test_completion_order_never_changes_the_round() -> None:
    """round_id is content-derived; a completion-order-dependent fold would
    break replay idempotence."""
    a = _parser_outcome("src/a.py")
    b = _parser_outcome("src/b.py")
    c = _skip_outcome("src/c.py")
    one = _fold((a, b, c))
    other = _fold((c, b, a))
    assert one.round.round_id == other.round.round_id
    assert one.round.findings == other.round.findings
    assert one == other


def test_zero_outcomes_folds_the_valid_empty_pass() -> None:
    """The zero-worker planner→aggregate route: one empty round, zero
    counters — the pass still exists."""
    fold = _fold(())
    assert fold.round.findings == ()
    assert fold.n_files_analyzed == 0
    assert fold.n_llm_calls == 0
    assert fold.round.pass_index == 0


def test_mixed_passes_and_duplicate_paths_fail_loud() -> None:
    with pytest.raises(FoldInputError, match="pass indices"):
        _fold((_parser_outcome("src/a.py"), _parser_outcome("src/b.py", pass_index=1)))
    with pytest.raises(FoldInputError, match="duplicate paths"):
        _fold((_parser_outcome("src/a.py"), _skip_outcome("src/a.py")))


# ---------------------------------------------------------------------------
# The sequential chain: pair dedup → hash collapse → cap → recompute.
# ---------------------------------------------------------------------------


def test_cross_file_pair_dedup_and_hash_collapse() -> None:
    """Same (content_hash, proposal_hash) admitted once; differing
    proposal_hash with same content_hash collapses first-wins — the
    sequential _admit_with_dedup + FUP-180 collapse chain."""
    f1 = _finding("src/a.py")
    twin = _finding("src/a.py")  # same content_hash, same proposal_hash
    variant = _finding("src/a.py", proposal_hash="b" * 64)  # same hash, new prose
    outcome = _parser_outcome(
        "src/a.py",
        n_proposals_seen=3,
        admitted_findings=(f1, twin, variant),
    )
    fold = _fold((outcome,))
    assert len(fold.round.findings) == 1  # pair-dedup then hash-collapse
    assert fold.n_findings_emitted == 1


def test_post_cap_recompute_by_origin_identity_feeds_a_valid_event() -> None:
    """The strongest pin: fold output constructs a REAL AnalyzeCompletedEvent
    and its proposal-accounting validator passes. Origin classification is
    by IDENTITY — the model-cited OBSERVED finding (valid registry id, not
    in producer_observed_hashes) counts as a surviving PROPOSAL, which the
    sequential tier+registry heuristic would have miscounted."""
    producer = _finding("src/a.py", line=1, tier=EvidenceTier.OBSERVED)
    cited = _finding("src/a.py", line=2, tier=EvidenceTier.OBSERVED)  # model-cited
    outcome = _parser_outcome(
        "src/a.py",
        n_proposals_seen=1,  # the cited proposal; producer finding isn't one
        admitted_findings=(producer, cited),
        producer_observed_hashes=(producer.content_hash,),
    )
    served_finding = _finding("src/b.py")
    serve = AnalyzeWorkerOutcome(
        path="src/b.py",
        pass_index=0,
        source="cache_serve",
        parse_status="clean",
        review_tier=ReviewTier.DEEP,
        admitted_findings=(served_finding,),
        served_content_hashes=(served_finding.content_hash,),
    )
    fold = _fold((outcome, serve))
    assert fold.n_findings_emitted == 3
    assert fold.n_findings_observed == 1  # producer only — NOT the cited one
    assert fold.n_findings_served == 1
    assert fold.n_proposals_dropped == 0
    event = AnalyzeCompletedEvent(
        review_id=_REVIEW_ID,
        pass_index=0,
        n_files_analyzed=fold.n_files_analyzed,
        n_files_skipped=fold.n_files_skipped,
        n_llm_calls=fold.n_llm_calls,
        n_proposals_seen=fold.n_proposals_seen,
        n_findings_emitted=fold.n_findings_emitted,
        n_findings_served=fold.n_findings_served,
        n_findings_observed=fold.n_findings_observed,
        n_proposals_superseded_by_observed=fold.n_proposals_superseded_by_observed,
        n_proposals_dropped=fold.n_proposals_dropped,
        n_findings_dropped_over_cap=fold.n_findings_dropped_over_cap,
        subsumed_matches=fold.subsumed_matches,
        n_proposals_rejected=fold.n_proposals_rejected,
        n_responses_rejected=fold.n_responses_rejected,
        n_trace_candidates_emitted=fold.n_trace_candidates_emitted,
        n_trace_candidates_dropped_malformed=fold.n_trace_candidates_dropped_malformed,
        total_input_tokens=fold.total_input_tokens,
        total_cache_read_tokens=fold.total_cache_read_tokens,
        total_cache_write_tokens=fold.total_cache_write_tokens,
        total_output_tokens=fold.total_output_tokens,
        total_cost_usd=float(fold.total_cost),
        pricing_version="v2",
        policy_version=ACTIVE_POLICY_VERSION,
        analyze_model="claude-sonnet-4-6",
        standard_analyze_model=None,
    )
    assert event.n_proposals_seen == 1  # equation validated at construction


def test_aggregate_dedup_drop_balances_the_equation() -> None:
    """Two workers admit the same proposal (same content+proposal hash from
    a shared template) — the survivor keeps the equation balanced via
    n_proposals_dropped."""
    shared = _finding("src/a.py")
    shared_b = _finding("src/a.py")  # identical hashes, second worker... same path
    # Same content from two DIFFERENT files is impossible (single-file
    # attribution), so model the drop with the same-file variant collapse:
    variant = _finding("src/a.py", proposal_hash="c" * 64)
    outcome = _parser_outcome(
        "src/a.py",
        n_proposals_seen=2,
        admitted_findings=(shared, variant),
    )
    del shared_b
    fold = _fold((outcome,))
    assert fold.n_findings_emitted == 1  # hash collapse dropped the variant
    assert fold.n_proposals_dropped == 1  # and the equation stays balanced


# ---------------------------------------------------------------------------
# Non-aliasing + per-source facts.
# ---------------------------------------------------------------------------


def test_round_findings_are_clones_not_aliases() -> None:
    """The 3b-2 acceptance gate: no live object shared between worker
    outcomes and the round."""
    outcome = _parser_outcome("src/a.py")
    fold = _fold((outcome,))
    assert fold.round.findings[0] == outcome.admitted_findings[0].model_copy(
        update={"finding_id": fold.round.findings[0].finding_id}
    )
    assert fold.round.findings[0] is not outcome.admitted_findings[0]


def test_serve_candidates_restore_without_counting_as_emitted() -> None:
    from outrider.policy.canonical import compute_candidate_id, compute_identity_hash

    sph = compute_identity_hash({"prop": "x"})
    from outrider.schemas.trace_candidate import TraceCandidate

    candidate = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=sph, import_string="app.db", reason="r"
        ),
        source_proposal_hash=sph,
        import_string="app.db",
        reason="r",
    )
    served_finding = _finding("src/b.py")
    serve = AnalyzeWorkerOutcome(
        path="src/b.py",
        pass_index=0,
        source="cache_serve",
        parse_status="clean",
        review_tier=ReviewTier.DEEP,
        admitted_findings=(served_finding,),
        served_content_hashes=(served_finding.content_hash,),
        trace_candidates=(candidate,),
    )
    fold = _fold((serve,))
    assert fold.trace_candidates == (candidate,)  # restored into state
    assert fold.n_trace_candidates_emitted == 0  # not emitted this pass


def test_budget_skips_and_standard_tier_surface() -> None:
    fold = _fold(
        (
            _skip_outcome("src/a.py"),
            _skip_outcome("src/b.py", skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS),
            _parser_outcome("src/c.py", review_tier=ReviewTier.STANDARD),
        )
    )
    assert fold.budget_skip_count == 1
    assert fold.n_files_skipped == 2
    assert fold.n_files_analyzed == 1
    assert fold.standard_tier_llm_used
