# Worker-outcome model pins per specs/2026-07-05-parallel-analyze.md + DECISIONS.md#063.
"""`AnalyzeWorkerOutcome` pins: facts and identities, never re-derived judgments.

The contracts: (1) pure JSON-round-trip state, no generated identities of
its own; (2) the discriminated `source` + explicit ORIGIN identity lists
(`producer_observed_hashes`, `served_content_hashes`) — origin is
identity, not evidence tier, so a model-cited OBSERVED proposal is a
legitimate proposal; (3) the #063 merge digest recomputes over the
canonical dump (the `AnalysisRound.round_id` precedent) — identical
retries with fresh finding UUIDs merge as no-ops, real divergence and
tampered state both fail loud.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, get_args, get_type_hints
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.agent.reducers import SlotDivergenceError
from outrider.ast_facts.models import SkipReason
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import FindingType, lookup_severity
from outrider.policy.versions import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewState
from outrider.schemas.analyze_worker import (
    WORKER_OUTCOME_EXCLUDE_PATHS,
    AnalyzeWorkerOutcome,
    worker_outcome_slot,
)
from outrider.schemas.observed_subsumption import ObservedSubsumedMatch
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.trace_candidate import TraceCandidate
from outrider.schemas.triage_result import ReviewTier

_REVIEW_ID = uuid4()


def _finding(
    title: str = "t",
    *,
    tier: EvidenceTier = EvidenceTier.JUDGED,
    line: int = 3,
) -> ReviewFinding:
    finding_type = FindingType.HARDCODED_SECRET
    return ReviewFinding(
        review_id=_REVIEW_ID,
        installation_id=42,
        finding_type=finding_type,
        severity=lookup_severity(finding_type),
        file_path="src/app.py",
        line_start=line,
        line_end=line,
        title=title,
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=tier,
        query_match_id=(
            "javascript.tls_env_verify_disabled" if tier is EvidenceTier.OBSERVED else None
        ),
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            "src/app.py", line_start=line, line_end=line, finding_type=finding_type
        ),
        proposal_hash="a" * 64,
    )


def _outcome(**overrides: Any) -> AnalyzeWorkerOutcome:
    """A valid parser-source outcome; override per test."""
    base: dict[str, Any] = {
        "path": "src/app.py",
        "pass_index": 0,
        "source": "parser",
        "parse_status": "clean",
        "review_tier": ReviewTier.DEEP,
        "n_proposals_seen": 1,
        "admitted_findings": (_finding(),),
        "cost": Decimal("0.0123"),
        "input_tokens": 100,
        "output_tokens": 50,
    }
    base.update(overrides)
    return AnalyzeWorkerOutcome(**base)


def _observed_skip(**overrides: Any) -> AnalyzeWorkerOutcome:
    """A valid observed_skip (ride-out) outcome."""
    finding = _finding(tier=EvidenceTier.OBSERVED)
    base: dict[str, Any] = {
        "path": "src/app.py",
        "pass_index": 0,
        "source": "observed_skip",
        "parse_status": "skipped",
        "skip_reason": SkipReason.COST_BUDGET_EXHAUSTED,
        "review_tier": ReviewTier.DEEP,
        "admitted_findings": (finding,),
        "producer_observed_hashes": (finding.content_hash,),
    }
    base.update(overrides)
    return AnalyzeWorkerOutcome(**base)


def _cache_serve(**overrides: Any) -> AnalyzeWorkerOutcome:
    """A valid cache_serve outcome."""
    finding = _finding()
    base: dict[str, Any] = {
        "path": "src/app.py",
        "pass_index": 0,
        "source": "cache_serve",
        "parse_status": "clean",
        "review_tier": ReviewTier.DEEP,
        "admitted_findings": (finding,),
        "served_content_hashes": (finding.content_hash,),
    }
    base.update(overrides)
    return AnalyzeWorkerOutcome(**base)


def _state_reducer() -> Any:
    hints = get_type_hints(ReviewState, include_extras=True)
    metadata = get_args(hints["analyze_worker_outcomes"])[1:]
    reducers = [m for m in metadata if callable(m)]
    assert reducers, "analyze_worker_outcomes carries no reducer — LangGraph would concat"
    return reducers[0]


# ---------------------------------------------------------------------------
# Skip + source coherence (fact-anchored).
# ---------------------------------------------------------------------------


def test_skip_coherence_iff_contract_both_directions() -> None:
    with pytest.raises(ValidationError, match="iff"):
        _observed_skip(skip_reason=None)
    with pytest.raises(ValidationError, match="iff"):
        _outcome(skip_reason=SkipReason.COST_BUDGET_EXHAUSTED)


def test_skipped_iff_skip_source() -> None:
    with pytest.raises(ValidationError, match="does not match source"):
        _outcome(source="observed_skip")


def _subsumed_match(path: str = "src/app.py") -> ObservedSubsumedMatch:
    return ObservedSubsumedMatch(
        file_path=path,
        query_match_id="javascript.tls_env_verify_disabled",
        finding_type=FindingType.HARDCODED_SECRET,
        subsumed_by_finding_type=FindingType.HARDCODED_SECRET,
        line_start=3,
        line_end=3,
        dropped_content_hash=compute_finding_content_hash(
            path, line_start=3, line_end=3, finding_type=FindingType.HARDCODED_SECRET
        ),
        subsumer_content_hash=compute_finding_content_hash(
            path, line_start=3, line_end=3, finding_type=FindingType.HARDCODED_SECRET
        ),
    )


def test_cache_serve_reconstructs_subsumption_records() -> None:
    """#055 proof retention survives the serve: _serve_cache_hit
    reconstructs subsumed matches from the cached payload — a serve
    outcome carrying them is valid; skip sources still reject."""
    outcome = _cache_serve(subsumed_matches=(_subsumed_match(),))
    assert outcome.subsumed_matches
    with pytest.raises(ValidationError, match="parser and cache_serve"):
        _observed_skip(subsumed_matches=(_subsumed_match(),))


def test_parser_only_facts_require_parser() -> None:
    """Tallies, spend, and candidate-on-skip all imply a parser ran."""
    with pytest.raises(ValidationError, match="parser tallies"):
        _cache_serve(n_proposals_seen=1)
    with pytest.raises(ValidationError, match="provider spend"):
        _cache_serve(input_tokens=100)
    with pytest.raises(ValidationError, match="provider spend"):
        _observed_skip(cost=Decimal("0.01"))


def test_response_rejection_zeroes_proposals_but_allows_producer_augment() -> None:
    """The parser's verified contract: a rejected response yields zero
    proposals. It does NOT preclude the deterministic producer's OBSERVED
    findings — augmentation rides the same pass."""
    with pytest.raises(ValidationError, match="zero proposals"):
        _outcome(n_responses_rejected=1)  # base carries n_proposals_seen=1
    finding = _finding(tier=EvidenceTier.OBSERVED)
    augmented = _outcome(
        n_responses_rejected=1,
        n_proposals_seen=0,
        admitted_findings=(finding,),
        producer_observed_hashes=(finding.content_hash,),
    )
    assert augmented.admitted_findings  # producer augmentation is legal
    with pytest.raises(ValidationError, match="one LLM call"):
        _outcome(n_responses_rejected=2, n_proposals_seen=0, admitted_findings=())
    with pytest.raises(ValidationError, match="zero proposals"):
        _outcome(
            n_responses_rejected=1,
            n_proposals_seen=0,
            admitted_findings=(),
            n_trace_candidates_dropped_malformed=1,
        )


# ---------------------------------------------------------------------------
# Origin identity (never inferred from evidence tier).
# ---------------------------------------------------------------------------


def test_model_cited_observed_is_a_proposal_not_producer_origin() -> None:
    """A model-cited OBSERVED finding (valid query_match_id, admitted through
    the proposal pipeline) stays OUT of producer_observed_hashes — origin is
    identity, not tier."""
    cited = _finding(tier=EvidenceTier.OBSERVED)
    outcome = _outcome(admitted_findings=(cited,))  # producer list empty
    assert outcome.producer_observed_hashes == ()


def test_producer_listed_findings_must_be_observed_tier() -> None:
    judged = _finding()  # JUDGED
    with pytest.raises(ValidationError, match="OBSERVED-tier"):
        _outcome(
            n_proposals_seen=0,
            admitted_findings=(judged,),
            producer_observed_hashes=(judged.content_hash,),
        )


def test_identity_lists_are_canonical_and_subsets() -> None:
    with pytest.raises(ValidationError, match="SHA-256 hex"):
        _outcome(producer_observed_hashes=("nope",))
    with pytest.raises(ValidationError, match="subset"):
        _outcome(producer_observed_hashes=("b" * 64,))
    # (format/subset errors fire before the accounting equation)
    with pytest.raises(ValidationError, match="subset"):
        _cache_serve(served_content_hashes=("b" * 64,))
    finding = _finding()
    with pytest.raises(ValidationError, match="sorted and unique"):
        _cache_serve(served_content_hashes=(finding.content_hash, finding.content_hash))


def test_served_is_cache_serve_exclusive_and_exact() -> None:
    finding = _finding()
    with pytest.raises(ValidationError, match="cache_serve"):
        _outcome(
            admitted_findings=(finding,),
            served_content_hashes=(finding.content_hash,),
        )
    with pytest.raises(ValidationError, match="exactly the served set"):
        _cache_serve(served_content_hashes=(), admitted_findings=(finding,))


def test_observed_skip_findings_are_all_producer_origin() -> None:
    """producer == admitted on the ride-out — which also forces every
    finding OBSERVED-tier via the producer-tier rule."""
    with pytest.raises(ValidationError, match="producer-origin"):
        _observed_skip(producer_observed_hashes=())
    with pytest.raises(ValidationError, match="OBSERVED-tier|producer-origin"):
        _observed_skip(
            admitted_findings=(_finding(),),  # JUDGED
            producer_observed_hashes=(_finding().content_hash,),
        )


def test_cache_serve_restores_prior_pass_trace_candidates() -> None:
    """The sequential serve branch extends state with the cached
    candidates WITHOUT counting them as newly emitted — a serve outcome
    carrying candidates is valid, and the aggregate derives this-pass
    emission from parser sources only."""
    from outrider.policy.canonical import compute_candidate_id, compute_identity_hash

    source_proposal_hash = compute_identity_hash({"prop": "cached"})
    candidate = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=source_proposal_hash,
            import_string="app.services.db",
            reason="cached",
        ),
        source_proposal_hash=source_proposal_hash,
        import_string="app.services.db",
        reason="cached",
    )
    outcome = _cache_serve(trace_candidates=(candidate,))
    assert outcome.trace_candidates
    with pytest.raises(ValidationError, match="parser and cache_serve"):
        _observed_skip(trace_candidates=(candidate,))


def test_worker_local_equation_from_identities() -> None:
    """seen == (admitted − producer) + rejected + superseded, every term an
    identity count — aggregate-only accounting would let invalid tallies
    cancel across files."""
    with pytest.raises(ValidationError, match="proposal accounting"):
        _outcome(n_proposals_seen=2)  # 1 admitted, 0 producer, 0 rejected
    ok = _outcome(
        n_proposals_seen=2,
        n_proposals_rejected=1,
    )
    assert ok.n_proposals_seen == 2


def test_trace_candidate_evidence_implies_proposals_seen() -> None:
    """Candidates are processed only inside the parser's proposal loop —
    both the well-formed and malformed forms imply seen > 0."""
    from outrider.policy.canonical import compute_candidate_id, compute_identity_hash

    source_proposal_hash = compute_identity_hash({"prop": "x"})
    candidate = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=source_proposal_hash,
            import_string="app.services.db",
            reason="r",
        ),
        source_proposal_hash=source_proposal_hash,
        import_string="app.services.db",
        reason="r",
    )
    with pytest.raises(ValidationError, match="imply n_proposals_seen"):
        _outcome(n_proposals_seen=0, admitted_findings=(), trace_candidates=(candidate,))
    with pytest.raises(ValidationError, match="imply n_proposals_seen"):
        _outcome(
            n_proposals_seen=0,
            admitted_findings=(),
            n_trace_candidates_dropped_malformed=1,
        )


def test_plain_skip_carries_nothing() -> None:
    with pytest.raises(ValidationError, match="plain_skip carries nothing"):
        _observed_skip(source="plain_skip")


def test_cross_file_attribution_is_unrepresentable() -> None:
    with pytest.raises(ValidationError, match="worker's file"):
        _outcome(path="src/other.py")


# ---------------------------------------------------------------------------
# Shape basics.
# ---------------------------------------------------------------------------


def test_path_is_canonicalized_like_analysis_round() -> None:
    outcome = _outcome(path="./src/app.py")
    assert outcome.path == "src/app.py"
    assert worker_outcome_slot(outcome) == ("src/app.py", 0)


def test_json_round_trip_is_exact_including_decimal_cost() -> None:
    outcome = _outcome(cost=Decimal("0.123456789"))
    restored = AnalyzeWorkerOutcome.model_validate_json(outcome.model_dump_json())
    assert restored.cost == Decimal("0.123456789")
    assert isinstance(restored.cost, Decimal)
    assert restored.admitted_findings == outcome.admitted_findings


def test_exclusion_paths_match_generated_fields_one_for_one() -> None:
    """Every default_factory-generated field in the model tree appears in
    WORKER_OUTCOME_EXCLUDE_PATHS and nothing else does."""
    generated: set[str] = set()
    for model, prefix in (
        (AnalyzeWorkerOutcome, ""),
        (ReviewFinding, "admitted_findings.[]."),
        (TraceCandidate, "trace_candidates.[]."),
        (ObservedSubsumedMatch, "subsumed_matches.[]."),
    ):
        for name, field in model.model_fields.items():
            if field.default_factory is not None and field.default_factory is not tuple:
                generated.add(f"{prefix}{name}")
    assert generated == set(WORKER_OUTCOME_EXCLUDE_PATHS)


# ---------------------------------------------------------------------------
# The #063 merge digest (recompute-at-merge; AnalysisRound.round_id precedent).
# ---------------------------------------------------------------------------


def test_identical_retry_with_fresh_finding_uuids_merges_as_noop() -> None:
    reducer = _state_reducer()
    first = _outcome(admitted_findings=(_finding("x"),))
    retry = _outcome(admitted_findings=(_finding("x"),))
    assert first.admitted_findings[0].finding_id != retry.admitted_findings[0].finding_id
    assert reducer([first], [retry]) == [first]


def test_semantic_divergence_fails_loud() -> None:
    reducer = _state_reducer()
    first = _outcome(admitted_findings=(_finding("x"),))
    diverged = _outcome(admitted_findings=(_finding("y"),))
    with pytest.raises(SlotDivergenceError):
        reducer([first], [diverged])


def test_post_insertion_mutation_fails_loud_never_merges_silently() -> None:
    """The #063 resolution (AnalysisRound.round_id precedent): state is
    never mutated post-insertion; if it IS, the recomputed digest diverges
    and the merge fails loud — tampered state must never silently pass as
    an identical retry."""
    reducer = _state_reducer()
    original = _outcome(admitted_findings=(_finding("x"),))
    original.admitted_findings[0].title = "mutated-after-insertion"
    retry = _outcome(admitted_findings=(_finding("x"),))  # pre-mutation semantics
    with pytest.raises(SlotDivergenceError):
        reducer([original], [retry])
