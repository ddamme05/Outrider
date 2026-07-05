# Worker-outcome model pins per specs/2026-07-05-parallel-analyze.md + DECISIONS.md#063.
"""`AnalyzeWorkerOutcome` shape, digest, and state-reducer wiring pins.

The three load-bearing contracts: (1) the outcome is pure JSON-round-trip
state with NO generated identities of its own; (2) the semantic digest
treats identical retries (fresh nested finding UUIDs) as equal and any
semantic change as divergent, with the exclusion-path list pinned
one-for-one against the models' generated fields; (3) `ReviewState`
carries the field under the slot-guard reducer, not first-wins dedup.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, get_args, get_type_hints
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.agent.reducers import SlotDivergenceError, semantic_digest
from outrider.ast_facts.models import SkipReason
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import FindingType, lookup_severity
from outrider.policy.versions import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewState
from outrider.schemas.analyze_worker import (
    WORKER_OUTCOME_EXCLUDE_PATHS,
    AnalyzeWorkerCounters,
    AnalyzeWorkerOutcome,
    worker_outcome_slot,
)
from outrider.schemas.observed_subsumption import ObservedSubsumedMatch
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.trace_candidate import TraceCandidate
from outrider.schemas.triage_result import ReviewTier

_REVIEW_ID = uuid4()


def _finding(title: str = "t") -> ReviewFinding:
    finding_type = FindingType.HARDCODED_SECRET
    return ReviewFinding(
        review_id=_REVIEW_ID,
        installation_id=42,
        finding_type=finding_type,
        severity=lookup_severity(finding_type),
        file_path="src/app.py",
        line_start=3,
        line_end=3,
        title=title,
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            "src/app.py", line_start=3, line_end=3, finding_type=finding_type
        ),
        proposal_hash="a" * 64,
    )


def _outcome(**overrides: Any) -> AnalyzeWorkerOutcome:
    base: dict[str, Any] = {
        "path": "src/app.py",
        "pass_index": 0,
        "parse_status": "clean",
        "review_tier": ReviewTier.DEEP,
        "llm_called": True,
        "counters": AnalyzeWorkerCounters(n_proposals_seen=1, n_findings_emitted=1),
        "admitted_findings": (_finding(),),
        "cost": Decimal("0.0123"),
        "input_tokens": 100,
        "output_tokens": 50,
    }
    base.update(overrides)
    return AnalyzeWorkerOutcome(**base)


def _digest(o: AnalyzeWorkerOutcome) -> str:
    return semantic_digest(o, exclude_paths=WORKER_OUTCOME_EXCLUDE_PATHS)


# ---------------------------------------------------------------------------
# Shape + validators.
# ---------------------------------------------------------------------------


_NO_SPEND: dict[str, Any] = {
    "llm_called": False,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "cost": Decimal("0"),
}


def test_skip_coherence_iff_contract_both_directions() -> None:
    """skip_reason non-None iff parse_status='skipped' (the #018 iff shape)."""
    with pytest.raises(ValidationError, match="iff"):
        _outcome(parse_status="skipped", **_NO_SPEND)  # skipped, no reason
    with pytest.raises(ValidationError, match="iff"):
        _outcome(skip_reason=SkipReason.COST_BUDGET_EXHAUSTED)  # reason, not skipped


def test_skipped_file_never_made_an_llm_call() -> None:
    with pytest.raises(ValidationError, match="never makes an LLM call"):
        _outcome(
            parse_status="skipped",
            skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
            llm_called=True,
        )


def test_ride_out_shape_is_legal_findings_on_a_skip() -> None:
    """The module-arm ride-out: a budget skip carrying OBSERVED findings is a
    legitimate outcome and must construct."""
    outcome = _outcome(
        parse_status="skipped",
        skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
        counters=AnalyzeWorkerCounters(n_findings_emitted=1, n_findings_observed=1),
        **_NO_SPEND,
    )
    assert outcome.admitted_findings


def test_no_llm_call_means_zero_spend() -> None:
    """llm_called=False with nonzero provider tokens or cost would let the
    aggregate's accounting contradict the LLMCallEvent stream."""
    with pytest.raises(ValidationError, match="zero provider"):
        _outcome(llm_called=False)  # base fixture carries tokens + cost
    with pytest.raises(ValidationError, match="zero provider"):
        _outcome(**{**_NO_SPEND, "cost": Decimal("0.01")})


def test_path_is_canonicalized_like_analysis_round() -> None:
    """Two spellings of one path must not occupy two slots."""
    outcome = _outcome(path="./src/app.py")
    assert outcome.path == "src/app.py"
    assert worker_outcome_slot(outcome) == ("src/app.py", 0)


def test_json_round_trip_is_exact_including_decimal_cost() -> None:
    outcome = _outcome(cost=Decimal("0.123456789"))
    restored = AnalyzeWorkerOutcome.model_validate_json(outcome.model_dump_json())
    assert restored.cost == Decimal("0.123456789")
    assert isinstance(restored.cost, Decimal)
    assert restored.admitted_findings == outcome.admitted_findings


# ---------------------------------------------------------------------------
# Semantic digest (#063).
# ---------------------------------------------------------------------------


def test_identical_retry_with_fresh_finding_uuids_digests_equal() -> None:
    """THE #063 case on the real model: a retry's findings carry fresh
    generated finding_ids; the digest must not see them."""
    first = _outcome(admitted_findings=(_finding("x"),))
    retry = _outcome(admitted_findings=(_finding("x"),))
    assert first.admitted_findings[0].finding_id != retry.admitted_findings[0].finding_id
    assert _digest(first) == _digest(retry)


def test_any_semantic_change_moves_the_digest() -> None:
    base = _outcome()
    assert _digest(base) != _digest(_outcome(admitted_findings=(_finding("other"),)))
    assert _digest(base) != _digest(_outcome(output_tokens=51))
    assert _digest(base) != _digest(_outcome(cost=Decimal("0.0124")))


def test_exclusion_paths_match_generated_fields_one_for_one() -> None:
    """The two-directional #063 pin: every default_factory-generated field in
    the outcome's model tree appears in WORKER_OUTCOME_EXCLUDE_PATHS, and
    every excluded path points at such a field. A generated field the list
    misses makes identical retries digest-divergent (crash on legitimate
    replay); a semantic field wrongly listed is silently ignored."""
    generated: set[str] = set()
    for model, prefix in (
        (AnalyzeWorkerOutcome, ""),
        (AnalyzeWorkerCounters, "counters."),
        (ReviewFinding, "admitted_findings.[]."),
        (TraceCandidate, "trace_candidates.[]."),
        (ObservedSubsumedMatch, "subsumed_matches.[]."),
    ):
        for name, field in model.model_fields.items():
            if field.default_factory is not None and field.default_factory is not tuple:
                generated.add(f"{prefix}{name}")
    assert generated == set(WORKER_OUTCOME_EXCLUDE_PATHS)


# ---------------------------------------------------------------------------
# ReviewState wiring: the slot-guard reducer, not first-wins dedup.
# ---------------------------------------------------------------------------


def _state_reducer() -> Any:
    hints = get_type_hints(ReviewState, include_extras=True)
    metadata = get_args(hints["analyze_worker_outcomes"])[1:]
    reducers = [m for m in metadata if callable(m)]
    assert reducers, "analyze_worker_outcomes carries no reducer — LangGraph would concat"
    return reducers[0]


def test_state_field_merges_identical_retries_and_rejects_divergence() -> None:
    reducer = _state_reducer()
    first = _outcome(admitted_findings=(_finding("x"),))
    retry = _outcome(admitted_findings=(_finding("x"),))  # fresh UUIDs, same semantics
    assert reducer([first], [retry]) == [first]
    diverged = _outcome(admitted_findings=(_finding("y"),))
    with pytest.raises(SlotDivergenceError):
        reducer([first], [diverged])


def test_served_hashes_canonical_form_enforced() -> None:
    """SHA-hex, sorted-unique, subset-of-admitted, counter-coupled — each
    direction pinned. Sorted-unique is digest stability (emission-order
    encodings would falsely diverge identical retries); subset + coupling is
    the post-cap kept_served accounting."""
    finding = _finding()
    h = finding.content_hash
    ok = _outcome(
        admitted_findings=(finding,),
        served_content_hashes=(h,),
        counters=AnalyzeWorkerCounters(n_findings_emitted=1, n_findings_served=1),
        **_NO_SPEND,
    )
    assert ok.served_content_hashes == (h,)
    with pytest.raises(ValidationError, match="SHA-256 hex"):
        _outcome(served_content_hashes=("nope",))
    with pytest.raises(ValidationError, match="sorted and unique"):
        _outcome(
            admitted_findings=(finding,),
            served_content_hashes=(h, h),
            counters=AnalyzeWorkerCounters(n_findings_emitted=1, n_findings_served=2),
            **_NO_SPEND,
        )
    with pytest.raises(ValidationError, match="subset"):
        _outcome(served_content_hashes=("b" * 64,))
    with pytest.raises(ValidationError, match="n_findings_served"):
        _outcome(
            admitted_findings=(finding,),
            served_content_hashes=(h,),
            counters=AnalyzeWorkerCounters(n_findings_emitted=1, n_findings_served=0),
            **_NO_SPEND,
        )


def test_cross_file_attribution_is_unrepresentable() -> None:
    """A worker outcome must not be able to express what the sequential
    per-file loop cannot: findings or subsumption records for another file."""
    with pytest.raises(ValidationError, match="worker's file"):
        _outcome(path="src/other.py")  # base finding names src/app.py
    match = ObservedSubsumedMatch(
        file_path="src/other.py",
        query_match_id="javascript.tls_env_verify_disabled",
        finding_type=FindingType.HARDCODED_SECRET,
        subsumed_by_finding_type=FindingType.HARDCODED_SECRET,
        line_start=3,
        line_end=3,
        dropped_content_hash=compute_finding_content_hash(
            "src/other.py",
            line_start=3,
            line_end=3,
            finding_type=FindingType.HARDCODED_SECRET,
        ),
        subsumer_content_hash=compute_finding_content_hash(
            "src/other.py",
            line_start=3,
            line_end=3,
            finding_type=FindingType.HARDCODED_SECRET,
        ),
    )
    with pytest.raises(ValidationError, match="worker's file"):
        _outcome(subsumed_matches=(match,))
