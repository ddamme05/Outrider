# Worker-outcome model pins per specs/2026-07-05-parallel-analyze.md + DECISIONS.md#063.
"""`AnalyzeWorkerOutcome` shape, source-coherence, digest, and reducer pins.

The load-bearing contracts: (1) pure JSON-round-trip state with NO
generated identities of its own; (2) the discriminated `source` makes the
sequential branch union explicit and every counter/finding coherence rule
hangs off it; (3) the #063 merge digest is snapshotted at construction —
nested findings are mutable by design, and a post-insertion mutation must
never falsely diverge a legitimate retry; (4) `ReviewState` carries the
field under the slot-guard reducer comparing stored snapshots.
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


def _observed_finding(title: str = "t") -> ReviewFinding:
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
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="javascript.tls_env_verify_disabled",
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            "src/app.py", line_start=3, line_end=3, finding_type=finding_type
        ),
        proposal_hash="b" * 64,
    )


def _outcome(**overrides: Any) -> AnalyzeWorkerOutcome:
    """A valid parser-source outcome; override per test."""
    base: dict[str, Any] = {
        "path": "src/app.py",
        "pass_index": 0,
        "source": "parser",
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


def _observed_skip(**overrides: Any) -> AnalyzeWorkerOutcome:
    """A valid observed_skip (ride-out) outcome."""
    base: dict[str, Any] = {
        "path": "src/app.py",
        "pass_index": 0,
        "source": "observed_skip",
        "parse_status": "skipped",
        "skip_reason": SkipReason.COST_BUDGET_EXHAUSTED,
        "review_tier": ReviewTier.DEEP,
        "llm_called": False,
        "counters": AnalyzeWorkerCounters(n_findings_emitted=1, n_findings_observed=1),
        "admitted_findings": (_observed_finding(),),
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
        "llm_called": False,
        "counters": AnalyzeWorkerCounters(n_findings_emitted=1, n_findings_served=1),
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
# Shape + skip coherence.
# ---------------------------------------------------------------------------


def test_skip_coherence_iff_contract_both_directions() -> None:
    with pytest.raises(ValidationError, match="iff"):
        _observed_skip(skip_reason=None)  # skipped, no reason
    with pytest.raises(ValidationError, match="iff"):
        _outcome(skip_reason=SkipReason.COST_BUDGET_EXHAUSTED)  # reason, not skipped


def test_skipped_iff_skip_source_and_llm_iff_parser() -> None:
    with pytest.raises(ValidationError, match="does not match source"):
        _outcome(source="observed_skip")  # clean status, skip source
    with pytest.raises(ValidationError, match="contradicts"):
        _observed_skip(llm_called=True)  # skip source, LLM call


def test_ride_out_shape_is_legal_findings_on_a_skip() -> None:
    """The module-arm ride-out: a budget skip carrying OBSERVED findings."""
    assert _observed_skip().admitted_findings


def test_skip_findings_must_be_observed_tier() -> None:
    """A JUDGED finding on a skip is a shape no sequential path produces."""
    with pytest.raises(ValidationError, match="OBSERVED"):
        _observed_skip(admitted_findings=(_finding(),))


def test_plain_skip_carries_nothing() -> None:
    with pytest.raises(ValidationError, match="plain_skip carries nothing"):
        _observed_skip(source="plain_skip")


def test_path_is_canonicalized_like_analysis_round() -> None:
    outcome = _outcome(path="./src/app.py")
    assert outcome.path == "src/app.py"
    assert worker_outcome_slot(outcome) == ("src/app.py", 0)


def test_json_round_trip_is_exact_including_decimal_cost_and_snapshot() -> None:
    outcome = _outcome(cost=Decimal("0.123456789"))
    restored = AnalyzeWorkerOutcome.model_validate_json(outcome.model_dump_json())
    assert restored.cost == Decimal("0.123456789")
    assert isinstance(restored.cost, Decimal)
    assert restored.admitted_findings == outcome.admitted_findings
    assert restored.semantic_snapshot == outcome.semantic_snapshot  # verified, not re-minted


# ---------------------------------------------------------------------------
# Source coherence: counters derived from the findings they describe.
# ---------------------------------------------------------------------------


def test_observed_counter_must_match_observed_tier_findings() -> None:
    """A JUDGED finding counted as observed breaks the accounting
    subtraction — the counter is DERIVED, not asserted."""
    with pytest.raises(ValidationError, match="OBSERVED-tier"):
        _outcome(
            counters=AnalyzeWorkerCounters(
                n_proposals_seen=0, n_findings_emitted=1, n_findings_observed=1
            ),
        )  # base finding is JUDGED


def test_response_rejection_is_exclusive() -> None:
    """A rejected response yields zero proposals, findings, and candidates
    (the parser's all-zero exclusive shape)."""
    ok = _outcome(
        counters=AnalyzeWorkerCounters(n_responses_rejected=1),
        admitted_findings=(),
    )
    assert ok.counters.n_responses_rejected == 1
    with pytest.raises(ValidationError, match="exclusive"):
        _outcome(
            counters=AnalyzeWorkerCounters(
                n_responses_rejected=1, n_proposals_seen=1, n_findings_emitted=1
            ),
        )


def test_trace_counter_tied_to_candidates_tuple() -> None:
    with pytest.raises(ValidationError, match="candidates carried"):
        _outcome(
            counters=AnalyzeWorkerCounters(
                n_proposals_seen=1, n_findings_emitted=1, n_trace_candidates_emitted=2
            ),
        )


def test_served_findings_cannot_coexist_with_an_llm_call() -> None:
    finding = _finding()
    with pytest.raises(ValidationError, match="cannot coexist"):
        _outcome(
            admitted_findings=(finding,),
            served_content_hashes=(finding.content_hash,),
            counters=AnalyzeWorkerCounters(
                n_proposals_seen=0, n_findings_emitted=1, n_findings_served=1
            ),
        )


def test_no_parser_means_no_proposal_or_trace_counters() -> None:
    with pytest.raises(ValidationError, match="require an LLM call"):
        _cache_serve(
            counters=AnalyzeWorkerCounters(
                n_proposals_seen=1, n_findings_emitted=1, n_findings_served=1
            ),
        )


def test_cache_serve_hashes_must_be_exactly_the_admitted_set() -> None:
    with pytest.raises(ValidationError, match="exactly the admitted"):
        _cache_serve(
            served_content_hashes=(),
            counters=AnalyzeWorkerCounters(n_findings_emitted=1, n_findings_served=0),
        )


# ---------------------------------------------------------------------------
# Served-hash canonical form (primitive checks fire before source rules).
# ---------------------------------------------------------------------------


def test_served_hashes_canonical_form_enforced() -> None:
    finding = _finding()
    h = finding.content_hash
    with pytest.raises(ValidationError, match="SHA-256 hex"):
        _cache_serve(served_content_hashes=("nope",))
    with pytest.raises(ValidationError, match="sorted and unique"):
        _cache_serve(
            served_content_hashes=(h, h),
            counters=AnalyzeWorkerCounters(n_findings_emitted=1, n_findings_served=2),
        )
    with pytest.raises(ValidationError, match="subset"):
        _cache_serve(served_content_hashes=("b" * 64,))
    with pytest.raises(ValidationError, match="n_findings_served"):
        _cache_serve(
            counters=AnalyzeWorkerCounters(n_findings_emitted=1, n_findings_served=0),
        )


# ---------------------------------------------------------------------------
# Cross-file attribution + accounting + spend.
# ---------------------------------------------------------------------------


def test_cross_file_attribution_is_unrepresentable() -> None:
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


def test_worker_accounting_equation_and_cardinality() -> None:
    with pytest.raises(ValidationError, match="findings admitted"):
        _outcome(counters=AnalyzeWorkerCounters(n_proposals_seen=2, n_findings_emitted=2))
    with pytest.raises(ValidationError, match="proposal accounting"):
        _outcome(counters=AnalyzeWorkerCounters(n_proposals_seen=2, n_findings_emitted=1))


def test_no_llm_call_means_zero_spend() -> None:
    with pytest.raises(ValidationError, match="zero provider"):
        _cache_serve(input_tokens=100)
    with pytest.raises(ValidationError, match="zero provider"):
        _observed_skip(cost=Decimal("0.01"))


# ---------------------------------------------------------------------------
# The #063 semantic snapshot.
# ---------------------------------------------------------------------------


def test_identical_retry_with_fresh_finding_uuids_snapshots_equal() -> None:
    first = _outcome(admitted_findings=(_finding("x"),))
    retry = _outcome(admitted_findings=(_finding("x"),))
    assert first.admitted_findings[0].finding_id != retry.admitted_findings[0].finding_id
    assert first.semantic_snapshot == retry.semantic_snapshot


def test_any_semantic_change_moves_the_snapshot() -> None:
    base = _outcome()
    assert (
        base.semantic_snapshot != _outcome(admitted_findings=(_finding("other"),)).semantic_snapshot
    )
    assert base.semantic_snapshot != _outcome(output_tokens=51).semantic_snapshot
    assert base.semantic_snapshot != _outcome(cost=Decimal("0.0124")).semantic_snapshot


def test_snapshot_is_immutable_at_birth_despite_nested_mutation() -> None:
    """THE F4 pin: nested findings are mutable by design; a post-insertion
    mutation must not change the stored snapshot, so the slot guard never
    falsely diverges a legitimate retry against mutated existing state."""
    original = _outcome(admitted_findings=(_finding("x"),))
    snapshot_at_birth = original.semantic_snapshot
    original.admitted_findings[0].title = "mutated-after-insertion"
    assert original.semantic_snapshot == snapshot_at_birth
    retry = _outcome(admitted_findings=(_finding("x"),))  # pre-mutation semantics
    reducer = _state_reducer()
    assert reducer([original], [retry]) == [original]  # replay no-op, no false divergence


def test_tampered_snapshot_is_verified_not_trusted() -> None:
    outcome = _outcome()
    payload = outcome.model_dump(mode="json")
    payload["semantic_snapshot"] = "f" * 64
    with pytest.raises(ValidationError, match="does not match"):
        AnalyzeWorkerOutcome.model_validate(payload)


def test_exclusion_paths_match_generated_fields_plus_self_reference() -> None:
    """Two-directional #063 pin: the exclusion list is exactly the model
    tree's default_factory-generated fields PLUS the digest's own storage
    field (self-referential — a digest cannot cover itself)."""
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
    assert generated | {"semantic_snapshot"} == set(WORKER_OUTCOME_EXCLUDE_PATHS)


# ---------------------------------------------------------------------------
# ReviewState wiring: the slot-guard reducer over stored snapshots.
# ---------------------------------------------------------------------------


def test_state_field_merges_identical_retries_and_rejects_divergence() -> None:
    reducer = _state_reducer()
    first = _outcome(admitted_findings=(_finding("x"),))
    retry = _outcome(admitted_findings=(_finding("x"),))  # fresh UUIDs, same semantics
    assert reducer([first], [retry]) == [first]
    diverged = _outcome(admitted_findings=(_finding("y"),))
    with pytest.raises(SlotDivergenceError):
        reducer([first], [diverged])
