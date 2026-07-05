# Worker-outcome builder pins per specs/2026-07-05-parallel-analyze.md (3b-2b).
"""Builder pins — the two 3b-2 acceptance gates live here.

Origin truth: `producer_observed_hashes` derives from the #054 merge's
object placement, never from evidence tier — a model-cited OBSERVED
proposal stays out, an evicted-slot producer survivor is in, a producer
finding dropped against a non-JUDGED incumbent is out. Non-aliasing:
builders clone findings INTO outcomes; no live object crosses.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from uuid import uuid4

from outrider.agent.nodes.analyze_worker_build import (
    worker_outcome_from_observed_skip,
    worker_outcome_from_parser,
    worker_outcome_from_plain_skip,
    worker_outcome_from_serve,
)
from outrider.ast_facts.models import SkipReason
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import FindingType, lookup_severity
from outrider.policy.versions import ACTIVE_POLICY_VERSION
from outrider.queries.registry import OBSERVED_QUERIES
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.triage_result import ReviewTier

_REVIEW_ID = uuid4()


def _finding(
    path: str = "src/app.py",
    *,
    line: int = 3,
    tier: EvidenceTier = EvidenceTier.JUDGED,
    query_match_id: str = "python.sql_injection_string_concat",
    finding_type: FindingType | None = None,
) -> ReviewFinding:
    """Production-possible OBSERVED shapes only. Producer fixtures default
    to a real Python SECURITY id and take their finding_type FROM THE
    REGISTRY BINDING (the producer maps query → type; a mismatched pair is
    an impossible raw producer object). The model-cited case must use a
    STRUCTURAL id — the model-citable set is structural-only, and empty
    for JS/TS entirely (the dispatch-arc security anchor); a structural
    citation carries whatever type the model proposed, so the default
    type stands there."""
    if finding_type is None:
        finding_type = FindingType.HARDCODED_SECRET
        if tier is EvidenceTier.OBSERVED and query_match_id in OBSERVED_QUERIES:
            finding_type = OBSERVED_QUERIES[query_match_id].finding_type
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
        query_match_id=(query_match_id if tier is EvidenceTier.OBSERVED else None),
        trace_path=(("caller",) if tier is EvidenceTier.INFERRED else None),
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            path, line_start=line, line_end=line, finding_type=finding_type
        ),
        proposal_hash=hashlib.sha256(f"{path}:{line}:{tier}".encode()).hexdigest(),
    )


def _parser_kwargs() -> dict[str, object]:
    return {
        "path": "src/app.py",
        "pass_index": 0,
        "review_tier": ReviewTier.DEEP,
        "parse_status": "clean",
        "trace_candidates": (),
        "subsumed_matches": (),
        "n_proposals_rejected": 0,
        "n_responses_rejected": 0,
        "n_proposals_superseded_by_observed": 0,
        "n_trace_candidates_dropped_malformed": 0,
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost": Decimal("0.01"),
        "estimated_tokens": 500,
    }


# ---------------------------------------------------------------------------
# Origin truth (the acceptance gate).
# ---------------------------------------------------------------------------


def test_origin_derives_from_merge_object_placement_not_tier() -> None:
    """Three producer fates, one derivation: an appended producer survivor
    is listed; a model-cited OBSERVED proposal (in admitted, not a producer
    object) is NOT; a producer finding dropped against an incumbent is NOT."""
    producer_survivor = _finding(line=1, tier=EvidenceTier.OBSERVED)
    # A REAL model-citable id: structural, Python (parser admission checks
    # membership in the structural citable set — a JS or security id here
    # would model an impossible input and make this pin vacuous).
    model_cited = _finding(
        line=2, tier=EvidenceTier.OBSERVED, query_match_id="python.function_definition"
    )
    dropped_producer = _finding(line=3, tier=EvidenceTier.OBSERVED)
    # A REAL #054 collision at the pass where the producer merge RUNS:
    # pass 0 rejects INFERRED (no trace context) and pass 1 never runs
    # the producer, so the reachable non-JUDGED incumbent is a
    # MODEL-CITED structural OBSERVED — content_hash excludes the query
    # id and the model chooses finding_type freely, so a structural
    # citation with the producer's type and line collides. The merge
    # keeps it (already carries proof) and drops the producer duplicate.
    incumbent = _finding(
        line=3,
        tier=EvidenceTier.OBSERVED,
        query_match_id="python.function_definition",
        finding_type=FindingType.SQL_INJECTION,
    )
    assert incumbent.content_hash == dropped_producer.content_hash  # collision is real
    admitted = (model_cited, incumbent, producer_survivor)  # post-#054 merge
    outcome = worker_outcome_from_parser(
        admitted_findings=admitted,
        producer_findings=(producer_survivor, dropped_producer),
        n_proposals_seen=2,  # the cited + the incumbent proposals
        **_parser_kwargs(),
    )
    assert outcome.producer_observed_hashes == (producer_survivor.content_hash,)
    # The cited finding is admitted but NOT producer-listed:
    admitted_hashes = {f.content_hash for f in outcome.admitted_findings}
    assert model_cited.content_hash in admitted_hashes
    assert model_cited.content_hash not in outcome.producer_observed_hashes


def test_evicting_producer_survivor_is_listed() -> None:
    """An eviction places the producer OBJECT into the admitted slot — the
    identity derivation sees it."""
    producer = _finding(line=5, tier=EvidenceTier.OBSERVED)
    admitted = (producer,)  # evicted the JUDGED collision in place
    kwargs = _parser_kwargs()
    kwargs["n_proposals_superseded_by_observed"] = 1
    outcome = worker_outcome_from_parser(
        admitted_findings=admitted,
        producer_findings=(producer,),
        n_proposals_seen=1,
        **kwargs,
    )
    assert outcome.producer_observed_hashes == (producer.content_hash,)


# ---------------------------------------------------------------------------
# Non-aliasing (the acceptance gate).
# ---------------------------------------------------------------------------


def test_builders_clone_findings_into_outcomes() -> None:
    finding = _finding()
    outcome = worker_outcome_from_parser(
        admitted_findings=(finding,),
        producer_findings=(),
        n_proposals_seen=1,
        **_parser_kwargs(),
    )
    assert outcome.admitted_findings[0] is not finding
    assert outcome.admitted_findings[0].content_hash == finding.content_hash
    served = worker_outcome_from_serve(
        path="src/app.py",
        pass_index=0,
        review_tier=ReviewTier.DEEP,
        served_findings=(finding,),
        trace_candidates=(),
        subsumed_matches=(),
        estimated_tokens=0,
    )
    assert served.admitted_findings[0] is not finding


# ---------------------------------------------------------------------------
# Per-source construction (through the model validators).
# ---------------------------------------------------------------------------


def test_serve_builder_produces_a_valid_cache_serve_outcome() -> None:
    finding = _finding()
    outcome = worker_outcome_from_serve(
        path="src/app.py",
        pass_index=1,
        review_tier=ReviewTier.STANDARD,
        served_findings=(finding,),
        trace_candidates=(),
        subsumed_matches=(),
        estimated_tokens=250,
    )
    assert outcome.source == "cache_serve"
    assert outcome.served_content_hashes == (finding.content_hash,)
    assert outcome.cost == Decimal("0")


def test_observed_skip_builder_lists_everything_as_producer() -> None:
    finding = _finding(tier=EvidenceTier.OBSERVED)
    outcome = worker_outcome_from_observed_skip(
        path="src/app.py",
        pass_index=0,
        review_tier=ReviewTier.DEEP,
        skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
        producer_findings=(finding,),
    )
    assert outcome.source == "observed_skip"
    assert outcome.producer_observed_hashes == (finding.content_hash,)
    assert outcome.admitted_findings[0] is not finding  # cloned


def test_plain_skip_builder() -> None:
    outcome = worker_outcome_from_plain_skip(
        path="src/app.py",
        pass_index=0,
        review_tier=ReviewTier.SKIM,
        skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS,
    )
    assert outcome.source == "plain_skip"
    assert outcome.admitted_findings == ()
