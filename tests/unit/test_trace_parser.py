# See specs/2026-05-23-trace-node.md M6.
"""Trace LLM-response parser — five rejection paths + happy path.

Covers all rejection-reason literals on `TraceRankingRejectedReason`
plus the happy-path reorder. Vendor-wire-format quirk (JSON fence) is
exercised via `strip_outer_json_fence` round-trip.
"""

from __future__ import annotations

from outrider.agent.nodes.trace_parser import (
    TraceRankingParsed,
    TraceRankingRejected,
    parse_trace_ranking,
)
from outrider.policy.canonical import compute_candidate_id
from outrider.schemas import TraceCandidate


def _build_candidate(
    *, source_proposal_hash: str = "a" * 64, import_string: str = "pkg.mod"
) -> TraceCandidate:
    reason = "x"
    return TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=source_proposal_hash,
            import_string=import_string,
            reason=reason,
        ),
        source_proposal_hash=source_proposal_hash,
        reason=reason,
        import_string=import_string,
    )


def test_happy_path_reorders_candidates_per_ranking() -> None:
    """Two candidates; response ranks them in reverse order; parser
    returns the candidates in that ranked order."""
    c1 = _build_candidate(import_string="pkg.first")
    c2 = _build_candidate(import_string="pkg.second")
    response = f'{{"ranked_candidate_ids": ["{c2.candidate_id}", "{c1.candidate_id}"]}}'

    result = parse_trace_ranking(response_text=response, candidates=(c1, c2))

    assert isinstance(result, TraceRankingParsed)
    assert result.ordered_candidates == (c2, c1)


def test_unparseable_response_rejected() -> None:
    """Non-JSON text → `raw_response_unparseable` rejection."""
    result = parse_trace_ranking(
        response_text="not json at all",
        candidates=(_build_candidate(),),
    )
    assert isinstance(result, TraceRankingRejected)
    assert result.reason == "raw_response_unparseable"


def test_count_mismatch_rejected() -> None:
    """LLM returns fewer ids than supplied → `ranking_count_mismatch`."""
    c1 = _build_candidate(import_string="pkg.alpha")
    c2 = _build_candidate(import_string="pkg.beta")
    response = f'{{"ranked_candidate_ids": ["{c1.candidate_id}"]}}'

    result = parse_trace_ranking(response_text=response, candidates=(c1, c2))
    assert isinstance(result, TraceRankingRejected)
    assert result.reason == "ranking_count_mismatch"


def test_duplicate_id_rejected() -> None:
    """LLM returns the same id twice → `ranking_id_duplicated`."""
    c1 = _build_candidate(import_string="pkg.alpha")
    c2 = _build_candidate(import_string="pkg.beta")
    response = f'{{"ranked_candidate_ids": ["{c1.candidate_id}", "{c1.candidate_id}"]}}'

    result = parse_trace_ranking(response_text=response, candidates=(c1, c2))
    assert isinstance(result, TraceRankingRejected)
    assert result.reason == "ranking_id_duplicated"


def test_fabricated_id_rejected() -> None:
    """LLM returns an id not in the supplied set → `ranking_id_fabricated`."""
    c1 = _build_candidate(import_string="pkg.alpha")
    c2 = _build_candidate(import_string="pkg.beta")
    fabricated = "f" * 64
    response = f'{{"ranked_candidate_ids": ["{c1.candidate_id}", "{fabricated}"]}}'

    result = parse_trace_ranking(response_text=response, candidates=(c1, c2))
    assert isinstance(result, TraceRankingRejected)
    assert result.reason == "ranking_id_fabricated"


def test_json_fence_wrapped_response_unwraps_cleanly() -> None:
    """Anthropic occasionally wraps structured-output JSON in ```json
    fences despite prompt instructions; `strip_outer_json_fence`
    defends. Vendor-wire-format-quirk regression."""
    c1 = _build_candidate(import_string="pkg.alpha")
    inner = f'{{"ranked_candidate_ids": ["{c1.candidate_id}"]}}'
    fenced = f"```json\n{inner}\n```"

    result = parse_trace_ranking(response_text=fenced, candidates=(c1,))
    assert isinstance(result, TraceRankingParsed)
    assert result.ordered_candidates == (c1,)
