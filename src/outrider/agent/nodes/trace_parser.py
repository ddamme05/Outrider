# See specs/2026-05-23-trace-node.md M6 + Q4.
"""Validate a model ranking as a permutation of the supplied trace candidates.

Normalize the outer JSON fence (`vendor-payloads-normalized-at-boundary`),
validate the envelope, then check count, duplicates and ID membership before
reordering the original candidate objects.
The trace node warns and retains input order on rejection. A dedicated
TraceRankingRejectedEvent remains deferred (FUP-076)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from outrider.llm.parsing import strip_outer_json_fence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.schemas.trace_candidate import TraceCandidate


# Stable rejection identifiers consumed by trace's warning/fallback path.
# Dedicated rejection-event attribution remains deferred (FUP-076).
TraceResponseRejectionReason = Literal[
    "raw_response_unparseable",
    "ranking_id_fabricated",
    "ranking_id_missing",
    "ranking_id_duplicated",
    "ranking_count_mismatch",
]


class TraceRankingResponseRaw(BaseModel):
    """Strict envelope for the configured trace model's ordered candidate IDs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ranked_candidate_ids: tuple[str, ...] = Field(
        description="Ordered candidate_ids in descending priority."
    )


@dataclass(frozen=True, slots=True)
class TraceRankingParsed:
    """Successful parse: candidates re-ordered per the LLM's ranking.

    `ordered_candidates` is the same set as the input list, just
    reshuffled to match `ranked_candidate_ids`. Length equality + id
    membership are verified at parse time, so consumers can iterate
    `ordered_candidates` without re-checking.
    """

    outcome: Literal["parsed"]
    ordered_candidates: tuple[TraceCandidate, ...]


@dataclass(frozen=True, slots=True)
class TraceRankingRejected:
    """Rejection: response did not satisfy the parser contract.

    `reason` is one of the `TraceResponseRejectionReason` literals.
    Consumers (trace node) WARN-log this reason and fall back to the
    input order. V1 does NOT emit a `TraceRankingRejectedEvent` —
    that audit surface is deferred per FUP-076.
    """

    outcome: Literal["rejected"]
    reason: TraceResponseRejectionReason


TraceRankingOutcome = TraceRankingParsed | TraceRankingRejected


def parse_trace_ranking(
    *,
    response_text: str,
    candidates: Sequence[TraceCandidate],
) -> TraceRankingOutcome:
    """Return the original candidates in ranking order, or a typed rejection.

    After JSON normalization and validation, check count before duplicates and
    membership. This order determines the reported reason for malformed rankings."""
    try:
        raw = TraceRankingResponseRaw.model_validate_json(strip_outer_json_fence(response_text))
    except ValidationError:
        return TraceRankingRejected(outcome="rejected", reason="raw_response_unparseable")

    ranked_ids = raw.ranked_candidate_ids
    candidate_id_set = {candidate.candidate_id for candidate in candidates}
    ranked_id_set = set(ranked_ids)

    # Count check FIRST — catches the obvious shape errors with one
    # comparison before the per-id checks.
    if len(ranked_ids) != len(candidates):
        return TraceRankingRejected(outcome="rejected", reason="ranking_count_mismatch")

    # Duplicate check — distinct sets vs ranked count.
    if len(ranked_id_set) != len(ranked_ids):
        return TraceRankingRejected(outcome="rejected", reason="ranking_id_duplicated")

    # Fabrication check — the ranked set must be a subset of the
    # supplied candidate ids. After the count + dup checks pass, this
    # implies set equality (same cardinality + ranked ⊆ supplied =
    # equality), so the missing-id branch is unreachable when both
    # prior checks pass; it stays as a defensive check.
    if not ranked_id_set.issubset(candidate_id_set):
        return TraceRankingRejected(outcome="rejected", reason="ranking_id_fabricated")
    if not candidate_id_set.issubset(ranked_id_set):
        return TraceRankingRejected(outcome="rejected", reason="ranking_id_missing")

    # Reorder per the LLM's ranking. `index_by_id` is the deterministic
    # lookup (input candidates are guaranteed unique by `candidate_id`
    # per the schema validator).
    index_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    ordered_candidates = tuple(index_by_id[candidate_id] for candidate_id in ranked_ids)
    return TraceRankingParsed(outcome="parsed", ordered_candidates=ordered_candidates)


__all__ = [
    "TraceRankingOutcome",
    "TraceRankingParsed",
    "TraceRankingRejected",
    "TraceRankingResponseRaw",
    "TraceResponseRejectionReason",
    "parse_trace_ranking",
]
