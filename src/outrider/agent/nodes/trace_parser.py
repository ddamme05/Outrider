# See specs/2026-05-23-trace-node.md M6 + Q4.
"""Trace LLM-response parser: validate + rank candidates per the response.

The trace node makes one Haiku call (`prompts/trace.py`) that returns
a JSON object with `ranked_candidate_ids: list[str]`. This parser:

  1. Strips the outer JSON code fence if present
     (`llm.parsing.strip_outer_json_fence` defends the
     vendor-wire-format quirk per `vendor-payloads-normalized-at-boundary`).
  2. Validates the response shape via Pydantic.
  3. Verifies the response's ranking includes every supplied
     candidate_id exactly once (no fabrication, no omission, no dup).
  4. Returns the input candidates re-ordered to match the LLM's
     ranking.

Failure surfaces (returned via `TraceRankingOutcome` discriminated
union rather than raised, mirroring the analyze-parser's discipline):

  - `parsed` — happy path; ranked candidates.
  - `rejected` — well-defined rejection reason (parse failure,
    fabricated id, missing id, duplicate id, count mismatch).

The trace node consumes `parsed` directly; on `rejected` it falls back
to the input order (deterministic) and audits the rejection — exactly
the shape analyze uses for `AnalyzeResponseRejectedEvent`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from outrider.llm.parsing import strip_outer_json_fence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.schemas.trace_candidate import TraceCandidate


# Rejection reasons span the LLM-response parser surface for trace.
# Each value is a stable identifier consumed by audit emission; do not
# rename without coordinating with `audit/events.py` rejection-reason
# enumerations.
TraceResponseRejectionReason = Literal[
    "raw_response_unparseable",
    "ranking_id_fabricated",
    "ranking_id_missing",
    "ranking_id_duplicated",
    "ranking_count_mismatch",
]


class TraceRankingResponseRaw(BaseModel):
    """Pydantic schema for the raw Haiku ranking response.

    Single field: `ranked_candidate_ids` — ordered list of strings,
    each MUST appear in the supplied candidate list. `extra="forbid"`
    catches schema drift (e.g., LLM adding a `reasoning` field that
    would otherwise silently land in an audit row).
    """

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
    Consumers (trace node) emit an audit row with this reason and
    fall back to the input order.
    """

    outcome: Literal["rejected"]
    reason: TraceResponseRejectionReason


TraceRankingOutcome = TraceRankingParsed | TraceRankingRejected


def parse_trace_ranking(
    *,
    response_text: str,
    candidates: Sequence[TraceCandidate],
) -> TraceRankingOutcome:
    """Validate the Haiku response + reorder candidates per its ranking.

    Step order (failure-path-significant):
      1. Strip outer JSON fence (vendor wire-format quirk).
      2. Pydantic-validate the JSON shape.
      3. Verify ranking is exactly the input candidate_id set (no
         fabrication, no omission, no duplicate).
      4. Reorder the input `candidates` list per the ranking.

    Empty input (`candidates=()`) is a producer bug — trace shouldn't
    invoke the parser without candidates. Caller asserts this before
    calling; the parser additionally short-circuits to a
    `count_mismatch` rejection if the LLM returns a non-empty ranking
    for an empty candidate set.
    """
    try:
        raw = TraceRankingResponseRaw.model_validate_json(strip_outer_json_fence(response_text))
    except ValidationError:
        return TraceRankingRejected(outcome="rejected", reason="raw_response_unparseable")

    ranked_ids = raw.ranked_candidate_ids
    candidate_id_set = {c.candidate_id for c in candidates}
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
    index_by_id = {c.candidate_id: c for c in candidates}
    ordered = tuple(index_by_id[cid] for cid in ranked_ids)
    return TraceRankingParsed(outcome="parsed", ordered_candidates=ordered)


__all__ = [
    "TraceRankingOutcome",
    "TraceRankingParsed",
    "TraceRankingRejected",
    "TraceRankingResponseRaw",
    "TraceResponseRejectionReason",
    "parse_trace_ranking",
]
