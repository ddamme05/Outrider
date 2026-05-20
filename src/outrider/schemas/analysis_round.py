# See specs/2026-05-19-analyze-foundation.md §1.
"""`AnalysisRound` per-pass results record.

Synthesize will consume the accumulated `list[AnalysisRound]` across all
analyze ⇄ trace iterations to build the final report. This module
defines the schema only; producers (the analyze node) and consumers
(synthesize) live in the sister analyze-implementation spec.

`round_id` is a SHA-256 hex digest used as the LangGraph reducer's merge
key (see `outrider.agent.reducers.append_with_dedup_by` in §2). The
hash is content-derived via `outrider.policy.canonical.compute_identity_hash`
so replay re-application is idempotent — same payload produces the same
id across processes, and duplicate rounds collapse on merge.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from outrider.policy.canonical import SHA256_HEX_PATTERN
from outrider.schemas.review_finding import (
    ReviewFinding,  # noqa: TC001 — Pydantic field type, needs runtime import
)


class AnalysisRound(BaseModel):
    """One analyze-pass record per analyze ⇄ trace iteration.

    Frozen + `extra="forbid"` matches the cross-boundary discipline
    in `docs/conventions.md`. `findings` is the admitted set from
    this pass; rejected proposals are recorded on
    `FindingProposalRejectedEvent` in the audit stream, not here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # SHA-256 hex digest, stable merge key for the dedup-by reducer.
    # Derived from the round's content (file set + finding hashes) via
    # `outrider.policy.canonical.compute_identity_hash`; same payload
    # produces same id, so replay re-application collapses duplicates.
    round_id: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
    pass_index: int = Field(ge=0)
    findings: tuple[ReviewFinding, ...]
    files_examined: tuple[Annotated[str, Field(max_length=1024)], ...]
    files_skipped: tuple[Annotated[str, Field(max_length=1024)], ...]
    started_at: AwareDatetime
    ended_at: AwareDatetime
