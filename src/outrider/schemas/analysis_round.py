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

from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from outrider.coordinates import validate_diff_path
from outrider.policy.canonical import SHA256_HEX_PATTERN, compute_round_id
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

    @field_validator("files_examined", "files_skipped")
    @classmethod
    def _enforce_canonical_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        """Reject paths that aren't `coordinates.validate_diff_path` output.

        Foundation-wide data-integrity audit F1: `round_id` is content-
        derived from this round's payload (per spec §1). If callers pass
        non-canonical paths (`./src/a.py` vs `src/a.py`, trailing slash,
        unvalidated metachars), the resulting `round_id` is non-
        deterministic across producers — replay re-application collides
        under a DIFFERENT key, defeating idempotency.

        Pushing the rule down to the schema means the parser, replay
        reconstructor, test fixture, and any future producer all get
        the same guarantee. `validate_diff_path` raises `CoordinateError`
        on invalid input — Pydantic surfaces that as `ValidationError`.
        """
        return tuple(validate_diff_path(p) for p in paths)

    @model_validator(mode="after")
    def _enforce_time_ordering(self) -> Self:
        """`ended_at >= started_at` per coherence-guard fold (Copilot/CodeRabbit
        convergent on PR review). A round whose end precedes its start is
        an impossible timing that would otherwise leak into replay /
        reporting and confuse latency aggregates.
        """
        if self.ended_at < self.started_at:
            raise ValueError(
                f"AnalysisRound.ended_at ({self.ended_at!r}) must be >= "
                f"started_at ({self.started_at!r})."
            )
        return self

    @model_validator(mode="after")
    def _enforce_round_id_matches_payload(self) -> Self:
        """Assert `round_id == compute_round_id(...)` over this round's payload.

        Post-foundation audit (high confidence): without this validator,
        `round_id` was only pattern-checked — a caller could supply ANY
        64-char hex string and Pydantic accepted it. The dedup-by-key
        reducer would then admit two logically-equivalent rounds under
        different `round_id`s and double-accumulate state on replay.

        The validator re-derives the canonical id from this round's
        actual payload and rejects mismatch. `compute_round_id` sorts
        inputs internally, so caller-side enumeration order doesn't
        matter — the validator is robust to the same loose-order risk
        the §1 spec named.

        Replay rehydrates `AnalysisRound` via `model_validate`, so this
        validator fires there too — a future change to the round-id
        recipe would surface as a loud replay failure rather than a
        silent dedup-key drift.
        """
        expected = compute_round_id(
            pass_index=self.pass_index,
            files_examined=self.files_examined,
            files_skipped=self.files_skipped,
            finding_content_hashes=tuple(f.content_hash for f in self.findings),
        )
        if self.round_id != expected:
            raise ValueError(
                f"AnalysisRound.round_id={self.round_id!r} does not match the "
                f"canonical id computed from this round's payload "
                f"({expected!r}). Construct via `compute_round_id(...)` rather "
                f"than passing an arbitrary hex string. If the recipe genuinely "
                f"changed, the change is a DECISIONS-level event — old audit "
                f"rows would otherwise fail replay validation here."
            )
        return self
