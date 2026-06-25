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

from typing import Annotated, Final, Self

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
from outrider.schemas.hitl import HITL_MAX_GATED_FINDINGS
from outrider.schemas.review_finding import (
    ReviewFinding,  # noqa: TC001 — Pydantic field type, needs runtime import
)

# Finding ceilings (FUP-180). A round aggregates admitted findings across ALL
# files in one analyze pass, so these are cross-file aggregate bounds — NOT the
# per-LLM-call ceiling (`AnalyzeResponseRaw.findings`, 50).
#
# `MAX_FINDINGS_PER_ROUND` is the SOFT cap: the analyze node truncates NON-gated
# findings to this bound before emission via `finding_cap.cap_findings_by_severity`.
# Gated (CRITICAL/HIGH) findings are NEVER dropped to fit it (FUP-180 review design
# call — silently dropping a gated finding would weaken hitl-gates-high-severity),
# so a round whose gated findings alone exceed the soft cap legitimately holds more
# than `MAX_FINDINGS_PER_ROUND`.
#
# `MAX_FINDINGS_HARD_CAP` is the runaway ceiling — ALIGNED to `HITL_MAX_GATED_FINDINGS`
# (the most gated findings the HITL request can carry). Gated findings are never
# dropped to fit it; instead `finding_cap.cap_findings_by_severity` raises
# `FindingCapOverflowError` (a clean crash, before any side effect) when gated findings
# exceed it — a review with more gated findings than HITL can hold fails loud rather
# than dropping a CRITICAL below the approval gate. Alignment matters: a hard cap LARGER
# than the HITL bound would keep gated findings the HITL request can't carry, crashing
# at HITL-partition construction (a strand) instead. The schema `max_length` below uses
# this so the kept set always satisfies it. See specs/2026-06-24-finding-cap-pre-side-effect.md.
MAX_FINDINGS_PER_ROUND: Final[int] = 200
MAX_FINDINGS_HARD_CAP: Final[int] = HITL_MAX_GATED_FINDINGS


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
    # max_length is the HARD runaway ceiling (not the soft cap) — gated findings
    # are never dropped to fit the soft `MAX_FINDINGS_PER_ROUND`, so the round can
    # legitimately exceed it; only `MAX_FINDINGS_HARD_CAP` bounds it. The analyze
    # node enforces the cap BEFORE emission, so this is the defense-in-depth
    # backstop that bounds checkpoint + audit-row payload size. See the constants
    # above.
    findings: tuple[ReviewFinding, ...] = Field(max_length=MAX_FINDINGS_HARD_CAP)
    files_examined: tuple[Annotated[str, Field(max_length=1024)], ...]
    files_skipped: tuple[Annotated[str, Field(max_length=1024)], ...]
    started_at: AwareDatetime
    ended_at: AwareDatetime

    @field_validator("files_examined", "files_skipped")
    @classmethod
    def _enforce_canonical_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        """Reject paths that aren't `coordinates.validate_diff_path` output.

        `round_id` is content-
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
        """`ended_at >= started_at` A round whose end precedes its start is
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
    def _enforce_files_examined_skipped_set_semantics(self) -> Self:
        """`files_examined` and `files_skipped` are set-semantic: each tuple
        carries distinct paths, and the two tuples are disjoint. Duplicate
        paths inside one tuple let logically identical rounds hash to
        different `round_id` values (one producer emits `("a.py",)`,
        another emits `("a.py", "a.py")`), defeating the dedup contract.
        Cross-tuple overlap is a separate semantic error: a file is either
        examined or skipped per pass, never both.
        """
        if len(self.files_examined) != len(set(self.files_examined)):
            raise ValueError(
                f"AnalysisRound.files_examined contains duplicates: {sorted(self.files_examined)!r}"
            )
        if len(self.files_skipped) != len(set(self.files_skipped)):
            raise ValueError(
                f"AnalysisRound.files_skipped contains duplicates: {sorted(self.files_skipped)!r}"
            )
        overlap = set(self.files_examined) & set(self.files_skipped)
        if overlap:
            raise ValueError(
                f"AnalysisRound: files cannot appear in both "
                f"files_examined and files_skipped; overlap: {sorted(overlap)!r}"
            )
        return self

    @model_validator(mode="after")
    def _enforce_findings_unique(self) -> Self:
        """`findings` is set-semantic by `finding_id`. Two ReviewFindings
        with the same `finding_id` is a producer bug — and the same
        `content_hash` collision changes the `round_id` digest (which
        sorts content_hashes), breaking dedup-by-round_id.
        """
        finding_ids = [f.finding_id for f in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError(
                f"AnalysisRound.findings contains duplicate finding_ids: "
                f"{sorted(str(fid) for fid in finding_ids)!r}"
            )
        content_hashes = [f.content_hash for f in self.findings]
        if len(content_hashes) != len(set(content_hashes)):
            raise ValueError(
                f"AnalysisRound.findings contains duplicate content_hashes: "
                f"{sorted(content_hashes)!r}"
            )
        return self

    @model_validator(mode="after")
    def _enforce_findings_proposal_hash_unique(self) -> Self:
        """`findings` is set-semantic by `proposal_hash` per
        `DECISIONS.md#025` point 4 — admitted findings within a review
        carry unique proposal_hashes. Two findings sharing a proposal_hash
        is a producer bug: `compute_proposal_hash` is content-derived
        from the raw proposal payload (per `DECISIONS.md#022`), so a
        collision means analyze emitted two findings from THE SAME logical
        proposal — which by definition is the same finding, not two.

        Load-bearing for trace's join contract per #025 point 5: trace
        builds a `{f.proposal_hash → f.finding_id}` lookup over
        admitted findings and raises `TraceJoinIntegrityError` on
        duplicates. Catching the collision here at construction time
        prevents the upstream producer bug from reaching trace
        (where its rejection would surface as a loud-fail at the
        join layer rather than at the source). Cross-round uniqueness
        is pinned by an analyze-side test, not by this validator
        (which sees only this round's findings).
        """
        proposal_hashes = [f.proposal_hash for f in self.findings]
        if len(proposal_hashes) != len(set(proposal_hashes)):
            raise ValueError(
                f"AnalysisRound.findings contains duplicate proposal_hashes: "
                f"{sorted(proposal_hashes)!r}. Per DECISIONS.md#025 point 4, "
                f"admitted findings within a round have unique proposal_hashes; "
                f"a collision means analyze emitted two findings from the same "
                f"raw proposal."
            )
        return self

    @model_validator(mode="after")
    def _enforce_round_id_matches_payload(self) -> Self:
        """Assert `round_id == compute_round_id(...)` over this round's payload.

        without this validator,
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
