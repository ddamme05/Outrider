# Cross-type subsumption proof-retention record per DECISIONS.md#055.
"""`ObservedSubsumedMatch` — the dropped-OBSERVED structural proof record.

Lives in `schemas/` per the placement rule (`DECISIONS.md#019`): the model
is produced by analyze, carried on `AnalyzeWorkerOutcome` (state) AND
`AnalyzeCompletedEvent` (audit) — cross-boundary with no single owning
subsystem. It previously lived in `audit/events.py`; the V1.5 worker
outcome created a true import cycle (audit.events ↔ schemas.analyze_worker
via the schemas package init), and this leaf is importable by both sides
cycle-free. `audit/events.py` re-exports it, so existing importers are
unchanged.
"""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outrider.coordinates import validate_diff_path
from outrider.policy import FindingType
from outrider.policy.canonical import SHA256_HEX_PATTERN as _SHA256_HEX_PATTERN

__all__ = ["ObservedSubsumedMatch"]


class ObservedSubsumedMatch(BaseModel):
    """An OBSERVED finding dropped by cross-type subsumption (DECISIONS.md#055):
    a same-span JUDGED subsumer of a more-specific `finding_type` absorbed it, so
    the broader OBSERVED finding is suppressed from the published set. This record
    RETAINS its replay-verifiable `query_match_id` in the audit stream — the
    existing `ObservedSkipShadowEvent` carries only `skip_safe` matches, and the
    subsumed query is `signal_only`, so without this record the structural proof
    would vanish.

    `file_path` is REQUIRED (unlike `ObservedSkipCoveringMatch`, which rides a
    per-file event): `AnalyzeCompletedEvent` is per-pass and aggregates over all
    files, so `query_match_id` + line span alone is ambiguous across files. The
    two content hashes cross-reference the dropped finding and the surviving
    subsumer's `FindingEvent`. Frozen + extra=forbid.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: Annotated[str, Field(max_length=1024)]
    query_match_id: Annotated[str, Field(max_length=200, min_length=1)]
    finding_type: FindingType
    subsumed_by_finding_type: FindingType
    line_start: Annotated[int, Field(ge=1)]
    line_end: Annotated[int, Field(ge=1)]
    dropped_content_hash: Annotated[str, Field(pattern=_SHA256_HEX_PATTERN)]
    subsumer_content_hash: Annotated[str, Field(pattern=_SHA256_HEX_PATTERN)]

    @field_validator("file_path")
    @classmethod
    def _validate_file_path(cls, path: str) -> str:
        # Re-run validate_diff_path so this record enforces the path boundary
        # even when reconstructed from a cached payload (paths-validated-before-use,
        # [security-critical]) — same shadow as FindingEvent / CacheServeEvent.
        return validate_diff_path(path)

    @model_validator(mode="after")
    def _enforce_line_order(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self

    @model_validator(mode="after")
    def _verify_content_hashes(self) -> Self:
        # Both hashes are derivable from this record's own (canonical file_path,
        # line span, finding_type) — recompute and verify so a spoofed/mismatched
        # hash (e.g. a tampered cache payload) fails construction, the same
        # integrity shape as ReviewFinding/FindingEvent._verify_content_hash.
        # Function-scope import: the hash helper lives in audit/events.py,
        # which imports schemas leaves at module level — a module-level
        # import here would re-create the cycle this leaf exists to sever.
        # By validation time both modules are fully initialized.
        from outrider.audit.events import compute_finding_content_hash

        # file_path is already canonical (the field validator ran first);
        # compute_finding_content_hash canonicalizes idempotently.
        expected_dropped = compute_finding_content_hash(
            self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            finding_type=self.finding_type,
        )
        if self.dropped_content_hash != expected_dropped:
            raise ValueError(
                f"dropped_content_hash mismatch: expected {expected_dropped}, "
                f"got {self.dropped_content_hash}"
            )
        expected_subsumer = compute_finding_content_hash(
            self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            finding_type=self.subsumed_by_finding_type,
        )
        if self.subsumer_content_hash != expected_subsumer:
            raise ValueError(
                f"subsumer_content_hash mismatch: expected {expected_subsumer}, "
                f"got {self.subsumer_content_hash}"
            )
        return self
