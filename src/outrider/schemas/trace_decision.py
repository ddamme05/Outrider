# See specs/2026-05-23-trace-node.md (Q3 / Q4) and DECISIONS.md#017 × #024.
"""`TraceDecision` — state-layer mirror of `TraceDecisionEvent`.

The trace node produces one `TraceDecision` per source finding it considers
per `DECISIONS.md#017` (Accepted 2026-04-29; amended same-day; amended again
2026-05-24 by `#024` for the field-shape rename). State-layer twin of the
audit-event `TraceDecisionEvent` defined in `audit/events.py`; same fields
and same cross-field validators so a producer cannot construct a
TraceDecision that would fail validation when lifted to the audit event.

Reducer key on `ReviewState.trace_decisions` is `source_finding_id` alone
(per #017 amended point 1 — explicitly rejects `(source_finding_id, target_file)`
because that key collapses unresolved/ambiguous decisions on
`(source_finding_id, None)`). The dedup-by-`source_finding_id` reducer
makes replay idempotent: the same trace decision applied twice (webhook
redelivery, checkpoint replay, retry) is a no-op.

Field shape per #024 amendment to #017:

- `proposed_import_strings: tuple[str, ...]` — the LLM-proposed dotted
  Python import strings (any cardinality).
- `resolved_candidate_paths: tuple[str, ...]` — the resolver outputs from
  `coordinates.resolve_candidate_paths` (any cardinality). Each element
  passes through `coordinates.validate_diff_path` at the schema boundary
  per #024 point 6 (audit-shadow discipline).
- `target_file: str | None` — selected candidate when `resolution_status
  == "resolved"`, None otherwise. When non-None, passes through
  `validate_diff_path`.

Three cross-field validator rules per #024 point 5:

- `resolved` → `len(resolved_candidate_paths) == 1` AND
  `target_file == resolved_candidate_paths[0]`
- `unresolved` → `len(resolved_candidate_paths) == 0` AND `target_file is None`
- `ambiguous` → `len(resolved_candidate_paths) > 1` AND `target_file is None`

Uniqueness validator split into two — one per tuple — per #024 amendment.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self
from uuid import UUID  # noqa: TC003 — Pydantic field type, needs runtime import

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outrider.coordinates import validate_diff_path


class TraceDecision(BaseModel):
    """One aggregate trace decision per source_finding_id.

    State-layer mirror of `TraceDecisionEvent` (`audit/events.py`).
    Frozen + `extra="forbid"` per the cross-boundary schema discipline
    in `docs/conventions.md`. Validator-set identical to the audit event
    so a producer's construction-time guarantees survive the lift.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_finding_id: UUID
    target_file: str | None
    reason: Annotated[str, Field(max_length=500)]
    resolution_status: Literal["resolved", "unresolved", "ambiguous"]
    proposed_import_strings: tuple[str, ...]
    resolved_candidate_paths: tuple[str, ...]
    trace_path: tuple[str, ...] | None = None

    @field_validator("target_file")
    @classmethod
    def _enforce_canonical_target_file(cls, value: str | None) -> str | None:
        """When non-None, run `validate_diff_path` so the schema layer
        enforces the canonical repo-relative POSIX shape per #024 point 6.
        Mirror of `TraceDecisionEvent._enforce_canonical_target_file`."""
        if value is None:
            return None
        return validate_diff_path(value)

    @field_validator("resolved_candidate_paths")
    @classmethod
    def _enforce_canonical_resolved_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        """Per-element `validate_diff_path` per #024 point 6. Load-bearing
        for the ambiguous branch where `target_file is None` but the
        tuple still carries multiple resolver-output paths that
        consumers (analyze round 2, replay reconstruction) rely on."""
        return tuple(validate_diff_path(p) for p in paths)

    @model_validator(mode="after")
    def _enforce_resolution_invariants(self) -> Self:
        """Three rules per #017 × #024 amendment (point 5)."""
        n_resolved = len(self.resolved_candidate_paths)
        if self.resolution_status == "resolved":
            if n_resolved != 1:
                raise ValueError(
                    f"resolved TraceDecision requires exactly one "
                    f"resolved_candidate_paths entry; got {n_resolved}"
                )
            if self.target_file is None:
                raise ValueError("resolved TraceDecision requires non-None target_file")
            if self.target_file != self.resolved_candidate_paths[0]:
                raise ValueError(
                    f"resolved target_file ({self.target_file!r}) must equal the "
                    f"single resolved_candidate_paths entry "
                    f"({self.resolved_candidate_paths[0]!r})"
                )
        elif self.resolution_status == "unresolved":
            if n_resolved != 0:
                raise ValueError(
                    f"unresolved TraceDecision requires zero "
                    f"resolved_candidate_paths entries; got {n_resolved}"
                )
            if self.target_file is not None:
                raise ValueError("unresolved TraceDecision requires target_file is None")
        else:  # ambiguous
            if n_resolved <= 1:
                raise ValueError(
                    f"ambiguous TraceDecision requires more than one "
                    f"resolved_candidate_paths entry; got {n_resolved}"
                )
            if self.target_file is not None:
                raise ValueError("ambiguous TraceDecision requires target_file is None")
        return self

    @model_validator(mode="after")
    def _enforce_proposed_import_strings_unique(self) -> Self:
        """`proposed_import_strings` is set-semantic — each LLM-proposed
        candidate is one consideration, not many. Duplicates would
        confuse audit-stream consumers and any future content-derived
        identifier over the tuple. Per #024 amendment: split uniqueness
        validator (one per tuple)."""
        if len(self.proposed_import_strings) != len(set(self.proposed_import_strings)):
            raise ValueError(
                f"TraceDecision.proposed_import_strings contains duplicates: "
                f"{sorted(self.proposed_import_strings)!r}"
            )
        return self

    @model_validator(mode="after")
    def _enforce_resolved_candidate_paths_unique(self) -> Self:
        """`resolved_candidate_paths` is set-semantic — each resolved
        candidate is one resolution outcome, not many. Per #024
        amendment: split uniqueness validator (one per tuple)."""
        if len(self.resolved_candidate_paths) != len(set(self.resolved_candidate_paths)):
            raise ValueError(
                f"TraceDecision.resolved_candidate_paths contains duplicates: "
                f"{sorted(self.resolved_candidate_paths)!r}"
            )
        return self
