# Audit event hierarchy per docs/spec.md §7.2.1 + §8.2.
# Append-only contract per docs/trust-boundaries.md §7.
"""Audit event class hierarchy + discriminated union.

`AuditEventBase` is the shared base. The hierarchy has fifteen
concrete subtypes: twelve V1 subtypes per spec §8.2 (`AgentTransitionEvent`,
`ReviewPhaseEvent`, `LLMCallEvent`, `FileExaminationEvent`,
`FindingEvent`, `TraceDecisionEvent`, `HITLRequestEvent`,
`HITLDecisionEvent`, `PublishEvent`, `PublishRoutingEvent`,
`PublishEligibilityEvent`, `PublishAttemptEvent`) plus three
analyze-foundation additions (`AnalyzeCompletedEvent`,
`FindingProposalRejectedEvent`, `AnalyzeResponseRejectedEvent`). Each
declares its own `event_type: Literal[...]` discriminator value. The
`AuditEvent` discriminated-union alias is what `audit/replay.py` uses to
reconstruct concrete events from `audit_events.payload` JSONB at read time:

    TypeAdapter(AuditEvent).validate_python({**payload, "sequence_number": row.sequence_number})

Every event uses `ConfigDict(frozen=True, extra="forbid")` per
`audit-events-frozen-extra-forbid`. Tuple-typed sequence fields
(`context_summary`, `trace_path`, the HITL containers, `candidates_considered`)
deliver true immutability — Pydantic `frozen=True` only blocks attribute
reassignment, not in-place container mutation. Nested Pydantic payload
classes (`ContextManifestEntry`) carry their own `frozen=True + extra=forbid`
because the outer model's frozen-ness does not propagate.

Six event types carry validators (plus `PerFindingDecision` inherited
by `HITLDecisionEvent.decisions`):

  - `LLMCallEvent` enforces the `degradation_reason` cross-field rule
    per `DECISIONS.md#016`: non-None iff `degraded_mode is True`.
  - `FileExaminationEvent` enforces the `skip_reason` cross-field rule
    per `DECISIONS.md#018`: `skip_reason is not None` ↔
    `parse_status == "skipped"`.
  - `FindingEvent` carries three validators — proof-boundary
    (`policy/findings.enforce_proof_boundary`, backs
    `evidence-tier-schema-enforced`), the line constraint
    (`line_end >= line_start`), and `_verify_content_hash` per spec §8.5
    (rejects emitter bugs that produce a mis-computed hash with the
    right shape).
  - `TraceDecisionEvent` enforces the three-rule resolution invariant
    per `DECISIONS.md#017` (Amended same-day, two clauses):
    (a) resolved ↔ non-None target_file;
    (b) unresolved/ambiguous ↔ target_file is None;
    (c) when resolved, target_file in candidates_considered.
  - `AnalyzeCompletedEvent` enforces two accounting equations per
    foundation §5: `n_proposals_seen == n_findings_emitted +
    n_proposals_rejected`, and `n_responses_rejected <= n_llm_calls`.
  - `FindingProposalRejectedEvent` enforces the bidirectional
    `claimed_evidence_tier` ↔ `rejection_reason ==
    "evidence_tier_not_in_enum"` coupling.
  - `PerFindingDecision` (referenced via `HITLDecisionEvent.decisions`)
    carries its own validator per `schemas/hitl.py`; the wrapping event
    inherits that gate.

Replay merges the row-level `sequence_number` (DB-assigned BIGSERIAL)
into the payload before validating; the emitter dumps with
`mode="json", exclude={"sequence_number"}` per the row-vs-payload split.
"""

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from outrider.ast_facts.models import SkipReason
from outrider.coordinates import validate_diff_path
from outrider.llm.pricing import PRICING_VERSION_PATTERN
from outrider.policy import (
    EvidenceTier,
    FindingSeverity,
    FindingType,
    enforce_proof_boundary,
)
from outrider.policy.canonical import SHA256_HEX_PATTERN, SHA256_HEX_PATTERN_SHORT
from outrider.policy.severity import BARE_SEMVER_PATTERN
from outrider.schemas import (
    PerFindingDecision,
    PublishDestination,
    ReviewDimension,
)

# SHA-256 hashes are 256 bits = 64 hex characters per spec §8.5
# (FindingEvent.finding_content_hash = SHA-256(file_path + line_start +
# line_end + finding_type)). Lowercase-hex is the canonical encoding;
# enforce at the schema layer so the audit log's deduplication contract
# can rely on a deterministic format. Lifted to `outrider.policy.canonical`
# per §1 of the analyze-foundation spec so both `schemas/` and `audit/`
# can consume without circular import; module-local alias preserved for
# the existing references below.
_SHA256_HEX_PATTERN: Final = SHA256_HEX_PATTERN

# `BARE_SEMVER_PATTERN` (re-exported via `outrider.policy.severity`)
# gates the bare-semver shape on every `policy_version` field below.
# Single source of truth — the runtime `_SEMVER_RE` in `policy.severity`
# AND the DB CHECK constraint added by migration `3d03bca7f2be` derive
# from the same string. `pricing_version` carries `PRICING_VERSION`
# which uses a distinct versioning scheme (`"v2"` not bare semver) per
# `llm/pricing.py`.


def compute_finding_content_hash(
    file_path: str,
    *,
    line_start: int,
    line_end: int,
    finding_type: FindingType,
) -> str:
    """Canonical SHA-256 hash of a finding's identity tuple per spec §8.5.

    Encoding: compact JSON of `[file_path, line_start, line_end, finding_type.value]`,
    UTF-8 bytes, SHA-256 hex digest (lowercase). JSON encoding handles
    file paths with special characters deterministically; compact separators
    `(",", ":")` produce a single canonical byte sequence per input tuple.

    `file_path` runs through `coordinates.validate_diff_path` BEFORE
    entering the hash payload — same shape as `compute_proposal_hash`
    and the path-canonicalization rule in spec.md §1. Without this,
    a caller that pre-computes the hash from a non-canonical path
    (`"./src/foo.py"`) and then constructs a `ReviewFinding` (which
    normalizes via the same validator) produces a `_verify_content_hash`
    mismatch — the wrapper recomputes from the canonical form and
    diverges from the caller's pre-computed hash. Pinning the
    canonical floor at the recipe means alias paths to the same file
    produce a SINGLE digest.

    `line_start`, `line_end`, and `finding_type` are keyword-only —
    `line_start`/`line_end` are adjacent same-typed `int` parameters, and
    a positional swap would silently produce a different hash, which IS
    the dedup key. Same misuse-resistance pattern as
    `outrider.llm.pricing.compute_cost_usd` (token args keyword-only)
    and `outrider.coordinates.tree_sitter_to_github` (full keyword-only).

    Both the emitter (when constructing `FindingEvent`) and the
    `_verify_content_hash` model_validator on `FindingEvent` use this
    helper. The validator verifies the supplied hash equals the helper's
    output — silent emitter bugs (wrong inputs, different encoding) raise
    at event-construction time rather than producing dedup false-negatives
    at audit-query time.

    Spec §8.5 originally said "SHA-256(file_path + line_start + line_end +
    finding_type)" with informal `+` notation; this helper pins down the
    encoding choice so type and event agree.
    """
    canonical_file_path = validate_diff_path(file_path)
    payload = json.dumps(
        [canonical_file_path, line_start, line_end, finding_type.value],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class AuditEventBase(BaseModel):
    """Shared fields for every audit event.

    Subclasses MUST declare an `event_type: Literal[...]` field with a
    default value matching their discriminator key (e.g.,
    `event_type: Literal["llm_call"] = "llm_call"`); the `AuditEvent`
    union below uses that field as the discriminator.

    `sequence_number` is nullable on the base because it is assigned by
    Postgres BIGSERIAL at INSERT time. The construct-then-insert path
    has `None`; the read-then-reconstruct path has the assigned int.

    Equality semantics: Pydantic compares all fields. `event_id` and
    `timestamp` both use `default_factory` (uuid4 + `datetime.now(UTC)`),
    so two events constructed back-to-back with otherwise-identical
    args compare UNEQUAL. This is the intended semantic (each event is
    a distinct point in time), but means tests asserting "the right
    event was emitted" must compare specific fields (`review_id`,
    `node_id`, `marker`, ...), NOT full model equality. The
    `(event_id, sequence_number)` pair is the durable identity once a
    row lands in `audit_events`; in-memory equality is rarely the
    operation you want.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    review_id: UUID
    event_type: str
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    sequence_number: int | None = None
    is_eval: bool = False


class ContextManifestEntry(BaseModel):
    """One scope-unit entry inside `LLMCallEvent.context_summary`.

    Frozen + extra=forbid because the outer event's `frozen=True` does
    not propagate to nested Pydantic models. Without this, an entry
    could be mutated post-construction (`entry.file_path = "..."`) even
    when the containing event is frozen and the tuple is immutable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # `file_path` cap matches `AnalysisRound.files_examined` per-element
    # max_length=1024. `scope_unit_name` carries a Python identifier
    # (function/method/class name) plus optional dotted qualifier;
    # 1024 chars accommodates deeply-nested closures while bounding
    # the audit-row growth.
    file_path: str = Field(max_length=1024)
    scope_unit_name: str = Field(max_length=1024)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    inclusion_reason: Literal[
        "changed_scope",
        "same_file_context",
        "trace_expansion",
    ]

    @field_validator("file_path")
    @classmethod
    def _enforce_canonical_file_path(cls, path: str) -> str:
        """Re-run `validate_diff_path` so the audit-event side enforces the
        same repo-relative-POSIX invariant `AnalysisRound.files_examined`
        does. Without this gate, a traversal-bearing or shell-metacharacter
        path on a `ContextManifestEntry` inside `LLMCallEvent.context_summary`
        could ride into the append-only audit log without going through
        the canonical path-validation gate.
        """
        return validate_diff_path(path)

    @model_validator(mode="after")
    def _enforce_line_constraint(self) -> Self:
        """line_end must be >= line_start (1-indexed per coordinates/)."""
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self


class AgentTransitionEvent(AuditEventBase):
    """Node-to-node transition in the LangGraph state machine."""

    event_type: Literal["agent_transition"] = "agent_transition"
    from_node: str
    to_node: str
    latency_ms: int = Field(ge=0)


class ReviewPhaseEvent(AuditEventBase):
    """Phase boundary marker; start/end pairs scope per-node work.

    Per `phase-events-bound-work`, replay groups events between matching
    start/end markers as belonging to one phase. `phase_key` is V1.5
    forward-compat: parallel-analyze workers in V1.5 emit per-file
    phase pairs keyed by file path.
    """

    event_type: Literal["review_phase"] = "review_phase"
    phase_id: str
    # The seven graph nodes per spec §4 (intake, triage, analyze, trace,
    # synthesize, hitl, publish). Each emits start/end ReviewPhaseEvent
    # pairs scoping the node's work. V1 only intake + triage have shipped
    # emit sites; the other five emit when their respective node specs
    # land. Tightening the Literal now is forward-compatible AND stops
    # an emission-site typo (`"analyse"`, `"sythesize"`) from landing in
    # the append-only audit log before the spec for that node arrives.
    node_id: Literal["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"]
    marker: Literal["start", "end"]
    phase_key: str | None = None


class LLMCallEvent(AuditEventBase):
    """Metadata for one LLM call. Content lives in `llm_call_content` per #016.

    Token / cost / latency fields carry `ge=0` constraints so the cost-budget
    anomaly (V1 sums LLMCallEvent.cost_usd, V1.5 estimates pre-flight) can't
    be poisoned by a malformed negative-cost event understating review cost.

    `pricing_version` records the `llm.pricing.PRICING_VERSION` value the
    wrapper used to compute `cost_usd`, per DECISIONS.md#016 Amended
    2026-05-05. Replay reads this field directly so reconstruction never
    depends on an external version-effective-range map.
    """

    event_type: Literal["llm_call"] = "llm_call"
    model: str
    # The four nodes that actually make LLM calls per spec §4.1. Mirrors
    # `LLMRequest.node_id` (`llm/base.py`) verbatim — the wrapper passes
    # `request.node_id` through unchanged, so the event field must admit
    # exactly the same set. Pre-sweep this was `str` and admitted any
    # value including typos that would land in the append-only audit log.
    node_id: Literal["triage", "analyze", "synthesize", "trace"]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    pricing_version: str = Field(pattern=PRICING_VERSION_PATTERN)
    latency_ms: int = Field(ge=0)
    prompt_hash: str = Field(pattern=_SHA256_HEX_PATTERN)
    cache_hit: bool
    context_summary: tuple[ContextManifestEntry, ...]
    prompt_template_version: str
    system_prompt_hash: str = Field(pattern=_SHA256_HEX_PATTERN)
    degraded_mode: bool
    # `degraded_mode: bool` alone loses provenance on metadata-only
    # replay (post-retention or partial-content). The two reasons
    # (`parse_failed` vs `tree_has_error_in_changed_regions`)
    # imply structurally different prompt content; collapsing them into
    # the bool means audit-stream queries like "how many parse_failed
    # analyze calls did we make this month" become unanswerable. Same
    # bidirectional coupling as `LLMRequest.degradation_reason` enforced
    # by `_enforce_degradation_reason_consistency` below. Spec gap
    # surfaced by the §0b ; landing in the same commit per
    # `feedback_spec_gaps_surface_as_suggestions` since omission would
    # corrupt replay reconstruction.
    degradation_reason: Literal["parse_failed", "tree_has_error_in_changed_regions"] | None = None

    @model_validator(mode="after")
    def _enforce_degradation_reason_consistency(self) -> Self:
        """Three-way coupling, mirroring `LLMRequest._enforce_degradation_provenance`:

          (a) analyze-only scoping: `degraded_mode=True` AND `degradation_reason`
              non-None are valid ONLY when `node_id == "analyze"`. Other nodes
              (triage/synthesize/trace) have no degraded-mode contract in V1.
          (b) bidirectional bool/reason coupling within analyze:
              `degraded_mode == (degradation_reason is not None)`.

        Provenance pairing across the request → event boundary: the
        wrapper copies these fields verbatim, so if a request was
        admissible (analyze-scoped AND bool/reason coupled), the event
        must be too. A divergent event would mean the wrapper lost the
        field mid-pipeline — a class of bug the persister's
        `_CHECKED_FIELDS` also catches but this validator surfaces at
        event-construction time, before any DB write AND at replay-time
        re-validation (the read/replay boundary
        was unguarded — request rejected `trace + degraded_mode=True`
        but the event admitted it).
        """
        # Rule (a): analyze-only scoping. Mirrors LLMRequest validator.
        if self.degraded_mode and self.node_id != "analyze":
            raise ValueError(
                f"LLMCallEvent.degraded_mode=True only valid for node_id='analyze' "
                f"in V1; got node_id={self.node_id!r}. Synthesize/trace/triage "
                f"have no degraded-mode contract."
            )
        if self.degradation_reason is not None and self.node_id != "analyze":
            raise ValueError(
                f"LLMCallEvent.degradation_reason is only valid for node_id='analyze' "
                f"in V1; got node_id={self.node_id!r}."
            )
        # Rule (b): bidirectional bool/reason coupling within analyze.
        if self.degraded_mode and self.degradation_reason is None:
            raise ValueError(
                "LLMCallEvent.degraded_mode=True requires a non-None degradation_reason; "
                "the wrapper must pass through LLMRequest.degradation_reason verbatim"
            )
        if (not self.degraded_mode) and self.degradation_reason is not None:
            raise ValueError(
                "LLMCallEvent.degradation_reason requires degraded_mode=True; "
                "reason without mode is inconsistent (wrapper drift?)"
            )
        return self

    @model_validator(mode="after")
    def _enforce_cache_hit_matches_cached_tokens(self) -> Self:
        """`cache_hit == (cached_tokens > 0)` — bidirectional coupling.

        Category F sweep — the producer (`anthropic_provider.py:520`)
        computes `cache_hit = (response.cache_read_tokens > 0)` and
        stores both fields on the event independently. Without this
        validator, the two could drift via a test fixture, a producer
        bug, or a partial replay-row corruption; downstream cache-rate
        metrics would silently report the wrong number. Same shape as
        the degraded_mode ↔ degradation_reason coupling above.
        """
        expected_hit = self.cached_tokens > 0
        if self.cache_hit != expected_hit:
            raise ValueError(
                f"LLMCallEvent.cache_hit={self.cache_hit!r} disagrees with "
                f"cached_tokens={self.cached_tokens!r} — `cache_hit` MUST equal "
                f"`cached_tokens > 0`. Producer (`anthropic_provider`) computes "
                f"cache_hit from cached_tokens; a divergent event row means the "
                f"wrapper drifted mid-pipeline."
            )
        return self

    @model_validator(mode="after")
    def _enforce_context_summary_unique(self) -> Self:
        """`context_summary` is set-semantic by `(file_path, scope_unit_name)`.
        The same scope unit shouldn't appear twice in one prompt's manifest;
        duplicates would inflate the audit row and confuse the
        context-attribution view (which scope grounded which finding).
        """
        keys = [(e.file_path, e.scope_unit_name) for e in self.context_summary]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"LLMCallEvent.context_summary contains duplicate "
                f"(file_path, scope_unit_name) entries: {sorted(keys)!r}"
            )
        return self


class FileExaminationEvent(AuditEventBase):
    """Records that a file was examined (parse status + node).

    `skip_reason` per `DECISIONS.md#018`: non-None iff
    `parse_status == "skipped"`. The cross-field validator below
    enforces the bidirectional rule. Same shape as
    `TraceDecisionEvent`'s `(target_file, resolution_status)` validator
    per #017 — one event, one related-but-nullable field, one
    cross-rule, deterministic on replay.
    """

    event_type: Literal["file_examination"] = "file_examination"
    file_path: Annotated[str, Field(max_length=1024)]
    # Bounded to the two actually-emitted values today (intake's per-file
    # fetch record at `agent/nodes/intake.py:703,737` and analyze's
    # per-file examination at the sister-spec node body). Adding a third
    # stage's emission site is a schema change — the explicit Literal forces
    # the discriminator-like field to be widened deliberately rather than
    # drifting on a string typo from a future contributor.
    examination_type: Literal["intake_fetch", "analyze"]
    # `node_id` is the graph-node that emitted this event; for V1
    # FileExaminationEvent fires only from intake (per-file fetch) and
    # analyze (per-file examination). Wider than examination_type only
    # because a future node might emit with a NEW examination_type while
    # sharing a node_id with an existing emitter — keep them independent
    # constants so widening one doesn't force widening the other.
    node_id: Literal["intake", "analyze"]
    parse_status: Literal["clean", "degraded", "failed", "skipped"]
    skip_reason: SkipReason | None = None

    @field_validator("file_path")
    @classmethod
    def _enforce_canonical_file_path(cls, path: str) -> str:
        """Audit-shadow mirror of `paths-validated-before-use`. Matches
        the `AnalysisRound.files_examined` validator at the schemas side
        so the audit-row contract is at least as strict as the in-memory
        shape — a traversal-bearing or shell-metacharacter path can't
        ride into the append-only log unvalidated.
        """
        return validate_diff_path(path)

    @model_validator(mode="after")
    def _enforce_skip_reason_outcome(self) -> Self:
        """Per DECISIONS.md#018: skip_reason non-None iff parse_status='skipped'."""
        skipped = self.parse_status == "skipped"
        has_reason = self.skip_reason is not None
        if skipped and not has_reason:
            raise ValueError(
                "FileExaminationEvent: parse_status='skipped' requires a non-None skip_reason"
            )
        if has_reason and not skipped:
            raise ValueError(
                f"FileExaminationEvent: skip_reason={self.skip_reason!r} "
                f"requires parse_status='skipped' "
                f"(got {self.parse_status!r})"
            )
        return self


class FindingEvent(AuditEventBase):
    """Metadata for one finding. Proof artifacts are validated here.

    `enforce_proof_boundary` runs as a model_validator so the boundary
    holds at the audit-event layer too, not just at `ReviewFinding`.
    Backs `evidence-tier-schema-enforced`.
    """

    event_type: Literal["finding"] = "finding"
    finding_id: UUID
    finding_type: FindingType
    severity: FindingSeverity
    file_path: Annotated[str, Field(max_length=1024)]
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    dimension: ReviewDimension
    # SHA-256 hex per spec §8.5: SHA-256(file_path + line_start + line_end + finding_type)
    finding_content_hash: str = Field(pattern=_SHA256_HEX_PATTERN)
    evidence_tier: EvidenceTier
    # `max_length=200` mirrors `ReviewFinding.query_match_id`; the audit
    # shadow enforces at least the same contract as the source schema so
    # a future direct emitter (replay reconstruction, alternate producer)
    # can't land an unbounded id in the append-only log. `trace_path` is
    # bounded symmetrically per element (256, min 1) and per tuple (32)
    # mirroring `ReviewFinding.trace_path` and the raw layer.
    query_match_id: Annotated[str | None, Field(max_length=200)] = None
    trace_path: (
        Annotated[
            tuple[Annotated[str, Field(max_length=256, min_length=1)], ...],
            Field(max_length=32),
        ]
        | None
    ) = None
    policy_version: str = Field(pattern=BARE_SEMVER_PATTERN)
    # Audit-shadow mirror of `ReviewFinding.proposal_hash` per
    # `DECISIONS.md#025` (Accepted 2026-05-24). Provenance link from
    # admitted findings to `TraceCandidate.source_proposal_hash` so
    # trace's join contract holds at replay time without consulting
    # the in-memory ReviewFinding. NOT part of `finding_content_hash`
    # recipe (#025 point 3): content hash stays stable across LLM
    # phrasing variation; proposal_hash carries provenance.
    proposal_hash: Annotated[str, Field(pattern=_SHA256_HEX_PATTERN)]

    @field_validator("file_path")
    @classmethod
    def _enforce_canonical_file_path(cls, path: str) -> str:
        """Audit-shadow mirror of `paths-validated-before-use`. Matches
        the `ReviewFinding.file_path` validator at the schemas side —
        the audit shadow must be at least as strict as the in-memory
        shape, or a traversal-bearing file_path could ride into the
        append-only log unvalidated.
        """
        return validate_diff_path(path)

    @model_validator(mode="after")
    def _enforce_proof_boundary(self) -> Self:
        """Wire policy/findings.enforce_proof_boundary into Pydantic validation."""
        enforce_proof_boundary(
            evidence_tier=self.evidence_tier,
            query_match_id=self.query_match_id,
            trace_path=self.trace_path,
        )
        return self

    @model_validator(mode="after")
    def _enforce_line_constraint(self) -> Self:
        """line_end must be >= line_start (1-indexed per coordinates/)."""
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self

    @model_validator(mode="after")
    def _enforce_dimension_lockstep(self) -> Self:
        """`dimension` must equal `FINDING_TYPE_TO_DIMENSION[finding_type]`.

        Mirror of `ReviewFinding._enforce_dimension_lockstep` at the
        audit-event layer. Without this, a row like
        `(finding_type=SQL_INJECTION, dimension=PERFORMANCE)` admits even
        though `FINDING_TYPE_TO_DIMENSION[SQL_INJECTION] == SECURITY` —
        the same gap class the schemas-side validator closes for
        `severity`. The module-load lockstep guard in
        `outrider.policy.dimensions` fires only at import; it can't
        detect an audit row ALREADY in `audit_events.payload` carrying
        a drifted dimension. This validator closes that hole at the
        audit-event layer too.

        Imported locally to avoid a circular import: `policy.dimensions`
        imports `ReviewDimension` from `schemas.review_finding`; this
        module imports `ReviewDimension` via the schemas re-export.
        """
        from outrider.policy.dimensions import (  # noqa: PLC0415
            FINDING_TYPE_TO_DIMENSION,
        )

        expected = FINDING_TYPE_TO_DIMENSION[self.finding_type]
        if self.dimension != expected:
            raise ValueError(
                f"FindingEvent.dimension={self.dimension.value!r} drifted from "
                f"FINDING_TYPE_TO_DIMENSION[{self.finding_type.value!r}]="
                f"{expected.value!r}. Same canonical rule as `ReviewFinding`. "
                f"Per DECISIONS.md#021, `FINDING_TYPE_TO_DIMENSION` is append-only "
                f"for existing FindingType members; a mapping change is a "
                f"DECISIONS-level ontology rewrite, not a quiet code edit."
            )
        return self

    @model_validator(mode="after")
    def _enforce_severity_matches_policy(self) -> Self:
        """`severity` must equal `SEVERITY_POLICY[finding_type]` under the
        live policy version. Backs `severity-set-by-policy`.

        FindingEvent is emitted by the analyze node BEFORE HITL; it
        carries the policy-computed baseline. HITL overrides do not
        rewrite FindingEvent — they emit a separate HITLDecisionEvent
        with `override_severity` + `original_severity`. So this event's
        severity must always match SEVERITY_POLICY[finding_type] at
        write time, no override case to consider.

        Replay-aware scoping. `model_validate` is the SAME path
        `TypeAdapter(AuditEvent).validate_python(...)` uses to
        reconstruct historical events (see module docstring at
        the top of this file). A historical event under an older
        `policy_version` MUST validate cleanly — the severity it
        carries was correct AT WRITE TIME under its frozen policy,
        and we have no synchronous loader for the historical
        mapping here (`policy/versions.py::load_policy_for_version`
        is async; it's the persister/replay layer's job, not the
        schema's).

        Scope: the live-policy match check below fires ONLY when
        `policy_version == ACTIVE_POLICY_VERSION`. Older versions
        skip and trust the historical row. The "fresh-write smuggle"
        concern (a producer setting `policy_version="0.9.0"` to
        dodge the live check) is NOT defended at the schema layer —
        it's a producer-side discipline enforced by the emitter and,
        when the replay/persister spec lands, by the persister's
        write-time check that incoming events carry the active
        version. The schema layer cannot distinguish fresh writes
        from replay reconstruction inside `model_validate`.
        """
        # Local import: policy modules cannot import from audit.events
        # at top-level (they don't), so this could move up; kept local
        # for symmetry with the schemas-side validator.
        from outrider.policy.severity import (  # noqa: PLC0415
            ACTIVE_POLICY_VERSION,
            SEVERITY_POLICY,
        )

        if self.policy_version != ACTIVE_POLICY_VERSION:
            # Historical event: trust the row. Versioned-replay
            # cross-check belongs in the persister/replay layer.
            return self
        expected = SEVERITY_POLICY.get(self.finding_type)
        if expected is None or self.severity != expected:
            raise ValueError(
                f"FindingEvent.severity={self.severity.value!r} does not match "
                f"SEVERITY_POLICY[{self.finding_type.value!r}]="
                f"{(expected.value if expected else None)!r} under policy_version "
                f"{self.policy_version!r}. Per `severity-set-by-policy` "
                f"(docs/invariants.md), baseline severity comes from SEVERITY_POLICY "
                f"keyed by finding_type, never from caller or model output."
            )
        return self

    @model_validator(mode="after")
    def _verify_content_hash(self) -> Self:
        """Spec §8.5: finding_content_hash MUST equal the canonical computation.

        Format gating alone (the Field pattern) accepts any 64-hex string for
        any input tuple, so an emitter bug producing a mis-computed hash
        would still pass and create dedup false-negatives at audit-query
        time. This validator computes the canonical hash and rejects mismatch.
        """
        expected = compute_finding_content_hash(
            file_path=self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            finding_type=self.finding_type,
        )
        if self.finding_content_hash != expected:
            raise ValueError(
                f"finding_content_hash mismatch: spec §8.5 requires "
                f"SHA-256 of canonical input tuple "
                f"(file_path, line_start, line_end, finding_type); got "
                f"{self.finding_content_hash!r}, expected {expected!r}. "
                "Use audit.events.compute_finding_content_hash() to compute "
                "the value at the call site."
            )
        return self


class TraceDecisionEvent(AuditEventBase):
    """One aggregate trace decision per source_finding_id (per DECISIONS.md#017).

    Three-rule cross-field validator per #017 (Amended same-day, two clauses):
    (a) resolved ↔ non-None target_file
    (b) unresolved / ambiguous ↔ target_file is None
    (c) when resolved, target_file in candidates_considered

    `candidates_considered` is the LLM-proposed candidate list (any
    cardinality); `resolution_status` describes how many resolved
    through ast_facts (zero / exactly one / multiple). Required field
    (no default) per #017 — defaults would silently absorb emitter bugs
    and undermine §8.7 replay equivalence; callers pass `()` explicitly
    for the zero-candidate case.
    """

    event_type: Literal["trace_decision"] = "trace_decision"
    source_finding_id: UUID
    target_file: str | None
    reason: str = Field(max_length=500)
    resolution_status: Literal["resolved", "unresolved", "ambiguous"]
    candidates_considered: tuple[str, ...]
    trace_path: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _enforce_resolution_invariants(self) -> Self:
        """Three rules per DECISIONS.md#017 (Amended same-day)."""
        if self.resolution_status == "resolved":
            if self.target_file is None:
                raise ValueError("resolved TraceDecisionEvent requires non-None target_file")
            if self.target_file not in self.candidates_considered:
                raise ValueError("resolved target_file must be a member of candidates_considered")
        else:
            if self.target_file is not None:
                raise ValueError(
                    f"{self.resolution_status} TraceDecisionEvent requires target_file is None"
                )
        return self

    @model_validator(mode="after")
    def _enforce_candidates_considered_unique(self) -> Self:
        """`candidates_considered` is set-semantic: each candidate is one
        consideration, not many. Duplicates would let the same logical
        trace decision hash differently (any future content-derived id
        over this field) and confuse audit-stream consumers.
        """
        if len(self.candidates_considered) != len(set(self.candidates_considered)):
            raise ValueError(
                f"TraceDecisionEvent.candidates_considered contains duplicates: "
                f"{sorted(self.candidates_considered)!r}"
            )
        return self


class HITLRequestEvent(AuditEventBase):
    """Records the HITL gate envelope at interrupt time.

    Audit-shadow mirror of `HITLRequest`: set-semantic partition of
    findings across the two tuples.
    """

    event_type: Literal["hitl_request"] = "hitl_request"
    findings_requiring_approval: tuple[UUID, ...]
    auto_post_findings: tuple[UUID, ...]
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def _enforce_finding_partition(self) -> Self:
        """Each finding appears at most once across the two tuples — mirror
        of `HITLRequest._enforce_finding_partition` at the audit-event layer.
        """
        if len(self.findings_requiring_approval) != len(set(self.findings_requiring_approval)):
            raise ValueError(
                f"HITLRequestEvent.findings_requiring_approval contains duplicate ids: "
                f"{sorted(str(u) for u in self.findings_requiring_approval)!r}"
            )
        if len(self.auto_post_findings) != len(set(self.auto_post_findings)):
            raise ValueError(
                f"HITLRequestEvent.auto_post_findings contains duplicate ids: "
                f"{sorted(str(u) for u in self.auto_post_findings)!r}"
            )
        overlap = set(self.findings_requiring_approval) & set(self.auto_post_findings)
        if overlap:
            raise ValueError(
                f"HITLRequestEvent: a finding cannot be in both "
                f"findings_requiring_approval and auto_post_findings; "
                f"overlap: {sorted(str(u) for u in overlap)!r}"
            )
        return self


class HITLDecisionEvent(AuditEventBase):
    """Records the reviewer's HITL submission.

    Field name `decisions` (not `per_finding_decisions`) matches the
    cross-boundary `HITLDecision.decisions` type per `DECISIONS.md#014`
    Amended 2026-04-29.
    """

    event_type: Literal["hitl_decision"] = "hitl_decision"
    # GitHub usernames are <=39 chars; SSO logins or future auth sources
    # might be longer, so 100 is generous-but-bounded. Without the cap a
    # malformed or attacker-supplied reviewer id could fill the audit row
    # arbitrarily and break replay aggregations keyed by reviewer.
    reviewer_id: str = Field(max_length=100)
    decisions: tuple[PerFindingDecision, ...]
    decision_latency_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _enforce_one_decision_per_finding(self) -> Self:
        """Mirror of `HITLDecision._enforce_one_decision_per_finding`: at
        most one `PerFindingDecision` per `finding_id` on the audit row.
        """
        finding_ids = [d.finding_id for d in self.decisions]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError(
                f"HITLDecisionEvent.decisions contains multiple decisions for the "
                f"same finding_id: {sorted(str(fid) for fid in finding_ids)!r}"
            )
        return self


class PublishEvent(AuditEventBase):
    """Records the GitHub publish operation outcome."""

    event_type: Literal["publish"] = "publish"
    github_review_id: int = Field(ge=1)  # GitHub review IDs are positive integers
    comments_posted: int = Field(ge=0)
    # The three GitHub PR review states — `event` param values on the
    # GitHub `POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews`
    # endpoint. Bounded so an emission-site typo or a stale snake_case
    # value can't drift into the append-only audit log. PENDING is a
    # GitHub-side draft state and is NEVER published from V1, so it's
    # deliberately omitted.
    review_status: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"]


# ---------------------------------------------------------------------------
# Publish-node §V1: PublishRoutingEvent extensions + PublishEligibilityEvent +
# PublishAttemptEvent + supporting enums + canonical decision-hash helpers.
# Per specs/2026-05-21-publish-node.md Q1-Q3.
# ---------------------------------------------------------------------------


class PublishRoutingReason(StrEnum):
    """Why the publisher routed a finding to its `PublishDestination`.

    StrEnum (not `Literal[...]`) so the routing reason mirrors the shape of
    `PublishEligibilityReason` and `PublishAttemptOutcome` below. Consumers
    branching on the value get `match`-statement exhaustiveness; the
    audit-stream queries can filter by enum members rather than raw strings.
    """

    # tree_sitter_to_github returned a GitHubCommentLocation. Destination
    # = INLINE_COMMENT.
    REVIEWABLE_DIFF_LINE = "reviewable_diff_line"

    # tree_sitter_to_github raised CoordinateError(kind=UNCHANGED_REGION).
    # Destination = REVIEW_BODY.
    UNCHANGED_REGION = "unchanged_region"

    # EITHER ChangedFile registry miss (coordinates not called) OR
    # CoordinateError(kind=FILE_NOT_IN_PATCH) — registry/patch disagreement.
    # Destination = DASHBOARD_ONLY.
    NON_DIFFED_FILE = "non_diffed_file"

    # Any other CoordinateError(kind=...); the kind itself rides on
    # PublishRoutingEvent.coordinate_error_kind so the audit stream can
    # group by structural failure class. Destination = DASHBOARD_ONLY.
    COORDINATE_ERROR = "coordinate_error"


class PublishEligibility(StrEnum):
    """Per-finding materialization decision, separate from routing.

    Per specs/2026-05-21-publish-node.md Q3: routing is coordinate-derived;
    eligibility is policy-derived; they're audited independently so a
    CRITICAL finding routed cleanly to INLINE_COMMENT carries
    destination=INLINE_COMMENT AND eligibility=withheld.
    """

    ELIGIBLE = "eligible"
    WITHHELD = "withheld"


class PublishEligibilityReason(StrEnum):
    """Why a finding was withheld at the eligibility gate (V1 reasons only).

    V1 ships before `hitl` is wired; the `hitl_required_node_absent` and
    `unexpected_override_fields_present` reasons cover the V1 trust gate.
    The `routing_emission_failed` reason covers the per-finding try/except
    recovery path. Post-V1 will add `hitl_pending`, `hitl_rejected`,
    `hitl_suppressed` when the hitl node lands.
    """

    # severity ∈ {CRITICAL, HIGH} and `hitl` node is absent.
    HITL_REQUIRED_NODE_ABSENT = "hitl_required_node_absent"

    # finding carries `original_severity is not None` despite no legitimate
    # HITL override path existing in V1 — defends against producer bugs
    # or replay-injected state forging a pre-approved downgrade.
    UNEXPECTED_OVERRIDE_FIELDS_PRESENT = "unexpected_override_fields_present"

    # Per-finding `try/except` in the publish node's routing+eligibility
    # interleaved loop caught an exception from `emit_publish_routing`;
    # the eligibility event still fires (withheld) so the per-finding
    # audit contract holds even when routing emission fails.
    ROUTING_EMISSION_FAILED = "routing_emission_failed"


class PublishAttemptOutcome(StrEnum):
    """Terminal outcome of one `publisher.create_review` attempt.

    Single emission per attempt, AFTER the GitHub call resolves. No
    `in_flight` outcome — an in-flight pre-call emission would be
    incompatible with `audit-events-append-only` because
    same-event_id-different-payload raises
    `AuditPersisterIdempotencyConflict` rather than acting as an update.
    Crash-after-success defense via `find_existing_review_on_head_sha`.
    """

    # Atomic GitHub POST returned 2xx.
    SUCCESS = "success"

    # GitHub call failed (HTTP error, app uninstalled, permission denied,
    # transient 5xx that retry middleware gave up on). `failure_class`
    # field carries the exception class name.
    FAILED = "failed"

    # Pre-flight check found a prior `PublishEvent` for this review_id;
    # no GitHub call made. The review's UNIQUE(repo_id, pr_number, head_sha)
    # canonical constraint means review_id ALREADY scopes to one head_sha.
    IDEMPOTENTLY_SKIPPED = "idempotently_skipped"

    # No prior `PublishEvent` but `find_existing_review_on_head_sha`
    # found an existing review on PR with our review_id's body marker —
    # crash-after-success path. No second GitHub call, no second PublishEvent.
    IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD = "idempotently_skipped_external_record"

    # Zero eligible+INLINE_COMMENT-routed findings; no GitHub call.
    NO_OP_EMPTY = "no_op_empty"


def compute_publish_routing_decision_hash(
    *,
    destination: PublishDestination,
    reason: PublishRoutingReason,
    coordinate_error_kind: object | None,
) -> str:
    """SHA-256 hex over the routing decision tuple per Q1.f.

    Canonical encoding: compact JSON of `[destination.value, reason.value,
    coordinate_error_kind.value | None]`. Mirrors `compute_finding_content_hash`
    above (the canonical reference recipe in this module).
    Two implementers OR two replay re-emissions of the same logical decision
    MUST produce identical hashes; the JSON encoding pins this.

    `coordinate_error_kind` is typed as `object | None` to avoid a runtime
    circular-import dependency on `outrider.coordinates.errors`; the helper
    accepts any StrEnum value (validated at the field-level on
    `PublishRoutingEvent.coordinate_error_kind`).

    Consumer-side dedup identity is `(review_id, finding_id, finding_content_hash,
    decision_content_hash)` — re-emission with the same decision collapses
    to one logical row; re-emission with a different decision produces two
    rows so an anomaly rule (V1.5 dashboard work, FOLLOWUPS.md FUP-063 —
    decision-drift saturation cap) can surface the drift.
    """
    kind_value: str | None
    if coordinate_error_kind is None:
        kind_value = None
    elif hasattr(coordinate_error_kind, "value"):
        kind_value = coordinate_error_kind.value
    else:
        kind_value = str(coordinate_error_kind)
    payload = json.dumps(
        [destination.value, reason.value, kind_value],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_publish_eligibility_decision_hash(
    *,
    eligibility: PublishEligibility,
    reason: PublishEligibilityReason | None,
) -> str:
    """SHA-256 hex over the eligibility decision tuple.

    `policy_version` is NOT in the hash even though it's on the event.
    A legitimate policy bump that doesn't change the gate logic for a
    given (finding_type, severity) must not surface as decision drift.
    `policy_version` is carried as a separate column for replay-equivalence
    queries (filter rows by policy_version) but the dedup identity treats
    two policy versions with identical gate outcomes as the SAME logical
    decision.
    """
    payload = json.dumps(
        [eligibility.value, reason.value if reason is not None else None],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_publish_attempt_content_hash(
    *,
    review_id: UUID,
    attempt_index: int,
    sorted_finding_ids: tuple[UUID, ...],
    outcome: PublishAttemptOutcome,
    status_code: int | None,
    failure_class: str | None,
    comments_attempted: int,
) -> str:
    """SHA-256 hex over the attempt content tuple.

    All attempt-distinguishing fields ride in the hash so divergent
    attempts don't collapse on read-time dedup:

    - `outcome` separates success-vs-failed-replay.
    - `status_code` + `failure_class` separate distinct failure modes
      under the same outcome (e.g., two FAILED attempts where one is
      a 422 validation rejection and the other is a 502 upstream
      error — these are LOGICALLY different attempts and should not
      collapse on the dedup join).
    - `comments_attempted` separates attempts where the eligible-
      finding set changed between retries (analyze re-ran with a
      different fixture, eligibility gate flipped a finding).
    - `sorted_finding_ids` ensures iteration-order permutations of
      the same set hash identically.

    `status_code` + `failure_class` are nullable (success attempts
    carry `None` on both); JSON encodes `None` → `null` distinct from
    any string, so the absence is itself a hash-distinguishing value.
    """
    payload = json.dumps(
        [
            str(review_id),
            attempt_index,
            [str(fid) for fid in sorted_finding_ids],
            outcome.value,
            status_code,
            failure_class,
            comments_attempted,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class PublishRoutingEvent(AuditEventBase):
    """Records the per-finding routing decision; backs publish-routes-through-coordinates.

    Q1-extended per specs/2026-05-21-publish-node.md:

    - `reason` is a `PublishRoutingReason` StrEnum (not a raw `Literal[...]`).
    - `coordinate_error_kind` carries the structurally distinct CoordinateError
      variants so they're queryable in the audit stream rather than collapsed
      into one opaque reason. Payload contains ONLY the enum value, NEVER the
      `CoordinateError.message` text — the `PATH_VALIDATION_FAILED` umbrella
      would otherwise leak the validate_diff_path rule set as an enumeration
      oracle to anyone with audit-log read access.
    - Identity fields `file_path`, `line_start`, `line_end`, `finding_content_hash`
      support post-retention metadata replay AND the `PublishEligibilityEvent`
      content-hash binding validator.
    - `decision_content_hash` is the consumer-side dedup identity so re-emission
      with the same decision collapses, but re-emission with a different
      decision surfaces as a second logical row rather than hiding behind
      `finding_content_hash`.
    """

    event_type: Literal["publish_routing"] = "publish_routing"
    finding_id: UUID
    destination: PublishDestination
    reason: PublishRoutingReason
    # Typed via `str | None` for JSONB-payload shape compatibility (the
    # field stores the enum's `.value` string, not a Python enum member,
    # so re-emission and replay round-trip cleanly through JSON). Runtime
    # membership against `CoordinateErrorKind` is enforced by the
    # `_enforce_coordinate_error_kind_membership` field validator below;
    # only the value lands in the audit row, never the
    # `CoordinateError.message` text (information-leak defense for the
    # `PATH_VALIDATION_FAILED` umbrella).
    #
    # `None` carries deliberate semantic on `reason=non_diffed_file`: the
    # registry-miss path short-circuits BEFORE coordinates is invoked, so
    # there's no `CoordinateError` to draw a kind from. Distinct from the
    # `FILE_NOT_IN_PATCH` case where coordinates IS invoked and reports
    # registry/patch disagreement. The two carry the same routing reason
    # but different diagnostic stories — replay can tell them apart via
    # this field.
    coordinate_error_kind: str | None = None
    file_path: Annotated[str, Field(max_length=1024)]
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    # Carried so `_verify_finding_content_hash` can recompute against the
    # canonical recipe; mirrors `PublishEligibilityEvent.finding_type` so
    # the two per-finding event types share the same identity tuple.
    finding_type: FindingType
    finding_content_hash: str = Field(pattern=_SHA256_HEX_PATTERN)
    decision_content_hash: str = Field(pattern=_SHA256_HEX_PATTERN)

    @field_validator("file_path")
    @classmethod
    def _enforce_canonical_file_path(cls, path: str) -> str:
        """Audit-shadow mirror of `paths-validated-before-use`."""
        return validate_diff_path(path)

    @field_validator("coordinate_error_kind")
    @classmethod
    def _enforce_coordinate_error_kind_membership(cls, value: str | None) -> str | None:
        """`coordinate_error_kind`, when set, MUST be the `.value` of a
        `CoordinateErrorKind` member. The field is typed as `str` (not the
        enum directly) for JSONB-payload shape compatibility, so the
        membership check has to live here rather than at the type layer.

        Without this validator, a JSON-replay row carrying
        `coordinate_error_kind="totally_made_up"` admits cleanly via
        `model_validate`, defeating the structural taxonomy that
        `coordinates/errors.py::CoordinateErrorKind` is supposed to be
        total over. Local-import the enum to avoid a top-level
        `audit -> coordinates` dependency.
        """
        if value is None:
            return value
        # Local import: keep `audit` independent of `coordinates` at module
        # import time. `coordinates/errors.py` is the canonical source for
        # the enum and intentionally has no `audit` dependency.
        from outrider.coordinates.errors import CoordinateErrorKind  # noqa: PLC0415

        valid_values = {member.value for member in CoordinateErrorKind}
        if value not in valid_values:
            raise ValueError(
                f"PublishRoutingEvent.coordinate_error_kind={value!r} is not a "
                f"CoordinateErrorKind member value (valid: {sorted(valid_values)})"
            )
        return value

    @model_validator(mode="after")
    def _enforce_line_constraint(self) -> Self:
        """line_end must be >= line_start (1-indexed per coordinates/)."""
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self

    @model_validator(mode="after")
    def _enforce_coordinate_error_kind_required_iff_coordinate_error(self) -> Self:
        """Enforce the `reason × destination × coordinate_error_kind` product.

        The three fields jointly identify the routing decision; an invalid
        tuple would silently corrupt audit semantics + drift analysis.

        Reason → destination → kind mapping (validator is total over the
        product):

        - reviewable_diff_line → INLINE_COMMENT → kind=None
          (coordinates returned success)
        - unchanged_region → REVIEW_BODY → kind=UNCHANGED_REGION
          (always; raise was caught)
        - non_diffed_file → DASHBOARD_ONLY → kind=None (registry miss;
          coordinates not called) OR kind=FILE_NOT_IN_PATCH
          (registry/patch disagreement)
        - coordinate_error → DASHBOARD_ONLY → kind required AND must
          NOT be UNCHANGED_REGION or FILE_NOT_IN_PATCH (those map to
          dedicated reasons above)
        """
        # Local import: kept inside the validator for the same reason as
        # `_enforce_coordinate_error_kind_membership` above — `audit` should
        # not gain a top-level `coordinates` dependency.
        from outrider.coordinates.errors import CoordinateErrorKind  # noqa: PLC0415

        kind = self.coordinate_error_kind

        if self.reason is PublishRoutingReason.REVIEWABLE_DIFF_LINE:
            if self.destination is not PublishDestination.INLINE_COMMENT:
                raise ValueError(
                    f"PublishRoutingEvent reason=reviewable_diff_line requires "
                    f"destination=INLINE_COMMENT, got {self.destination!r}"
                )
            if kind is not None:
                raise ValueError(
                    f"PublishRoutingEvent reason=reviewable_diff_line is the success path; "
                    f"coordinate_error_kind must be None, got {kind!r}"
                )
        elif self.reason is PublishRoutingReason.UNCHANGED_REGION:
            if self.destination is not PublishDestination.REVIEW_BODY:
                raise ValueError(
                    f"PublishRoutingEvent reason=unchanged_region requires "
                    f"destination=REVIEW_BODY, got {self.destination!r}"
                )
            if kind != CoordinateErrorKind.UNCHANGED_REGION.value:
                expected = CoordinateErrorKind.UNCHANGED_REGION.value
                raise ValueError(
                    f"PublishRoutingEvent reason=unchanged_region requires "
                    f"coordinate_error_kind={expected!r} (coordinates raised "
                    f"UNCHANGED_REGION; the kind is part of the routing "
                    f"identity), got {kind!r}"
                )
        elif self.reason is PublishRoutingReason.NON_DIFFED_FILE:
            if self.destination is not PublishDestination.DASHBOARD_ONLY:
                raise ValueError(
                    f"PublishRoutingEvent reason=non_diffed_file requires "
                    f"destination=DASHBOARD_ONLY, got {self.destination!r}"
                )
            allowed = {None, CoordinateErrorKind.FILE_NOT_IN_PATCH.value}
            if kind not in allowed:
                fnip = CoordinateErrorKind.FILE_NOT_IN_PATCH.value
                raise ValueError(
                    f"PublishRoutingEvent reason=non_diffed_file accepts only "
                    f"coordinate_error_kind in {{None (registry miss), {fnip!r} "
                    f"(registry/patch disagreement)}}, got {kind!r}"
                )
        elif self.reason is PublishRoutingReason.COORDINATE_ERROR:
            if self.destination is not PublishDestination.DASHBOARD_ONLY:
                raise ValueError(
                    f"PublishRoutingEvent reason=coordinate_error requires "
                    f"destination=DASHBOARD_ONLY, got {self.destination!r}"
                )
            if kind is None:
                raise ValueError(
                    "PublishRoutingEvent reason=coordinate_error requires coordinate_error_kind"
                )
            forbidden = {
                CoordinateErrorKind.UNCHANGED_REGION.value,
                CoordinateErrorKind.FILE_NOT_IN_PATCH.value,
            }
            if kind in forbidden:
                raise ValueError(
                    f"PublishRoutingEvent reason=coordinate_error must use the dedicated "
                    f"reason for {kind!r} (UNCHANGED_REGION → reason=unchanged_region; "
                    f"FILE_NOT_IN_PATCH → reason=non_diffed_file)"
                )
        return self

    @model_validator(mode="after")
    def _verify_finding_content_hash(self) -> Self:
        """`finding_content_hash` must equal `compute_finding_content_hash(
        file_path, line_start, line_end, finding_type)`.

        Same recipe + same canonical helper as `FindingEvent._verify_content_hash`
        and `PublishEligibilityEvent._verify_content_hash_binding`, so the
        three event types ride on identical identity hashes for the same
        finding — joins between routing/eligibility/finding rows compare by
        this hash directly without recomputation drift.
        """
        expected = compute_finding_content_hash(
            self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            finding_type=self.finding_type,
        )
        if self.finding_content_hash != expected:
            raise ValueError(
                f"PublishRoutingEvent.finding_content_hash="
                f"{self.finding_content_hash!r} does not match "
                f"compute_finding_content_hash(file_path={self.file_path!r}, "
                f"line_start={self.line_start}, line_end={self.line_end}, "
                f"finding_type={self.finding_type.value!r})={expected!r}."
            )
        return self

    @model_validator(mode="after")
    def _verify_decision_content_hash(self) -> Self:
        """`decision_content_hash` must equal `compute_publish_routing_decision_hash(...)`
        over this event's decision tuple.

        Pinning the recipe at the in-memory event layer prevents two
        emissions of the same logical decision from disagreeing on hash
        — which would surface as phantom drift in the consumer-side
        dedup. Pattern mirrors `FindingEvent._verify_content_hash`.
        """
        # CoordinateErrorKind is a StrEnum so the helper's hasattr branch
        # handles both the raw string and enum cases. We pass the field
        # value as-is; the helper canonicalizes.
        expected = compute_publish_routing_decision_hash(
            destination=self.destination,
            reason=self.reason,
            coordinate_error_kind=self.coordinate_error_kind,
        )
        if self.decision_content_hash != expected:
            raise ValueError(
                f"PublishRoutingEvent.decision_content_hash={self.decision_content_hash!r} "
                f"does not match compute_publish_routing_decision_hash over "
                f"(destination={self.destination.value!r}, reason={self.reason.value!r}, "
                f"coordinate_error_kind={self.coordinate_error_kind!r})={expected!r}. "
                f"Use compute_publish_routing_decision_hash(...) to compute the hash."
            )
        return self


class PublishEligibilityEvent(AuditEventBase):
    """Records the per-finding materialization decision, separate from routing.

    Per Q3: eligibility is policy-derived (gates on severity + HITL absence).
    Fires AFTER `PublishRoutingEvent` for each finding under the interleaved
    per-finding routing+eligibility loop. Carries identity fields so
    `_verify_content_hash_binding` can recompute `finding_content_hash` via
    the canonical helper; carries `severity` + `finding_type` + `policy_version`
    for severity-versioned replay (`severity-policy-versioned-for-replay`).
    """

    event_type: Literal["publish_eligibility"] = "publish_eligibility"
    finding_id: UUID
    file_path: Annotated[str, Field(max_length=1024)]
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    finding_type: FindingType
    severity: FindingSeverity
    # V1 requires None: no legitimate HITL override path exists yet, so a
    # non-None value indicates either a producer bug or replay-injected
    # state forging a pre-approved downgrade. Enforced at the gate
    # (is_eligible_for_v1_publish returns withheld with
    # unexpected_override_fields_present) and validated here on the event.
    original_severity: FindingSeverity | None = None
    finding_content_hash: str = Field(pattern=_SHA256_HEX_PATTERN)
    decision_content_hash: str = Field(pattern=_SHA256_HEX_PATTERN)
    eligibility: PublishEligibility
    reason: PublishEligibilityReason | None = None
    policy_version: str = Field(pattern=BARE_SEMVER_PATTERN)

    @field_validator("file_path")
    @classmethod
    def _enforce_canonical_file_path(cls, path: str) -> str:
        return validate_diff_path(path)

    @model_validator(mode="after")
    def _enforce_line_constraint(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self

    @model_validator(mode="after")
    def _enforce_eligibility_reason_required_when_withheld(self) -> Self:
        """`reason` required iff `eligibility == withheld`. An eligible
        finding has no withholding reason; a withheld finding must record
        one."""
        if self.eligibility is PublishEligibility.WITHHELD and self.reason is None:
            raise ValueError("PublishEligibilityEvent eligibility=withheld requires a reason")
        if self.eligibility is PublishEligibility.ELIGIBLE and self.reason is not None:
            raise ValueError(
                f"PublishEligibilityEvent eligibility=eligible must have reason=None, "
                f"got {self.reason!r}"
            )
        return self

    @model_validator(mode="after")
    def _enforce_severity_matches_policy(self) -> Self:
        """`severity` must equal `SEVERITY_POLICY[finding_type]` under the
        live policy version. Backs `severity-set-by-policy`. Mirror of
        `FindingEvent._enforce_severity_matches_policy` so the
        eligibility-side audit shadow is at least as strict as the
        analyze-side source row.

        Replay-aware scoping (identical reasoning to `FindingEvent`):
        the live-policy match check fires ONLY when
        `policy_version == ACTIVE_POLICY_VERSION`. Historical rows under
        older policy versions trust the row — the severity it carries
        was correct AT WRITE TIME under its frozen policy, and there is
        no synchronous loader for the historical mapping at this layer.
        """
        from outrider.policy.severity import (  # noqa: PLC0415
            ACTIVE_POLICY_VERSION,
            SEVERITY_POLICY,
        )

        if self.policy_version != ACTIVE_POLICY_VERSION:
            return self
        expected = SEVERITY_POLICY.get(self.finding_type)
        if expected is None or self.severity != expected:
            raise ValueError(
                f"PublishEligibilityEvent.severity={self.severity.value!r} does not match "
                f"SEVERITY_POLICY[{self.finding_type.value!r}]="
                f"{(expected.value if expected else None)!r} under policy_version "
                f"{self.policy_version!r}. Per `severity-set-by-policy`, baseline severity "
                f"comes from SEVERITY_POLICY keyed by finding_type, never from caller."
            )
        return self

    @model_validator(mode="after")
    def _enforce_v1_no_overrides(self) -> Self:
        """V1 publish ships BEFORE the `hitl` node is wired, so no legitimate
        HITL override path exists yet. A non-None `original_severity` on an
        eligibility event indicates either a producer bug or replay-injected
        state forging a pre-approved downgrade — reject at the schema layer
        so the audit row cannot lie.

        When the hitl-node spec lands, this validator relaxes to mirror
        `ReviewFinding._enforce_override_triplet_coherence` (`original_severity`
        is set iff override happened, with reviewer identity + reason). For
        now the schema's "all three or none" override triplet is "none"
        unconditionally.
        """
        if self.original_severity is not None:
            raise ValueError(
                f"PublishEligibilityEvent.original_severity={self.original_severity.value!r} "
                f"but V1 publish ships before the hitl node — no legitimate path produces "
                f"override fields. A non-None value indicates a producer bug or "
                f"replay-injected state forging a pre-approved downgrade; the eligibility "
                f"gate would normally withhold via `unexpected_override_fields_present`, "
                f"but the event-side validator rejects too so the audit row cannot lie."
            )
        return self

    @model_validator(mode="after")
    def _verify_content_hash_binding(self) -> Self:
        """MUST call `compute_finding_content_hash(...)` — never re-implement
        the recipe. Mirror of `FindingEvent._verify_content_hash` above.
        """
        expected = compute_finding_content_hash(
            self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            finding_type=self.finding_type,
        )
        if self.finding_content_hash != expected:
            raise ValueError(
                f"PublishEligibilityEvent.finding_content_hash={self.finding_content_hash!r} "
                f"does not match compute_finding_content_hash over "
                f"(file_path={self.file_path!r}, line_start={self.line_start}, "
                f"line_end={self.line_end}, finding_type={self.finding_type.value!r})="
                f"{expected!r}."
            )
        return self

    @model_validator(mode="after")
    def _verify_decision_content_hash(self) -> Self:
        """MUST call `compute_publish_eligibility_decision_hash(...)`. NOTE:
        `policy_version` is NOT in the hash — legitimate policy bumps must
        not surface as decision drift, so the dedup identity treats two
        policy versions with identical gate outcomes as one logical decision.
        """
        expected = compute_publish_eligibility_decision_hash(
            eligibility=self.eligibility,
            reason=self.reason,
        )
        if self.decision_content_hash != expected:
            raise ValueError(
                f"PublishEligibilityEvent.decision_content_hash={self.decision_content_hash!r} "
                f"does not match compute_publish_eligibility_decision_hash over "
                f"(eligibility={self.eligibility.value!r}, "
                f"reason={self.reason.value if self.reason else None!r})={expected!r}."
            )
        return self


class PublishAttemptEvent(AuditEventBase):
    """Records the terminal outcome of one publish-attempt to GitHub.

    Per Q2: single emission per attempt, AFTER the GitHub call resolves.
    No in_flight pre-call emission (would conflict with append-only
    audit semantics — same-event_id-different-payload raises rather than
    updates). Carries `attempt_content_hash` (which includes `outcome`)
    so consumer-side dedup distinguishes attempts with the same finding
    set but different outcomes (e.g., success-then-failed-replay).
    """

    event_type: Literal["publish_attempt"] = "publish_attempt"
    attempt_index: int = Field(ge=1)
    outcome: PublishAttemptOutcome
    status_code: int | None = None
    # Bounded to defend against attacker-influenced 422 response strings
    # being interpolated into `failure_class` by a producer bug (a
    # GitHub 422 body's `errors[].message` is attacker-controlled when
    # the PR author crafts an invalid request shape). Practical
    # `type(exc).__name__` strings are well under 128 chars; the cap
    # exists to bound append-only audit-row size on the pathological case.
    failure_class: Annotated[str, Field(max_length=128)] | None = None
    comments_attempted: int = Field(ge=0)
    sorted_finding_ids: tuple[UUID, ...] = ()
    attempt_content_hash: str = Field(pattern=_SHA256_HEX_PATTERN)

    @field_validator("sorted_finding_ids")
    @classmethod
    def _enforce_sorted_finding_ids(cls, ids: tuple[UUID, ...]) -> tuple[UUID, ...]:
        """`sorted_finding_ids` must be sorted AND unique at construction.

        Two constraints, both load-bearing for the attempt-content-hash
        recipe:

        1. **Sorted.** `compute_publish_attempt_content_hash` encodes
           the tuple positionally; an unsorted producer would compute
           a hash that diverges from the same logical attempt encoded
           in sorted order, producing two consumer-dedup rows for one
           logical attempt.

        2. **Set-semantic (no duplicates).** A tuple like `(a, a, b)`
           is already "sorted" but encodes a different hash than
           `(a, b)` for what is logically the same attempted-finding
           set. The publish node already de-duplicates upstream, but
           schema-level enforcement is the loud-failure floor that
           catches a future producer that drops the dedup.

        Enforced here rather than auto-coerced via `sorted(set(ids))`:
        silent coercion would mask the producer bug, defeating the
        loud-failure pattern documented for V1 audit-event factories.
        """
        if len(ids) != len(set(ids)):
            raise ValueError(
                "PublishAttemptEvent.sorted_finding_ids must not contain "
                "duplicate finding IDs; the field is set-semantic for "
                "attempt-content-hash determinism. Producer-side bug — "
                "de-duplicate the tuple before constructing the event."
            )
        if tuple(sorted(ids)) != ids:
            raise ValueError(
                "PublishAttemptEvent.sorted_finding_ids must be sorted "
                "at construction (per `compute_publish_attempt_content_hash` "
                "recipe). Producer-side bug — sort the tuple before "
                "constructing the event."
            )
        return ids

    @model_validator(mode="after")
    def _enforce_failure_class_required_when_failed(self) -> Self:
        """`failure_class` required iff `outcome == failed`. A successful
        attempt has no failure to record; a failed attempt must record one."""
        if self.outcome is PublishAttemptOutcome.FAILED and self.failure_class is None:
            raise ValueError("PublishAttemptEvent outcome=failed requires failure_class")
        if self.outcome is not PublishAttemptOutcome.FAILED and self.failure_class is not None:
            raise ValueError(
                f"PublishAttemptEvent outcome={self.outcome.value!r} must have "
                f"failure_class=None, got {self.failure_class!r}"
            )
        return self

    @model_validator(mode="after")
    def _verify_attempt_content_hash(self) -> Self:
        """MUST call `compute_publish_attempt_content_hash(...)` — recipe pinning
        for consumer-side dedup correctness."""
        expected = compute_publish_attempt_content_hash(
            review_id=self.review_id,
            attempt_index=self.attempt_index,
            sorted_finding_ids=self.sorted_finding_ids,
            outcome=self.outcome,
            status_code=self.status_code,
            failure_class=self.failure_class,
            comments_attempted=self.comments_attempted,
        )
        if self.attempt_content_hash != expected:
            raise ValueError(
                f"PublishAttemptEvent.attempt_content_hash={self.attempt_content_hash!r} "
                f"does not match compute_publish_attempt_content_hash="
                f"{expected!r}."
            )
        return self


# ---------------------------------------------------------------------------
# Analyze-foundation §5: three new event subclasses for the analyze node.
# Schema-only — emission semantics live in the analyze-implementation
# sister spec.
# ---------------------------------------------------------------------------


# Module-local alias for the canonical short pattern. Single source lives
# in `outrider.policy.canonical`; redefining the regex here would
# reintroduce the per-call-site drift class the chokepoint module exists
# to prevent.
_SHA256_HEX_PATTERN_SHORT: Final = SHA256_HEX_PATTERN_SHORT


class AnalyzeCompletedEvent(AuditEventBase):
    """Per-pass aggregate emitted at the end of each analyze ⇄ trace iteration.

    Counter fields are cross-validated by two model validators so a counter
    that lies (`n_findings_emitted=5` with only 3 findings actually fired)
    fails Pydantic construction, not just reads weird. Per §5 of
    `specs/2026-05-19-analyze-foundation.md` and
    """

    event_type: Literal["analyze_completed"] = "analyze_completed"
    pass_index: int = Field(ge=0)
    node_id: Literal["analyze"] = "analyze"
    n_files_analyzed: int = Field(ge=0)
    n_files_skipped: int = Field(ge=0)
    n_llm_calls: int = Field(ge=0)
    n_proposals_seen: int = Field(ge=0)
    n_findings_emitted: int = Field(ge=0)
    n_proposals_rejected: int = Field(ge=0)
    n_responses_rejected: int = Field(ge=0)
    n_trace_candidates_emitted: int = Field(ge=0)
    """Count of `TraceCandidate` records the parser emitted across this
    pass's per-file calls — pre-dedup. The state-side reducer on
    `ReviewState.trace_candidates` is `append_with_dedup_by(candidate_id)`
    (`schemas/review_state.py`), so the unique-candidate count visible
    to downstream consumers equals `len(state.trace_candidates_delta)`,
    not this counter. Keeping the counter pre-dedup preserves the audit
    signal "model proposed the same candidate twice" which would
    otherwise be invisible — a model emitting N duplicate candidate_ids
    indicates either prompt confusion or a model behavior worth tracking.
    Reviewers periodically suggest deduping here; that would erase the
    signal, hence the explicit contract."""
    n_trace_candidates_dropped_malformed: int = Field(ge=0, default=0)
    """Count of raw `trace_candidates` entries the parser DROPPED because
    `coordinates.is_valid_import_string` rejected the `import_string_raw`
    (sharp-edges F1 audit-fold per `specs/2026-05-23-trace-node.md` arc).
    Per `DECISIONS.md#024` trace candidates are dotted Python import
    strings; the parser silently drops malformed candidates (rather than
    crashing the whole pass) to preserve the n_proposals_seen accounting
    equation, but the count surfaces here so operators can distinguish
    "model proposed no trace candidates" from "every proposal was
    malformed" — drift in this metric over time signals prompt drift or
    a model retraining that warrants prompt-template review.
    Default=0 for backward compatibility with pre-fold AnalyzeCompletedEvent
    payloads (none exist in production yet — analyze hasn't shipped — so
    the default is a forward-compat hedge, not a backfill substitute)."""
    total_input_tokens: int = Field(ge=0)
    total_cache_read_tokens: int = Field(ge=0)
    """Sum of `LLMResponse.cache_read_tokens` across this pass's LLM calls.
    Cache reads bill at 0.1× the base input rate (per Anthropic's published
    pricing); kept separate from writes so a `total_cost_usd` recomputation
    from raw token counts can reconcile to the per-call `LLMCallEvent` sum.
    Matches `LLMCallEvent.cached_tokens` semantics (reads-only)."""
    total_cache_write_tokens: int = Field(ge=0)
    """Sum of `LLMResponse.cache_write_tokens` across this pass's LLM calls.
    Cache writes bill at 1.25× the base input rate. Separate from reads
    because the 12.5× cost differential is material — combining them
    obscures the cost driver. NOT mirrored on `LLMCallEvent` (which
    carries only `cached_tokens=reads`); `total_cache_write_tokens` is
    aggregate-event-only and the divergence is intentional, surfaced
    here so a reader doesn't expect `sum(LLMCallEvent.cached_tokens) ==
    total_cache_read_tokens + total_cache_write_tokens`."""
    total_output_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    pricing_version: str = Field(pattern=PRICING_VERSION_PATTERN)
    policy_version: str = Field(pattern=BARE_SEMVER_PATTERN)
    analyze_model: str

    @model_validator(mode="after")
    def _enforce_proposal_accounting(self) -> Self:
        """`n_proposals_seen == n_findings_emitted + n_proposals_rejected`.

        Every raw proposal either becomes a finding or gets rejected; total
        accounting must hold. Response-level rejections (`n_responses_rejected`)
        are separate — those don't have a proposal to count, so they don't
        enter this equation.
        """
        expected = self.n_findings_emitted + self.n_proposals_rejected
        if self.n_proposals_seen != expected:
            raise ValueError(
                f"Proposal accounting mismatch: n_proposals_seen={self.n_proposals_seen} "
                f"!= n_findings_emitted({self.n_findings_emitted}) + "
                f"n_proposals_rejected({self.n_proposals_rejected}) = {expected}. "
                f"Response-level rejections (n_responses_rejected={self.n_responses_rejected}) "
                f"do NOT enter this equation — only proposal-level rejections do. "
                f"If counting raw-response-unparseable cases, those increment "
                f"n_responses_rejected (separate)."
            )
        return self

    @model_validator(mode="after")
    def _enforce_response_accounting(self) -> Self:
        """`n_responses_rejected <= n_llm_calls`.

        Rejected responses are a subset of LLM calls: a response only
        exists if the call succeeded enough to return text. More
        rejected-responses than calls is incoherent.
        """
        if self.n_responses_rejected > self.n_llm_calls:
            raise ValueError(
                f"n_responses_rejected={self.n_responses_rejected} cannot exceed "
                f"n_llm_calls={self.n_llm_calls}; rejected responses are a subset of calls"
            )
        return self


class FindingProposalRejectedEvent(AuditEventBase):
    """Proposal-level rejection — one per model proposal that failed admission.

    Stores `claimed_finding_type_hash` (SHA-256 short prefix) + length
    rather than the raw model string per `DECISIONS.md#014` point 1:
    every model-originated value is hostile until validated, so audit
    rows must not carry user code or prompt/completion content.
    Cross-field validator pairs `claimed_evidence_tier` with the
    `evidence_tier_not_in_enum` reason bidirectionally.

    `proposal_hash` carries the PR/file-scoped `compute_proposal_hash`
    digest per `DECISIONS.md#022`. Two analyze passes over DIFFERENT
    source files that emit logically-identical proposals produce
    DISTINCT `proposal_hash` values — preserving per-source-file
    audit provenance on the join with `TraceCandidate.source_proposal_hash`.
    """

    event_type: Literal["finding_proposal_rejected"] = "finding_proposal_rejected"
    node_id: Literal["analyze"] = "analyze"
    file_path: Annotated[str, Field(max_length=1024)]
    proposal_hash: Annotated[str, Field(pattern=_SHA256_HEX_PATTERN)]
    claimed_evidence_tier: EvidenceTier | None = None
    claimed_finding_type_hash: Annotated[str, Field(pattern=_SHA256_HEX_PATTERN_SHORT)]
    claimed_finding_type_len: int = Field(ge=0, le=128)
    rejection_reason: Literal[
        "query_match_id_not_in_registry",
        "trace_path_not_admissible",
        "finding_type_not_in_enum",
        "evidence_tier_not_in_enum",
        "span_outside_scope_unit",
        "span_outside_file",
        "schema_construction_failed",
    ]
    rejection_detail: Annotated[str, Field(max_length=500)]

    @field_validator("file_path")
    @classmethod
    def _enforce_canonical_file_path(cls, path: str) -> str:
        """Audit-shadow `paths-validated-before-use`. Even rejected
        proposals carry a file_path that names the analyze target;
        a traversal-bearing path here would land in the audit log
        with no in-memory ReviewFinding to gate it.
        """
        return validate_diff_path(path)

    @model_validator(mode="after")
    def _enforce_claimed_evidence_tier_coupling(self) -> Self:
        """`claimed_evidence_tier is None` iff `rejection_reason == "evidence_tier_not_in_enum"`.

        Bidirectional rule per §5: when the model returned a tier value
        that didn't parse to `EvidenceTier`, there's no admitted tier to
        record (the field is None and the reason names that exact case).
        For ALL other rejection reasons, the model's claimed tier DID
        parse (the rejection happened on a different axis — bad
        query_match_id, bad span, etc.), so the parsed tier MUST be
        recorded.
        """
        is_tier_failure = self.rejection_reason == "evidence_tier_not_in_enum"
        tier_is_none = self.claimed_evidence_tier is None
        if is_tier_failure and not tier_is_none:
            raise ValueError(
                f"rejection_reason='evidence_tier_not_in_enum' requires "
                f"claimed_evidence_tier is None (the model's tier didn't parse); "
                f"got claimed_evidence_tier={self.claimed_evidence_tier!r}"
            )
        if (not is_tier_failure) and tier_is_none:
            raise ValueError(
                f"rejection_reason={self.rejection_reason!r} requires a non-None "
                f"claimed_evidence_tier (the model's tier parsed successfully on "
                f"this code path; rejection happened on a different axis)"
            )
        return self


class AnalyzeResponseRejectedEvent(AuditEventBase):
    """Response-level rejection — the LLM response failed to parse as `AnalyzeResponseRaw`.

    Distinct event from `FindingProposalRejectedEvent` because that event
    presupposes a proposal; no proposal exists when the raw response
    itself fails to parse. `response_hash` is the SHA-256 of the FULL
    raw response text encoded as UTF-8 bytes (no truncation prefix).
    Hash-only — no content leak per `DECISIONS.md#014`.
    """

    event_type: Literal["analyze_response_rejected"] = "analyze_response_rejected"
    node_id: Literal["analyze"] = "analyze"
    file_path: Annotated[str, Field(max_length=1024)]
    response_hash: Annotated[str, Field(pattern=_SHA256_HEX_PATTERN)]
    rejection_reason: Literal["raw_response_unparseable"]
    rejection_detail: Annotated[str, Field(max_length=500)]

    @field_validator("file_path")
    @classmethod
    def _enforce_canonical_file_path(cls, path: str) -> str:
        """Audit-shadow `paths-validated-before-use`. Same canonical
        gate as `FindingProposalRejectedEvent.file_path` above."""
        return validate_diff_path(path)


# Discriminated union for replay: TypeAdapter(AuditEvent).validate_python({...})
# selects the right concrete subtype using the event_type field.
AuditEvent = Annotated[
    AgentTransitionEvent
    | ReviewPhaseEvent
    | LLMCallEvent
    | FileExaminationEvent
    | FindingEvent
    | TraceDecisionEvent
    | HITLRequestEvent
    | HITLDecisionEvent
    | PublishEvent
    | PublishRoutingEvent
    | PublishEligibilityEvent
    | PublishAttemptEvent
    | AnalyzeCompletedEvent
    | FindingProposalRejectedEvent
    | AnalyzeResponseRejectedEvent,
    Field(discriminator="event_type"),
]

# Module-level TypeAdapter so callers don't have to construct one each time
# (TypeAdapter construction is comparatively expensive; reuse is the documented
# Pydantic V2 pattern).
AuditEventAdapter: Final[
    TypeAdapter[
        AgentTransitionEvent
        | ReviewPhaseEvent
        | LLMCallEvent
        | FileExaminationEvent
        | FindingEvent
        | TraceDecisionEvent
        | HITLRequestEvent
        | HITLDecisionEvent
        | PublishEvent
        | PublishRoutingEvent
        | PublishEligibilityEvent
        | PublishAttemptEvent
        | AnalyzeCompletedEvent
        | FindingProposalRejectedEvent
        | AnalyzeResponseRejectedEvent
    ]
] = TypeAdapter(AuditEvent)


__all__ = [
    "AgentTransitionEvent",
    "AnalyzeCompletedEvent",
    "AnalyzeResponseRejectedEvent",
    "AuditEvent",
    "AuditEventAdapter",
    "AuditEventBase",
    "ContextManifestEntry",
    "FileExaminationEvent",
    "FindingEvent",
    "FindingProposalRejectedEvent",
    "HITLDecisionEvent",
    "HITLRequestEvent",
    "LLMCallEvent",
    "PublishAttemptEvent",
    "PublishAttemptOutcome",
    "PublishEligibility",
    "PublishEligibilityEvent",
    "PublishEligibilityReason",
    "PublishEvent",
    "PublishRoutingEvent",
    "PublishRoutingReason",
    "ReviewPhaseEvent",
    "TraceDecisionEvent",
    "compute_publish_attempt_content_hash",
    "compute_publish_eligibility_decision_hash",
    "compute_publish_routing_decision_hash",
]
