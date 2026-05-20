# Audit event hierarchy per docs/spec.md §7.2.1 + §8.2.
# Append-only contract per docs/trust-boundaries.md §7.
"""Audit event class hierarchy + discriminated union.

`AuditEventBase` is the shared base; the ten V1 subtypes per spec §8.2 each
declare their own `event_type: Literal[...]` discriminator value. The
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

Four event types carry validators:

  - `FindingEvent` runs `policy/findings.enforce_proof_boundary` so the
    proof boundary holds at the audit-event layer, not just on
    `ReviewFinding`. Backs `evidence-tier-schema-enforced`.
  - `TraceDecisionEvent` enforces the three-rule resolution invariant
    per `DECISIONS.md#017` (Amended same-day, two clauses):
    (a) resolved ↔ non-None target_file;
    (b) unresolved/ambiguous ↔ target_file is None;
    (c) when resolved, target_file in candidates_considered.
  - `FileExaminationEvent` enforces the `skip_reason` cross-field rule
    per `DECISIONS.md#018`: `skip_reason is not None` ↔
    `parse_status == "skipped"`.
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
from typing import Annotated, Final, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

from outrider.ast_facts.models import SkipReason
from outrider.policy import (
    EvidenceTier,
    FindingSeverity,
    FindingType,
    enforce_proof_boundary,
)
from outrider.policy.canonical import SHA256_HEX_PATTERN
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

# Bare-semver pattern matching `outrider.policy.severity._SEMVER_RE` AND
# the DB CHECK constraint added by migration `3d03bca7f2be`. Audit events
# carrying `policy_version` apply this pattern so a bogus value (e.g.,
# "banana") fails at construction rather than landing in the append-only
# audit log and breaking replay reconstruction downstream. Foundation-
# wide adversarial audit M1, narrowed to policy_version only:
# `pricing_version` carries `PRICING_VERSION` which uses a distinct
# versioning scheme (`"v2"` not bare semver) per `llm/pricing.py`.
_BARE_SEMVER_PATTERN: Final = r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"


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
    payload = json.dumps(
        [file_path, line_start, line_end, finding_type.value],
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

    file_path: str
    scope_unit_name: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    inclusion_reason: Literal[
        "changed_scope",
        "same_file_context",
        "trace_expansion",
    ]

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
    node_id: str
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
    node_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    pricing_version: str
    latency_ms: int = Field(ge=0)
    prompt_hash: str = Field(pattern=_SHA256_HEX_PATTERN)
    cache_hit: bool
    context_summary: tuple[ContextManifestEntry, ...]
    prompt_template_version: str
    system_prompt_hash: str = Field(pattern=_SHA256_HEX_PATTERN)
    degraded_mode: bool
    # Per §0b of `specs/2026-05-19-analyze-foundation.md` + the audit's
    # convergent finding: `degraded_mode: bool` alone loses provenance on
    # metadata-only replay (post-retention or partial-content). The two
    # reasons (`parse_failed` vs `tree_has_error_in_changed_regions`)
    # imply structurally different prompt content; collapsing them into
    # the bool means audit-stream queries like "how many parse_failed
    # analyze calls did we make this month" become unanswerable. Same
    # bidirectional coupling as `LLMRequest.degradation_reason` enforced
    # by `_enforce_degradation_reason_consistency` below. Spec gap
    # surfaced by the §0b crazy-audit; landing in the same commit per
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
        re-validation (post-foundation audit: the read/replay boundary
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
    file_path: str
    examination_type: str
    node_id: str
    parse_status: Literal["clean", "degraded", "failed", "skipped"]
    skip_reason: SkipReason | None = None

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
    file_path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    dimension: ReviewDimension
    # SHA-256 hex per spec §8.5: SHA-256(file_path + line_start + line_end + finding_type)
    finding_content_hash: str = Field(pattern=_SHA256_HEX_PATTERN)
    evidence_tier: EvidenceTier
    query_match_id: str | None = None
    trace_path: tuple[str, ...] | None = None
    policy_version: str = Field(pattern=_BARE_SEMVER_PATTERN)

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


class HITLRequestEvent(AuditEventBase):
    """Records the HITL gate envelope at interrupt time."""

    event_type: Literal["hitl_request"] = "hitl_request"
    findings_requiring_approval: tuple[UUID, ...]
    auto_post_findings: tuple[UUID, ...]
    expires_at: AwareDatetime


class HITLDecisionEvent(AuditEventBase):
    """Records the reviewer's HITL submission.

    Field name `decisions` (not `per_finding_decisions`) matches the
    cross-boundary `HITLDecision.decisions` type per `DECISIONS.md#014`
    Amended 2026-04-29.
    """

    event_type: Literal["hitl_decision"] = "hitl_decision"
    reviewer_id: str
    decisions: tuple[PerFindingDecision, ...]
    decision_latency_seconds: float = Field(ge=0)


class PublishEvent(AuditEventBase):
    """Records the GitHub publish operation outcome."""

    event_type: Literal["publish"] = "publish"
    github_review_id: int = Field(ge=1)  # GitHub review IDs are positive integers
    comments_posted: int = Field(ge=0)
    review_status: str


class PublishRoutingEvent(AuditEventBase):
    """Records the per-finding routing decision; backs publish-routes-through-coordinates."""

    event_type: Literal["publish_routing"] = "publish_routing"
    finding_id: UUID
    destination: PublishDestination
    reason: Literal["reviewable_diff_line", "unchanged_region", "non_diffed_file"]


# ---------------------------------------------------------------------------
# Analyze-foundation §5: three new event subclasses for the analyze node.
# Schema-only — emission semantics live in the analyze-implementation
# sister spec.
# ---------------------------------------------------------------------------


# Short SHA-256-hex prefix pattern for hostile-string fingerprinting per
# `specs/2026-05-19-analyze-foundation.md` §5 (FindingProposalRejectedEvent).
# `_SHA256_HEX_PATTERN` matches the full 64-char digest; this matches the
# 16-char prefix used when storing `sha256(raw_value)[:16]` to dedup audit
# rows without leaking model-controlled raw values per `DECISIONS.md#014`.
_SHA256_HEX_PATTERN_SHORT: Final = r"^[a-f0-9]{16}$"


class AnalyzeCompletedEvent(AuditEventBase):
    """Per-pass aggregate emitted at the end of each analyze ⇄ trace iteration.

    Counter fields are cross-validated by two model validators so a counter
    that lies (`n_findings_emitted=5` with only 3 findings actually fired)
    fails Pydantic construction, not just reads weird. Per §5 of
    `specs/2026-05-19-analyze-foundation.md` and post-split audit S7.
    """

    event_type: Literal["analyze_completed"] = "analyze_completed"
    pass_index: int = Field(ge=0)
    node_id: str = "analyze"
    n_files_analyzed: int = Field(ge=0)
    n_files_skipped: int = Field(ge=0)
    n_llm_calls: int = Field(ge=0)
    n_proposals_seen: int = Field(ge=0)
    n_findings_emitted: int = Field(ge=0)
    n_proposals_rejected: int = Field(ge=0)
    n_responses_rejected: int = Field(ge=0)
    n_trace_candidates_emitted: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_cached_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    pricing_version: str
    policy_version: str = Field(pattern=_BARE_SEMVER_PATTERN)
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
    """

    event_type: Literal["finding_proposal_rejected"] = "finding_proposal_rejected"
    node_id: str = "analyze"
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
    raw response text encoded as UTF-8 bytes (no 8 KiB prefix per
    post-split audit S11). Hash-only — no content leak per
    `DECISIONS.md#014`.
    """

    event_type: Literal["analyze_response_rejected"] = "analyze_response_rejected"
    node_id: str = "analyze"
    file_path: Annotated[str, Field(max_length=1024)]
    response_hash: Annotated[str, Field(pattern=_SHA256_HEX_PATTERN)]
    rejection_reason: Literal["raw_response_unparseable"]
    rejection_detail: Annotated[str, Field(max_length=500)]


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
    "PublishEvent",
    "PublishRoutingEvent",
    "ReviewPhaseEvent",
    "TraceDecisionEvent",
]
