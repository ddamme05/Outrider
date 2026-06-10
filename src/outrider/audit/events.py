# Audit event hierarchy per docs/spec.md §7.2.1 + §8.2.
# Append-only contract per docs/trust-boundaries.md §7.
"""Audit event class hierarchy + discriminated union.

`AuditEventBase` is the shared base. The hierarchy has seventeen
concrete subtypes: twelve V1 subtypes per spec §8.2 (`AgentTransitionEvent`,
`ReviewPhaseEvent`, `LLMCallEvent`, `FileExaminationEvent`,
`FindingEvent`, `TraceDecisionEvent`, `HITLRequestEvent`,
`HITLDecisionEvent`, `PublishEvent`, `PublishRoutingEvent`,
`PublishEligibilityEvent`, `PublishAttemptEvent`), three
analyze-foundation additions (`AnalyzeCompletedEvent`,
`FindingProposalRejectedEvent`, `AnalyzeResponseRejectedEvent`), the
synthesize-node addition (`SynthesizeCompletedEvent`), and the
replay-verdict-projection addition (`ReplayVerdictEvent`). Each
declares its own `event_type: Literal[...]` discriminator value. The
`AuditEvent` discriminated-union alias is what `audit/replay.py` uses to
reconstruct concrete events from `audit_events.payload` JSONB at read time:

    TypeAdapter(AuditEvent).validate_python({**payload, "sequence_number": row.sequence_number})

Every event uses `ConfigDict(frozen=True, extra="forbid")` per
`audit-events-frozen-extra-forbid`. Tuple-typed sequence fields
(`context_summary`, `trace_path`, the HITL containers,
`proposed_import_strings`, `resolved_candidate_paths`)
deliver true immutability — Pydantic `frozen=True` only blocks attribute
reassignment, not in-place container mutation. Nested Pydantic payload
classes (`ContextManifestEntry`) carry their own `frozen=True + extra=forbid`
because the outer model's frozen-ness does not propagate.

Seven event types carry validators (plus `PerFindingDecision` inherited
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
    per `DECISIONS.md#017` × `#024` amendment:
    (a) resolved ↔ `len(resolved_candidate_paths) == 1` AND
        `target_file == resolved_candidate_paths[0]`;
    (b) unresolved ↔ `len(resolved_candidate_paths) == 0` AND
        `target_file is None`;
    (c) ambiguous ↔ `len(resolved_candidate_paths) > 1` AND
        `target_file is None`.
    Plus per-element `is_valid_import_string` on `proposed_import_strings`
    and `validate_diff_path` on `resolved_candidate_paths` + `target_file`.
  - `AnalyzeCompletedEvent` enforces two accounting equations per
    foundation §5: `n_proposals_seen == n_findings_emitted +
    n_proposals_rejected`, and `n_responses_rejected <= n_llm_calls`.
  - `FindingProposalRejectedEvent` enforces the bidirectional
    `claimed_evidence_tier` ↔ `rejection_reason ==
    "evidence_tier_not_in_enum"` coupling.
  - `ReplayVerdictEvent` carries two validators: `reason` paired iff
    inequivalent, and the reconstruction-metadata envelope (`mode` + the
    three counts) all-present-or-all-absent — present iff reconstruction
    succeeded, so an equivalent verdict requires it.
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
    ValidationInfo,
    field_validator,
    model_validator,
)

from outrider.ast_facts.models import SkipReason
from outrider.coordinates import is_valid_import_string, validate_diff_path
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
    RiskLevel,
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

# Reserved replay sentinel — see DECISIONS.md#032. The all-zero SHA-256 is
# RESERVED to mark a `proposal_hash` that was absent on a persisted pre-#025
# historical event and defaulted by the read-side replay normalizer
# (`audit/replay.py::_normalize_historical_payload`). It is pattern-valid hex,
# NOT impossible — so write-time validators on every real `proposal_hash` field
# reject it, making it unambiguous by construction (no real event can carry it).
# Single-sourced here so the reserve (these validators) and the inject (replay)
# reference the same constant and cannot drift.
RESERVED_HISTORICAL_PROPOSAL_HASH: Final = "0" * 64

# Validation-context key the replay normalizer sets so the reserved-sentinel
# guards PERMIT the sentinel during historical reconstruction while still
# REJECTING it at every normal write (no context). Single-sourced so replay and
# the guards agree on the flag name. See DECISIONS.md#032.
REPLAY_HISTORICAL_CONTEXT_KEY: Final = "replay_historical"

# The (event_type, field) pairs whose reserved sentinel the replay normalizer
# may inject — and which the write-time guards therefore PERMIT under replay
# context. The permission is pair-scoped, NOT context-wide (DECISIONS.md#032):
# a sentinel on an UNREGISTERED pair (e.g. finding_proposal_rejected) stays a
# loud failure even at replay, because the normalizer never injects it there, so
# its presence would be corruption. Must equal the key set of replay's
# `_HISTORICAL_FIELD_DEFAULTS` — pinned by the registry-allowlist test.
REPLAY_TOLERABLE_SENTINEL_FIELDS: Final = frozenset({("finding", "proposal_hash")})


def _sentinel_permitted(info: ValidationInfo, event_type: str, field: str) -> bool:
    """True only under the replay context AND for a registered tolerable pair.
    Unregistered pairs are rejected even under replay context (DECISIONS.md#032).
    """
    context = info.context
    if not (context and context.get(REPLAY_HISTORICAL_CONTEXT_KEY)):
        return False
    return (event_type, field) in REPLAY_TOLERABLE_SENTINEL_FIELDS


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
    """Node-to-node transition in the LangGraph state machine.

    `from_node` and `to_node` are constrained Literals over the seven
    graph nodes plus `"webhook"` — the latter is the seed source for
    the FIRST transition in a review (webhook → intake), emitted by
    `api/webhooks/router.py` when the review row is created. Every
    subsequent transition is between two graph nodes.

    Sharing the Literal with `ReviewPhaseEvent.node_id` (graph nodes
    only — webhook doesn't have a phase) plus the seed exception stops
    a typo (`"analyse"`, `"sythesize"`) from landing in the append-only
    audit log without admitting arbitrary strings.
    """

    event_type: Literal["agent_transition"] = "agent_transition"
    from_node: Literal[
        "webhook", "intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"
    ]
    to_node: Literal["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"]
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
    # replay (post-retention or partial-content). The distinct reasons
    # (the `_DegradationReason` literals) imply structurally different
    # prompt content; collapsing them into
    # the bool means audit-stream queries like "how many parse_failed
    # analyze calls did we make this month" become unanswerable. Same
    # bidirectional coupling as `LLMRequest.degradation_reason` enforced
    # by `_enforce_degradation_reason_consistency` below. Modeled as an
    # enum rather than a bool because dropping the reason would corrupt
    # replay reconstruction.
    # `"tree_has_error_no_scope"` added per DECISIONS.md#033, in lockstep with
    # `LLMRequest.degradation_reason` and `_DegradationReason` (degradation.py).
    degradation_reason: (
        Literal[
            "parse_failed",
            "tree_has_error_in_changed_regions",
            "tree_has_error_no_scope",
        ]
        | None
    ) = None

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
    # Bounded to the two actually-emitted values today: intake's
    # per-file fetch record (`emit_file_examination` sites in
    # `agent/nodes/intake.py`) and analyze's per-file examination (the
    # `_emit_skip` / `_emit_examined` helpers in `agent/nodes/analyze.py`).
    # Adding a third stage's emission site is a schema change — the
    # explicit Literal forces the discriminator-like field to be
    # widened deliberately rather than drifting on a string typo from
    # a future contributor.
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

    @model_validator(mode="after")
    def _enforce_examination_type_matches_node(self) -> Self:
        """`examination_type` is bound to the emitting `node_id`: intake
        emits `intake_fetch`; analyze emits `analyze`. Without this
        cross-field rule, the two independent Literals admit
        contradictory combinations (`node_id="intake"` +
        `examination_type="analyze"`, or the reverse) — a self-
        contradictory row would land in the append-only log.
        Tightening the discriminator-pair here keeps the audit
        contract honest: the LP-emitter and the per-emitter stage
        always agree.
        """
        valid_pairs = {("intake", "intake_fetch"), ("analyze", "analyze")}
        if (self.node_id, self.examination_type) not in valid_pairs:
            raise ValueError(
                f"FileExaminationEvent: examination_type="
                f"{self.examination_type!r} is not valid for "
                f"node_id={self.node_id!r}. Valid pairs: "
                f"intake/intake_fetch, analyze/analyze."
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

    @field_validator("proposal_hash")
    @classmethod
    def _reject_reserved_proposal_hash(cls, value: str, info: ValidationInfo) -> str:
        """Reserve the all-zero sentinel for the replay normalizer (see
        DECISIONS.md#032). A real event must never carry
        `RESERVED_HISTORICAL_PROPOSAL_HASH`, so its appearance on a
        reconstructed event unambiguously means "pre-#025 historical event,
        provenance defaulted" rather than a genuine hash. The replay
        normalizer injects it under `REPLAY_HISTORICAL_CONTEXT_KEY`, the only
        place the sentinel is permitted — and only for this registered pair.
        """
        if value == RESERVED_HISTORICAL_PROPOSAL_HASH and not _sentinel_permitted(
            info, "finding", "proposal_hash"
        ):
            raise ValueError(
                "proposal_hash must not be the reserved all-zero sentinel "
                "(reserved for replay of pre-#025 historical events; see DECISIONS.md#032)"
            )
        return value

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
    """One aggregate trace decision per source_finding_id (per DECISIONS.md#017,
    amended by DECISIONS.md#024 — Accepted 2026-05-24).

    Cross-field validator rules per #017 × #024:
    (a) resolved → len(resolved_candidate_paths) == 1 AND
        target_file == resolved_candidate_paths[0]
    (b) unresolved → len(resolved_candidate_paths) == 0 AND target_file is None
    (c) ambiguous → len(resolved_candidate_paths) > 1 AND target_file is None

    Two parallel tuples:
    - `proposed_import_strings`: the LLM-proposed dotted Python import strings
      (any cardinality). Per DECISIONS.md#024 trace candidates are import
      strings, not file paths.
    - `resolved_candidate_paths`: the resolution outputs — file paths
      the import strings resolved to (any cardinality, including zero /
      one / multiple). V1 source per M8: GitHub fetch-probes (paths
      whose `fetch_file_content_at` returned content). V1.5+ source:
      filesystem-aware `coordinates.resolve_candidate_paths` (per
      `DECISIONS.md#024` point 4 Amended 2026-05-24). Each element is
      post-`validate_diff_path` per the audit-shadow rule (defense in
      depth at the append-only log against a hypothetical future
      direct emitter bypassing the resolution mechanism).

    `resolution_status` describes how many resolved (zero / exactly one /
    multiple). `target_file`, when non-None, ALSO passes through
    `validate_diff_path` at the audit-event boundary. All tuples
    required (no default) per #017 — defaults would silently absorb
    emitter bugs and undermine §8.7 replay equivalence; callers pass
    `()` explicitly for the zero-candidate case.
    """

    event_type: Literal["trace_decision"] = "trace_decision"
    source_finding_id: UUID
    # max_length=1024 mirrors `AnalysisRound.files_examined` /
    # `FileExaminationEvent.file_path` / `TraceFetchedFile.path` —
    # the path-bearing audit surfaces all cap at 1024. Without the
    # cap, a direct emitter constructing a TraceDecisionEvent could
    # push an unbounded path through the audit boundary even though
    # `validate_diff_path` rejects traversal / shell-metas / etc.
    target_file: Annotated[str, Field(max_length=1024)] | None
    reason: str = Field(max_length=500)
    resolution_status: Literal["resolved", "unresolved", "ambiguous"]
    # Per-element max_length on the tuples mirrors the singleton-field
    # bounds: `proposed_import_strings` matches
    # `TraceCandidate.import_string` (max_length=1024);
    # `resolved_candidate_paths` matches the same 1024-cap as
    # `target_file` above. The per-element validators
    # (`_enforce_canonical_*` below) already NFC-normalize and
    # `validate_diff_path` each entry, but Pydantic's per-element
    # length constraint is a separate gate that fires BEFORE the
    # validators — defense in depth at the append-only audit boundary.
    # Outer-container caps mirror the state-layer twin
    # `TraceDecision.proposed_import_strings` /
    # `.resolved_candidate_paths` at `schemas/trace_decision.py:80-87`.
    # Module docstring there declares "same fields and same cross-field
    # validators so a producer cannot construct a TraceDecision that
    # would fail validation when lifted to the audit event" — the cap
    # belongs on both sides for the lockstep contract to hold. 32
    # matches `ReviewFinding.trace_path` at
    # `schemas/review_finding.py:164`.
    proposed_import_strings: Annotated[
        tuple[Annotated[str, Field(max_length=1024)], ...],
        Field(max_length=32),
    ]
    resolved_candidate_paths: Annotated[
        tuple[Annotated[str, Field(max_length=1024)], ...],
        Field(max_length=32),
    ]
    trace_path: tuple[str, ...] | None = None

    @field_validator("target_file")
    @classmethod
    def _enforce_canonical_target_file(cls, value: str | None) -> str | None:
        """Audit-shadow `validate_diff_path` on the target_file when non-None.
        Per DECISIONS.md#024 point 6: even though the resolver produces safe
        repo-relative paths, the audit-event schema must shadow the
        boundary the same way other path-bearing events do (defense in
        depth against a hypothetical future direct emitter bypassing
        the resolver). None passes through unchanged (the validator only
        canonicalizes when there's a path to canonicalize).
        """
        if value is None:
            return None
        return validate_diff_path(value)

    @field_validator("resolved_candidate_paths")
    @classmethod
    def _enforce_canonical_resolved_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        """Per-element audit-shadow `validate_diff_path` on every
        resolved candidate path + deterministic sort. Per DECISIONS.md#024
        point 6: load-bearing for the ambiguous branch where target_file
        is None but the tuple carries multiple resolver-output paths
        that otherwise enter audit storage unvalidated. Sorted ordering
        keeps the stored row bytes canonical so dashboard reads + replay
        reconstructors don't see resolver-probe-order noise (probe order
        is non-deterministic). The field is NOT in
        `_SET_SEMANTIC_IDENTITY_FIELDS` per `DECISIONS.md#026` (point 3:
        excluded from `trace_decision`'s identity subset because it is
        derived from LLM-ranking-order-variant `proposed_import_strings`),
        so this sort exists for on-disk readability only — not for
        natural-key identity comparison.
        """
        return tuple(sorted(validate_diff_path(p) for p in paths))

    @field_validator("proposed_import_strings")
    @classmethod
    def _enforce_canonical_proposed_import_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Per-element audit-shadow `is_valid_import_string` on every
        LLM-proposed import string + deterministic sort. Defense in depth
        against a direct emitter (replay path, test fixture, future
        callsite) bypassing `TraceCandidate.import_string`'s field
        validator. The upstream singleton field validates one import
        string at construction; this tuple gathers many, and the audit
        boundary must rerun the same validation so malformed /
        non-canonical strings cannot persist into `audit_events`.
        Sorted ordering matches `_SET_SEMANTIC_IDENTITY_FIELDS` treating
        this field as set-semantic at persister-identity compare —
        canonical bytes on disk + identity-stable across LLM proposal
        ordering noise.
        """
        return tuple(sorted(is_valid_import_string(value) for value in values))

    @model_validator(mode="after")
    def _enforce_resolution_invariants(self) -> Self:
        """Three rules per DECISIONS.md#017 × #024 amendment (point 5).
        Consults `resolved_candidate_paths` cardinality (not the old
        `candidates_considered` membership) and asserts `target_file`
        matches the single resolved path when resolved.
        """
        n_resolved = len(self.resolved_candidate_paths)
        if self.resolution_status == "resolved":
            if n_resolved != 1:
                raise ValueError(
                    f"resolved TraceDecisionEvent requires exactly one "
                    f"resolved_candidate_paths entry; got {n_resolved}"
                )
            if self.target_file is None:
                raise ValueError("resolved TraceDecisionEvent requires non-None target_file")
            if self.target_file != self.resolved_candidate_paths[0]:
                raise ValueError(
                    f"resolved target_file ({self.target_file!r}) must equal the "
                    f"single resolved_candidate_paths entry "
                    f"({self.resolved_candidate_paths[0]!r})"
                )
        elif self.resolution_status == "unresolved":
            if n_resolved != 0:
                raise ValueError(
                    f"unresolved TraceDecisionEvent requires zero "
                    f"resolved_candidate_paths entries; got {n_resolved}"
                )
            if self.target_file is not None:
                raise ValueError("unresolved TraceDecisionEvent requires target_file is None")
        else:  # ambiguous
            if n_resolved <= 1:
                raise ValueError(
                    f"ambiguous TraceDecisionEvent requires more than one "
                    f"resolved_candidate_paths entry; got {n_resolved}"
                )
            if self.target_file is not None:
                raise ValueError("ambiguous TraceDecisionEvent requires target_file is None")
        return self

    @model_validator(mode="after")
    def _enforce_proposed_import_strings_unique(self) -> Self:
        """`proposed_import_strings` is set-semantic: each LLM-proposed
        import string is one consideration, not many. Duplicates would
        let the same logical trace decision hash differently (any future
        content-derived id over this field) and confuse audit-stream
        consumers. Per #024 amendment to #017's uniqueness validator —
        split into two (one per tuple)."""
        if len(self.proposed_import_strings) != len(set(self.proposed_import_strings)):
            raise ValueError(
                f"TraceDecisionEvent.proposed_import_strings contains duplicates: "
                f"{sorted(self.proposed_import_strings)!r}"
            )
        return self

    @model_validator(mode="after")
    def _enforce_resolved_candidate_paths_unique(self) -> Self:
        """`resolved_candidate_paths` is set-semantic: each resolved
        candidate is one resolution outcome, not many. Per #024
        amendment to #017's uniqueness validator — split into two."""
        if len(self.resolved_candidate_paths) != len(set(self.resolved_candidate_paths)):
            raise ValueError(
                f"TraceDecisionEvent.resolved_candidate_paths contains duplicates: "
                f"{sorted(self.resolved_candidate_paths)!r}"
            )
        return self


class HITLRequestEvent(AuditEventBase):
    """Records the HITL gate envelope at interrupt time.

    Audit-shadow mirror of `HITLRequest`: set-semantic partition of
    findings across the two tuples + the deterministic timestamp pair
    (`created_at`, `expires_at`) derived from `state.received_at`. The
    natural-key idempotency on `(review_id)` requires the producer-side
    derivation to be stable across body re-runs; both timestamps appear
    in `_IDENTITY_SUBSETS["hitl_request"]` so drift in either derivation
    surfaces as a persister conflict.
    """

    event_type: Literal["hitl_request"] = "hitl_request"
    findings_requiring_approval: tuple[UUID, ...]
    auto_post_findings: tuple[UUID, ...]
    created_at: AwareDatetime
    expires_at: AwareDatetime

    @field_validator("findings_requiring_approval", "auto_post_findings", mode="after")
    @classmethod
    def _canonicalize_finding_tuple(cls, v: tuple[UUID, ...]) -> tuple[UUID, ...]:
        """Sort the tuple deterministically by `str(UUID)` so semantically-
        equal sets (same membership, different ordering) produce
        byte-identical persisted payloads.

        Canonical ordering matters at TWO layers:

          1. Persister-side natural-key identity-subset comparison
             (`_SET_SEMANTIC_IDENTITY_FIELDS`) already treats these
             tuples as sets, so order-only divergence does NOT trigger
             spurious conflicts. This validator covers the OTHER half:
             downstream consumers (dashboard reads, replay
             reconstructors, third-party audit-log readers) that
             rely on raw tuple iteration would see ordering drift
             without it.
          2. The canonical body upstream (`_partition_findings` in
             `agent/nodes/hitl.py`) already sorts; this validator is
             defense-in-depth for direct-construction paths (replay,
             tests, future call sites).

        Sort by `str(UUID)` (lexicographic on the canonical 8-4-4-4-12
        hex form) for stability across Python versions; UUID's own
        ordering is well-defined but documenting the canonical form
        explicitly avoids subtle bugs if the field type ever changes.
        """
        return tuple(sorted(v, key=str))

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

    Canonical stream record of HITL override provenance (`decisions[*]` by
    `finding_id` + `reviewer_id`); the `findings`-table override columns are
    read-model projections of it. See DECISIONS.md#034.

    Field name `decisions` (not `per_finding_decisions`) matches the
    cross-boundary `HITLDecision.decisions` type per `DECISIONS.md#014`
    Amended 2026-04-29.

    Carries the full `HITLDecision` field set so audit-only replay
    reconstructs the state-layer object: `reviewer_id`, `decisions`,
    `annotation`, `decided_at`. `decision_latency_seconds` is a derived
    metric distinct from `decided_at` (the canonical time field for state
    reconstruction). `decisions_content_hash` is the audit-shadow of
    `compute_hitl_decision_content_hash(decisions, annotation)`; the
    persister's natural-key idempotency on `(review_id)` reads this hash
    via the `_IDENTITY_SUBSETS["hitl_decision"]` registry — divergent
    submissions for the same review fail loudly at the persister.
    """

    event_type: Literal["hitl_decision"] = "hitl_decision"
    # GitHub usernames are <=39 chars; SSO logins or future auth sources
    # might be longer, so 100 is generous-but-bounded. Without the cap a
    # malformed or attacker-supplied reviewer id could fill the audit row
    # arbitrarily and break replay aggregations keyed by reviewer. V1
    # operator scope (per DECISIONS.md#011) means `reviewer_id` is
    # server-set to the literal `"admin"`; the endpoint refuses any
    # body-supplied `reviewer_id` via `HITLDecisionPayload.extra="forbid"`.
    reviewer_id: str = Field(max_length=100)
    decisions: tuple[PerFindingDecision, ...]
    # Forensic note attached at submit time; bounded to keep the audit
    # row size reasonable. `None` means the reviewer attached no note.
    # Hashed alongside `decisions` per
    # `compute_hitl_decision_content_hash` so two decisions with
    # identical per-finding decisions but different annotations remain
    # logically distinct under the natural-key identity subset.
    annotation: str | None = Field(default=None, max_length=2000)
    decided_at: AwareDatetime
    decision_latency_seconds: float = Field(ge=0)
    # Content-derived audit-shadow of `compute_hitl_decision_content_hash`.
    # The persister's natural-key idempotency on `(review_id)` includes
    # this field via `_IDENTITY_SUBSETS["hitl_decision"]`; divergent
    # `decisions_content_hash` on a re-emit raises
    # `AuditPersisterHITLDecisionNaturalKeyConflict`.
    decisions_content_hash: str = Field(pattern=_SHA256_HEX_PATTERN)

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

    @model_validator(mode="after")
    def _enforce_decisions_content_hash(self) -> Self:
        """`decisions_content_hash` MUST equal
        `compute_hitl_decision_content_hash(decisions, annotation)`.

        Defense against forged or producer-mismatched audit rows: the
        canonical recipe is single-sourced in `policy/canonical.py`;
        an event constructed with a hand-typed hash that doesn't match
        the content raises here at construction time, BEFORE the row
        reaches the persister's natural-key check. Pinned by
        `test_audit_events.py`'s validator test.
        """
        from outrider.policy.canonical import (  # noqa: PLC0415
            compute_hitl_decision_content_hash,
        )

        expected = compute_hitl_decision_content_hash(
            decisions=self.decisions,
            annotation=self.annotation,
        )
        if self.decisions_content_hash != expected:
            raise ValueError(
                f"HITLDecisionEvent.decisions_content_hash mismatches "
                f"compute_hitl_decision_content_hash(decisions, annotation): "
                f"got {self.decisions_content_hash!r}, expected {expected!r}"
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
    """Why a finding was withheld at the eligibility gate.

    The HITL node landed per `specs/2026-05-26-hitl-node.md`; the gate now
    consults `hitl_decision` for CRITICAL/HIGH severities. Pre-HITL
    reasons (`hitl_required_node_absent`, `unexpected_override_fields_present`,
    `routing_emission_failed`) remain in the enum: the first as a
    defense-in-depth signal for the bypass case where publish runs
    without HITL having gated (graph wiring bug); the second as a
    fabricated-override defense; the third as the routing-emission
    recovery path. New HITL-driven reasons cover REJECT / SUPPRESS /
    no-matching-decision outcomes.
    """

    # severity ∈ {CRITICAL, HIGH} and `hitl` node did not run for this
    # review (state.hitl_request is None). Defense-in-depth — the graph
    # always routes through HITL post-trace/analyze; reaching this branch
    # indicates a wiring bypass.
    HITL_REQUIRED_NODE_ABSENT = "hitl_required_node_absent"

    # finding carries `original_severity is not None` despite no matching
    # SEVERITY_OVERRIDE decision being present in the HITL decision set —
    # defends against producer bugs or replay-injected state forging a
    # pre-approved downgrade.
    UNEXPECTED_OVERRIDE_FIELDS_PRESENT = "unexpected_override_fields_present"

    # Per-finding `try/except` in the publish node's routing+eligibility
    # interleaved loop caught an exception from `emit_publish_routing`;
    # the eligibility event still fires (withheld) so the per-finding
    # audit contract holds even when routing emission fails.
    ROUTING_EMISSION_FAILED = "routing_emission_failed"

    # severity ∈ {CRITICAL, HIGH}, HITL request landed but no decision
    # arrived (e.g., publish reached via a future graph path bypassing
    # the resume step, OR no matching PerFindingDecision for this
    # finding_id in the submitted decision set — defense-in-depth
    # against an endpoint mismatch check that missed something).
    HITL_DECISION_MISSING = "hitl_decision_missing"

    # severity ∈ {CRITICAL, HIGH}, HITL decision landed and the
    # reviewer's outcome for this finding was REJECT — the finding is
    # withheld from GitHub per the reviewer's explicit decision.
    HITL_REJECTED = "hitl_rejected"

    # severity ∈ {CRITICAL, HIGH}, HITL decision landed and the
    # reviewer's outcome for this finding was SUPPRESS — same
    # withhold semantic as REJECT but signals the finding is a
    # known false-positive class the reviewer wants tagged forensically.
    HITL_SUPPRESSED = "hitl_suppressed"


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
    recovered_github_review_id: int | None,
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
    - `recovered_github_review_id` is **integrity-protecting** for the
      IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD outcome: this is the only
      audit row that binds the recovered github_review_id (no paired
      PublishEvent lands on that path per DECISIONS.md#023 Amended
      2026-05-27), so a forged/replay emit could swap the id and still
      pass `_verify_attempt_content_hash` without this inclusion —
      breaking the audit-only recovery contract. For every other
      outcome the field is None; JSON encodes `null` distinct from
      any int, so absence is itself a hash-distinguishing value (same
      shape as `status_code` + `failure_class`).

    `status_code` + `failure_class` + `recovered_github_review_id`
    are all nullable; JSON encodes `None` → `null` distinct from
    any string/int, so the absence is itself a hash-distinguishing
    value.
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
            recovered_github_review_id,
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

    `policy_version` mirrors `finding.policy_version` (the snapshot under
    which the finding's severity was classified) per
    DECISIONS.md#028-per-review-policy-version-snapshot-anchor-on-triageresult.
    Stamping the live process-current `ACTIVE_POLICY_VERSION` would break
    `severity-policy-versioned-for-replay` across HITL-pause-then-deploy
    boundaries.

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
    # Post-HITL ship (per `DECISIONS.md#023` Amended 2026-05-27): non-None
    # `original_severity` is admissible when the gated finding carries a
    # reviewer-issued `PerFindingDecision(outcome=SEVERITY_OVERRIDE)` — that
    # IS the legitimate override path. Non-None WITHOUT a matching
    # HITLDecision override is a producer bug or replay-injected forged
    # downgrade. Enforced at the gate (`is_eligible_for_v1_publish` returns
    # withheld with `unexpected_override_fields_present` when override
    # fields are present but unauthorized) and validated here on the event
    # via `_enforce_override_legitimacy` below.
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
        # Compute baseline per the post-HITL convention (mirror of
        # `ReviewFinding._enforce_severity_matches_policy`):
        #   - When override is in effect: `severity` carries the
        #     override; `original_severity` carries the baseline.
        #   - When no override: `severity` IS the baseline.
        baseline = self.original_severity if self.original_severity is not None else self.severity
        expected = SEVERITY_POLICY.get(self.finding_type)
        if expected is None or baseline != expected:
            raise ValueError(
                f"PublishEligibilityEvent.severity baseline={baseline.value!r} does not match "
                f"SEVERITY_POLICY[{self.finding_type.value!r}]="
                f"{(expected.value if expected else None)!r} under policy_version "
                f"{self.policy_version!r}. Per `severity-set-by-policy`, baseline severity "
                f"comes from SEVERITY_POLICY keyed by finding_type, never from caller. "
                f"If a HITL override is in effect, set `original_severity` to the policy "
                f"baseline and put the override on `severity`."
            )
        return self

    @model_validator(mode="after")
    def _enforce_override_legitimacy(self) -> Self:
        """`original_severity` is set iff a HITL `SEVERITY_OVERRIDE`
        decision is in effect for this finding. Backs
        `severity-set-by-policy` + `hitl-gates-high-severity`.

        Semantics post-HITL (mirror of `ReviewFinding`'s convention):

          - When override is in effect:
            `severity` carries the OVERRIDE value (reviewer's choice);
            `original_severity` carries the POLICY BASELINE (what
            SEVERITY_POLICY would map this finding_type to under the
            event's `policy_version`).
          - When no override:
            `severity` carries the policy baseline;
            `original_severity` is None.

        The override REASON + reviewer identity live on the paired
        `HITLDecisionEvent.decisions[i]` (joined on `finding_id`). The
        publish-eligibility event records only the override SIGNAL +
        baseline so replay can reconstruct the "what severity did the
        published comment show" answer from publish-event alone, and
        the "who authorized + why" answer by joining to the HITL
        decision event.

        Pre-HITL audit rows (`original_severity=None`) remain valid:
        they encode "no override was in effect", which is correct for
        all rows written before the HITL node landed.
        """
        # A real override implies `severity != original_severity` (the
        # reviewer's choice differs from the policy baseline). When
        # both are equal AND original_severity is set, the row claims
        # "an override happened" but the applied severity matches the
        # baseline — semantically a no-op override and very likely a
        # producer bug. Reject loudly so the schema doesn't admit fake
        # overrides that pass policy-baseline checks at
        # `_enforce_severity_matches_policy` (which only requires
        # `baseline == SEVERITY_POLICY[finding_type]`, not that
        # `baseline != severity`).
        if self.original_severity is not None and self.original_severity == self.severity:
            raise ValueError(
                f"PublishEligibilityEvent claims an override "
                f"(original_severity={self.original_severity.value!r}) but "
                f"severity matches original_severity. A real override "
                f"requires severity != original_severity. If no override "
                f"is in effect, set original_severity=None."
            )
        # Gated-set defense: HITL only fires for CRITICAL/HIGH per
        # `_V1_SEVERITY_GATE`. A baseline severity outside the gated
        # set could not have produced a legitimate SEVERITY_OVERRIDE
        # decision — therefore `original_severity ∉ gated set` is a
        # producer bug or replay-injected forge. Shares the gated-set
        # definition with the runtime gate via `is_hitl_gated_severity`.
        if self.original_severity is not None:
            from outrider.policy.publish_eligibility import (  # noqa: PLC0415
                is_hitl_gated_severity,
            )

            if not is_hitl_gated_severity(self.original_severity):
                raise ValueError(
                    f"PublishEligibilityEvent.original_severity="
                    f"{self.original_severity.value!r} is not a HITL-gated "
                    f"severity. HITL only fires for CRITICAL/HIGH, so a "
                    f"baseline outside that set cannot have produced a "
                    f"legitimate SEVERITY_OVERRIDE. If no override is in "
                    f"effect, set original_severity=None."
                )
        # Eligibility-coherence defense: a non-None `original_severity`
        # records that a SEVERITY_OVERRIDE was applied to the published
        # comment. WITHHELD rows by definition never published, so a
        # WITHHELD row carrying `original_severity` is a producer-bug or
        # replay-injected forge — the publish path's
        # `_resolve_effective_severity` returns `original_severity_for_
        # audit=None` whenever no matching `PerFindingDecision(outcome=
        # SEVERITY_OVERRIDE)` exists. Reject loudly so the schema doesn't
        # admit override metadata on rows that never reached GitHub.
        if self.original_severity is not None and self.eligibility is PublishEligibility.WITHHELD:
            raise ValueError(
                f"PublishEligibilityEvent.original_severity="
                f"{self.original_severity.value!r} set on a WITHHELD row "
                f"(reason={self.reason.value if self.reason else None!r}). "
                f"Override metadata records a SEVERITY_OVERRIDE applied at "
                f"publish time; a WITHHELD row never published, so the "
                f"override claim has no referent. If no override is in "
                f"effect, set original_severity=None."
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
    # The canonical GitHub review id recovered via body-marker scan when
    # `outcome == IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD` (the crash-after-
    # success-before-emit recovery path at publish.py Step 6). Required
    # for that outcome AND forbidden for every other outcome — the
    # external-record skip is the only path that recovers a github review
    # id without emitting a paired PublishEvent. Without this field,
    # audit-only replay for the recovery path loses the github_review_id
    # binding (PublishAttemptEvent alone otherwise carries no github
    # identifier, and PublishEvent is the canonical review-summary record
    # per `DECISIONS.md#023`). INCLUDED in
    # `compute_publish_attempt_content_hash` + verified at
    # `_verify_attempt_content_hash` — the IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD
    # outcome is the only audit row that binds the recovered
    # github_review_id (no paired PublishEvent on that path), so a
    # forged/replay emit could swap the id and still pass the hash check
    # without inclusion; integrity-protecting binding closes that surface.
    # For every other outcome the field is None (JSON `null` is a
    # hash-distinguishing value, same shape as `status_code` +
    # `failure_class`).
    recovered_github_review_id: int | None = Field(default=None, ge=1)

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
    def _enforce_recovered_github_review_id_iff_external_record_skip(self) -> Self:
        """`recovered_github_review_id` required iff
        `outcome == IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD` AND forbidden
        for every other outcome.

        The external-record-skip path discovers a prior GitHub review
        via body-marker scan and recovers its id WITHOUT emitting a
        paired PublishEvent (the canonical review-summary record):
        without this field, audit-only replay loses the github_review_id
        binding for that recovery path. For every other outcome the
        attempt either DID emit a PublishEvent (success path carries
        the id in PublishEvent.github_review_id) or didn't discover a
        github review at all (failed / no_op_empty / idempotently_skipped
        — the prior PublishEvent is the canonical record).
        """
        is_external_skip = (
            self.outcome is PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD
        )
        if is_external_skip and self.recovered_github_review_id is None:
            raise ValueError(
                "PublishAttemptEvent outcome=idempotently_skipped_external_record "
                "requires recovered_github_review_id (the github review id "
                "discovered via body-marker scan); audit-only replay needs the "
                "binding."
            )
        if not is_external_skip and self.recovered_github_review_id is not None:
            raise ValueError(
                f"PublishAttemptEvent outcome={self.outcome.value!r} must have "
                f"recovered_github_review_id=None, got {self.recovered_github_review_id!r} "
                "(field is exclusive to the external-record skip path)."
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
            recovered_github_review_id=self.recovered_github_review_id,
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
    fails Pydantic construction, not just reads weird. See §5 of
    `specs/2026-05-19-analyze-foundation.md`.
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
    # le=100.0 matches SynthesizeCompletedEvent.total_cost_usd + ReviewMetrics.total_cost_usd —
    # a float('inf') / JSONB-poisoning upper bound (real V1 reviews land well under $1).
    total_cost_usd: float = Field(ge=0, le=100.0)
    pricing_version: str = Field(pattern=PRICING_VERSION_PATTERN)
    policy_version: str = Field(pattern=BARE_SEMVER_PATTERN)
    analyze_model: str
    """The DEEP-tier (and trace-fetched-file) model used this pass. Historically "the
    analyze model"; NARROWED to DEEP-tier when tiered routing landed
    (`specs/2026-06-08-analyze-tiered-model-routing.md`) — the STANDARD-tier model is
    on `standard_analyze_model`."""
    standard_analyze_model: str | None = None
    """The STANDARD-tier model used this pass, or `None` when NO STANDARD-tier LLM call
    ran — a pass with no STANDARD-tier files, or a pass-1 trace round (trace-fetched
    files have no tier and stay on `analyze_model`). When a STANDARD file IS analyzed the
    field records the model used EVEN under inert config where it equals the DEEP
    `analyze_model`: `None` means "no STANDARD call ran," NOT "same model as DEEP"
    (observability over distinctness — you can always tell which model handled the
    STANDARD tier). Additive field; `None` on historical `AnalyzeCompletedEvent` payloads
    (none exist in production yet — analyze hasn't shipped — so the default is a
    forward-compat hedge), so persister / replay / dashboard readers tolerate its
    absence. `specs/2026-06-08-analyze-tiered-model-routing.md`."""

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
        "span_outside_degraded_context",
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

    @field_validator("proposal_hash")
    @classmethod
    def _reject_reserved_proposal_hash(cls, value: str, info: ValidationInfo) -> str:
        """Reserve the all-zero sentinel (see DECISIONS.md#032). This pair is NOT
        in `REPLAY_TOLERABLE_SENTINEL_FIELDS`, so `_sentinel_permitted` is False
        even under replay context — the sentinel is rejected always here. The
        normalizer never injects it for `finding_proposal_rejected`, so its
        presence would be corruption, which must stay loud.
        """
        if value == RESERVED_HISTORICAL_PROPOSAL_HASH and not _sentinel_permitted(
            info, "finding_proposal_rejected", "proposal_hash"
        ):
            raise ValueError(
                "proposal_hash must not be the reserved all-zero sentinel "
                "(reserved for replay of pre-#025 historical events; see DECISIONS.md#032)"
            )
        return value

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


class SynthesizeCompletedEvent(AuditEventBase):
    """Per-review aggregate emitted at the end of the synthesize node.

    `policy_version` mirrors the triage-captured snapshot per
    DECISIONS.md#028-per-review-policy-version-snapshot-anchor-on-triageresult.
    See also DECISIONS.md#029 for the V2 durable-retry idempotency
    + content-binding via `llm_call_event_id` (out of scope for V1).

    One per review (not per-pass like `AnalyzeCompletedEvent`). Carries
    the canonical `ReviewMetrics` fields (mirror of
    `outrider.schemas.review_report.ReviewMetrics`), plus the binding
    metadata (`summary_content_hash`, `overall_risk`, `n_findings`)
    that lets replay reconstruct what summary text + what report shape
    this synthesize-node invocation produced. Replay joins the
    paired `LLMCallEvent` by `review_id` + `node_id="synthesize"` —
    `LLMCallEvent` carries its own `event_id` AND scans `llm_call_content`
    by that key per `DECISIONS.md#016`; no explicit `llm_call_event_id`
    FK is needed on this event because the synthesize node emits
    exactly one LLM call per review (joinable unambiguously) and the
    `summary_content_hash` here cross-validates the content match.

    Metadata-only per `DECISIONS.md#016`: the summary prose lives in
    `llm_call_content` (audit-side TTL) AND in the LangGraph checkpoint
    payload (operational-side; see spec gate #6 option (c) retention
    model). `summary_content_hash` is the sha256 of the RAW LLM
    `response.text` (matches the canon `llm_call_content.completion`
    persister stores — see synthesize.py `_compute_summary_content_hash`
    docstring for the display-vs-canon split rationale). Within the
    LLM-content TTL window, replay can reconstruct prose by joining on
    this hash; outside the window, metadata-only replay is the
    canonical claim.

    Idempotency: event_id-PK (default per `DECISIONS.md#026`). Natural-
    key was rejected at pre-spec gate #1 because the natural-key
    persister cannot return enough payload to reconstruct
    `ReviewReport` (the summary text lives in `llm_call_content`, not
    in the audit-row payload) — state-lockstep gate iii fails. The
    event_id-PK contract catches CONCURRENT re-emit of the same logical
    event (e.g., dispatcher fires twice for one checkpoint state) but
    does NOT catch crash-recovery re-emit (Synthesize body crashes
    AFTER this event lands but BEFORE node return → resume mints a
    fresh UUID → second row lands). V1's in-process BackgroundTasks
    dispatcher does not durably retry; V2 Celery + Redis adds durable
    retry semantics and will need a natural-key add-on before durable
    retry lands (sibling of `emit_phase` which already keys natural).
    Per-review-aggregate dashboard queries should DISTINCT ON
    `(review_id, event_type)` or use MAX(timestamp) to be robust.

    No content-hash binding on findings here (unlike `FindingEvent`):
    every finding in the deduplicated `ReviewReport.findings` is already
    recorded as its own `FindingEvent` upstream. `n_findings` is the
    aggregate; replay can join to per-finding events via `review_id`.
    """

    event_type: Literal["synthesize_completed"] = "synthesize_completed"
    node_id: Literal["synthesize"] = "synthesize"
    summary_content_hash: Annotated[str, Field(pattern=_SHA256_HEX_PATTERN)]
    """SHA-256 hex over the RAW LLM `response.text` (UTF-8 bytes).
    Binds to `llm_call_content.completion` which persists raw — NOT
    the post-`strip_outer_json_fence` display text used by
    `ReviewReport.summary`. Hashing stripped would break replay
    identity the moment Anthropic wraps a response in ```json```
    fences. Identity check for replay-conditional reconstruction:
    within the LLM-content TTL window, an audit reader can join on
    this hash to fetch the prose from `llm_call_content`; outside it,
    the hash is the only proof of which summary was produced. See
    `agent/nodes/synthesize.py::_compute_summary_content_hash` for the
    display-vs-canon split."""
    overall_risk: RiskLevel
    """Mirror of `ReviewReport.overall_risk`. PR-level risk classification
    carried forward from triage's `RiskLevel` ladder; synthesize does
    NOT re-derive it from the Sonnet call (the summary call produces
    prose, not classification — `severity-set-by-policy` analog at the
    PR level)."""
    n_findings: int = Field(ge=0)
    """Total count of deduplicated findings in the `ReviewReport.findings`
    tuple. Aggregate; per-finding details are joinable via the
    upstream `FindingEvent` rows."""
    # ReviewMetrics fields — mirror of
    # `outrider.schemas.review_report.ReviewMetrics`. Field names match
    # the canonical ReviewMetrics shape (spec.md:1106-1114), NOT the
    # `n_*` per-pass-counter convention of `AnalyzeCompletedEvent` —
    # these are review-level aggregates, not per-pass counters.
    files_examined: int = Field(ge=0)
    # Union of trace_decisions[*].(target_file | resolved_candidate_paths)
    # and trace_fetched_files[*].path, minus pr_context.changed_files
    # paths. See `_compute_files_traced_beyond_diff` for the recipe and
    # the "beyond diff = outside changed-files set, NOT Phase-2-fetched
    # specifically" semantic. Mirror of `ReviewMetrics.files_traced_beyond_diff`.
    files_traced_beyond_diff: int = Field(ge=0)
    # LLM-aggregate metrics. Populated at synthesize-emit time from the
    # audit-stream SUM over this review's `LLMCallEvent` rows (FUP-093) —
    # mirror of `ReviewMetrics`. Kept nullable for append-only read-compat:
    # pre-FUP-093 rows serialize `null` here and replay re-validates historical
    # payloads through the strict adapter, so a required type would reject them
    # (#030, amended). A `None` now means "historical row, predates population."
    # See review_report.py for the full rationale.
    llm_calls_made: int | None = Field(default=None, ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    total_cost_usd: float | None = Field(default=None, ge=0, le=100.0)
    """Upper cap matches `ReviewMetrics.total_cost_usd` (le=100.0) —
    defense against `float('inf')` propagating into JSONB. Real V1
    reviews land well under $1; le=100 is "this would already be a
    runaway." Optional+None per the same read-compat rationale above
    (populated going forward; nullable for historical rows)."""
    wall_clock_seconds: float = Field(ge=0, le=86400)
    """Upper cap matches `ReviewMetrics.wall_clock_seconds` (le=86400,
    24h). A multi-day review is a bug, not a workload."""
    pricing_version: str = Field(pattern=PRICING_VERSION_PATTERN)
    """Active LLM pricing version at synthesize-emit time, mirrored from
    `LLMCallEvent.pricing_version` semantics. Enables cost-recomputation
    audits — total_cost_usd should equal the sum across this review's
    LLMCallEvent rows under the pinned pricing_version."""
    policy_version: str = Field(pattern=BARE_SEMVER_PATTERN)
    """Severity policy version captured at the triage-snapshot
    (review-start). Mirrors `state.triage_result.policy_version` so a
    single review's findings, summary, and replay share one policy
    snapshot regardless of mid-deploy `ACTIVE_POLICY_VERSION` bumps.
    Per pre-spec gate #1 + the canonical-amendment route,
    `policy_version` is scoped to this event (NOT promoted to
    `ReviewReport`-on-state). Replay reads this field; per-finding
    `FindingEvent.policy_version` must match (audit-side
    cross-consistency check)."""
    synthesize_model: str
    """Model string for the summary call (e.g., 'claude-haiku-4-5' —
    the DECISIONS.md#043 default). From config per
    `model-strings-from-config-not-hardcoded`. Replay needs this to
    know which model produced the canonicalized summary text the hash
    binds."""


class ReplayVerdictEvent(AuditEventBase):
    """Records the outcome of a replay-equivalence check over a judged prefix.

    Emitted by the background replay-verdict projector AFTER a review completes —
    phase-unbounded replay metadata, NOT graph-node work, so it is exempt from
    `_verify_phase_wellformed` phase containment via
    `audit.replay._PHASE_UNBOUNDED_EVENTS`. `target_max_sequence_number` pins the
    `sequence_number` high-water mark of the prefix the verdict covers (the judged
    stream, EXCLUDING any prior `replay_verdict` events — a verdict is never
    computed over a stream containing its own kind). `mode` + the `*_count` fields
    mirror the on-demand `ReplayVerdict` shape (`api/dashboard/reviews.py`); they
    form an all-present-or-all-absent envelope, `None` only when reconstruction
    itself raised (see `_enforce_metadata_envelope`). `reason` is set iff the
    verdict is inequivalent.
    """

    event_type: Literal["replay_verdict"] = "replay_verdict"
    replay_equivalent: bool
    # Bare Literal mirroring `audit.replay.ReplayMode` values — NOT the enum, to
    # avoid a circular import (`replay` already imports `events`); a drift test
    # pins the two in sync. `None` ONLY when reconstruction itself raised.
    mode: Literal["full", "metadata_only", "mixed"] | None = None
    event_count: int | None = Field(default=None, ge=0)
    finding_count: int | None = Field(default=None, ge=0)
    orphan_finding_count: int | None = Field(default=None, ge=0)
    reason: str | None = Field(default=None, max_length=500)
    # >= 1: `sequence_number` is a BIGINT IDENTITY starting at 1, so the judged
    # prefix's high-water mark always names a real row (a review always has events).
    target_max_sequence_number: int = Field(ge=1)

    @model_validator(mode="after")
    def _enforce_reason_paired_with_inequivalence(self) -> Self:
        """`reason` is set iff `replay_equivalent is False` — an equivalent verdict
        has nothing to explain; an inequivalent one must record why (mirrors the
        on-demand `ReplayVerdict`: reason on failure, `None` on success)."""
        if not self.replay_equivalent and self.reason is None:
            raise ValueError("ReplayVerdictEvent with replay_equivalent=False requires a reason")
        if self.replay_equivalent and self.reason is not None:
            raise ValueError("ReplayVerdictEvent with replay_equivalent=True must have reason=None")
        return self

    @model_validator(mode="after")
    def _enforce_metadata_envelope(self) -> Self:
        """The reconstruction metadata (`mode` + the three counts) is ALL-present or
        ALL-absent. It is present when reconstruction SUCCEEDED — whether the verdict
        is equivalent or an `assert_equivalent` failure — and absent (all `None`) ONLY
        when reconstruction itself raised, which is necessarily inequivalent. So a
        partial envelope is malformed, and an EQUIVALENT verdict must carry the full
        envelope (it can only be equivalent if reconstruction succeeded)."""
        envelope = (self.mode, self.event_count, self.finding_count, self.orphan_finding_count)
        present = [field is not None for field in envelope]
        if any(present) and not all(present):
            raise ValueError(
                "ReplayVerdictEvent metadata envelope (mode + event/finding/orphan counts) "
                "must be all-present or all-absent; got a partial envelope"
            )
        if self.replay_equivalent and not all(present):
            raise ValueError(
                "ReplayVerdictEvent with replay_equivalent=True requires the full reconstruction "
                "metadata envelope (mode + counts) — an equivalent verdict means reconstruction "
                "succeeded"
            )
        return self


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
    | AnalyzeResponseRejectedEvent
    | SynthesizeCompletedEvent
    | ReplayVerdictEvent,
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
        | SynthesizeCompletedEvent
        | ReplayVerdictEvent
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
    "ReplayVerdictEvent",
    "ReviewPhaseEvent",
    "SynthesizeCompletedEvent",
    "TraceDecisionEvent",
    "compute_publish_attempt_content_hash",
    "compute_publish_eligibility_decision_hash",
    "compute_publish_routing_decision_hash",
]
