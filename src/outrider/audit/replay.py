# See DECISIONS.md#014-audit-events-are-metadata-only-content-purge-targets-reviews-and-findings
# See also DECISIONS.md#016-llm-exchanges-stored-locally-under-retention-logs-stay-metadata-only
# See also DECISIONS.md#031-replay-v1-verification-is-verify-only-no-source-span-re-run
"""Replay-equivalence reconstruction over the append-only audit stream.

Read-only reconstructor: rebuilds a review from `audit_events` plus the
content tables (`reviews` / `findings` / `llm_call_content`) and exposes
`AuditReplayer.assert_replay_equivalent`. This turns the append-only audit
rows from observability into a verifiable reconstruction surface.

Two design commitments:

  - **Verify-only.** Replay re-verifies the proof boundary, recomputes
    content hashes, and reconstructs severity under the historical policy
    version — it does NOT re-run the graph, call the LLM, or re-run a
    tree-sitter query against source bytes (a full `match(id, source)`
    re-run needs a durable source store, routed to future scope).
  - **Mode by content-row presence.** `reconstruct` selects full vs
    metadata-only vs mixed by whether the content row physically exists,
    NOT by a `retention_expires_at` comparison and NOT by a NULL column.
    `DECISIONS.md#016`'s single-transaction insert makes "audit row present,
    content row absent" mean unambiguously "purged per retention," so
    row-absence is a sound signal. Findings and LLM content can carry
    different TTLs (the ordering `llm_content <= findings <= review` holds;
    all three default to 90d but an operator may raise findings above llm),
    so a single review can be legitimately MIXED; replay labels every item
    rather than silently producing a hybrid.

The canonical ordered reconstruction DTO (`ReconstructedReview`) is the
single read model consumed by both `assert_replay_equivalent` and the
future timeline-playback surface (`ROADMAP.md` feature 6).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from outrider.audit.events import (
    AgentTransitionEvent,
    AuditEvent,
    AuditEventAdapter,
    AuditEventBase,
    FindingEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    LLMCallEvent,
    PublishAttemptEvent,
    PublishEligibilityEvent,
    PublishEvent,
    PublishRoutingEvent,
    ReviewPhaseEvent,
    TraceDecisionEvent,
    compute_finding_content_hash,
)
from outrider.db.models.audit_events import AuditEvent as AuditEventRow
from outrider.db.models.findings import Finding
from outrider.db.models.llm_call_content import LLMCallContent
from outrider.db.models.reviews import Review
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import (
    ACTIVE_POLICY_VERSION,
    SEVERITY_POLICY,
    FindingSeverity,
    FindingType,
)
from outrider.policy.versions import (
    PolicyVersionShapeError,
    UnknownPolicyVersionError,
    load_policy_for_version,
)

# ---------------------------------------------------------------------------
# Typed errors (functions-raise-typed-exceptions)
# ---------------------------------------------------------------------------


class ReplayError(Exception):
    """Base class for replay failures."""


class ReplayReviewNotFoundError(ReplayError):
    """No `audit_events` rows exist for the requested review_id."""


class ReplayEquivalenceError(ReplayError):
    """A replay-equivalence assertion failed; the message names the check."""


# ---------------------------------------------------------------------------
# Reconstruction DTO (the read model)
# ---------------------------------------------------------------------------


class ReplayMode(StrEnum):
    """Which reconstruction mode applied, per `DECISIONS.md#014` point 4.

    FULL: every content row is present (review + all findings + all LLM
        exchanges) — reconstructs with content. METADATA_ONLY: the review
        row is purged, so all content is gone — findings as stubs, LLM as
        metadata + surviving hashes. MIXED: some content present, some
        purged (the legitimate window where llm_call_content's shorter-or-equal
        TTL has elapsed but findings remain — non-empty only when an operator
        sets findings TTL above the llm-content TTL) — labeled per item,
        never silently hybridized.
    """

    FULL = "full"
    METADATA_ONLY = "metadata_only"
    MIXED = "mixed"


class FindingContent(BaseModel):
    """Full-mode hydration of a finding from the `findings` content table.

    Present only within retention; `None` on a `ReconstructedFinding` means
    the row was purged (metadata-only stub). Carries the fields the
    `FindingEvent` audit shadow does not (title/description/evidence text +
    override provenance), plus the duplicated metadata replay cross-checks
    against the event in full mode.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_type: FindingType
    severity: FindingSeverity
    evidence_tier: EvidenceTier
    file_path: str
    line_start: int
    line_end: int
    title: str
    description: str
    evidence: str
    suggested_fix: str | None
    query_match_id: str | None
    trace_path: tuple[str, ...] | None
    original_severity: FindingSeverity | None
    override_reason: str | None
    overrider_id: UUID | None
    publish_destination: str | None
    policy_version: str
    content_hash: str


class ReconstructedFinding(BaseModel):
    """A finding reconstructed from its `FindingEvent` + optional content.

    `event` is always present (the audit stream survives forever);
    `content` is the full-mode hydration (`None` ⇒ metadata-only stub).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: FindingEvent
    content: FindingContent | None


class ReconstructedLLMExchange(BaseModel):
    """An LLM call reconstructed from its `LLMCallEvent` + optional content.

    `prompt`/`completion` are the full-mode hydration from
    `llm_call_content` (joined by `event_id`); both `None` ⇒ metadata-only
    (content purged, but the surviving event carries token counts + hashes).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: LLMCallEvent
    prompt: str | None
    completion: str | None


class ReconstructedPhase(BaseModel):
    """A graph-node phase bounded by a `ReviewPhaseEvent` start/end pair.

    `phase-events-bound-work`: start/end markers (keyed by `phase_id`) are
    the causal barriers. `end` is `None` for a phase that never closed (a
    real crash state). `events` are the per-operation events that occurred
    between the barriers, in sequence order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase_id: str
    node_id: str
    phase_key: str | None
    start: ReviewPhaseEvent | None
    end: ReviewPhaseEvent | None
    events: tuple[AuditEvent, ...]


class ReconstructedReviewMetadata(BaseModel):
    """The `reviews` content-table row, reconstructed in full/mixed mode.

    `None` on a `ReconstructedReview` when the review row is purged (the
    metadata-only signal — the audit stream survives the review).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_id: UUID
    installation_id: int
    status: str
    is_eval: bool
    repo_id: int
    pr_number: int
    head_sha: str
    files_examined: int
    files_traced_beyond_diff: int
    llm_calls_made: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Decimal
    wall_clock_seconds: Decimal
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    expires_at: datetime | None


class ReconstructedReview(BaseModel):
    """Canonical ordered reconstruction of a review (the read model).

    Consumed by `assert_replay_equivalent` (verification) and the future
    timeline-playback surface (`ROADMAP.md` feature 6) — one reconstruction
    surface, not re-interpreted per consumer. `events` is the complete
    ordered stream; `phases` is the phase-grouped view over it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_id: UUID
    mode: ReplayMode
    is_eval: bool
    review: ReconstructedReviewMetadata | None
    events: tuple[AuditEvent, ...]
    phases: tuple[ReconstructedPhase, ...]
    findings: tuple[ReconstructedFinding, ...]
    llm_exchanges: tuple[ReconstructedLLMExchange, ...]
    # Stored `findings`-table rows whose finding_id has no FindingEvent in the
    # audit stream — an append-only-guarantee violation (a finding exists that
    # was never audit-logged). Empty in a faithful reconstruction.
    orphan_finding_ids: tuple[UUID, ...] = ()


# ---------------------------------------------------------------------------
# Pure reconstruction + verification helpers (DB-free, unit-testable)
# ---------------------------------------------------------------------------


@dataclass
class _PhaseBuilder:
    """Mutable accumulator for a phase under construction during grouping."""

    start: ReviewPhaseEvent
    end: ReviewPhaseEvent | None = None
    events: list[AuditEvent] = field(default_factory=list)


def _group_phases(events: tuple[AuditEvent, ...]) -> tuple[ReconstructedPhase, ...]:
    """Group events into phases by `ReviewPhaseEvent` start/end markers.

    Sequential (non-nested) phases per V1; events outside any open phase
    (e.g. the leading webhook→intake transition) stay in `events` but are
    omitted from the grouped view. Keys on `phase_id`, not the
    V1.5-nullable `phase_key`.
    """
    builders: list[_PhaseBuilder] = []
    open_builder: _PhaseBuilder | None = None
    for event in events:
        if isinstance(event, ReviewPhaseEvent):
            if event.marker == "start":
                open_builder = _PhaseBuilder(start=event)
                builders.append(open_builder)
            else:  # marker == "end"
                match = next(
                    (b for b in builders if b.start.phase_id == event.phase_id and b.end is None),
                    None,
                )
                if match is not None:
                    match.end = event
                    if match is open_builder:
                        open_builder = None
            continue
        if open_builder is not None:
            open_builder.events.append(event)
    return tuple(
        ReconstructedPhase(
            phase_id=b.start.phase_id,
            node_id=b.start.node_id,
            phase_key=b.start.phase_key,
            start=b.start,
            end=b.end,
            events=tuple(b.events),
        )
        for b in builders
    )


def _classify_mode(
    *,
    review_present: bool,
    findings: tuple[ReconstructedFinding, ...],
    llm_exchanges: tuple[ReconstructedLLMExchange, ...],
) -> ReplayMode:
    """Select the reconstruction mode from content-row presence.

    Three legitimate states under the retention ordering
    (`llm_content <= findings <= review`, per `DECISIONS#014`/`#016`; all three
    default to 90d, operator-overridable within that ordering):

    - **FULL** — review row present and every finding + LLM call carries content
      (within the shortest TTL). A review with no findings/LLM calls and a
      present row classifies FULL vacuously.
    - **MIXED** — review row present but not every content item is full. The
      canonical case is the window where the shorter-or-equal-lived LLM content
      has purged while finding content remains (non-empty only when findings
      TTL is set above the llm-content TTL); more generally it is the
      residual partial-presence state that is neither FULL nor one of the two
      impossible shapes below. Every item is labelled individually rather than
      silently hybridized.
    - **METADATA_ONLY** — review row absent ⇒ all content purged before it.

    Raises `ReplayEquivalenceError` on three impossible states:

    - **Review absent + any content survives.** Because content (LLM and
      findings) purges no later than the review (`llm <= findings <= review`),
      a purged review implies all its content already purged — surviving
      content with no review row is corruption (partial / out-of-order purge
      or tampering).
    - **LLM content survives + any finding content purged.** Because LLM
      content purges no later than finding content (`llm <= findings`),
      surviving LLM content guarantees the findings window is still open — so
      a purged finding alongside surviving LLM content is an out-of-order
      purge / tampering, the sibling of the case above. The legitimate MIXED
      window is the opposite shape: findings present, the shorter-lived LLM
      purged.
    - **Half-present LLM content row.** `prompt` and `completion` are both
      NOT NULL in `llm_call_content` and co-inserted in one transaction, so
      they purge together. A row with one side present and the other absent
      is a torn/corrupt row — rejected here so mode classification can't key
      off `prompt` alone and silently mis-bucket the review.
    """
    for exchange in llm_exchanges:
        if (exchange.prompt is None) != (exchange.completion is None):
            raise ReplayEquivalenceError(
                f"llm_call_content for event {exchange.event.event_id} is half-present "
                f"(prompt={'set' if exchange.prompt is not None else 'None'}, "
                f"completion={'set' if exchange.completion is not None else 'None'}); "
                "prompt and completion purge together — a one-sided row is corruption"
            )
    any_finding_content = any(f.content is not None for f in findings)
    all_finding_content = all(f.content is not None for f in findings)
    # Post the half-present guard above, `prompt is not None` ⟺
    # `completion is not None`, so keying on `prompt` covers both sides.
    any_llm_content = any(x.prompt is not None for x in llm_exchanges)
    all_llm_content = all(x.prompt is not None for x in llm_exchanges)
    if not review_present and (any_finding_content or any_llm_content):
        raise ReplayEquivalenceError(
            "review row is purged but content rows survive "
            f"(finding_content={any_finding_content}, llm_content={any_llm_content}); "
            "under the retention ordering a purged review implies all content already "
            "purged — surviving content with no review row is corruption, not a "
            "legitimate mixed window"
        )
    if any_llm_content and not all_finding_content:
        raise ReplayEquivalenceError(
            "LLM content survives while finding content is purged "
            f"(any_llm_content={any_llm_content}, all_finding_content={all_finding_content}); "
            "the retention ordering (LLM content TTL ≤ findings TTL) requires LLM "
            "content to purge no later than finding content — surviving LLM content "
            "with a purged finding is corruption, not a legitimate mixed window "
            "(MIXED is the opposite shape: findings present, LLM purged)"
        )
    all_present = review_present and all_finding_content and all_llm_content
    if all_present:
        return ReplayMode.FULL
    if not review_present:  # guard above guarantees no content survives here
        return ReplayMode.METADATA_ONLY
    return ReplayMode.MIXED


def _verify_sequence_monotonic(events: tuple[AuditEvent, ...]) -> None:
    """Assert sequence numbers are present, strictly ascending, and unique."""
    seqs = [e.sequence_number for e in events]
    if any(s is None for s in seqs):
        raise ReplayEquivalenceError(
            "reconstructed event missing sequence_number; the row-level "
            "sequence_number was not merged into the payload on read"
        )
    for prev, curr in zip(seqs, seqs[1:], strict=False):
        if curr <= prev:  # type: ignore[operator]
            raise ReplayEquivalenceError(
                f"sequence_number not strictly ascending: {prev} then {curr}; "
                "the audit stream must replay in a deterministic total order"
            )


# Graph-faithfulness: which node owns each NODE-LESS event type (the types
# that do not carry their own `node_id` field). Verified against the production
# emit sites — each owner emits the event between its own ReviewPhaseEvent
# start/end markers:
#   FindingEvent           → analyze  (analyze.py emit_finding, in analyze phase)
#   TraceDecisionEvent     → trace    (trace.py emit_trace_decision, in trace phase)
#   HITLRequestEvent       → hitl     (hitl.py emit_hitl_request, in hitl phase)
#   HITLDecisionEvent      → hitl     (hitl.py emit_hitl_decision on resume re-entry,
#                                      in hitl phase; the dashboard endpoint does NOT
#                                      emit the audit row — see api/dashboard/hitl.py)
#   PublishEvent / PublishRoutingEvent / PublishEligibilityEvent /
#   PublishAttemptEvent    → publish  (publish.py, phase-start emitted before any work)
# An event NOT in this map and carrying no `node_id` is phase-unbounded-exempt
# (see `_PHASE_UNBOUNDED_EVENTS`) or fails the completeness guard
# (test_node_less_events_have_owner_or_exemption) — a new node-less event type
# can't silently skip the node-containment check. The guard catches NEW types,
# not a MOVED emit site (an existing type emitted from a different node would
# make replay reject a valid stream); that drift is tracked by FUP-112.
# Wrapped in MappingProxyType + Final per the repo constant-immutability pattern
# (`policy.severity.SEVERITY_POLICY`, `llm.pricing.RATE_TABLE`): a bare dict could
# be mutated at runtime by a buggy caller and silently change node-containment for
# the rest of the process. Keyed by `type[AuditEventBase]` (the shared base), not
# the `AuditEvent` discriminated-union alias — `type[...]` wants a class.
_NODE_LESS_EVENT_OWNER: Final[Mapping[type[AuditEventBase], str]] = MappingProxyType(
    {
        FindingEvent: "analyze",
        TraceDecisionEvent: "trace",
        HITLRequestEvent: "hitl",
        HITLDecisionEvent: "hitl",
        PublishEvent: "publish",
        PublishRoutingEvent: "publish",
        PublishEligibilityEvent: "publish",
        PublishAttemptEvent: "publish",
    }
)

# Node-less event types that legitimately occur outside any single node's phase
# and so are exempt from node-containment. `AgentTransitionEvent` records a
# transition BETWEEN phases (it carries from_node/to_node, not a single node_id).
# `ReviewPhaseEvent` is the phase marker itself, handled before this check.
# Keyed by `type[AuditEventBase]`, not the `AuditEvent` union alias (`type[...]`
# wants a class).
_PHASE_UNBOUNDED_EVENTS: Final[tuple[type[AuditEventBase], ...]] = (AgentTransitionEvent,)


def _required_phase_node(event: AuditEvent) -> str | None:
    """The node whose phase must enclose `event`, or None if unconstrained.

    Prefers the event's own `node_id` (LLMCallEvent, FileExaminationEvent, the
    analyze/synthesize aggregates); falls back to the node-less owner map
    (FindingEvent → analyze, etc.). Returns None for phase-unbounded events
    (`AgentTransitionEvent`) — they are bounded by nothing.
    """
    own = getattr(event, "node_id", None)
    if own is not None:
        # `node_id` is a `Literal[...]` (str subtype) on the events that carry
        # it; `getattr` widens to Any, so narrow back to str for the caller.
        return str(own)
    return _NODE_LESS_EVENT_OWNER.get(type(event))


def _verify_phase_wellformed(
    events: tuple[AuditEvent, ...], *, require_all_terminated: bool = False
) -> None:
    """Assert phases are well-formed and bound every node-work event.

    `phase-events-bound-work` (spec §8.4): per-operation work events must fall
    within a `ReviewPhaseEvent` start/end pair — the causal barriers replay
    relies on. Walking in sequence order, this enforces:

    - **Boundedness.** Every work event occurs while a phase is open.
      `AgentTransitionEvent` and the phase markers themselves are exempt —
      transitions legitimately occur before/between phases.
    - **Node containment.** A work event must occur inside a phase for the
      node that owns it — its own `node_id` when it carries one (`LLMCallEvent`,
      `FileExaminationEvent`, the analyze/synthesize aggregates), else the
      node-less owner map (`_NODE_LESS_EVENT_OWNER`: `FindingEvent` → analyze,
      `TraceDecisionEvent` → trace, HITL → hitl, publish events → publish). An
      `analyze` LLM call belongs in an `analyze` phase, not a `triage` one; an
      analyze-owned `FindingEvent` likewise. This makes the stream
      graph-faithful, not merely phase-bounded. Only `AgentTransitionEvent` is
      unbounded (it records a transition BETWEEN phases); the completeness
      guard test asserts every other node-less type has an owner.
    - **Ordering.** An end never precedes its start (an end whose phase_id has
      no prior start raises — this is the end-before-start case in sequence
      order).
    - **Uniqueness.** A phase_id has ≤1 start and ≤1 end.
    - **Non-nesting (sequential phases).** A phase may not start while another
      is still open — V1 runs phases one at a time. Distinct from Uniqueness: a
      *reused* phase_id raises "more than one start marker"; a *different*
      phase_id opened concurrently raises the non-nesting error. (V1.5
      parallel-analyze will rekey this around `(node_id, phase_key)`.)
    - **Marker agreement.** An end's `node_id` / `phase_key` match its start.
    - **Termination on success.** When `require_all_terminated` is set, every
      started phase must also have an end. The invariant's "missing phase end
      events on success are violations" clause: the caller sets this only for a
      `completed` review. A trailing start-without-end is tolerated otherwise —
      for a crashed / in-flight / failed review it is a real state, not a
      corruption — and for a metadata-only reconstruction the review row is
      purged so success can't be observed (so it can't be required).
    """
    started: dict[str, ReviewPhaseEvent] = {}
    ended: set[str] = set()
    open_phases: dict[str, ReviewPhaseEvent] = {}
    for event in events:
        if isinstance(event, ReviewPhaseEvent):
            if event.marker == "start":
                # Uniqueness before non-nesting: a *reused* phase_id is the more
                # specific diagnosis (it fires for both consecutive start→start
                # and reopen-after-close), so check it first; the non-nesting
                # guard then catches a *different* phase_id opened while one is
                # still open. Together they enforce "≤1 open phase, no reuse".
                if event.phase_id in started:
                    raise ReplayEquivalenceError(
                        f"phase {event.phase_id!r} has more than one start marker"
                    )
                if open_phases:
                    open_ids = sorted(open_phases)
                    raise ReplayEquivalenceError(
                        f"phase {event.phase_id!r} starts while phase(s) {open_ids} "
                        "are still open; V1 phases must be sequential/non-nested "
                        "(phase-events-bound-work). V1.5 parallel-analyze will redesign "
                        "this around (node_id, phase_key)."
                    )
                started[event.phase_id] = event
                open_phases[event.phase_id] = event
            else:  # marker == "end"
                start = started.get(event.phase_id)
                if start is None:
                    raise ReplayEquivalenceError(
                        f"phase {event.phase_id!r} has an end marker with no preceding start"
                    )
                if event.phase_id in ended:
                    raise ReplayEquivalenceError(
                        f"phase {event.phase_id!r} has more than one end marker"
                    )
                if event.node_id != start.node_id:
                    raise ReplayEquivalenceError(
                        f"phase {event.phase_id!r} end node_id {event.node_id!r} disagrees "
                        f"with start node_id {start.node_id!r}"
                    )
                if event.phase_key != start.phase_key:
                    raise ReplayEquivalenceError(
                        f"phase {event.phase_id!r} end phase_key {event.phase_key!r} disagrees "
                        f"with start phase_key {start.phase_key!r}"
                    )
                ended.add(event.phase_id)
                del open_phases[event.phase_id]
        elif isinstance(event, AgentTransitionEvent):
            continue  # transitions legitimately occur before/between phases
        elif not open_phases:
            raise ReplayEquivalenceError(
                f"{type(event).__name__} (sequence {event.sequence_number}) occurs outside "
                f"any open review phase; node work must be bounded by ReviewPhaseEvent "
                f"start/end markers (phase-events-bound-work)"
            )
        else:
            # Node containment (graph-faithfulness): the event must sit in an
            # open phase for the node that owns it — its own `node_id` if it has
            # one, else the node-less owner map (`_required_phase_node`). An
            # analyze-owned FindingEvent in a triage phase is not a stream any
            # graph node would emit. `AgentTransitionEvent` returns None (it is
            # already `continue`d above) and so is never constrained here.
            required_node = _required_phase_node(event)
            if required_node is not None and not any(
                phase.node_id == required_node for phase in open_phases.values()
            ):
                open_node_ids = sorted({phase.node_id for phase in open_phases.values()})
                raise ReplayEquivalenceError(
                    f"{type(event).__name__} (sequence {event.sequence_number}) is owned by "
                    f"node {required_node!r} but no open phase matches that node "
                    f"(open phases: {open_node_ids}); a node's work must be bounded by its "
                    f"own phase markers (phase-events-bound-work)"
                )
    if require_all_terminated:
        unterminated = sorted(phase_id for phase_id in started if phase_id not in ended)
        if unterminated:
            raise ReplayEquivalenceError(
                f"completed review has unterminated phase(s) {unterminated}; "
                "phase-events-bound-work requires a phase end event on success"
            )


def _verify_proof_boundary(events: tuple[AuditEvent, ...]) -> None:
    """Re-verify the proof boundary for every finding (verify-only).

    The schema-level `enforce_proof_boundary` already re-fired on
    deserialization. This adds the registry-membership and hash-recompute
    checks the schema layer does not do: an OBSERVED finding's
    `query_match_id` must resolve via the deprecation-aware registry
    surface (so a historical finding citing a since-deprecated id is NOT
    flagged), and `finding_content_hash` must recompute. No source re-run.
    """
    # Lazy import: `queries.registry` pulls tree_sitter at module load.
    # Importing it eagerly would make the whole `audit` package drag
    # tree_sitter and break the ast_facts import-light firewall, so defer
    # it to first proof-verification.
    from outrider.ast_facts.errors import UnknownQueryMatchId
    from outrider.queries.registry import get_query_source

    for event in events:
        if not isinstance(event, FindingEvent):
            continue
        if event.evidence_tier == EvidenceTier.OBSERVED:
            # `query_match_id` is guaranteed non-None for OBSERVED by the
            # proof-boundary validator that re-fired on deserialization.
            try:
                get_query_source(event.query_match_id)  # type: ignore[arg-type]
            except UnknownQueryMatchId as exc:
                raise ReplayEquivalenceError(
                    f"OBSERVED finding {event.finding_id} cites query_match_id "
                    f"{event.query_match_id!r} that is not in the registry or "
                    f"deprecation ledger: {exc}"
                ) from exc
        recomputed = compute_finding_content_hash(
            event.file_path,
            line_start=event.line_start,
            line_end=event.line_end,
            finding_type=event.finding_type,
        )
        if recomputed != event.finding_content_hash:
            raise ReplayEquivalenceError(
                f"finding {event.finding_id} content_hash recompute mismatch: "
                f"stored {event.finding_content_hash}, recomputed {recomputed}"
            )


def _verify_cross_event_refs(events: tuple[AuditEvent, ...]) -> None:
    """Assert cross-event finding references resolve (FUP-041, focused subset).

    Every `finding_id` referenced by a publish-routing, publish-eligibility,
    HITL-request, HITL-decision, or trace-decision event must resolve to a
    `FindingEvent` in the stream; routing/eligibility `finding_content_hash`
    must agree with the referenced finding's.
    """
    finding_hashes: dict[UUID, str] = {
        e.finding_id: e.finding_content_hash for e in events if isinstance(e, FindingEvent)
    }
    known = finding_hashes.keys()

    def _require(finding_id: UUID, source: str) -> None:
        if finding_id not in known:
            raise ReplayEquivalenceError(
                f"{source} references finding_id {finding_id} with no FindingEvent in the stream"
            )

    for event in events:
        if isinstance(event, PublishRoutingEvent | PublishEligibilityEvent):
            _require(event.finding_id, type(event).__name__)
            if event.finding_content_hash != finding_hashes[event.finding_id]:
                raise ReplayEquivalenceError(
                    f"{type(event).__name__} for finding {event.finding_id} carries "
                    f"finding_content_hash {event.finding_content_hash} disagreeing with "
                    f"the FindingEvent's {finding_hashes[event.finding_id]}"
                )
        elif isinstance(event, TraceDecisionEvent):
            _require(event.source_finding_id, "TraceDecisionEvent.source_finding_id")
        elif isinstance(event, HITLRequestEvent):
            for finding_id in (*event.findings_requiring_approval, *event.auto_post_findings):
                _require(finding_id, "HITLRequestEvent")
        elif isinstance(event, HITLDecisionEvent):
            for decision in event.decisions:
                _require(decision.finding_id, "HITLDecisionEvent.decisions")


def _verify_mode_consistency(review: ReconstructedReview) -> None:
    """Assert the classified mode matches the per-item content presence.

    The mode-aware content guarantees: FULL ⇒ review present + every item
    full; METADATA_ONLY ⇒ review absent + every item a stub (NO
    content-equality claim — there is no content after retention); MIXED ⇒
    the explicit label is the non-silent-hybrid signal.
    """
    recomputed = _classify_mode(
        review_present=review.review is not None,
        findings=review.findings,
        llm_exchanges=review.llm_exchanges,
    )
    if recomputed != review.mode:
        raise ReplayEquivalenceError(
            f"mode {review.mode} disagrees with content presence (recomputed {recomputed})"
        )
    if review.mode == ReplayMode.FULL:
        for finding in review.findings:
            _verify_full_finding(finding)
        for exchange in review.llm_exchanges:
            if exchange.prompt is None or exchange.completion is None:
                raise ReplayEquivalenceError(
                    f"FULL mode but LLM exchange {exchange.event.event_id} is missing content"
                )
    elif review.mode == ReplayMode.METADATA_ONLY:
        if review.review is not None or any(f.content is not None for f in review.findings):
            raise ReplayEquivalenceError("METADATA_ONLY mode but content is present")
    else:  # MIXED — per-item: full items get the full check, stubs are left as stubs.
        for finding in review.findings:
            if finding.content is not None:
                _verify_full_finding(finding)


def _verify_full_finding(finding: ReconstructedFinding) -> None:
    """Assert a full-mode finding's content row agrees with its audit event."""
    content = finding.content
    if content is None:
        raise ReplayEquivalenceError(
            f"finding {finding.event.finding_id} expected full content but is a stub"
        )
    event = finding.event
    mismatches = [
        field_name
        for field_name, content_value, event_value in (
            ("content_hash", content.content_hash, event.finding_content_hash),
            ("finding_type", content.finding_type, event.finding_type),
            ("severity", content.severity, event.severity),
            ("evidence_tier", content.evidence_tier, event.evidence_tier),
            ("file_path", content.file_path, event.file_path),
            ("line_start", content.line_start, event.line_start),
            ("line_end", content.line_end, event.line_end),
            ("policy_version", content.policy_version, event.policy_version),
            # Proof artifacts — the content row must agree with the canonical
            # FindingEvent on the evidence the proof boundary turns on.
            ("query_match_id", content.query_match_id, event.query_match_id),
            ("trace_path", content.trace_path, event.trace_path),
        )
        if content_value != event_value
    ]
    if mismatches:
        raise ReplayEquivalenceError(
            f"finding {event.finding_id} content row disagrees with audit event "
            f"on: {', '.join(mismatches)}"
        )


def _verify_row_consistent(
    event: AuditEvent,
    *,
    event_id: UUID,
    review_id: UUID,
    event_type: str,
    timestamp: datetime,
    is_eval: bool,
    phase_key: str | None,
) -> None:
    """Assert an `audit_events` row's base columns agree with its payload.

    The persister mirrors `event_id` / `review_id` / `event_type` / `timestamp`
    / `is_eval` / `phase_key` into dedicated columns AND into the JSONB payload
    from the same event (`persister._row_kwargs_from_event`): the columns are
    the query surface, the payload is the durable record, and they must match.
    Replay reconstructs from the payload, so a column that drifts from the
    payload (direct DB tampering, a future persister bug) would otherwise go
    undetected. `phase_key` is only populated for `ReviewPhaseEvent` (NULL for
    every other event type), matching the persister. Timestamps compare by
    instant (Python aware-datetime equality), so a column and payload that
    encode the same moment in different tz offsets still agree.
    """
    expected_phase_key = event.phase_key if isinstance(event, ReviewPhaseEvent) else None
    mismatches = [
        name
        for name, column_value, payload_value in (
            ("event_id", event_id, event.event_id),
            ("review_id", review_id, event.review_id),
            ("event_type", event_type, event.event_type),
            ("timestamp", timestamp, event.timestamp),
            ("is_eval", is_eval, event.is_eval),
            ("phase_key", phase_key, expected_phase_key),
        )
        if column_value != payload_value
    ]
    if mismatches:
        raise ReplayEquivalenceError(
            f"audit row {event_id} base column(s) disagree with the payload on: "
            f"{', '.join(mismatches)}"
        )


# ---------------------------------------------------------------------------
# The reconstructor
# ---------------------------------------------------------------------------


class AuditReplayer:
    """Read-only reconstructor over `audit_events` + the content tables.

    Mirrors `AuditPersister`'s dependency-injection shape: the
    `async_sessionmaker` is injected, not an open session
    (`nodes-receive-deps-via-closure`). Every method opens its own
    read-only `AsyncSession` (no `session.begin()` — single read
    transactions); `AsyncSession` is not concurrent-safe, so a fresh one
    per call keeps replay safe under concurrent reviews. `reconstruct`
    additionally pins its transaction to REPEATABLE READ so its four
    content-table reads observe one consistent snapshot even if a
    retention purge commits mid-reconstruct.
    """

    def __init__(self, *, session_factory: async_sessionmaker[AsyncSession]) -> None:
        if session_factory is None:
            raise ReplayError("session_factory is required")
        self._session_factory = session_factory

    async def reconstruct(self, review_id: UUID) -> ReconstructedReview:
        """Reconstruct a review into the canonical ordered read model.

        Reads the `audit_events` stream ascending by `sequence_number`,
        rebuilds each event via the shared `AuditEventAdapter` (re-merging
        the DB-assigned `sequence_number` the emitter excluded on write),
        joins the content tables, and classifies the mode by content-row
        presence. Raises `ReplayReviewNotFoundError` if no audit rows exist.
        A corrupted payload surfaces as `pydantic.ValidationError` at this
        read boundary (the frozen + extra=forbid validator chain re-fires);
        a row whose base columns drift from its payload surfaces as
        `ReplayEquivalenceError` (see `_verify_row_consistent`).

        Also enforced here (so direct read-model consumers such as the
        timeline UI get them without calling `assert_replay_equivalent`):
        `is_eval` coherence across the stream and every joined content row
        (raising `ReplayEquivalenceError` on drift, see
        `_verify_is_eval_consistent`), and population of `orphan_finding_ids`
        (stored `findings` rows with no `FindingEvent` in the stream).
        """
        async with self._session_factory() as session:
            # Pin all four content-table reads (audit_events, reviews, findings,
            # llm_call_content) to ONE consistent snapshot. The default READ
            # COMMITTED isolation gives each statement a fresh snapshot, so a
            # retention purge committing mid-reconstruct (sweep/purge_expired.py
            # commits per-table independently) could let replay combine pre- and
            # post-purge rows into a reconstruction that never existed at any DB
            # instant. REPEATABLE READ takes the snapshot at the first statement
            # and holds it for the transaction — read-only, so it needs no
            # write-skew protection (SERIALIZABLE would be overkill). Must be set
            # before the first statement autobegins the transaction.
            await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            rows = (
                await session.execute(
                    select(
                        AuditEventRow.event_id,
                        AuditEventRow.review_id,
                        AuditEventRow.event_type,
                        AuditEventRow.timestamp,
                        AuditEventRow.is_eval,
                        AuditEventRow.phase_key,
                        AuditEventRow.payload,
                        AuditEventRow.sequence_number,
                    )
                    .where(AuditEventRow.review_id == review_id)
                    .order_by(AuditEventRow.sequence_number.asc())
                )
            ).all()
            if not rows:
                raise ReplayReviewNotFoundError(f"no audit_events rows for review_id {review_id}")
            reconstructed: list[AuditEvent] = []
            for row in rows:
                event = AuditEventAdapter.validate_python(
                    {**row.payload, "sequence_number": row.sequence_number}
                )
                _verify_row_consistent(
                    event,
                    event_id=row.event_id,
                    review_id=row.review_id,
                    event_type=row.event_type,
                    timestamp=row.timestamp,
                    is_eval=row.is_eval,
                    phase_key=row.phase_key,
                )
                reconstructed.append(event)
            events: tuple[AuditEvent, ...] = tuple(reconstructed)

            review_row = (
                await session.execute(select(Review).where(Review.id == review_id))
            ).scalar_one_or_none()
            finding_rows = {
                row.finding_id: row
                for row in (
                    await session.execute(select(Finding).where(Finding.review_id == review_id))
                ).scalars()
            }
            llm_event_ids = [e.event_id for e in events if isinstance(e, LLMCallEvent)]
            content_rows: dict[UUID, LLMCallContent] = {}
            if llm_event_ids:
                content_rows = {
                    row.event_id: row
                    for row in (
                        await session.execute(
                            select(LLMCallContent).where(LLMCallContent.event_id.in_(llm_event_ids))
                        )
                    ).scalars()
                }

        findings = tuple(
            ReconstructedFinding(
                event=event,
                content=_finding_content(finding_rows.get(event.finding_id)),
            )
            for event in events
            if isinstance(event, FindingEvent)
        )
        # Orphans: stored findings the append-only audit stream never recorded.
        event_finding_ids = {e.finding_id for e in events if isinstance(e, FindingEvent)}
        orphan_finding_ids = tuple(
            fid for fid in sorted(finding_rows, key=str) if fid not in event_finding_ids
        )
        llm_exchanges = tuple(
            ReconstructedLLMExchange(
                event=event,
                prompt=content.prompt if (content := content_rows.get(event.event_id)) else None,
                completion=(
                    content.completion if (content := content_rows.get(event.event_id)) else None
                ),
            )
            for event in events
            if isinstance(event, LLMCallEvent)
        )
        # is_eval coherence (eval-isolation, docs/testing.md): the audit stream's
        # is_eval (events[0], the canonical review-level flag) must agree across
        # every event AND every joined content-table row (reviews / findings /
        # llm_call_content). A table-vs-stream drift would mis-bucket the
        # reconstructed review; checked here in reconstruct() so the timeline-UI
        # read-model consumer is protected, not only assert_replay_equivalent.
        _verify_is_eval_consistent(
            stream_is_eval=events[0].is_eval,
            events=events,
            review_row=review_row,
            finding_rows=tuple(finding_rows.values()),
            content_rows=tuple(content_rows.values()),
        )
        mode = _classify_mode(
            review_present=review_row is not None,
            findings=findings,
            llm_exchanges=llm_exchanges,
        )
        return ReconstructedReview(
            review_id=review_id,
            mode=mode,
            is_eval=events[0].is_eval,
            review=_review_metadata(review_row),
            events=events,
            phases=_group_phases(events),
            findings=findings,
            llm_exchanges=llm_exchanges,
            orphan_finding_ids=orphan_finding_ids,
        )

    async def assert_replay_equivalent(self, review_id: UUID) -> None:
        """Reconstruct and assert the review replays faithfully (verify-only).

        Runs the mode-aware checklist: deserialization, row-vs-payload
        base-column consistency, and `is_eval` coherence across the stream +
        content tables (all three via `reconstruct`), then sequence
        monotonicity, phase well-formedness (work bounded by phase markers +
        node-containment, ordering, marker agreement), proof re-verification
        (registry membership + hash recompute + proof-artifact agreement in
        full mode), cross-event reference resolution, no-orphan-stored-findings,
        historical-policy severity reconstruction, and the mode-appropriate
        content checks (full content equality only in FULL mode; metadata-only
        mode asserts shape/stubs, never content equality). Raises
        `ReplayEquivalenceError` naming the failing check; returns `None` on
        success.
        """
        review = await self.reconstruct(review_id)
        _verify_sequence_monotonic(review.events)
        # A completed review must have terminated every phase ("missing phase
        # end events on success are violations"). A review whose row is purged
        # (metadata-only) or non-completed can't assert success, so it tolerates
        # a trailing unterminated phase.
        require_all_terminated = review.review is not None and review.review.status == "completed"
        _verify_phase_wellformed(review.events, require_all_terminated=require_all_terminated)
        _verify_proof_boundary(review.events)
        _verify_cross_event_refs(review.events)
        _verify_mode_consistency(review)
        if review.orphan_finding_ids:
            raise ReplayEquivalenceError(
                f"review {review_id} has {len(review.orphan_finding_ids)} stored finding(s) "
                f"with no FindingEvent in the audit stream (append-only violation): "
                f"{[str(fid) for fid in review.orphan_finding_ids]}"
            )
        # is_eval coherence (stream + content tables) is enforced inside
        # reconstruct() via _verify_is_eval_consistent, so it guards direct
        # read-model consumers (timeline UI) too — not only this path.
        await self._verify_historical_severity(review)

    async def _verify_historical_severity(self, review: ReconstructedReview) -> None:
        """Assert each finding's severity matches its historical policy version.

        For `policy_version == ACTIVE_POLICY_VERSION` the live `SEVERITY_POLICY`
        is authoritative (and the schema validator already enforced it); for
        older versions the snapshot is loaded from `severity_policies` via
        `load_policy_for_version` (FUP-040 — the schema validator deliberately
        skips this for non-ACTIVE rows). `FindingEvent.severity` is the
        policy-assigned (pre-override) severity, so the comparison is direct.
        """
        finding_events = [e for e in review.events if isinstance(e, FindingEvent)]
        if not finding_events:
            return
        historical_versions = {
            e.policy_version for e in finding_events if e.policy_version != ACTIVE_POLICY_VERSION
        }
        snapshots: dict[str, dict[FindingType, FindingSeverity]] = {
            ACTIVE_POLICY_VERSION: dict(SEVERITY_POLICY)
        }
        if historical_versions:
            async with self._session_factory() as session:
                conn = await session.connection()
                for version in historical_versions:
                    try:
                        snapshots[version] = await load_policy_for_version(version, conn)
                    except (UnknownPolicyVersionError, PolicyVersionShapeError) as exc:
                        raise ReplayEquivalenceError(
                            f"cannot load policy version {version!r} for replay: {exc}"
                        ) from exc
        for event in finding_events:
            expected = snapshots[event.policy_version].get(event.finding_type)
            if event.severity != expected:
                raise ReplayEquivalenceError(
                    f"finding {event.finding_id} severity {event.severity} does not match "
                    f"policy {event.policy_version} assignment {expected} for "
                    f"finding_type {event.finding_type}"
                )


def _finding_content(row: Finding | None) -> FindingContent | None:
    """Project a `findings` ORM row into the full-mode content DTO (or None)."""
    if row is None:
        return None
    return FindingContent(
        finding_type=FindingType(row.finding_type),
        severity=FindingSeverity(row.severity),
        evidence_tier=EvidenceTier(row.evidence_tier),
        file_path=row.file_path,
        line_start=row.line_start,
        line_end=row.line_end,
        title=row.title,
        description=row.description,
        evidence=row.evidence,
        suggested_fix=row.suggested_fix,
        query_match_id=row.query_match_id,
        trace_path=tuple(row.trace_path) if row.trace_path is not None else None,
        original_severity=(
            FindingSeverity(row.original_severity) if row.original_severity is not None else None
        ),
        override_reason=row.override_reason,
        overrider_id=row.overrider_id,
        publish_destination=row.publish_destination,
        policy_version=row.policy_version,
        content_hash=row.content_hash,
    )


def _review_metadata(row: Review | None) -> ReconstructedReviewMetadata | None:
    """Project a `reviews` ORM row into the metadata DTO (or None if purged)."""
    if row is None:
        return None
    return ReconstructedReviewMetadata(
        review_id=row.id,
        installation_id=row.installation_id,
        status=row.status,
        is_eval=row.is_eval,
        repo_id=row.repo_id,
        pr_number=row.pr_number,
        head_sha=row.head_sha,
        files_examined=row.files_examined,
        files_traced_beyond_diff=row.files_traced_beyond_diff,
        llm_calls_made=row.llm_calls_made,
        total_input_tokens=row.total_input_tokens,
        total_output_tokens=row.total_output_tokens,
        total_cost_usd=row.total_cost_usd,
        wall_clock_seconds=row.wall_clock_seconds,
        created_at=row.created_at,
        updated_at=row.updated_at,
        completed_at=row.completed_at,
        expires_at=row.expires_at,
    )


def _verify_is_eval_consistent(
    *,
    stream_is_eval: bool,
    events: tuple[AuditEvent, ...],
    review_row: Review | None,
    finding_rows: tuple[Finding, ...],
    content_rows: tuple[LLMCallContent, ...],
) -> None:
    """Assert the audit stream and every joined content row agree on `is_eval`.

    `stream_is_eval` is the canonical review-level flag (the first audit
    event's). docs/testing.md's eval-isolation discipline requires
    `reviews` / `findings` / `llm_call_content` / `audit_events` to share one
    `is_eval`; a row that drifts would mis-bucket the reconstructed review (the
    dashboard, sweep, and anomaly queue all filter on it). Raises
    `ReplayEquivalenceError` on any disagreement. Called from `reconstruct()`
    so the read model is coherent for every consumer, not just
    `assert_replay_equivalent`.
    """
    if any(e.is_eval != stream_is_eval for e in events):
        raise ReplayEquivalenceError(
            f"audit stream has mixed is_eval flags across its events "
            f"(stream is_eval={stream_is_eval})"
        )
    if review_row is not None and review_row.is_eval != stream_is_eval:
        raise ReplayEquivalenceError(
            f"reviews row is_eval={review_row.is_eval} disagrees with the audit "
            f"stream is_eval={stream_is_eval} (eval-isolation drift)"
        )
    if any(r.is_eval != stream_is_eval for r in finding_rows):
        raise ReplayEquivalenceError(
            f"a findings row's is_eval disagrees with the audit stream "
            f"is_eval={stream_is_eval} (eval-isolation drift)"
        )
    if any(r.is_eval != stream_is_eval for r in content_rows):
        raise ReplayEquivalenceError(
            f"an llm_call_content row's is_eval disagrees with the audit stream "
            f"is_eval={stream_is_eval} (eval-isolation drift)"
        )
