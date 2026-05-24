# See specs/2026-05-19-analyze-node.md §6
"""Analyze parser — proof-boundary admission for model proposals.

**Boundary contract.** Pure function. Takes a raw provider response
plus the per-file context the node body already assembled, applies
the spec §6 10-step admission flow, and returns a `ParserResult`:

- admitted `ReviewFinding`s (one per proposal passing every gate),
- collected `TraceCandidate`s (from both admitted and proposal-rejected
  raw proposals; response-level rejections produce none),
- proposal-level rejection payloads (one per proposal that failed
  admission — finding-type-not-in-enum, evidence-tier-not-in-enum,
  query-match-id-not-in-registry, trace-path-not-admissible,
  span-outside-scope-unit, span-outside-file, schema-construction-
  failed),
- a single optional response-level rejection (set iff step 0 —
  `AnalyzeResponseRaw.model_validate_json` — failed),
- counters for `AnalyzeCompletedEvent`.

**No IO.** The parser does NOT call the persister and does NOT emit
events. The node body lifts each `ProposalRejection` to a
`FindingProposalRejectedEvent` and each `ResponseRejection` to an
`AnalyzeResponseRejectedEvent` by adding audit-context fields at
construction. The parser is exercisable as a pure-data transformation
independent of the persister mock surface.

Spec uses "emit" in step descriptions; the shipped shape reads that as
"produces the event content," with persistence owned by the node body.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import ValidationError

from outrider.audit.events import compute_finding_content_hash
from outrider.coordinates import is_valid_import_string
from outrider.coordinates.errors import CoordinateError
from outrider.coordinates.spans import (
    span_is_nonempty,
    span_to_line_range,
    span_within_file,
    span_within_scope_unit,
)
from outrider.llm.parsing import strip_outer_json_fence
from outrider.policy.canonical import (
    compute_candidate_id,
    compute_proposal_hash,
    compute_response_hash,
)
from outrider.policy.dimensions import FINDING_TYPE_TO_DIMENSION
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import SEVERITY_POLICY, FindingType
from outrider.schemas import ReviewFinding, TraceCandidate
from outrider.schemas.llm.analyze import AnalyzeResponseRaw

if TYPE_CHECKING:
    from uuid import UUID

    from outrider.ast_facts.models import ScopeUnit
    from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw

# Mirrors `FindingProposalRejectedEvent.rejection_reason` literal at
# `audit/events.py:893`. Duplicated here so this module doesn't depend
# on `audit/events.py` at import time (keeps the parser's pure-data
# discipline visible at the import graph). If these drift, parser-
# produced `ProposalRejection.rejection_reason` values that the event
# rejects at construction fire at node-lift time — the wide audit
# test suite catches the drift on first run.
_ProposalRejectionReason = Literal[
    "query_match_id_not_in_registry",
    "trace_path_not_admissible",
    "finding_type_not_in_enum",
    "evidence_tier_not_in_enum",
    "span_outside_scope_unit",
    "span_outside_file",
    "schema_construction_failed",
]


@dataclass(frozen=True, slots=True)
class ProposalRejection:
    """Parser output for a proposal that failed admission.

    Lifted to `FindingProposalRejectedEvent` by the node body. The
    parser produces these fields; the node adds audit-context
    (`review_id`, `event_id`, `timestamp`, `sequence_number`,
    `is_eval`, `node_id`, `event_type`) at lift time.

    Frozen + slots: positional unpacking raises `TypeError` (dataclass
    is not a tuple); attribute reassignment raises
    `dataclasses.FrozenInstanceError`. Same swap-impossibility
    discipline as `AnalyzePromptParts` and `TriagePromptParts`.
    """

    proposal_hash: str
    file_path: str
    claimed_finding_type_hash: str
    claimed_finding_type_len: int
    claimed_evidence_tier: EvidenceTier | None
    rejection_reason: _ProposalRejectionReason
    rejection_detail: str


@dataclass(frozen=True, slots=True)
class ResponseRejection:
    """Parser output for a response that failed `AnalyzeResponseRaw` parsing.

    Lifted to `AnalyzeResponseRejectedEvent` by the node body. Same
    audit-context fields added at lift time as for `ProposalRejection`.
    At most one `ResponseRejection` per analyze pass — parser step 0
    either parses the response (then `ParserResult.response_rejection
    is None`) or fails (then it's set and no proposals exist).
    """

    file_path: str
    response_hash: str
    rejection_reason: Literal["raw_response_unparseable"]
    rejection_detail: str


@dataclass(frozen=True, slots=True)
class ParserCounters:
    """Per-call counters the node body sums into `AnalyzeCompletedEvent`.

    `n_proposals_seen == n_findings_emitted + n_proposals_rejected`
    holds at the parser layer (response-level rejections produce zero
    proposals); the equation is also enforced by
    `AnalyzeCompletedEvent._enforce_proposal_accounting` at construction.
    """

    n_proposals_seen: int
    n_findings_emitted: int
    n_proposals_rejected: int
    n_responses_rejected: int
    n_trace_candidates_emitted: int
    n_trace_candidates_dropped_malformed: int = 0
    """Count of raw trace_candidates entries DROPPED because
    `is_valid_import_string` rejected the import_string_raw (sharp-edges
    F1 audit-fold; sister field on AnalyzeCompletedEvent has the full
    rationale). Default=0 so existing fixture-driven constructions
    don't break; the parser sets it explicitly when emitting."""


@dataclass(frozen=True, slots=True)
class ParserResult:
    """Per-call parser output. The node body consumes this to:

    - persist a `FindingEvent` per `admitted_findings` entry,
    - append `trace_candidates` to `ReviewState.trace_candidates`,
    - persist a `FindingProposalRejectedEvent` per `proposal_rejections` entry,
    - persist a `AnalyzeResponseRejectedEvent` if `response_rejection is not None`,
    - sum `counters` into the per-pass `AnalyzeCompletedEvent`.

    The parser populates fields in one of two shapes:

    1. Response-level rejection (parser step 0 failed):
       `admitted_findings == ()`, `trace_candidates == ()`,
       `proposal_rejections == ()`, `response_rejection is not None`,
       counters all zero except `n_responses_rejected = 1`.
    2. Proposal-by-proposal (parser step 0 succeeded):
       `response_rejection is None`; the other tuples carry the
       per-proposal outcomes.
    """

    admitted_findings: tuple[ReviewFinding, ...]
    trace_candidates: tuple[TraceCandidate, ...]
    proposal_rejections: tuple[ProposalRejection, ...]
    response_rejection: ResponseRejection | None
    counters: ParserCounters


def parse_analyze_response(
    response_text: str,
    *,
    review_id: UUID,
    installation_id: int,
    file_path: str,
    file_content: str,
    file_byte_length: int,
    included_scope_units: tuple[ScopeUnit, ...],
    query_match_id_set: frozenset[str],
    degraded_mode: bool,
    active_policy_version: str,
    pass_index: int = 0,
) -> ParserResult:
    """Apply the spec §6 10-step admission flow to a raw analyze response.

    Pure function — no IO. All inputs are passed; outputs go into the
    returned `ParserResult`. The node body owns persistence and state
    updates.

    Inputs:

    - `response_text` — raw text from `LLMResponse.text` for this file.
    - `review_id` — for `ReviewFinding.review_id` on admitted proposals.
    - `installation_id` — for `ReviewFinding.installation_id`.
    - `file_path` — repo-relative path; goes into every emitted
      payload, already canonicalized at intake.
    - `file_content` — full file source as `str`. Needed for
      `coordinates.span_to_line_range(...)` translation.
    - `file_byte_length` — `len(file_content.encode("utf-8"))`
      computed ONCE in the node body and passed here; the parser does
      NOT recompute per proposal.
    - `included_scope_units` — the scope units this call's prompt
      included (their byte ranges define the `span_within_scope_unit`
      check for clean outcomes).
    - `query_match_id_set` — the pre-fired registry IDs the prompt
      supplied; OBSERVED admission rejects any claimed id not in this
      set. Empty for degraded outcomes.
    - `degraded_mode` — branches parser step 5: clean uses
      `span_within_scope_unit`, degraded uses `span_within_file`.
    - `active_policy_version` — closure-captured per
      `nodes-receive-deps-via-closure`; goes into
      `ReviewFinding.policy_version` on admitted proposals.

    The node body owns `pass_index` for `AnalyzeCompletedEvent.pass_index`
    and now threads it here for INFERRED admission: pass 0 (the original
    PR-diff analyze pass) rejects every INFERRED proposal — no trace
    context exists yet — while pass 1+ (post-trace re-entry per M8 loop)
    admits INFERRED proposals carrying a non-empty `trace_path` (the
    proof-boundary validator at `policy/findings.py::_trace_path_is_valid`
    is the schema-level gate for trace_path shape).
    """
    try:
        # Strip a single outer ```json...``` wrapper if present — the
        # model sometimes adds one despite the system-prompt instruction.
        # Malformed wrappers fall through unchanged so Pydantic produces
        # a clean ResponseRejection. Previously, fenced output was
        # silently rejected as `raw_response_unparseable` and the file's
        # findings were lost; this defense closes that coverage hole.
        raw = AnalyzeResponseRaw.model_validate_json(strip_outer_json_fence(response_text))
    except ValidationError as e:
        return ParserResult(
            admitted_findings=(),
            trace_candidates=(),
            proposal_rejections=(),
            response_rejection=ResponseRejection(
                file_path=file_path,
                response_hash=compute_response_hash(response_text),
                rejection_reason="raw_response_unparseable",
                rejection_detail=_format_validation_error_detail(e),
            ),
            counters=ParserCounters(
                n_proposals_seen=0,
                n_findings_emitted=0,
                n_proposals_rejected=0,
                n_responses_rejected=1,
                n_trace_candidates_emitted=0,
            ),
        )

    # Per-proposal admission. Evidence-tier admission runs FIRST (spec §6
    # step 2) so the bidirectional cross-field validator on the lifted
    # `FindingProposalRejectedEvent` holds: `claimed_evidence_tier is None`
    # iff `rejection_reason == "evidence_tier_not_in_enum"`. Finding-type
    # admission runs SECOND (spec §6 step 3) and carries the parsed enum
    # value through to its rejection event when it fires.
    admitted_findings: list[ReviewFinding] = []
    proposal_rejections: list[ProposalRejection] = []
    trace_candidates: list[TraceCandidate] = []
    n_proposals_seen = 0
    n_trace_candidates_dropped_malformed = 0
    for raw_proposal in raw.findings:
        n_proposals_seen += 1

        # Pre-compute `proposal_hash` once per iteration. The same hash
        # feeds the rejection payload (if any) AND the trace-candidate
        # collection's `source_proposal_hash` — the audit-trail join
        # between rejection events and trace candidates depends on
        # both using the identical hash value.
        proposal_hash = compute_proposal_hash(
            source_file_path=file_path,
            finding_type=raw_proposal.finding_type,
            evidence_tier=raw_proposal.evidence_tier,
            query_match_id=raw_proposal.query_match_id,
            trace_path=raw_proposal.trace_path,
            title=raw_proposal.title,
            description=raw_proposal.description,
            evidence=raw_proposal.evidence,
            byte_start=raw_proposal.span.byte_start,
            byte_end=raw_proposal.span.byte_end,
        )

        # Trace candidates are collected from BOTH admitted and
        # proposal-level-rejected raw proposals (spec §6 step 10) so
        # a rejected JUDGED-claim might still surface a legitimate
        # cross-file signal. Pre-compute here and `.extend(...)` on
        # whichever branch the iteration takes.
        proposal_trace_candidates, n_dropped = _collect_trace_candidates_for(
            raw_proposal, proposal_hash=proposal_hash
        )
        n_trace_candidates_dropped_malformed += n_dropped

        # Step 2: evidence_tier enum admission (runs first per the
        # bidirectional-validator requirement above).
        try:
            evidence_tier = EvidenceTier(raw_proposal.evidence_tier)
        except ValueError:
            proposal_rejections.append(
                _build_proposal_rejection(
                    raw_proposal,
                    proposal_hash=proposal_hash,
                    file_path=file_path,
                    rejection_reason="evidence_tier_not_in_enum",
                    rejection_detail="no_near_enum_match",
                    claimed_evidence_tier=None,
                )
            )
            trace_candidates.extend(proposal_trace_candidates)
            continue

        # Step 3: finding_type enum admission (runs second so
        # claimed_evidence_tier carries the parsed enum value).
        try:
            finding_type = FindingType(raw_proposal.finding_type)
        except ValueError:
            proposal_rejections.append(
                _build_proposal_rejection(
                    raw_proposal,
                    proposal_hash=proposal_hash,
                    file_path=file_path,
                    rejection_reason="finding_type_not_in_enum",
                    rejection_detail="no_near_enum_match",
                    claimed_evidence_tier=evidence_tier,
                )
            )
            trace_candidates.extend(proposal_trace_candidates)
            continue

        # Step 4: producer admission for OBSERVED / INFERRED. JUDGED
        # skips the producer check (model can claim JUDGED unilaterally;
        # carries no structural artifact).
        if evidence_tier == EvidenceTier.OBSERVED:
            claimed_id = raw_proposal.query_match_id
            if claimed_id is None or claimed_id not in query_match_id_set:
                proposal_rejections.append(
                    _build_proposal_rejection(
                        raw_proposal,
                        proposal_hash=proposal_hash,
                        file_path=file_path,
                        rejection_reason="query_match_id_not_in_registry",
                        # Spec §3 names `[A-Za-z0-9_./:-]+` as the
                        # safety pattern for storing query_match_id
                        # verbatim in rejection_detail, BUT the raw
                        # schema at `schemas/llm/analyze.py:87` only
                        # ships `max_length=256` — no pattern. Without
                        # sanitization, ANSI escapes / Trojan-Source /
                        # XSS payloads would land verbatim. Apply the
                        # spec-named pattern here as a sanitization
                        # step: replace any out-of-class char with
                        # `?` so the structural shape is preserved for
                        # operator visibility while the dangerous
                        # bytes are stripped. Tracked as FUP-046 to
                        # align the raw schema with spec §3 intent.
                        rejection_detail=_sanitize_query_match_id_for_detail(claimed_id),
                        claimed_evidence_tier=evidence_tier,
                    )
                )
                trace_candidates.extend(proposal_trace_candidates)
                continue
        elif evidence_tier == EvidenceTier.INFERRED:
            # Pass-conditional admission per the trace-node arc (M8 loop).
            # Pass 0: trace hasn't run yet — no trace context exists —
            # so every INFERRED proposal is rejected. The model should
            # emit JUDGED for cross-file or walk-derived reasoning on
            # pass 0 (mirrors the pass-0 prompt instruction at
            # prompts/analyze.py).
            # Pass 1+: trace ran + fetched files; INFERRED is admitted
            # when `trace_path` is non-empty (the proof-boundary check
            # at `policy/findings.py::_trace_path_is_valid` enforces
            # the shape — non-empty list-or-tuple of non-empty strs).
            # Empty / wrong-shape trace_path on pass 1+ is rejected with
            # the same `trace_path_not_admissible` reason.
            if pass_index == 0 or not _raw_trace_path_is_admissible(raw_proposal.trace_path):
                detail = (
                    "INFERRED rejected on pass 0 (no trace context yet)"
                    if pass_index == 0
                    else "INFERRED requires non-empty trace_path of non-empty strs"
                )
                proposal_rejections.append(
                    _build_proposal_rejection(
                        raw_proposal,
                        proposal_hash=proposal_hash,
                        file_path=file_path,
                        rejection_reason="trace_path_not_admissible",
                        rejection_detail=detail,
                        claimed_evidence_tier=evidence_tier,
                    )
                )
                trace_candidates.extend(proposal_trace_candidates)
                continue
            # Pass 1+ with valid trace_path: fall through to step 5
            # (span admission). The proof-boundary validator at
            # `policy/findings.py::enforce_proof_boundary` runs again
            # at ReviewFinding construction.
        # JUDGED falls through to step 5 (span admission).

        # Step 5: span admission (per-outcome branch). The
        # `byte_start < byte_end` predicate is part of admission on both
        # paths — `Span` itself admits zero-width (`byte_end >= byte_start`)
        # so the parser carries the prompt's stricter rule. A zero-width
        # finding anchors to no bytes; the rejection_detail prefix
        # `zero_width:` distinguishes it from EOF-overflow on the same
        # rejection reason.
        is_nonempty_span = span_is_nonempty(raw_proposal.span)
        if degraded_mode:
            # Degraded outcomes have no scope-unit context (the file
            # didn't parse, or had has_error nodes in changed regions).
            # The deterministic guard is "within file" — model can't
            # fabricate a span pointing past EOF or before BOF.
            if not is_nonempty_span or not span_within_file(
                raw_proposal.span, file_byte_length=file_byte_length
            ):
                detail = (
                    f"zero_width:({raw_proposal.span.byte_start},{raw_proposal.span.byte_end})"
                    if not is_nonempty_span
                    else f"({raw_proposal.span.byte_start},{raw_proposal.span.byte_end})"
                )
                proposal_rejections.append(
                    _build_proposal_rejection(
                        raw_proposal,
                        proposal_hash=proposal_hash,
                        file_path=file_path,
                        rejection_reason="span_outside_file",
                        rejection_detail=detail,
                        claimed_evidence_tier=evidence_tier,
                    )
                )
                trace_candidates.extend(proposal_trace_candidates)
                continue
        else:
            # Clean outcome — span must land inside one of the file's
            # included scope units. `any(...)` over the included set;
            # rejection if none match.
            if not is_nonempty_span or not any(
                span_within_scope_unit(raw_proposal.span, su) for su in included_scope_units
            ):
                detail = (
                    f"zero_width:({raw_proposal.span.byte_start},{raw_proposal.span.byte_end})"
                    if not is_nonempty_span
                    else f"({raw_proposal.span.byte_start},{raw_proposal.span.byte_end})"
                )
                proposal_rejections.append(
                    _build_proposal_rejection(
                        raw_proposal,
                        proposal_hash=proposal_hash,
                        file_path=file_path,
                        rejection_reason="span_outside_scope_unit",
                        rejection_detail=detail,
                        claimed_evidence_tier=evidence_tier,
                    )
                )
                trace_candidates.extend(proposal_trace_candidates)
                continue

        # Steps 6-9: admission succeeded; construct `ReviewFinding`.
        # `line_start`/`line_end` translate via `coordinates.span_to_line_range`
        # (coordinate translation lives in `coordinates/` per
        # `coordinates-module-is-sole-translator`). The byte→line
        # conversion can raise `CoordinateError` — defense-in-depth,
        # since step-5 admission already gated EOF-overflow spans.
        # Narrow to `CoordinateError` so a `MemoryError` /
        # `RecursionError` / etc. propagates loud (root-cause
        # forensics depends on not folding unexpected exceptions into
        # `schema_construction_failed`).
        try:
            line_start, line_end = span_to_line_range(raw_proposal.span, source=file_content)
        except CoordinateError as exc:
            proposal_rejections.append(
                _build_proposal_rejection(
                    raw_proposal,
                    proposal_hash=proposal_hash,
                    file_path=file_path,
                    rejection_reason="schema_construction_failed",
                    rejection_detail=f"{type(exc).__name__} x1",
                    claimed_evidence_tier=evidence_tier,
                )
            )
            trace_candidates.extend(proposal_trace_candidates)
            continue

        # Severity from `SEVERITY_POLICY[finding_type]` per
        # `severity-set-by-policy`. Dimension from
        # `FINDING_TYPE_TO_DIMENSION[finding_type]` per
        # `evidence-tier-schema-enforced` + the dimensions table. Both
        # lookups are total (totality is pinned by
        # `policy.dimensions.verify_lockstep` at import time) — a
        # missing key here means the lockstep guard was bypassed,
        # which is a load-bearing module-load assertion failure, not
        # a parser fallback.
        try:
            finding = ReviewFinding(
                review_id=review_id,
                installation_id=installation_id,
                policy_version=active_policy_version,
                finding_type=finding_type,
                dimension=FINDING_TYPE_TO_DIMENSION[finding_type],
                severity=SEVERITY_POLICY[finding_type],
                evidence_tier=evidence_tier,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                title=raw_proposal.title,
                description=raw_proposal.description,
                evidence=raw_proposal.evidence,
                query_match_id=raw_proposal.query_match_id,
                trace_path=raw_proposal.trace_path,
                # Per DECISIONS.md#025: admitted findings carry the
                # `proposal_hash` for trace's join contract. Same value
                # the rejected branch (proposal_rejections.append(...))
                # passes to FindingProposalRejectedEvent — both branches
                # use the SAME pre-computed `proposal_hash` from the
                # per-proposal compute_proposal_hash call above. Provenance
                # link between TraceCandidate.source_proposal_hash and
                # admitted finding_id is closed at admission time.
                proposal_hash=proposal_hash,
                content_hash=compute_finding_content_hash(
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    finding_type=finding_type,
                ),
            )
        except ValidationError as exc:
            # Spec §6 step 8 fallback: Pydantic ValidationError after
            # passing every parser step is structurally unexpected —
            # fold to `schema_construction_failed` so the rejection is
            # auditable rather than crashing the whole pass. Narrowed
            # to `ValidationError` so genuinely-unexpected failures
            # (MemoryError, RecursionError, etc.) propagate as bugs.
            proposal_rejections.append(
                _build_proposal_rejection(
                    raw_proposal,
                    proposal_hash=proposal_hash,
                    file_path=file_path,
                    rejection_reason="schema_construction_failed",
                    rejection_detail=f"{type(exc).__name__} x1",
                    claimed_evidence_tier=evidence_tier,
                )
            )
            trace_candidates.extend(proposal_trace_candidates)
            continue

        admitted_findings.append(finding)
        # Step 10: collect trace candidates from THIS admitted proposal.
        # Same shape as the rejection branches above; only the
        # accumulator updates differ.
        trace_candidates.extend(proposal_trace_candidates)

    return ParserResult(
        admitted_findings=tuple(admitted_findings),
        trace_candidates=tuple(trace_candidates),
        proposal_rejections=tuple(proposal_rejections),
        response_rejection=None,
        counters=ParserCounters(
            n_proposals_seen=n_proposals_seen,
            n_findings_emitted=len(admitted_findings),
            n_proposals_rejected=len(proposal_rejections),
            n_responses_rejected=0,
            n_trace_candidates_emitted=len(trace_candidates),
            n_trace_candidates_dropped_malformed=n_trace_candidates_dropped_malformed,
        ),
    )


# `claimed_finding_type_hash` width matches the schema's pattern at
# `audit/events.py:891` (`_SHA256_HEX_PATTERN_SHORT` — 16 hex chars).
# Per `DECISIONS.md#014` point 1, the raw model string never lands in
# the audit row; the hash+length pair lets operators reason about
# identity without admitting content.
_CLAIMED_FINDING_TYPE_HASH_WIDTH: Final[int] = 16


# Spec §3 named character class for `query_match_id` (and `trace_path`
# step values). Raw schema at `schemas/llm/analyze.py:87` ships only
# `max_length=256` — no pattern. This regex enforces the spec-promised
# safety class as a parser-side sanitization step.
_QUERY_MATCH_ID_SAFE_CLASS: Final = re.compile(r"[^A-Za-z0-9_./:\-]")


def _raw_trace_path_is_admissible(trace_path: object) -> bool:
    """Pre-construction shape check on a raw `trace_path` proposal value.

    Mirrors `policy/findings.py::_trace_path_is_valid` exactly so the
    parser's pass-1 INFERRED admission gate and the ReviewFinding
    proof-boundary validator agree on what "non-empty trace_path of
    non-empty strs" means. Type-agnostic on input (the raw proposal
    schema admits `list[str] | None` but the parser must guard against
    a future schema relaxation that admits other types).
    """
    if not isinstance(trace_path, (list, tuple)) or not trace_path:
        return False
    return all(isinstance(item, str) and item for item in trace_path)


def _sanitize_query_match_id_for_detail(claimed_id: str | None) -> str:
    """Replace any char outside the spec-named safety class with `?`.

    Per `DECISIONS.md#014` point 1: audit rows must not carry
    user code or prompt/completion content. Spec §3 cites the
    `[A-Za-z0-9_./:-]+` pattern as the safety floor that makes the
    raw value safe to record verbatim — but the raw schema didn't
    ship the pattern (FUP-046 tracks the alignment). This helper
    enforces the spec-promised character class so attacker-controlled
    ANSI escapes / Trojan-Source / shell-metachars cannot land in
    `FindingProposalRejectedEvent.rejection_detail`.

    The structural shape is preserved (length + safe chars intact),
    so operators investigating a registry-mismatch rejection still
    see the structural form. The dropped chars are replaced with `?`
    rather than stripped to keep the length signal accurate.
    """
    if claimed_id is None:
        return "<absent>"
    return _QUERY_MATCH_ID_SAFE_CLASS.sub("?", claimed_id)


def _hash_claimed_finding_type(raw_value: str) -> str:
    """sha256(raw_value.encode("utf-8")).hexdigest()[:16] — short-prefix
    hash for `ProposalRejection.claimed_finding_type_hash`. Lifted to
    `FindingProposalRejectedEvent.claimed_finding_type_hash` by the
    node body."""
    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:_CLAIMED_FINDING_TYPE_HASH_WIDTH]


def _build_proposal_rejection(
    raw: AnalyzeFindingProposalRaw,
    *,
    proposal_hash: str,
    file_path: str,
    rejection_reason: _ProposalRejectionReason,
    rejection_detail: str,
    claimed_evidence_tier: EvidenceTier | None,
) -> ProposalRejection:
    """Construct a `ProposalRejection` from a raw proposal + admission outcome.

    Caller pre-computes `proposal_hash` via `compute_proposal_hash`
    once per iteration (the same hash feeds
    `_collect_trace_candidates_for`'s `source_proposal_hash` so the
    rejection-event-↔-trace-candidate join holds). The helper composes
    the rejection payload from the raw + branch-specific fields
    (reason, detail, claimed evidence-tier where parsed) and the
    `claimed_finding_type_hash` / `_len` per `DECISIONS.md#014`.
    """
    return ProposalRejection(
        proposal_hash=proposal_hash,
        file_path=file_path,
        claimed_finding_type_hash=_hash_claimed_finding_type(raw.finding_type),
        claimed_finding_type_len=len(raw.finding_type),
        claimed_evidence_tier=claimed_evidence_tier,
        rejection_reason=rejection_reason,
        rejection_detail=rejection_detail,
    )


def _collect_trace_candidates_for(
    raw: AnalyzeFindingProposalRaw,
    *,
    proposal_hash: str,
) -> tuple[list[TraceCandidate], int]:
    """Build the `TraceCandidate` list from a raw proposal's
    `trace_candidates`. Each candidate's `candidate_id` is content-
    derived via `compute_candidate_id`, `source_proposal_hash` is the
    parent proposal's hash (same value whether the parent was admitted
    or proposal-level-rejected — the audit-trail join is preserved
    across both outcomes per spec §6 step 10).

    Returns a list (caller extends its accumulator). Response-level
    rejections never reach this helper because no proposals exist at
    that point.
    """
    # Canonicalize `import_string_raw` ONCE up-front and use the result
    # for both `compute_candidate_id` and `TraceCandidate(...)`. Two
    # failure shapes are guarded:
    #
    # - Malformed import string (path separator, shell metachar, empty
    #   part, Python keyword, non-identifier part) →
    #   `is_valid_import_string` raises `ValueError`.
    # - Decomposed-Unicode import (e.g. `café.bar` in NFD) canonicalizes
    #   successfully via NFC normalization, but feeding the RAW value
    #   to `compute_candidate_id` would produce an id the schema's
    #   recomputation rejects (since the schema validator NFC-normalizes
    #   first; raw NFD bytes hash differently from NFC bytes).
    #
    # Without the up-front canonicalize + try/except, a single bad
    # candidate crashes the whole pass and breaks the
    # `n_proposals_seen == admitted + rejected` accounting equation.
    # Per DECISIONS.md#024 trace candidates are dotted Python import
    # strings (V1; no file-path fallback) — this helper switched from
    # `validate_diff_path` (path-shaped) to `is_valid_import_string`
    # (identifier-shaped) in the same DECISIONS-aligned commit.
    out: list[TraceCandidate] = []
    n_dropped_malformed = 0
    for raw_cand in raw.trace_candidates:
        try:
            canonical_import = is_valid_import_string(raw_cand.import_string_raw)
        except ValueError:
            # Malformed import string — drop silently AND increment the
            # forensic counter (sharp-edges F1 audit-fold). Per spec §6
            # step 10 trace candidates are advisory; dropping one bad
            # candidate is preferable to crashing the whole parser pass.
            # The dropped candidate's parent proposal still produces its
            # own rejection/admission outcome independently. The counter
            # surfaces aggregate drift so operators can distinguish
            # "model proposed nothing" from "every proposal was
            # malformed" without sanitizing raw model output into an
            # audit row.
            n_dropped_malformed += 1
            continue
        try:
            candidate = TraceCandidate(
                candidate_id=compute_candidate_id(
                    source_proposal_hash=proposal_hash,
                    import_string=canonical_import,
                    reason=raw_cand.reason,
                ),
                source_proposal_hash=proposal_hash,
                reason=raw_cand.reason,
                import_string=canonical_import,
            )
        except ValidationError:
            # Defense-in-depth: should not fire given `canonical_import`
            # just passed `is_valid_import_string` AND the candidate_id
            # is computed canonically. If it does, drop AND count as
            # malformed (same forensic bucket).
            n_dropped_malformed += 1
            continue
        out.append(candidate)
    return out, n_dropped_malformed


# Max length matches `Field(max_length=500)` on
# `AnalyzeResponseRejectedEvent.rejection_detail`. Truncate just under
# the schema cap so the lifted event constructs cleanly even when the
# Pydantic error count is pathological.
_REJECTION_DETAIL_MAX_LEN: Final[int] = 500


def _format_validation_error_detail(error: ValidationError) -> str:
    """Render a `ValidationError` as JSON-pointer paths + per-path counts.

    Per `DECISIONS.md#014` point 1: audit rows must not carry user code
    or prompt/completion content. The standard `str(error)` rendering
    includes `input_value=` snippets of the model response — exactly
    the leak this gate exists to close. This formatter walks
    `error.errors()` and emits only the location path + count — never
    the `input` field — for every error grouped by JSON pointer.

    Output shape matches the spec §3 example:
    `findings[0].finding_type x1, findings[0].evidence_tier x1`.
    Result is truncated to `_REJECTION_DETAIL_MAX_LEN` chars (with a
    trailing `"..."` marker when truncation fires) so the lifted
    `AnalyzeResponseRejectedEvent.rejection_detail` Field(max_length=500)
    accepts it.
    """
    path_counts: dict[str, int] = {}
    for err in error.errors():
        loc = _format_loc(err["loc"])
        path_counts[loc] = path_counts.get(loc, 0) + 1
    parts = [f"{path} x{count}" for path, count in sorted(path_counts.items())]
    rendered = ", ".join(parts) if parts else "no_errors"
    if len(rendered) > _REJECTION_DETAIL_MAX_LEN:
        rendered = rendered[: _REJECTION_DETAIL_MAX_LEN - 3] + "..."
    return rendered


def _format_loc(loc: tuple[str | int, ...]) -> str:
    """Render a Pydantic error location tuple as a JSON-pointer path.

    `("findings", 0, "finding_type")` → `"findings[0].finding_type"`.
    Integer segments attach to the preceding string as `[N]`; string
    segments separate with `.`. Pure formatter; no IO.
    """
    parts: list[str] = []
    for seg in loc:
        if isinstance(seg, int):
            if parts:
                parts[-1] = parts[-1] + f"[{seg}]"
            else:
                parts.append(f"[{seg}]")
        else:
            parts.append(str(seg))
    return ".".join(parts)


__all__ = [
    "ParserCounters",
    "ParserResult",
    "ProposalRejection",
    "ResponseRejection",
    "parse_analyze_response",
]
