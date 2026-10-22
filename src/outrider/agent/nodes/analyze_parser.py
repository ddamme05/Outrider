# See specs/2026-05-19-analyze-node.md §6
"""Admit model-proposed findings through deterministic proof and location checks.

This pure parser returns findings, trace candidates, rejection payloads, and
counters. The analyze node owns persistence and state updates.

Candidates survive proposal rejection, but a rejected response has no proposals
to collect from. OBSERVED and INFERRED require supplied query or trace proof
beyond the model's tier label.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import ValidationError

from outrider.audit.events import compute_finding_content_hash
from outrider.coordinates import (
    is_valid_import_string,
    is_valid_relative_specifier,
    relative_specifier_candidate_paths,
)
from outrider.coordinates.errors import CoordinateError
from outrider.coordinates.spans import (
    line_range_to_span,
    line_range_vetoed_by_parameterized_call,
    line_range_within_scope_unit,
    span_is_nonempty,
    span_within_degraded_context,
    span_within_file,
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
    from collections.abc import Mapping
    from uuid import UUID

    from outrider.ast_facts.models import ImportRef, ScopeUnit
    from outrider.ast_facts.parameterized_calls import ParameterizedCallScan
    from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw

# Versions the findings and trace candidates stored in the analyze cache
# (specs/2026-06-11-file-hash-analyze-cache.md; FUP-166).
# Bump on any change to `admitted_findings` or `trace_candidates`: rejection
# rules, span gates, schema construction, candidate admission/correction,
# language dispatch, or the node's post-parse OBSERVED merge.
# Merge rules: DECISIONS.md#054 and DECISIONS.md#055.
# Relative trace specifiers: DECISIONS.md#024.
# SUBSUMES_DIGEST and from_import_map_digest separately key the subsumption
# relation and import map; this version covers the rules that use them.
# Keep this version beside the parser, never injected through configuration.
ANALYZE_PARSER_VERSION: Final = "analyze-parser-v7"

# Keep these values aligned with `FindingProposalRejectedEvent.rejection_reason`
# in `audit/events.py`; the node lifts parser rejections into that event.
_ProposalRejectionReason = Literal[
    "query_match_id_not_in_registry",
    "trace_path_not_admissible",
    "finding_type_not_in_enum",
    "evidence_tier_not_in_enum",
    "span_outside_scope_unit",
    "span_outside_file",
    "span_outside_degraded_context",
    "sql_injection_on_parameterized_call",
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
    """One parser call failed to validate its response envelope.

    No proposals or candidates exist for that call. The node adds audit context
    when lifting this payload to AnalyzeResponseRejectedEvent."""

    file_path: str
    response_hash: str
    rejection_reason: Literal["raw_response_unparseable"]
    rejection_detail: str


@dataclass(frozen=True, slots=True)
class ParserCounters:
    """Per-call counters the node body sums into `AnalyzeCompletedEvent`.

    At the PARSER layer the invariant is the 2-term form
    `n_proposals_seen == n_findings_emitted + n_proposals_rejected`
    (response-level rejections produce zero proposals; `n_findings_observed`
    and `n_proposals_superseded_by_observed` are NODE-set via
    `dataclasses.replace` and are always 0 as the parser produces them). The
    node's post-merge form subtracts the observed/served terms and ADDS the
    superseded term; that full equation is enforced by
    `AnalyzeCompletedEvent._enforce_proposal_accounting` at construction.
    """

    n_proposals_seen: int
    n_findings_emitted: int
    n_proposals_rejected: int
    n_responses_rejected: int
    n_trace_candidates_emitted: int
    n_trace_candidates_dropped_malformed: int = 0
    """Count of raw trace_candidates entries DROPPED because the
    per-language form validation rejected the import_string_raw — or,
    for specifier form, admission-time repo-escape containment did
    (sharp-edges F1 audit-fold; sister field on AnalyzeCompletedEvent
    has the full rationale). Default=0 so existing fixture-driven
    constructions don't break; the parser sets it explicitly when
    emitting."""
    n_trace_candidates_module_corrected: int = 0
    """Count of corrected module-form SIBLING candidates emitted
    alongside originals whose module prefix the analyzed file's
    from-imports contradict (`_correct_against_from_imports`). NOT
    mirrored to `AnalyzeCompletedEvent` — node-log telemetry only."""
    n_findings_observed: int = 0
    """Deterministic OBSERVED-tier findings the analyze NODE merged into
    `admitted_findings` after parsing (Cost Lever 3). The parser never sets
    this (always 0 from `parse_analyze_response`); the node sets it via
    `dataclasses.replace` when it merges the producer's output. These fire
    real `FindingEvent`s (so they ride the aggregate `n_findings_emitted`)
    but have NO proposal in this pass, so the AnalyzeCompletedEvent
    accounting equation subtracts them, exactly like `n_findings_served`."""
    n_proposals_superseded_by_observed: int = 0
    """Model JUDGED proposals SEEN this pass but EVICTED at the merge point
    because a deterministic OBSERVED producer finding shared their
    `content_hash` (prefer-OBSERVED, DECISIONS.md#054). The node sets this
    via `dataclasses.replace`. A superseded proposal was counted in
    `n_proposals_seen` but is neither emitted as a finding nor rejected —
    a third proposal disposition. The accounting equation ADDS it (it sits
    on the same side as `n_proposals_rejected`); it fires no event. Default
    0 (no superseding when the producer finds nothing colliding)."""


@dataclass(frozen=True, slots=True)
class ParserResult:
    """Findings, candidates, rejection payloads and counters from one model call.

    The node audits rejections and retains candidates, merges deterministic findings,
    then emits kept findings after the round cap. Response rejection returns empty
    tuples and zero proposal counters, with n_responses_rejected=1; successful
    envelope parsing returns the per-proposal admission outcomes."""

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
    degraded_context_byte_ranges: tuple[tuple[int, int], ...] = (),
    pass_index: int = 0,
    valid_trace_path_elements: frozenset[str] = frozenset(),
    finish_reason: str = "unknown",
    parameterized_call_scan: ParameterizedCallScan | None = None,
    import_refs: tuple[ImportRef, ...] = (),
    collect_trace_candidates: bool = True,
    trace_candidate_form: Literal["module", "specifier"] = "module",
) -> ParserResult:
    """Validate the response envelope, then admit proposals in their original order.

    Gates run in order: evidence tier, finding type, structural proof, location,
    parameterized-call veto, and finding schema. The node persists the returned
    rejection payloads and supplies all file context and the policy version.

    Clean locations use the included scope units' line ranges. Degraded locations
    use full source, its precomputed UTF-8 byte length, and deterministic diff-byte
    ranges; an empty degraded context admits no findings. Degraded callers must
    supply no parameterized-call scan because their parse tree is untrustworthy.

    Pass 0 collects trace candidates unless disabled and rejects INFERRED claims.
    Later INFERRED claims require a valid trace-path shape and every element to match an
    exact rendered scope-unit label in valid_trace_path_elements. OBSERVED claims
    require an ID in the supplied query-match registry set.
    """
    # Reject negative pass indexes before parsing or admitting structural evidence.
    if pass_index < 0:
        raise ValueError(f"pass_index must be >= 0; got {pass_index}")

    try:
        # Normalize only an outer fence; malformed envelopes remain response rejections.
        raw_response = AnalyzeResponseRaw.model_validate_json(strip_outer_json_fence(response_text))
    except ValidationError as e:
        # Name token truncation directly; otherwise record bounded schema-error metadata.
        if finish_reason == "max_tokens":
            rejection_detail = (
                "LLM response truncated at the max_tokens limit; output is "
                "incomplete, so JSON validation failed. Raise max_tokens or "
                "narrow the per-file analysis scope."
            )
        else:
            rejection_detail = _format_validation_error_detail(e)
        return ParserResult(
            admitted_findings=(),
            trace_candidates=(),
            proposal_rejections=(),
            response_rejection=ResponseRejection(
                file_path=file_path,
                response_hash=compute_response_hash(response_text),
                rejection_reason="raw_response_unparseable",
                rejection_detail=rejection_detail,
            ),
            counters=ParserCounters(
                n_proposals_seen=0,
                n_findings_emitted=0,
                n_proposals_rejected=0,
                n_responses_rejected=1,
                n_trace_candidates_emitted=0,
            ),
        )

    # Evidence tier precedes finding type: only an invalid tier records a missing tier.
    admitted_findings: list[ReviewFinding] = []
    proposal_rejections: list[ProposalRejection] = []
    trace_candidates: list[TraceCandidate] = []
    n_proposals_seen = 0
    n_trace_candidates_dropped_malformed = 0
    n_trace_candidates_module_corrected = 0
    # Build correction context once. Later passes cannot feed another trace round.
    collecting_trace_candidates = pass_index == 0 and collect_trace_candidates
    from_import_modules = _build_from_import_map(import_refs) if collecting_trace_candidates else {}

    def reject_proposal(
        *,
        proposal: AnalyzeFindingProposalRaw,
        proposal_hash: str,
        candidates: list[TraceCandidate],
        rejection_reason: _ProposalRejectionReason,
        rejection_detail: str,
        claimed_evidence_tier: EvidenceTier | None,
    ) -> None:
        """Record a proposal's rejection and keep its trace candidates.

        Candidates survive rejection per spec §6 step 10 — a rejected JUDGED
        claim can still carry a legitimate cross-file signal. The caller writes
        its own `continue`, so the loop's exits stay visible at the step that
        decides them.
        """
        proposal_rejections.append(
            _build_proposal_rejection(
                proposal,
                proposal_hash=proposal_hash,
                file_path=file_path,
                rejection_reason=rejection_reason,
                rejection_detail=rejection_detail,
                claimed_evidence_tier=claimed_evidence_tier,
            )
        )
        trace_candidates.extend(candidates)

    def _structural_evidence_rejection(
        proposal: AnalyzeFindingProposalRaw,
        evidence_tier: EvidenceTier,
    ) -> tuple[_ProposalRejectionReason, str] | None:
        """Require registry proof for OBSERVED and trace proof for INFERRED.

        JUDGED needs no structural artifact. Return safe rejection metadata;
        the main loop owns recording the rejection and retaining candidates.
        """
        if evidence_tier == EvidenceTier.OBSERVED:
            claimed_id = proposal.query_match_id
            if claimed_id is None or claimed_id not in query_match_id_set:
                # The raw schema bounds length but not characters (FUP-046).
                return (
                    "query_match_id_not_in_registry",
                    _sanitize_query_match_id_for_detail(claimed_id),
                )
        elif evidence_tier == EvidenceTier.INFERRED:
            inferred_rejection_detail: str | None = None
            if pass_index == 0:
                inferred_rejection_detail = "INFERRED rejected on pass 0 (no trace context yet)"
            elif not _raw_trace_path_is_admissible(proposal.trace_path):
                inferred_rejection_detail = (
                    "INFERRED requires non-empty trace_path of non-empty strs"
                )
            else:
                ungrounded = _trace_path_elements_not_in_proof(
                    proposal.trace_path, valid_trace_path_elements
                )
                if ungrounded:
                    # Record counts, not model-supplied trace labels, in the audit detail.
                    inferred_rejection_detail = (
                        f"INFERRED trace_path has {len(ungrounded)} element(s) "
                        f"not in the trace-fetched file's scope-unit set "
                        f"(valid_trace_path_elements has "
                        f"{len(valid_trace_path_elements)} entries)"
                    )
            if inferred_rejection_detail is not None:
                return "trace_path_not_admissible", inferred_rejection_detail
        return None

    for raw_proposal in raw_response.findings:
        n_proposals_seen += 1

        # One hash joins this raw proposal's finding/rejection and trace candidates.
        proposal_hash = compute_proposal_hash(
            source_file_path=file_path,
            finding_type=raw_proposal.finding_type,
            evidence_tier=raw_proposal.evidence_tier,
            query_match_id=raw_proposal.query_match_id,
            trace_path=raw_proposal.trace_path,
            title=raw_proposal.title,
            description=raw_proposal.description,
            evidence=raw_proposal.evidence,
            line_start=raw_proposal.line_start,
            line_end=raw_proposal.line_end,
        )

        # Collect before admission: candidates survive a proposal-level rejection.
        proposal_trace_candidates: list[TraceCandidate] = []
        if collecting_trace_candidates:
            proposal_trace_candidates, n_candidates_dropped, n_candidates_corrected = (
                _collect_trace_candidates_for(
                    raw_proposal,
                    proposal_hash=proposal_hash,
                    from_import_modules=from_import_modules,
                    candidate_form=trace_candidate_form,
                    source_file_path=file_path,
                )
            )
            n_trace_candidates_dropped_malformed += n_candidates_dropped
            n_trace_candidates_module_corrected += n_candidates_corrected

        # Step 2: admit the evidence-tier enum before any other proposal gate.
        try:
            evidence_tier = EvidenceTier(raw_proposal.evidence_tier)
        except ValueError:
            reject_proposal(
                proposal=raw_proposal,
                proposal_hash=proposal_hash,
                candidates=proposal_trace_candidates,
                rejection_reason="evidence_tier_not_in_enum",
                rejection_detail="no_near_enum_match",
                claimed_evidence_tier=None,
            )
            continue

        # Step 3: type rejections carry the evidence tier already admitted above.
        try:
            finding_type = FindingType(raw_proposal.finding_type)
        except ValueError:
            reject_proposal(
                proposal=raw_proposal,
                proposal_hash=proposal_hash,
                candidates=proposal_trace_candidates,
                rejection_reason="finding_type_not_in_enum",
                rejection_detail="no_near_enum_match",
                claimed_evidence_tier=evidence_tier,
            )
            continue

        # Step 4: structural claims need deterministic proof before location checks.
        structural_rejection = _structural_evidence_rejection(raw_proposal, evidence_tier)
        if structural_rejection is not None:
            rejection_reason, rejection_detail = structural_rejection
            reject_proposal(
                proposal=raw_proposal,
                proposal_hash=proposal_hash,
                candidates=proposal_trace_candidates,
                rejection_reason=rejection_reason,
                rejection_detail=rejection_detail,
                claimed_evidence_tier=evidence_tier,
            )
            continue

        # Step 5: check raw source lines. Clean containment stays indentation-robust.
        line_start, line_end = raw_proposal.line_start, raw_proposal.line_end
        if degraded_mode:
            # Translate degraded ranges to bytes; reject past EOF or an empty span.
            try:
                span = line_range_to_span(line_start, line_end, file_content)
            except CoordinateError:
                reject_proposal(
                    proposal=raw_proposal,
                    proposal_hash=proposal_hash,
                    candidates=proposal_trace_candidates,
                    rejection_reason="span_outside_file",
                    rejection_detail=f"({line_start},{line_end})",
                    claimed_evidence_tier=evidence_tier,
                )
                continue
            if not span_is_nonempty(span) or not span_within_file(
                span, file_byte_length=file_byte_length
            ):
                detail = (
                    f"zero_width:({span.byte_start},{span.byte_end})"
                    if not span_is_nonempty(span)
                    else f"({span.byte_start},{span.byte_end})"
                )
                reject_proposal(
                    proposal=raw_proposal,
                    proposal_hash=proposal_hash,
                    candidates=proposal_trace_candidates,
                    rejection_reason="span_outside_file",
                    rejection_detail=detail,
                    claimed_evidence_tier=evidence_tier,
                )
                continue
            # Degraded evidence must intersect deterministic diff context (FUP-138).
            if not span_within_degraded_context(span, degraded_context_byte_ranges):
                reject_proposal(
                    proposal=raw_proposal,
                    proposal_hash=proposal_hash,
                    candidates=proposal_trace_candidates,
                    rejection_reason="span_outside_degraded_context",
                    rejection_detail=f"({span.byte_start},{span.byte_end})",
                    claimed_evidence_tier=evidence_tier,
                )
                continue
        elif not any(
            line_range_within_scope_unit(line_start, line_end, su) for su in included_scope_units
        ):
            # Clean ranges outside all included units share this reason, including past EOF.
            reject_proposal(
                proposal=raw_proposal,
                proposal_hash=proposal_hash,
                candidates=proposal_trace_candidates,
                rejection_reason="span_outside_scope_unit",
                rejection_detail=f"({line_start},{line_end})",
                claimed_evidence_tier=evidence_tier,
            )
            continue

        # Veto JUDGED SQL injection on a provably parameterized call (FUP-162).
        # Run after proof/location gates; degraded callers supply no trusted scan.
        if (
            parameterized_call_scan is not None
            and finding_type is FindingType.SQL_INJECTION
            and evidence_tier is EvidenceTier.JUDGED
            and line_range_vetoed_by_parameterized_call(
                line_start, line_end, parameterized_call_scan
            )
        ):
            reject_proposal(
                proposal=raw_proposal,
                proposal_hash=proposal_hash,
                candidates=proposal_trace_candidates,
                rejection_reason="sql_injection_on_parameterized_call",
                rejection_detail=f"({line_start},{line_end})",
                claimed_evidence_tier=evidence_tier,
            )
            continue

        # Steps 6-9: preserve raw lines; derive severity and dimension from policy.
        # Mapping totality is guarded at import time, so missing keys must propagate.
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
                # Preserve the same proposal identity as its trace candidates (#025).
                proposal_hash=proposal_hash,
                content_hash=compute_finding_content_hash(
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    finding_type=finding_type,
                ),
            )
        except ValidationError as exc:
            # Only schema failures become rejections; unexpected errors remain bugs.
            reject_proposal(
                proposal=raw_proposal,
                proposal_hash=proposal_hash,
                candidates=proposal_trace_candidates,
                rejection_reason="schema_construction_failed",
                rejection_detail=f"{type(exc).__name__} x1",
                claimed_evidence_tier=evidence_tier,
            )
            continue

        admitted_findings.append(finding)
        # Step 10: retain candidates from admitted proposals as well as rejected ones.
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
            n_trace_candidates_module_corrected=n_trace_candidates_module_corrected,
        ),
    )


# `claimed_finding_type_hash` width matches the schema's pattern in
# `audit/events.py` (`_SHA256_HEX_PATTERN_SHORT` — 16 hex chars).
# Per `DECISIONS.md#014` point 1, the raw model string never lands in
# the audit row; the hash+length pair lets operators reason about
# identity without admitting content.
_CLAIMED_FINDING_TYPE_HASH_WIDTH: Final[int] = 16


# Spec §3 named character class for `query_match_id` (and `trace_path`
# step values). Raw schema in `schemas/llm/analyze.py` ships only
# `max_length=256` — no pattern. This regex enforces the spec-promised
# safety class as a parser-side sanitization step.
_QUERY_MATCH_ID_SAFE_CLASS: Final = re.compile(r"[^A-Za-z0-9_./:\-]")


def _trace_path_elements_not_in_proof(
    trace_path: object,
    valid_trace_path_elements: frozenset[str],
) -> list[str]:
    """Return the trace_path elements that are NOT in the deterministic-
    proof set. Empty list iff every element is grounded.

    Pre-condition: `_raw_trace_path_is_admissible(trace_path)` returned
    True (caller already validated shape). This helper assumes `trace_path`
    is a list-or-tuple of non-empty strings; the `isinstance` guard
    inside is defensive against direct callers that bypassed the shape
    check.
    """
    if not isinstance(trace_path, (list, tuple)):
        return []  # defensive — shape-check should have rejected
    ungrounded_elements: list[str] = []
    for trace_element in trace_path:
        if trace_element not in valid_trace_path_elements:
            ungrounded_elements.append(trace_element)
    return ungrounded_elements


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


def _build_from_import_map(import_refs: tuple[ImportRef, ...]) -> dict[str, str]:
    """Map each from-imported NAME in the analyzed file to its importing
    module, later imports shadowing earlier ones (Python runtime
    semantics). Only absolute `from M import name` refs participate:
    direct imports bind module paths (nothing to correct against),
    relative modules can't be rewritten into absolute dotted candidates
    without package context, and star imports name no symbols.

    Keys are NFC-normalized so lookups from canonicalized candidate
    strings (`is_valid_import_string` NFC-normalizes) can't miss on a
    decomposed-Unicode imported name. Values are validated through
    `is_valid_import_string` AFTER shadowing resolves (last write wins,
    THEN invalid modules drop) so a later producer-bug-shaped ref can't
    resurrect an earlier shadowed module.

    Deliberately does NOT filter on `ImportRef.is_value_import`: trace
    correction asks WHERE a name's module lives, not whether runtime can
    call through it — a type-only import is still evidence of the
    defining module, so non-value refs stay in the map (unlike OBSERVED
    binding admission, which requires value imports).
    """
    mapping: dict[str, str] = {}
    for ref in import_refs:
        if ref.import_kind != "from":
            continue
        for name in ref.names:
            mapping[unicodedata.normalize("NFC", name)] = ref.module
    validated: dict[str, str] = {}
    for name, module in mapping.items():
        try:
            validated[name] = is_valid_import_string(module)
        except ValueError:
            continue
    return validated


def from_import_map_digest(import_refs: tuple[ImportRef, ...]) -> str:
    """SHA-256 over the validated from-import name→module map — the
    analyze-cache-key component pinning candidate correction's per-file
    input (the `scan_digest` / FUP-171 precedent).

    Corrected sibling candidates depend on module-level imports the
    rendered prompt does NOT carry (scope-unit bodies + hunks only), so
    two reviews with byte-identical prompts but different from-imports
    admit different trace_candidates and must never share a cache entry.
    Hashes the POST-validation map — exactly the input the correction
    consumes — as sorted `{name}={module};` pairs with a count header
    (collision-safe: `is_valid_import_string` output contains neither
    `=` nor `;`, and the header delimits the set). The empty map (no
    from-imports, failed/degraded parse) digests distinctly from any
    populated map.
    """
    mapping = _build_from_import_map(import_refs)
    body = ";".join(f"{name}={module}" for name, module in sorted(mapping.items()))
    payload = f"{len(mapping)}:{body}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _correct_against_from_imports(
    import_string: str,
    from_import_modules: Mapping[str, str],
) -> str | None:
    """Suggest the from-import-recorded module for a candidate whose
    module prefix disagrees with the analyzed file's own imports.

    Real models sometimes hallucinate the module for a symbol the file
    visibly imports (live instance: `svc.utils.normalize_owner` proposed
    while the file says `from svc.db import normalize_owner`). When the
    trailing component matches a from-imported name AND the candidate's
    prefix disagrees with the recorded importing module, that module —
    deterministic ast_facts parse output, not model output — is returned
    as a SUGGESTION (module-form, the DECISIONS.md#024 canonical shape).
    The caller emits it ALONGSIDE the original candidate, never instead:
    the trailing-name match is a heuristic, and a name collision
    (`myproject.settings` vs `from django.conf import settings`) must
    not clobber a valid candidate — the probe ladder resolves whichever
    exists. Returns None (no suggestion) for consistent candidates: a
    symbol-form prefix already matching the import, a module-form
    candidate that IS the from-imported submodule (`svc.db` under
    `from svc import db` — prefix equals the mapped module), and the
    importing module itself.
    """
    prefix, _, trailing = import_string.rpartition(".")
    module = from_import_modules.get(trailing)
    if module is None or prefix == module or import_string == module:
        return None
    return module


def _collect_trace_candidates_for(
    raw: AnalyzeFindingProposalRaw,
    *,
    proposal_hash: str,
    from_import_modules: Mapping[str, str],
    candidate_form: Literal["module", "specifier"] = "module",
    source_file_path: str = "",
) -> tuple[list[TraceCandidate], int, int]:
    """Return candidates, malformed-count and corrected-sibling count for a proposal.

    Candidate IDs bind content to source_proposal_hash whether the parent is admitted
    or rejected (spec §6 step 10). Response-level rejection has no proposals here.

    Python admits dotted modules; JS/TS admits leading-dot specifiers. Wrong forms
    and specifiers escaping the repository count as malformed (DECISIONS.md#024,
    Amended 2026-07-03). Probe construction checks containment again.

    Python from-import correction adds a module-form sibling beside the original,
    never replaces it. The final counter counts emitted synthetic siblings."""
    # Validate first, then hash and store the same canonical import. Raw NFD
    # bytes would produce an ID that the schema's NFC recomputation rejects.
    out: list[TraceCandidate] = []
    n_dropped_malformed = 0
    n_module_corrected = 0
    for raw_cand in raw.trace_candidates:
        try:
            if candidate_form == "module":
                canonical_import = is_valid_import_string(raw_cand.import_string_raw)
            else:
                canonical_import = is_valid_relative_specifier(raw_cand.import_string_raw)
        except ValueError:
            # Count malformed advisory candidates without rejecting their parent
            # proposal or copying raw model output into audit (sharp-edges F1).
            n_dropped_malformed += 1
            continue
        if candidate_form == "specifier" and not relative_specifier_candidate_paths(
            canonical_import, source_file_path
        ):
            # Reject repository escapes at admission (DECISIONS.md#024,
            # Amended 2026-07-03). Probe construction checks containment again.
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
            # Keep schema failures in the same malformed-candidate counter.
            n_dropped_malformed += 1
            continue
        out.append(candidate)

        # Add a corrected module sibling; the trailing-name match is a heuristic,
        # so keep the original for the probe ladder (DECISIONS.md#024).
        if candidate_form != "module":
            continue
        suggested_module = _correct_against_from_imports(canonical_import, from_import_modules)
        if suggested_module is not None:
            try:
                sibling = TraceCandidate(
                    candidate_id=compute_candidate_id(
                        source_proposal_hash=proposal_hash,
                        import_string=suggested_module,
                        reason=raw_cand.reason,
                    ),
                    source_proposal_hash=proposal_hash,
                    reason=raw_cand.reason,
                    import_string=suggested_module,
                )
            except ValidationError:
                # Only rejected raw candidates count as malformed. This sibling
                # is synthetic, and its original candidate already survived.
                continue
            out.append(sibling)
            n_module_corrected += 1
    return out, n_dropped_malformed, n_module_corrected


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
        if not loc:
            # Root-level error (empty location) — name it by error TYPE, which
            # only this caller has: `json_invalid` is a JSON-SYNTAX failure
            # (e.g. an unescaped quote inside a string — the 2026-06-12 live
            # run); anything else at the root (e.g. `model_type` for a valid
            # `[]`/`null` where the response object is expected) is a schema
            # failure of well-formed JSON. Conflating them would misdirect
            # the diagnosis the audit detail exists to enable.
            loc = "json_syntax" if err["type"] == "json_invalid" else "root_schema"
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
    segments separate with `.`. An EMPTY location renders as the empty
    string here — the CALLER (`_format_validation_error_detail`) names
    root-level errors by error type (`json_syntax` vs `root_schema`),
    which only it can see. Pure formatter; no IO.
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
