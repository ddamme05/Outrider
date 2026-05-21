# Analyze response parser per specs/2026-05-19-analyze-node.md §6
"""Analyze parser — proof-boundary admission for model proposals.

**Boundary contract.** The parser is a PURE function. It takes a raw
provider response text plus the per-file context the node body has
already assembled, applies the spec §6 10-step admission flow, and
returns a `ParserResult` carrying:

- admitted `ReviewFinding`s (one per proposal that passed every gate),
- collected `TraceCandidate`s (from both admitted and proposal-level-
  rejected raw proposals; response-level rejections produce none),
- proposal-level rejection payloads (one per proposal that failed
  admission — finding-type-not-in-enum, evidence-tier-not-in-enum,
  query-match-id-not-in-registry, trace-path-not-admissible,
  span-outside-scope-unit, span-outside-file, schema-construction-
  failed),
- a single optional response-level rejection payload (set iff parser
  step 0 — `AnalyzeResponseRaw.model_validate_json` — failed),
- counters for `AnalyzeCompletedEvent`.

**No IO.** The parser does NOT call the audit persister and does NOT
emit events. The node body lifts each `ProposalRejection` to a
`FindingProposalRejectedEvent` and each `ResponseRejection` to an
`AnalyzeResponseRejectedEvent` by adding the audit-context fields
(`review_id`, `event_id`, `timestamp`, `sequence_number`, `is_eval`,
`node_id`, `event_type`) at construction, then persists. This keeps
the proof-boundary admission tests independent of the persister
mock surface — the parser is exercisable as a pure-data
transformation.

Spec divergence (recorded for Actual Outcome): spec §6 uses "emit"
throughout the step descriptions. The shipped shape interprets that
as "produces the event content"; the node body owns persistence per
the locked "boring node body" framing in the user-direction memo
2026-05-20.

**Scaffolding status.** This module ships the public surface (frozen
dataclasses + the `parse_analyze_response` signature) but the
admission flow itself raises `NotImplementedError`. Subsequent commits
land the 10 steps. The spec already owns the step-by-step description;
the code scaffold only creates the surface that later commits fill.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import ValidationError

from outrider.policy.canonical import compute_proposal_hash, compute_response_hash
from outrider.schemas.llm.analyze import AnalyzeResponseRaw

if TYPE_CHECKING:
    from uuid import UUID

    from outrider.ast_facts.models import ScopeUnit
    from outrider.policy.findings import EvidenceTier
    from outrider.schemas import ReviewFinding, TraceCandidate
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
    pass_index: int,
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
    - `pass_index` — for any rejection-detail metadata that names the
      pass; the node body uses it for `AnalyzeCompletedEvent.pass_index`.
    """
    try:
        raw = AnalyzeResponseRaw.model_validate_json(response_text)
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

    # Per-proposal admission lands incrementally. Each iteration must
    # decide the proposal's fate (admitted → ReviewFinding; rejected →
    # ProposalRejection). Until the admission checks land, the loop
    # raises NotImplementedError with the proposal index so the next
    # commit's insertion point is obvious. The empty-findings happy
    # path returns the zero-counter result without entering the loop.
    for idx, _raw_proposal in enumerate(raw.findings):
        raise NotImplementedError(
            f"parse_analyze_response: admission for findings[{idx}] not yet implemented"
        )
    return ParserResult(
        admitted_findings=(),
        trace_candidates=(),
        proposal_rejections=(),
        response_rejection=None,
        counters=ParserCounters(
            n_proposals_seen=0,
            n_findings_emitted=0,
            n_proposals_rejected=0,
            n_responses_rejected=0,
            n_trace_candidates_emitted=0,
        ),
    )


# `claimed_finding_type_hash` width matches the schema's pattern at
# `audit/events.py:891` (`_SHA256_HEX_PATTERN_SHORT` — 16 hex chars).
# Per `DECISIONS.md#014` point 1, the raw model string never lands in
# the audit row; the hash+length pair lets operators reason about
# identity without admitting content.
_CLAIMED_FINDING_TYPE_HASH_WIDTH: Final[int] = 16


def _hash_claimed_finding_type(raw_value: str) -> str:
    """sha256(raw_value.encode("utf-8")).hexdigest()[:16] — short-prefix
    hash for `ProposalRejection.claimed_finding_type_hash`. Lifted to
    `FindingProposalRejectedEvent.claimed_finding_type_hash` by the
    node body."""
    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:_CLAIMED_FINDING_TYPE_HASH_WIDTH]


def _build_proposal_rejection(
    raw: AnalyzeFindingProposalRaw,
    *,
    file_path: str,
    rejection_reason: _ProposalRejectionReason,
    rejection_detail: str,
    claimed_evidence_tier: EvidenceTier | None,
) -> ProposalRejection:
    """Construct a `ProposalRejection` from a raw proposal + admission outcome.

    Computes the identity-bearing fields shared by every rejection
    branch (proposal_hash via the canonical wrapper; claimed
    finding-type hash + length per `DECISIONS.md#014`). The caller
    supplies the branch-specific fields (reason, detail, claimed
    evidence-tier where parsed).

    `proposal_hash` runs through `policy.canonical.compute_proposal_hash`
    so `source_file_path` canonicalizes via `coordinates.validate_diff_path`
    before folding (alias-equivalence per DECISIONS#022), and
    `trace_path=None`/`()` normalize to the same logical state. Caller
    MUST pass `file_path` already canonicalized at intake — the wrapper
    runs it through `validate_diff_path` again as defense-in-depth.
    """
    proposal_hash = compute_proposal_hash(
        source_file_path=file_path,
        finding_type=raw.finding_type,
        evidence_tier=raw.evidence_tier,
        query_match_id=raw.query_match_id,
        trace_path=raw.trace_path,
        title=raw.title,
        description=raw.description,
        evidence=raw.evidence,
        byte_start=raw.span.byte_start,
        byte_end=raw.span.byte_end,
    )
    return ProposalRejection(
        proposal_hash=proposal_hash,
        file_path=file_path,
        claimed_finding_type_hash=_hash_claimed_finding_type(raw.finding_type),
        claimed_finding_type_len=len(raw.finding_type),
        claimed_evidence_tier=claimed_evidence_tier,
        rejection_reason=rejection_reason,
        rejection_detail=rejection_detail,
    )


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
