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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from uuid import UUID

    from outrider.ast_facts.models import ScopeUnit
    from outrider.policy.findings import EvidenceTier
    from outrider.schemas import ReviewFinding, TraceCandidate

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
    raise NotImplementedError("parse_analyze_response: parser admission flow not implemented")


__all__ = [
    "ParserCounters",
    "ParserResult",
    "ProposalRejection",
    "ResponseRejection",
    "parse_analyze_response",
]
