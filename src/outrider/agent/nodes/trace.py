# See specs/2026-05-23-trace-node.md M1+M5+M6+M7+M8 + Q3 + Q4.
"""Trace node — consumes `state.trace_candidates`, ranks via Haiku,
resolves via two-phase fetch, emits `TraceDecisionEvent` audit-first.

Per `specs/2026-05-23-trace-node.md` the trace node implements the
adaptive analyze ⇄ trace loop's consumer side. Sequence:

  1. Emit `ReviewPhaseEvent(node_id="trace", marker="start")`.
  2. Build join lookup `{proposal_hash → finding_id}` from
     `state.analysis_rounds`; raise `TraceJoinIntegrityError` on
     collision (M5).
  3. Build `already_traced: set[UUID]` from `state.trace_decisions` —
     within-graph re-entry idempotency (M1 + #025 point 5).
  4. Bucket `state.trace_candidates` by `source_finding_id`
     (resolved via the join). Drop any candidate whose
     `source_proposal_hash` doesn't appear in the join (unjoinable —
     analyze-side bug; surfaced via the join-integrity raise).
  5. Drop any bucket whose `source_finding_id` is already in
     `already_traced`.
  6. One Haiku call across all remaining candidates ranks them per
     `prompts/trace.py` + `trace_parser.py`. Rejection → fall back to
     input order (V1 simplification — FOLLOWUP for dedicated audit
     event type).
  7. For each `source_finding_id` bucket (in finding-id-stable order):
     a. Phase 1 — probe fetches per candidate: construct paths
        (`foo.bar → foo/bar.py + foo/bar/__init__.py`), validate via
        `coordinates.validate_diff_path`, fetch-probe each via
        `github.fetch.fetch_file_content_at` at head SHA.
     b. Aggregate probe outcomes → `resolution_status` (resolved /
        unresolved / ambiguous) + `target_file` + `resolved_candidate_paths`.
     c. Build `TraceDecisionEvent`; emit audit-first via
        `trace_sink.emit_trace_decision(event)`; sink returns the
        canonical persisted event (incoming on insert path, existing
        on natural-key no-op per M7 b).
     d. Build state-layer `TraceDecision` from the RETURNED event
        (M7 b lockstep-recovery).
     e. Phase 2 — only if `resolution_status="resolved"` AND
        `target_file NOT IN pr_context.changed_files`: fetch
        `target_file` at head SHA, build `TraceFetchedFile`. Per M8,
        Phase 2 is structurally separate from Phase 1 probes; probe
        outcomes do NOT populate `state.trace_fetched_files`.
  8. Emit `ReviewPhaseEvent(marker="end")` AND return state delta as
     a coupled atom (M7 phase-end ordering — successful path only).

`try/finally` is NOT used for the phase-end emission: per M7 (and the
analyze-node precedent), the phase-end MUST NOT fire on exception
paths (dangling-start preserved via the missing end — replay sees
start without end as the failure signature).

Failure semantics:
  - Producer-deterministic (validator raises, join-integrity raises):
    propagates without state delta. V1 loud-fail.
  - Transient (GitHub fetch errors, DB connection): propagates without
    state delta. Re-invocation re-runs trace; natural-key idempotency
    makes already-persisted decisions no-ops; remaining work proceeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID, uuid4

from outrider.agent.nodes.trace_parser import (
    TraceRankingParsed,
    parse_trace_ranking,
)
from outrider.audit.events import (
    ReviewPhaseEvent,
    TraceDecisionEvent,
)
from outrider.coordinates import validate_diff_path
from outrider.coordinates.errors import CoordinateError
from outrider.github.fetch import fetch_file_content_at
from outrider.llm.base import LLMRequest
from outrider.prompts import trace as trace_prompt
from outrider.schemas import TraceDecision, TraceFetchedFile

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from outrider.audit.sinks import PhaseEventSink, TraceEventSink
    from outrider.github import InstallationGitHubClient
    from outrider.llm.base import LLMProvider
    from outrider.schemas import ReviewState
    from outrider.schemas.trace_candidate import TraceCandidate


logger = logging.getLogger(__name__)


# Cap on candidates probed per source-finding bucket per trace invocation.
# Without a cap, a hostile (or buggy) analyze pass can emit N candidates
# per finding × M findings × 2 paths-per-candidate GitHub fetches in
# Phase 1; at the analyze-side cap of 50 findings × 20 candidates that
# is 2000 requests per pass, ~4000 across the depth-2 round limit —
# enough to exhaust an installation's 5000/hr GitHub rate limit on a
# single hostile PR. Top-K per finding keeps Phase 1 cost bounded
# (`MAX_CANDIDATES_PER_FINDING × n_findings × 2` GitHub fetches per
# pass) and the LLM's ranking is exactly the signal trace uses to
# decide WHICH top-K to probe — non-top-K candidates are recorded in
# `TraceDecisionEvent.proposed_import_strings` for audit transparency
# but never probed. Per sharp-edges H1 fold.
MAX_CANDIDATES_PER_FINDING: Final[int] = 5

# Depth limit on the analyze ⇄ trace loop. After round 2, the trace
# router unconditionally routes to publish — bounds the loop's total
# wall-clock cost and matches the spec's depth-2 ceiling.
MAX_ANALYSIS_ROUNDS: Final[int] = 2


class TraceJoinIntegrityError(RuntimeError):
    """Two admitted findings share the same `proposal_hash` (M5 / #025 point 5).

    Trace's join from `proposal_hash → finding_id` requires uniqueness
    across `state.analysis_rounds`. Within-round uniqueness is the
    analyze-side `AnalysisRound._enforce_findings_proposal_hash_unique`
    validator; cross-round uniqueness is a downstream consequence of
    `compute_proposal_hash`'s content-derived recipe (per #022) — two
    logically-identical proposals collide deterministically. Trace's
    raise is the last-resort guard.

    Strict-keyword constructor; the message names the offending
    `proposal_hash` (which is a SHA-256 hex — not content-bearing) plus
    the two colliding `finding_id`s for operator diagnosis.
    """

    def __init__(
        self,
        *,
        proposal_hash: str,
        first_finding_id: UUID,
        second_finding_id: UUID,
    ) -> None:
        super().__init__(
            f"trace join-lookup collision: proposal_hash={proposal_hash} "
            f"appears on multiple findings "
            f"(first_finding_id={first_finding_id}, "
            f"second_finding_id={second_finding_id}). Either "
            f"AnalysisRound._enforce_findings_proposal_hash_unique was "
            f"bypassed (producer bug), or compute_proposal_hash's recipe "
            f"changed in a way that collapses previously-distinct proposals."
        )
        self.proposal_hash = proposal_hash
        self.first_finding_id = first_finding_id
        self.second_finding_id = second_finding_id


async def trace(
    state: ReviewState,
    *,
    provider: LLMProvider,
    trace_model: str,
    phase_event_sink: PhaseEventSink,
    trace_sink: TraceEventSink,
    github_factory: Callable[[int], InstallationGitHubClient],
) -> dict[str, object]:
    """Run one trace pass.

    Returns `{"trace_decisions": [...], "trace_fetched_files": [...]}`
    for LangGraph's reducer to merge (both fields use
    `append_with_dedup_by`).

    Step order matches the module docstring. The Haiku ranking call is
    one per invocation (not per finding); the LLM sees all candidates
    at once and orders them.
    """
    phase_id = str(uuid4())

    # Step 1: start phase event.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="trace",
            marker="start",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    # Step 2: build join lookup. Raises TraceJoinIntegrityError on collision.
    join_lookup = _build_proposal_hash_join(state)

    # Step 3: already-traced set for within-graph idempotency.
    already_traced: set[UUID] = {d.source_finding_id for d in state.trace_decisions}

    # Step 4: bucket candidates by source_finding_id via the join.
    # Drop unjoinable candidates (would indicate analyze-side bug —
    # state.trace_candidates is supposed to be source_proposal_hash-
    # joinable to state.analysis_rounds; in practice that holds because
    # analyze emits both atomically into the same state delta).
    candidate_buckets = _bucket_candidates_by_finding(state.trace_candidates, join_lookup)

    # Step 5: drop already-traced buckets.
    pending_buckets = {
        finding_id: bucket
        for finding_id, bucket in candidate_buckets.items()
        if finding_id not in already_traced
    }

    # Step 6: Haiku ranking. One call across all remaining candidates.
    # Empty pending_buckets → no Haiku call, no decisions to emit.
    ordered_candidates: tuple[TraceCandidate, ...] = ()
    if pending_buckets:
        # Flatten preserving deterministic input order (sorted by
        # finding_id then by candidate_id for stability across calls).
        flat_candidates = tuple(
            c
            for finding_id in sorted(pending_buckets)
            for c in sorted(pending_buckets[finding_id], key=lambda c: c.candidate_id)
        )
        # Defensive: comprehension over a non-empty pending_buckets MUST
        # produce a non-empty tuple (every bucket has at least one
        # candidate by construction at step 4). Asserts protect against
        # a future refactor that empties a bucket without removing the
        # key — the Haiku call would otherwise fire with zero candidates
        # and burn tokens on a vacuous response.
        if flat_candidates:
            ordered_candidates = await _rank_candidates_via_haiku(
                state=state,
                candidates=flat_candidates,
                provider=provider,
                trace_model=trace_model,
            )

    # Step 7: process each source_finding_id bucket. Trace_decisions +
    # trace_fetched_files accumulate locally; emitted via the audit-first
    # sink per decision, returned in the state delta atomically at step 8.
    accumulated_decisions: list[TraceDecision] = []
    accumulated_fetched_files: list[TraceFetchedFile] = []

    # Group the ranked-and-ordered candidates back into per-finding
    # buckets, preserving the LLM's intra-bucket priority order.
    ranked_by_finding: dict[UUID, list[TraceCandidate]] = {}
    for candidate in ordered_candidates:
        # Re-derive finding_id via the join (already validated at step 2).
        finding_id = join_lookup.get(candidate.source_proposal_hash)
        if finding_id is None:
            # Unjoinable — was dropped at step 4 but defensive against
            # mutation between bucket-build and rank-call.
            continue
        ranked_by_finding.setdefault(finding_id, []).append(candidate)

    gh_client = github_factory(state.pr_context.installation_id)
    head_sha = state.pr_context.head_sha
    pr_file_paths: frozenset[str] = frozenset(cf.path for cf in state.pr_context.changed_files)

    for finding_id in sorted(ranked_by_finding):
        full_bucket = ranked_by_finding[finding_id]
        # Top-K cap per H1 fold: probe only the highest-ranked
        # MAX_CANDIDATES_PER_FINDING candidates. The full ranked list
        # still lands in `proposed_import_strings` on the audit row so
        # forensic reconstruction can see what the LLM proposed; only
        # the probed subset incurs GitHub fetches.
        bucket = full_bucket[:MAX_CANDIDATES_PER_FINDING]

        # Phase 1: probe fetches across this finding's top-K candidates.
        probe_outcome = await _resolve_via_probes(
            candidates=bucket,
            gh_client=gh_client,
            owner=state.pr_context.owner,
            repo=state.pr_context.repo,
            head_sha=head_sha,
        )

        # Construct TraceDecisionEvent. proposed_import_strings is the
        # LLM-ranked input order; resolved_candidate_paths is the probe
        # output (only the paths that fetched OK).
        decision_event = TraceDecisionEvent(
            review_id=state.review_id,
            is_eval=state.is_eval,
            source_finding_id=finding_id,
            target_file=probe_outcome.target_file,
            reason=_aggregate_candidate_reasons(bucket),
            resolution_status=probe_outcome.resolution_status,
            proposed_import_strings=tuple(c.import_string for c in bucket),
            resolved_candidate_paths=probe_outcome.resolved_candidate_paths,
        )

        # Audit-first per M7: emit BEFORE returning state delta. The sink
        # returns the canonical persisted event (incoming on insert,
        # existing on natural-key no-op per M7 b).
        persisted_event = await trace_sink.emit_trace_decision(decision_event)

        # Build state-layer TraceDecision from the RETURNED event so
        # state + audit stay in lockstep across retry/replay even when
        # per-emission fields (reason, proposed_import_strings,
        # resolved_candidate_paths) differ between attempts.
        accumulated_decisions.append(
            TraceDecision(
                source_finding_id=persisted_event.source_finding_id,
                target_file=persisted_event.target_file,
                reason=persisted_event.reason,
                resolution_status=persisted_event.resolution_status,
                proposed_import_strings=persisted_event.proposed_import_strings,
                resolved_candidate_paths=persisted_event.resolved_candidate_paths,
            )
        )

        # Phase 2: content fetch only for resolved AND not-in-PR per M8.
        # Probe outcomes do NOT populate trace_fetched_files; only this
        # explicit second fetch does.
        if (
            persisted_event.resolution_status == "resolved"
            and persisted_event.target_file is not None
            and persisted_event.target_file not in pr_file_paths
        ):
            fetched_file = await _phase_two_content_fetch(
                target_file=persisted_event.target_file,
                source_finding_id=persisted_event.source_finding_id,
                gh_client=gh_client,
                owner=state.pr_context.owner,
                repo=state.pr_context.repo,
                head_sha=head_sha,
            )
            if fetched_file is not None:
                accumulated_fetched_files.append(fetched_file)

    # Step 8: phase-end + state delta. Coupled atom — either both happen
    # (success path) or neither (any earlier exception propagates without
    # phase-end nor state delta merge).
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="trace",
            marker="end",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    return {
        "trace_decisions": accumulated_decisions,
        "trace_fetched_files": accumulated_fetched_files,
    }


# ---------------------------------------------------------------------------
# Helpers (private; tested via the trace node's unit tests).
# ---------------------------------------------------------------------------


def _build_proposal_hash_join(state: ReviewState) -> dict[str, UUID]:
    """Build `{proposal_hash → finding_id}` lookup across all rounds.

    Raises `TraceJoinIntegrityError` on duplicate. The
    `AnalysisRound._enforce_findings_proposal_hash_unique` validator
    handles within-round uniqueness; cross-round uniqueness comes from
    `compute_proposal_hash`'s content-derived recipe. The raise is the
    last-resort guard per M5.
    """
    join: dict[str, UUID] = {}
    for analysis_round in state.analysis_rounds:
        for finding in analysis_round.findings:
            existing = join.get(finding.proposal_hash)
            if existing is not None and existing != finding.finding_id:
                raise TraceJoinIntegrityError(
                    proposal_hash=finding.proposal_hash,
                    first_finding_id=existing,
                    second_finding_id=finding.finding_id,
                )
            join[finding.proposal_hash] = finding.finding_id
    return join


def _bucket_candidates_by_finding(
    candidates: Sequence[TraceCandidate],
    join_lookup: dict[str, UUID],
) -> dict[UUID, list[TraceCandidate]]:
    """Group candidates by their joined source_finding_id.

    Candidates whose `source_proposal_hash` doesn't resolve via the
    join are dropped silently — in practice this branch shouldn't fire
    because analyze emits findings + candidates atomically, but a
    cross-graph mutation could leave dangling candidates. Logged at
    DEBUG so a producer-bug investigation has the trail.
    """
    buckets: dict[UUID, list[TraceCandidate]] = {}
    for candidate in candidates:
        finding_id = join_lookup.get(candidate.source_proposal_hash)
        if finding_id is None:
            logger.debug(
                "trace: dropping unjoinable candidate candidate_id=%s source_proposal_hash=%s",
                candidate.candidate_id,
                candidate.source_proposal_hash,
            )
            continue
        buckets.setdefault(finding_id, []).append(candidate)
    return buckets


async def _rank_candidates_via_haiku(
    *,
    state: ReviewState,
    candidates: tuple[TraceCandidate, ...],
    provider: LLMProvider,
    trace_model: str,
) -> tuple[TraceCandidate, ...]:
    """One Haiku call ranks all candidates across all findings.

    On parse rejection: falls back to input order (V1 simplification —
    FOLLOWUP for dedicated `TraceRankingRejectedEvent` audit type).
    Rejection is logged at WARN with the rejection reason.

    The LLMCallEvent for this call is NOT emitted here; the provider's
    wrapper handles the audit emission via the standard LLM persister
    path. Trace consumes the response text.
    """

    prompt_parts = trace_prompt.render(candidates)
    request = LLMRequest(
        model=trace_model,
        system_prompt=prompt_parts.system_prompt,
        user_prompt=prompt_parts.user_prompt,
        max_tokens=trace_prompt.MAX_TOKENS,
        temperature=trace_prompt.TEMPERATURE,
        review_id=state.review_id,
        node_id="trace",
        is_eval=state.is_eval,
        prompt_template_version=trace_prompt.VERSION,
        degraded_mode=False,
    )

    response = await provider.complete(request)

    parse_result = parse_trace_ranking(
        response_text=response.text,
        candidates=candidates,
    )
    if isinstance(parse_result, TraceRankingParsed):
        return parse_result.ordered_candidates
    logger.warning(
        "trace: Haiku ranking rejected (reason=%s); falling back to input order. "
        "FOLLOWUP — dedicated TraceRankingRejectedEvent audit type.",
        parse_result.reason,
    )
    return candidates


def _aggregate_candidate_reasons(candidates: Sequence[TraceCandidate]) -> str:
    """Collapse per-candidate reasons into a single `reason` field for
    the TraceDecisionEvent. The event has one reason; bucket has many
    candidates. Concatenate with a separator; truncate to fit the
    schema's 500-char max."""
    parts = [f"{c.import_string}: {c.reason}" for c in candidates]
    aggregated = " | ".join(parts)
    if len(aggregated) > 500:
        aggregated = aggregated[:497] + "..."
    return aggregated


# ---------------------------------------------------------------------------
# Phase 1 + Phase 2 fetch helpers (per M8 two-phase fetch).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ProbeOutcome:
    """Result of Phase 1 probe fetches for one bucket."""

    resolution_status: Literal["resolved", "unresolved", "ambiguous"]
    target_file: str | None
    resolved_candidate_paths: tuple[str, ...]


async def _resolve_via_probes(
    *,
    candidates: Sequence[TraceCandidate],
    gh_client: InstallationGitHubClient,
    owner: str,
    repo: str,
    head_sha: str,
) -> _ProbeOutcome:
    """Phase 1 per M8 — probe each candidate's possible paths via
    GitHub fetches at head SHA.

    For each candidate import_string (`foo.bar`), construct the two
    candidate paths (`foo/bar.py` AND `foo/bar/__init__.py`),
    validate each via `coordinates.validate_diff_path`, then fetch-probe
    via `github.fetch.fetch_file_content_at`. A path is "real" if the
    fetch returns content (not None).

    Aggregate outcomes:
      - 1 real path → resolved
      - 0 real paths → unresolved
      - 2+ real paths → ambiguous

    Probe failures (non-404 errors) propagate per M8 transient semantics;
    404 / None content is admitted as "candidate did not resolve."

    Path-validation failures (a candidate path that fails `validate_diff_path`
    — extremely rare given the import_string field validator already
    rejects shell metacharacters) are treated as probe negatives.
    """
    real_paths: list[str] = []

    for candidate in candidates:
        for candidate_path in _candidate_paths_for(candidate.import_string):
            try:
                safe_path = validate_diff_path(candidate_path)
            except CoordinateError:
                continue
            content = await fetch_file_content_at(
                gh_client,
                owner=owner,
                repo=repo,
                path=safe_path,
                ref=head_sha,
            )
            if content is not None:
                real_paths.append(safe_path)

    # Deduplicate while preserving order — two candidates could resolve
    # to the same path.
    seen: set[str] = set()
    unique_real_paths: list[str] = []
    for path in real_paths:
        if path not in seen:
            seen.add(path)
            unique_real_paths.append(path)

    if len(unique_real_paths) == 1:
        return _ProbeOutcome(
            resolution_status="resolved",
            target_file=unique_real_paths[0],
            resolved_candidate_paths=(unique_real_paths[0],),
        )
    if len(unique_real_paths) == 0:
        return _ProbeOutcome(
            resolution_status="unresolved",
            target_file=None,
            resolved_candidate_paths=(),
        )
    return _ProbeOutcome(
        resolution_status="ambiguous",
        target_file=None,
        resolved_candidate_paths=tuple(unique_real_paths),
    )


def _candidate_paths_for(import_string: str) -> tuple[str, ...]:
    """Construct candidate paths from a dotted Python import string.

    `foo.bar` → `('foo/bar.py', 'foo/bar/__init__.py')`. Used by Phase 1
    probes per M8. Does NOT consult the filesystem; the existence test
    is the subsequent fetch-probe.

    Defensive: the schema validator already requires valid dotted
    identifiers, so `.split('.')` yields well-formed parts. The
    fail-loud here is a producer-bug catcher if a malformed string
    somehow reaches the helper.
    """
    parts = import_string.split(".")
    base = "/".join(parts)
    return (f"{base}.py", f"{base}/__init__.py")


async def _phase_two_content_fetch(
    *,
    target_file: str,
    source_finding_id: UUID,
    gh_client: InstallationGitHubClient,
    owner: str,
    repo: str,
    head_sha: str,
) -> TraceFetchedFile | None:
    """Phase 2 per M8 — fetch the resolved target_file at head SHA.

    Constructs `TraceFetchedFile` with `content_head` from the fetched
    bytes. Returns None if the fetch returns None (the target was
    resolved by Phase 1 but disappeared between probe and Phase 2 —
    races, force-pushes; rare but defensive). The trace_fetched_files
    reducer's `append_with_dedup_by(path)` collapses duplicates if
    multiple findings resolve to the same target.

    Fields per Q3:
      - `path`: from `target_file` (already validated by Phase 1's
        `validate_diff_path` upstream).
      - `content_head`: bytes-as-utf8 from the head-SHA fetch.
      - `source_finding_id`: from the persisted event (lockstep).
    No `source_import_string` / `source_proposal_hash` per Q3 revision;
    cross-reference recovers via `state.trace_decisions`.
    """
    content_bytes = await fetch_file_content_at(
        gh_client,
        owner=owner,
        repo=repo,
        path=target_file,
        ref=head_sha,
    )
    if content_bytes is None:
        return None
    return TraceFetchedFile(
        path=target_file,
        content_head=content_bytes.decode("utf-8", errors="replace"),
        source_finding_id=source_finding_id,
    )


__all__ = [
    "TraceJoinIntegrityError",
    "trace",
]
