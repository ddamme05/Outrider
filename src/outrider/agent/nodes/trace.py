# See specs/2026-05-23-trace-node.md and DECISIONS.md#017, #024, #025, #026.
"""Trace node — consumes `state.trace_candidates`, ranks via Haiku,
resolves via two-phase fetch (probe + content), emits
`TraceDecisionEvent` audit-first.

Audit-first emission contract + two-phase fetch design + failure
semantics: see `specs/2026-05-23-trace-node.md` (the spec is the
source of truth for the audit-boundary invariants). The step-by-step
flow is also visible inline in the `trace()` body via step comments.

Security-relevant: the Haiku-flatten step (in `trace()`'s step 6 +
the comment block above `flat_candidates`) carries the candidate-
ordering attack rationale — cross-bucket sort vs intra-bucket
insertion order; insertion-order's PR-content influence surface;
the FOLLOWUP for `TraceRankingRejectedEvent` audit-attribution.
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
# but never probed.
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

    Returns `{"trace_decisions": [...], "trace_fetched_files": [...],
    "last_trace_pass_fetched_count": int}` for LangGraph's reducer to
    merge (the two list fields use `append_with_dedup_by`;
    `last_trace_pass_fetched_count` is a scalar overwrite per invocation
    — the per-pass-delta signal `_trace_router` reads).

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

    # Step 6: Haiku ranking. One call across the capped candidate set.
    # Empty pending_buckets → no Haiku call, no decisions to emit.
    #
    # DEDUPE-THEN-CAP: each bucket is FIRST deduped by `import_string`
    # (first-occurrence-wins, order-stable),
    # THEN truncated to MAX_CANDIDATES_PER_FINDING. Two candidates with
    # the same `import_string` but different `reason` are distinct
    # `candidate_id`s (content-hash) and both survive the reducer's
    # `append_with_dedup_by(candidate_id)`; without pre-cap dedup, a
    # benign-but-loquacious LLM that proposed the same import 5+ times
    # with different rationales would fill every slot with the same
    # import_string, crowding out unique candidates from ranking,
    # probe-fetch fanout, and the audit row's set-semantic
    # `proposed_import_strings`. Dedupe-then-cap means
    # MAX_CANDIDATES_PER_FINDING bounds UNIQUE imports per finding —
    # matching the cap's stated intent + the downstream audit-event
    # validator's set semantics. PRE-RANK (per earlier F3) is preserved:
    # the Haiku call still receives ≤ MAX_CANDIDATES_PER_FINDING ×
    # len(pending_buckets) candidates; the dedup tightens the bound by
    # eliminating duplicate work. Trade-off remains: the audit row's
    # `proposed_import_strings` carries the post-dedup-and-cap set, NOT
    # the full LLM-proposed list. Operators wanting the full pre-cap
    # list read `state.trace_candidates` (reducer-deduped per finding).
    #
    # Flatten: cross-bucket order is `sorted(pending_buckets)` (by
    # `source_finding_id` UUID — content-derived from `proposal_hash`
    # which #022 + M5 make collision-resistant); intra-bucket order
    # is the reducer's insertion order (the order analyze emitted
    # candidates into `state.trace_candidates`).
    #
    # Intra-bucket is deliberately NOT sorted by `candidate_id`
    # (SHA-256 of LLM-controlled fields): a hostile analyze-LLM could
    # grind candidate_ids to win lexicographic order, then force
    # parser fallback (e.g., via a malformed ranking response) to
    # smuggle attacker-chosen candidates into the top-K. Insertion
    # order is reducer-controlled.
    #
    # Residual attack surface (see FOLLOWUPS — TraceRankingRejectedEvent):
    # PR-author content can influence analyze-LLM's emission order,
    # so insertion order is influenced (but not deterministic) by
    # PR content. The dedicated rejected-event audit type is the V1.5
    # mitigation that lets operators observe rejection-rate spikes.
    ordered_candidates: tuple[TraceCandidate, ...] = ()
    if pending_buckets:
        # Dedupe each bucket BEFORE applying the per-finding cap so
        # MAX_CANDIDATES_PER_FINDING bounds unique imports, not raw
        # entries. Downstream consumers (rank, probe, audit) all see
        # this same deduped+capped set.
        capped_buckets: dict[UUID, tuple[TraceCandidate, ...]] = {
            finding_id: _dedupe_by_import_string(bucket)[:MAX_CANDIDATES_PER_FINDING]
            for finding_id, bucket in pending_buckets.items()
        }
        flat_candidates = tuple(
            c for finding_id in sorted(capped_buckets) for c in capped_buckets[finding_id]
        )
        # Defensive: a non-empty `pending_buckets` MUST flatten non-empty.
        # Guard against a future refactor that empties a bucket without
        # removing the key.
        if flat_candidates:
            # Skip the Haiku ranking call when there's only one candidate
            # — the ranking is trivial (the one candidate is already the
            # top-K). A PR with N findings × 1 unique import each fires
            # one ranking call per finding under the old shape; this
            # guard turns the per-trace-invocation single-candidate case
            # into a zero-LLM-call pass. Order is preserved (input ==
            # output). Tested as `len == 1` (not `<= 1`) because the
            # outer `if flat_candidates:` already excludes len == 0.
            if len(flat_candidates) == 1:
                ordered_candidates = flat_candidates
            else:
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
        # Bucket here is dedup'd-then-capped to MAX_CANDIDATES_PER_FINDING:
        # step 6 deduped each `pending_buckets[finding_id]` by
        # `import_string` (first-occurrence-wins, order-stable) BEFORE
        # the cap, then truncated to MAX_CANDIDATES_PER_FINDING. Ranking
        # reorders but does not introduce duplicates. So
        # `ranked_by_finding[finding_id]` already satisfies the audit
        # event's `_enforce_proposed_import_strings_unique` validator
        # (set-semantic per #024) by construction — no further dedup
        # call needed here. The full pre-cap candidate set lives in
        # `state.trace_candidates` (reducer-deduped per source finding);
        # operators wanting the LLM's full proposal for forensic
        # inspection read it from there.
        bucket = ranked_by_finding[finding_id]

        # Phase 1: probe fetches across this finding's ranked candidates.
        probe_outcome = await _resolve_via_probes(
            candidates=bucket,
            gh_client=gh_client,
            owner=state.pr_context.owner,
            repo=state.pr_context.repo,
            head_sha=head_sha,
        )

        # Construct TraceDecisionEvent. `proposed_import_strings` carries
        # the dedup'd-then-capped Haiku-ranked import strings (≤
        # MAX_CANDIDATES_PER_FINDING UNIQUE imports per bucket; see
        # step 6 for the ordering rationale). The full pre-cap LLM
        # proposal lives in `state.trace_candidates` for forensic
        # inspection. `resolved_candidate_paths` is the probe output
        # (only the paths that fetched OK).
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

    # Single return path. `last_trace_pass_fetched_count` is the router's
    # per-invocation delta signal — see ReviewState's field comment.
    # MUST be present in EVERY return delta from this function (including
    # zero-fetch calls AND any future early-return). The default-overwrite
    # reducer means a delta missing the key leaves the prior invocation's
    # value in state, and `_trace_router` would route on stale data
    # (infinite loop / unwanted re-entry under depth > 2). If you add an
    # early return below, include the key.
    state_delta: dict[str, object] = {
        "trace_decisions": accumulated_decisions,
        "trace_fetched_files": accumulated_fetched_files,
        "last_trace_pass_fetched_count": len(accumulated_fetched_files),
    }
    return state_delta


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
    join are dropped. `_filter_to_admitted_proposals` in
    `agent/nodes/analyze.py` filters rejected-proposal candidates at
    the analyze→state edge, so in normal flow this branch fires only
    on:

      1. Replay of state checkpointed before that filter landed
         (transient — goes away once those checkpoints age out).
      2. A direct `state.trace_candidates.append` bypassing the reducer
         + analyze admission gate (genuine producer bug).
      3. A future code path that emits into `state.trace_candidates`
         without flowing through analyze's admission gate (genuine
         producer bug if introduced).

    Logged at WARN so any of the three is visible at default log level —
    DEBUG would make the drop invisible in production exactly when the
    underlying scenario (replay or producer bug) needs investigation.
    """
    buckets: dict[UUID, list[TraceCandidate]] = {}
    for candidate in candidates:
        finding_id = join_lookup.get(candidate.source_proposal_hash)
        if finding_id is None:
            logger.warning(
                "trace: dropping unjoinable candidate candidate_id=%s "
                "source_proposal_hash=%s — either a replay-of-old-state "
                "transition case (the analyze→state filter now prevents "
                "new emissions of unjoinable candidates) or a producer "
                "bug (state mutation bypassing the reducer + analyze "
                "admission gate)",
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


def _dedupe_by_import_string(
    candidates: Sequence[TraceCandidate],
) -> tuple[TraceCandidate, ...]:
    """Dedupe candidates by `import_string`, preserving first-occurrence
    order. Two candidates with the same import_string but different
    `reason` are distinct `candidate_id`s (content-derived) and both
    survive `state.trace_candidates`'s `append_with_dedup_by(candidate_id)`
    reducer — but `TraceDecisionEvent.proposed_import_strings` is set-
    semantic per #024's `_enforce_proposed_import_strings_unique`
    validator. First-occurrence-wins gives a deterministic, order-stable
    dedup that matches the LLM's ranked preference.
    """
    seen: set[str] = set()
    out: list[TraceCandidate] = []
    for candidate in candidates:
        if candidate.import_string in seen:
            continue
        seen.add(candidate.import_string)
        out.append(candidate)
    return tuple(out)


def _aggregate_candidate_reasons(candidates: Sequence[TraceCandidate]) -> str:
    """Collapse per-candidate reasons into a single `reason` field for
    the TraceDecisionEvent. The event has one reason; bucket has many
    candidates. Concatenate with a separator; truncate to fit the
    schema's 500-char max.

    Caller's responsibility to pre-dedupe by `import_string` (the audit
    event treats `proposed_import_strings` as set-semantic; the
    aggregated `reason` mirrors the same set). The structured-tuple
    field shape is the long-term fix for forensic-loss-via-truncation;
    the 500-char cap here is the V1 schema floor (see FUP-075 for the
    structured-field follow-up).
    """
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
            try:
                content = await fetch_file_content_at(
                    gh_client,
                    owner=owner,
                    repo=repo,
                    path=safe_path,
                    ref=head_sha,
                )
            except Exception as exc:
                # 404 is the COMMON probe outcome: the LLM proposed a
                # path that doesn't exist in this repo. Treat as
                # "candidate did not resolve" (i.e., None content). All
                # other HTTP errors (5xx / 403 / timeout / connection)
                # propagate per M8 transient semantics — they signal
                # GitHub-side issues that should abort the trace pass
                # rather than silently miss a real path. Duck-typed
                # `status_code` access mirrors the pattern at
                # `github/publisher.py:355` (avoids importing the
                # SDK-exception class outside the wrapper).
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 404:
                    continue
                raise
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
    races, force-pushes; rare but defensive) OR if the bytes don't
    decode as UTF-8 (a `.py`-named path that's actually a binary
    blob — compiled `.pyc` misnamed, vendor-injected bytes-as-`.py`,
    generated-stub binary). The decode-failure case is logged at WARN
    so an operator can investigate; trace continues for other decisions
    rather than failing the whole pass on a single producer-bug-shaped
    candidate.

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
    try:
        content_head = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Skip — a binary blob masquerading as Python would corrupt
        # the audit trail (mojibake string in content_head) AND mislead
        # analyze pass 2 (LLM hallucinates findings off replacement
        # chars). Skipping keeps the TraceDecision audit row intact
        # (resolution_status="resolved" reflects what Phase 1 saw)
        # while preventing the bad bytes from entering state.
        logger.warning(
            "trace: Phase 2 fetched bytes at %s (source_finding_id=%s) do "
            "not decode as UTF-8; skipping TraceFetchedFile construction. "
            "Producer-bug or binary masquerading as .py — operator "
            "investigation needed.",
            target_file,
            source_finding_id,
        )
        return None
    return TraceFetchedFile(
        path=target_file,
        content_head=content_head,
        source_finding_id=source_finding_id,
    )


__all__ = [
    "MAX_ANALYSIS_ROUNDS",
    "TraceJoinIntegrityError",
    "trace",
]
