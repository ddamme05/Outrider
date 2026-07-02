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

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

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
from outrider.policy.canonical import compute_phase_id
from outrider.prompts import trace as trace_prompt
from outrider.schemas import TraceDecision, TraceFetchedFile

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from uuid import UUID

    from outrider.audit.sinks import PhaseEventSink, TraceEventSink
    from outrider.github import InstallationGitHubClient
    from outrider.llm.base import LLMProvider
    from outrider.schemas import ReviewState
    from outrider.schemas.trace_candidate import TraceCandidate


logger = logging.getLogger(__name__)


# Cap on candidates probed per source-finding bucket per trace invocation.
# Without a cap, a hostile (or buggy) analyze pass can emit N candidates
# per finding × M findings × up to 6 paths-per-candidate GitHub fetches
# in Phase 1 (the suffix-strip ladder: 2 module-form probes + 2 per
# strip level × `MAX_SUFFIX_STRIP_LEVELS`, FUP-209). The cap is applied
# to insertion order (the order analyze emitted candidates into
# `state.trace_candidates`, reducer-controlled) BEFORE the Haiku
# ranking call, so membership in the probed set is
# reducer-deterministic — a hostile analyze-LLM cannot use ranking to
# smuggle attacker-chosen candidates past the cap. The HARD per-pass
# fetch ceiling is `MAX_PROBE_FETCHES_PER_PASS` (below) — the
# per-candidate ladder bounds shape, the pass budget bounds total.
# The Haiku ranking step orders the post-cap bucket (informs
# intra-bucket probe order; reserved for future probe-behavior changes
# per spec M6 where order will gate early-exit). The full pre-cap
# LLM-proposed list lives in `state.trace_candidates` for forensic
# inspection. `TraceDecisionEvent.proposed_import_strings` carries
# ONLY the deduped+capped bucket per finding.
MAX_CANDIDATES_PER_FINDING: Final[int] = 5

# Suffix-strip ladder depth for Phase 1 resolution (FUP-209). Level 0
# reads the whole import_string as a module; level k strips the trailing
# k components (symbol-form fallback: `svc.queries.run_query` needs
# strip 1; `app.views.UserView.get` — method on a class — needs strip
# 2). Depth 2 covers the module.symbol and module.Class.method emission
# shapes real models produce; deeper nesting is not a real Python
# module shape.
MAX_SUFFIX_STRIP_LEVELS: Final[int] = 2

# Hard ceiling on Phase 1 probe fetches per trace pass. The deterministic
# control that keeps a hostile analyze pass (max findings × max
# candidates, all misses — every miss pays the full ladder) from
# exhausting the installation's 5000/hr GitHub rate limit: without it
# the ladder worst case is ~6000-7680 requests/pass. 1024/pass ×
# depth-2 rounds ≈ 2048/review, ~40% of the hourly budget. Legitimate
# passes sit far below (a handful of findings with candidates; the
# probe memo dedupes shared-parent paths; in-PR paths are pre-seeded
# and cost zero fetches). On exhaustion, remaining paths count as
# not-real (candidates land `unresolved`) and a WARN is logged once
# per pass. Per-installation cross-review budget tracking stays
# FUP-077.
MAX_PROBE_FETCHES_PER_PASS: Final[int] = 1024

# Depth limit on the analyze ⇄ trace loop. After round 2, the trace
# router unconditionally routes to `hitl` (the next non-trace
# destination; `hitl` then pass-throughs or interrupts depending on
# the finding set) — bounds the loop's total wall-clock cost and
# matches the spec's depth-2 ceiling.
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
    # Deterministic per-invocation phase_id: same `state.analysis_rounds`
    # length on resume re-runs produces the same key, so the PhaseEventSink
    # idempotency collapses re-emissions.
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="trace",
        attempt_key=f"trace-pass-{len(state.analysis_rounds)}",
    )

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
    # Unjoinable candidates are the documented forensic-only case per
    # DECISIONS.md#025 point 6 (rejected-parent proposals whose
    # candidates stay in state.trace_candidates for replay but produce
    # no TraceDecisionEvent and no GitHub fetch). They are skipped here,
    # logged at INFO inside the helper, and remain visible in state.
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
            # Skip the Haiku ranking call when ranking can't change the
            # outcome:
            #
            #   (a) a single flat candidate is trivially its own top-K
            #       (the original short-circuit), and
            #   (b) every per-finding bucket already has exactly one
            #       candidate — ranking only reorders WITHIN buckets
            #       (Step 7 re-groups by `source_finding_id`), so
            #       singleton buckets get an inconsequential reordering.
            #
            # (b) is the common shape: many findings × 1 unique import
            # each. Without it, every trace pass with N≥2 distinct
            # findings (one candidate each) fires a Haiku call whose
            # ordering output is discarded by the re-grouping. The
            # post-cap-flatten order is preserved as-is on both
            # short-circuits.
            if len(flat_candidates) == 1 or all(
                len(bucket) == 1 for bucket in capped_buckets.values()
            ):
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
    # Seed the within-pass fetch-dedup set with paths already in state
    # (replay / multi-pass) so a Phase-2 fetch never repeats a
    # `(target_file, head_sha)` round-trip that would land on a path
    # the reducer's first-write-wins dedup-by-path is going to drop
    # anyway. Also drives the `last_trace_pass_fetched_count` scalar
    # the router reads: the count must reflect NEW state, not all
    # successful Phase-2 calls (two findings resolving to the same
    # target_file would otherwise count as 2 when only 1 file lands).
    fetched_target_files: set[str] = {f.path for f in state.trace_fetched_files}

    # Pass-level probe memo + budget (FUP-209 review). The memo dedupes
    # byte-identical Phase 1 probes across candidates AND buckets
    # (sibling symbol-form candidates share parent-module paths), and is
    # seeded with the head content state already carries for PR-diff
    # files — probing a path the PR itself changed needs no GitHub
    # round-trip (removed files have no head content and are deliberately
    # not seeded, so a candidate naming a deleted module still probes and
    # misses). The budget hard-caps Phase 1 fetches per pass; see the
    # `MAX_PROBE_FETCHES_PER_PASS` comment for the hostile-PR math.
    probe_memo: dict[str, bytes | None] = {
        cf.path: cf.content_head.encode("utf-8")
        for cf in state.pr_context.changed_files
        if cf.content_head is not None
    }
    probe_budget = _ProbeBudget(remaining=MAX_PROBE_FETCHES_PER_PASS)

    # Crash-resume recovery (audit-ahead-of-state window). A decision row
    # persisted by a run that crashed after `emit_trace_decision` but
    # before this node's state delta merged is invisible to the step-5
    # `already_traced` gate (state lags audit). Re-probing would re-decide
    # the finding under possibly-newer resolver semantics and trip the
    # persister's natural-key identity guard, aborting the resumed
    # review. Adopt the persisted row instead — it is canonical per
    # M7 (b) — and skip probe + emit for that finding; the loud guard
    # keeps firing for genuine same-run nondeterminism.
    persisted_decisions: dict[UUID, TraceDecisionEvent] = {}
    if ranked_by_finding:
        persisted_decisions = {
            event.source_finding_id: event
            for event in await trace_sink.get_trace_decisions(review_id=state.review_id)
        }

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

        existing_event = persisted_decisions.get(finding_id)
        if existing_event is not None:
            # Adopt without re-probing (crash-resume recovery above).
            logger.warning(
                "trace: adopting persisted trace_decision for finding %s "
                "(state lagged audit — crash-resume recovery; no re-probe)",
                finding_id,
            )
            persisted_event = existing_event
        else:
            # Phase 1: probe fetches across this finding's ranked candidates.
            probe_outcome = await _resolve_via_probes(
                candidates=bucket,
                gh_client=gh_client,
                owner=state.pr_context.owner,
                repo=state.pr_context.repo,
                head_sha=head_sha,
                probe_memo=probe_memo,
                probe_budget=probe_budget,
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
                # Mirror the full persisted event — `trace_path` is None
                # in V1 trace emission but the audit→state lift must
                # carry whatever the persister returns so state and
                # audit don't diverge if the persisted row ever has a
                # non-None trace_path (V1.5 + replay reconstruction).
                trace_path=persisted_event.trace_path,
            )
        )

        # Phase 2: content fetch only for resolved AND not-in-PR per M8
        # AND not-already-fetched (within-pass OR carried-from-state).
        # Probe outcomes do NOT populate trace_fetched_files; only this
        # explicit second fetch does. The `fetched_target_files` set
        # prevents duplicate Phase-2 round-trips when multiple findings
        # resolve to the same target (the reducer would drop dupes via
        # first-write-wins on `path`, but the GitHub call already
        # happened, and the scalar would have over-counted).
        if (
            persisted_event.resolution_status == "resolved"
            and persisted_event.target_file is not None
            and persisted_event.target_file not in pr_file_paths
            and persisted_event.target_file not in fetched_target_files
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
                fetched_target_files.add(fetched_file.path)

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
    join are dropped at this point per `DECISIONS.md#025` point 6:
    `state.trace_candidates` may contain TraceCandidates from
    REJECTED PARENT PROPOSALS (the parser preserves them per
    `test_step10_trace_candidates_collected_from_rejected_proposal`)
    — those candidates stay in state for replay/forensic visibility
    but produce no `TraceDecisionEvent` and no GitHub fetch. The
    drop here is the documented enforcement of "unjoined candidates
    remain forensic-only." Logged at INFO (not WARN) because this
    is normal forensic behavior, not a producer bug. A genuine
    producer bug would manifest as a candidate whose
    `source_proposal_hash` doesn't match any proposal the parser
    has ever seen — distinguishable only at the
    `TraceJoinIntegrityError` site upstream.
    """
    buckets: dict[UUID, list[TraceCandidate]] = {}
    for candidate in candidates:
        finding_id = join_lookup.get(candidate.source_proposal_hash)
        if finding_id is None:
            # INFO level: per DECISIONS#025 point 6 this is the
            # documented forensic-only path for rejected-parent
            # candidates; promote to WARN/ERROR only if a future
            # invariant says "every state.trace_candidate must join."
            logger.info(
                "trace: candidate not joined to admitted finding "
                "candidate_id=%s source_proposal_hash=%s — "
                "forensic-only per DECISIONS.md#025 point 6 "
                "(rejected-parent proposal). No TraceDecisionEvent "
                "or GitHub fetch will fire for this candidate.",
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


@dataclass(slots=True)
class _ProbeBudget:
    """Mutable per-pass Phase 1 fetch budget (`MAX_PROBE_FETCHES_PER_PASS`).

    Consumed in deterministic bucket/candidate/path order, so which
    probes get funded is reducer-deterministic. `exhausted_logged`
    keeps the WARN to one line per pass.
    """

    remaining: int
    exhausted_logged: bool = False


async def _resolve_via_probes(
    *,
    candidates: Sequence[TraceCandidate],
    gh_client: InstallationGitHubClient,
    owner: str,
    repo: str,
    head_sha: str,
    probe_memo: dict[str, bytes | None],
    probe_budget: _ProbeBudget,
) -> _ProbeOutcome:
    """Phase 1 per M8 — resolve the bucket via a suffix-strip probe ladder.

    Level 0 reads each candidate's whole import_string as a module
    (`foo.bar` → `foo/bar.py` + `foo/bar/__init__.py`). Level k
    (1..`MAX_SUFFIX_STRIP_LEVELS`) strips the trailing k components and
    probes the remaining prefix as a module — the symbol-form fallback
    (FUP-209): real models emit `svc.queries.run_query` (function in
    module) and `app.views.UserView.get` (method on class) despite the
    prompt's module-form instruction.

    Levels are bucket-level barriers: the ladder stops at the SHALLOWEST
    level where any candidate has a verified real path, so a deeper
    fallback can never demote a sibling's clean module-form resolution
    to ambiguous (a hallucinated candidate's parent-package
    `__init__.py` cannot pollute level 0).

    A level-k hit (k >= 1) must pass symbol verification: the fetched
    module text must name the first stripped component in a defining
    context — def/class, module-level binding, or import
    (`_symbol_in_content`). Existence alone would resolve any
    hallucinated `pkg.ghost` to `pkg/__init__.py` — which exists in
    essentially every package — so verification keeps hallucinated and
    PR-deleted module candidates `unresolved`, exactly as before the
    fallback existed.

    Every path is validated via `coordinates.validate_diff_path` before
    any fetch (validation failure = probe negative). Unique unfetched
    paths per level fetch concurrently in deterministic order via
    `_fetch_paths_into_memo`; `probe_memo` (pass-level, seeded with
    in-PR head content) dedupes byte-identical probes across candidates
    and buckets; `probe_budget` enforces the per-pass fetch ceiling —
    unfunded paths count as not-real.

    Aggregate outcomes (within the winning level, across the bucket):
      - 1 unique verified real path → resolved
      - 2+ → ambiguous (M8's single-target contract)
      - no level yields any → unresolved

    Probe failures (non-404 errors) propagate per M8 transient semantics;
    404 / None content is admitted as "candidate did not resolve."
    """
    for strip_level in range(MAX_SUFFIX_STRIP_LEVELS + 1):
        candidate_paths: list[tuple[TraceCandidate, tuple[str, ...]]] = []
        for candidate in candidates:
            validated_paths: list[str] = []
            for candidate_path in _tier_paths_for(candidate.import_string, strip_level):
                try:
                    validated_paths.append(validate_diff_path(candidate_path))
                except CoordinateError:
                    continue
            candidate_paths.append((candidate, tuple(validated_paths)))

        await _fetch_paths_into_memo(
            [path for _, paths in candidate_paths for path in paths],
            gh_client=gh_client,
            owner=owner,
            repo=repo,
            head_sha=head_sha,
            probe_memo=probe_memo,
            probe_budget=probe_budget,
        )

        level_real: list[str] = []
        for candidate, safe_paths in candidate_paths:
            parts = candidate.import_string.split(".")
            for path in safe_paths:
                content = probe_memo.get(path)
                if content is None:
                    # Probed-and-missing OR unfunded (budget) — both
                    # count as not-real.
                    continue
                if strip_level >= 1 and not _symbol_in_content(parts[-strip_level], content):
                    continue
                level_real.append(path)

        if not level_real:
            continue

        # Deduplicate while preserving order — two candidates can verify
        # against the same path (shared parent module).
        seen: set[str] = set()
        unique_real_paths: list[str] = []
        for path in level_real:
            if path not in seen:
                seen.add(path)
                unique_real_paths.append(path)

        if len(unique_real_paths) == 1:
            return _ProbeOutcome(
                resolution_status="resolved",
                target_file=unique_real_paths[0],
                resolved_candidate_paths=(unique_real_paths[0],),
            )
        return _ProbeOutcome(
            resolution_status="ambiguous",
            target_file=None,
            resolved_candidate_paths=tuple(unique_real_paths),
        )

    return _ProbeOutcome(
        resolution_status="unresolved",
        target_file=None,
        resolved_candidate_paths=(),
    )


async def _fetch_paths_into_memo(
    paths: Sequence[str],
    *,
    gh_client: InstallationGitHubClient,
    owner: str,
    repo: str,
    head_sha: str,
    probe_memo: dict[str, bytes | None],
    probe_budget: _ProbeBudget,
) -> None:
    """Fetch-probe validated paths into `probe_memo`, deduped + budgeted.

    Paths already in the memo (probed earlier this pass, or seeded from
    in-PR head content) cost nothing. A deterministic prefix of the
    remaining unique paths is funded from `probe_budget`; unfunded paths
    stay out of the memo and read as not-real to the caller.

    Funded paths fetch concurrently via `asyncio.gather` — probes are
    independent, and results are processed in path order so the FIRST
    non-404 error in path order propagates deterministically per M8
    transient semantics. Deliberately gather-not-TaskGroup (intake's
    fan-out pattern): sibling cancellation buys nothing for a handful
    of probes, and ExceptionGroup unwrapping would make which-error-
    propagates nondeterministic.
    """
    to_fetch: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in probe_memo and path not in seen:
            seen.add(path)
            to_fetch.append(path)
    if not to_fetch:
        return

    funded = to_fetch[: probe_budget.remaining] if probe_budget.remaining > 0 else []
    if len(funded) < len(to_fetch) and not probe_budget.exhausted_logged:
        logger.warning(
            "trace: probe budget exhausted (MAX_PROBE_FETCHES_PER_PASS=%d); "
            "%d probe path(s) skipped this pass — affected candidates "
            "resolve as unresolved",
            MAX_PROBE_FETCHES_PER_PASS,
            len(to_fetch) - len(funded),
        )
        probe_budget.exhausted_logged = True
    if not funded:
        return
    probe_budget.remaining -= len(funded)

    results = await asyncio.gather(
        *(
            _probe_single_path(
                path,
                gh_client=gh_client,
                owner=owner,
                repo=repo,
                head_sha=head_sha,
            )
            for path in funded
        ),
        return_exceptions=True,
    )
    for path, result in zip(funded, results, strict=True):
        if isinstance(result, BaseException):
            raise result
        probe_memo[path] = result


async def _probe_single_path(
    path: str,
    *,
    gh_client: InstallationGitHubClient,
    owner: str,
    repo: str,
    head_sha: str,
) -> bytes | None:
    """Fetch-probe one validated path at head SHA; None = not real.

    404 is the COMMON probe outcome: the LLM proposed a path that
    doesn't exist in this repo — admitted as "did not resolve". All
    other HTTP errors (5xx / 403 / timeout / connection) propagate per
    M8 transient semantics — they signal GitHub-side issues that should
    abort the trace pass rather than silently miss a real path.
    Duck-typed `status_code` access mirrors the pattern at
    `github/publisher.py:355` (avoids importing the SDK-exception class
    outside the wrapper).
    """
    try:
        return await fetch_file_content_at(
            gh_client,
            owner=owner,
            repo=repo,
            path=path,
            ref=head_sha,
        )
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            return None
        raise


def _symbol_in_content(symbol: str, content: bytes) -> bool:
    """Context-restricted check that the module text binds `symbol`.

    Guards the suffix-strip fallback against existence-only false
    resolution: a level-k hit counts only if the fetched text names the
    first stripped component in a DEFINING context — `def`/`async def`/
    `class` statements, a module-level binding or annotation
    (`name = …`, `name: T`), an import line naming it, or a bare name
    on its own line (a parenthesized multi-line from-import
    continuation). Incidental uses reject: attribute access
    (`session.get(url)`), comments (`# process data`), and string
    literals (`{"id": 1}`) are not bindings — a bare anywhere-in-text
    word match would falsely resolve candidates with common trailing
    names (`get`/`data`/`id`) to modules that never define them.
    Deliberately textual, not parsed: scope-level extraction would put
    an ast_facts parse on every fallback probe for a yes/no gate.
    Wildcard re-exports (`from .x import *`) don't name the symbol and
    reject — the same `unresolved` outcome as before the fallback
    existed, never a regression. Non-UTF-8 content rejects (a binary
    blob is not the module the candidate names).
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    escaped = re.escape(symbol)
    binding_contexts = re.compile(
        # def SYMBOL / async def SYMBOL / class SYMBOL
        rf"^\s*(?:async\s+)?def\s+{escaped}\b"
        rf"|^\s*class\s+{escaped}\b"
        # SYMBOL: T (annotation) | SYMBOL = … (binding); rejects == and :=
        rf"|^\s*{escaped}\s*(?::(?!=)|=(?!=))"
        # import SYMBOL | from x import … SYMBOL (single-line forms)
        rf"|^\s*(?:from\s+\S+\s+)?import\b[^\n]*\b{escaped}\b"
        # bare name line — parenthesized multi-line from-import continuation
        rf"|^\s*{escaped}\s*,?\s*$",
        re.MULTILINE,
    )
    return binding_contexts.search(text) is not None


def _candidate_paths_for(import_string: str) -> tuple[str, ...]:
    """Construct module-form probe paths from a dotted import string.

    `foo.bar` → `('foo/bar.py', 'foo/bar/__init__.py')` — the string
    read as a module. THE single module→path mapping rule; the ladder's
    deeper levels reach it through `_tier_paths_for`. Does NOT consult
    the filesystem; the existence test is the subsequent fetch-probe.

    Defensive: the schema validator already requires valid dotted
    identifiers, so `.split('.')` yields well-formed parts. The
    fail-loud here is a producer-bug catcher if a malformed string
    somehow reaches the helper.
    """
    parts = import_string.split(".")
    base = "/".join(parts)
    return (f"{base}.py", f"{base}/__init__.py")


def _tier_paths_for(import_string: str, strip_level: int) -> tuple[str, ...]:
    """Probe paths at suffix-strip level `strip_level` (FUP-209 ladder).

    Level 0 = the whole string read as a module; level k >= 1 = the
    trailing k components read as a symbol chain defined in the
    remaining prefix (`svc.queries.run_query` at level 1 →
    `svc/queries.py` + `svc/queries/__init__.py`). Delegates to
    `_candidate_paths_for` so the module→path mapping rule lives in
    exactly one place. Returns () when stripping would consume the
    whole string.
    """
    if strip_level == 0:
        return _candidate_paths_for(import_string)
    parts = import_string.split(".")
    if strip_level >= len(parts):
        return ()
    return _candidate_paths_for(".".join(parts[:-strip_level]))


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
    # Phase-2 race window: Phase 1 probe confirmed the path existed at
    # head_sha, but the file can disappear between probe and Phase 2
    # (force-push, file deletion in a concurrent push to the same SHA
    # — rare but real). Treat 404 here as a soft miss to match
    # Phase 1's 404→not-real probe contract; aborting the
    # whole trace pass on a single race would defeat the M8 design.
    # Non-404 errors (5xx / 403 / timeout) still propagate per the
    # transient-failure contract. Same duck-typed status pattern as
    # `_probe_single_path` and `github/publisher.py:355`.
    try:
        content_bytes = await fetch_file_content_at(
            gh_client,
            owner=owner,
            repo=repo,
            path=target_file,
            ref=head_sha,
        )
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            return None
        raise
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
