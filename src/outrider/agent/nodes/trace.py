# See specs/2026-05-23-trace-node.md and DECISIONS.md#017, #024, #025, #026.
"""Find other files that may explain an analysis finding.

`trace()` groups suggested imports by finding, chooses candidates to check,
probes their paths, records each decision, and fetches resolved files for the
next analysis pass. Probing checks which path exists; the separate content
fetch supplies the text analysis will read.

`_rank_pending_candidates` owns candidate selection and ordering.
`_resolve_via_probes` owns path resolution. Decisions are persisted before
they enter graph state, so retries can recover from the audit record.
See `specs/2026-05-23-trace-node.md` for the audit and failure contracts.
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
from outrider.coordinates import (
    is_relative_specifier_form,
    relative_specifier_candidate_paths,
    validate_diff_path,
)
from outrider.coordinates.errors import CoordinateError
from outrider.github.fetch import fetch_file_content_at
from outrider.llm.base import LLMRequest
from outrider.policy.canonical import compute_phase_id
from outrider.prompts import trace as trace_prompt
from outrider.schemas import TraceDecision, TraceFetchedFile

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from uuid import UUID

    from outrider.audit.sinks import PhaseEventSink, TraceEventSink
    from outrider.github import InstallationGitHubClient
    from outrider.llm.base import LLMProvider
    from outrider.schemas import ReviewState
    from outrider.schemas.trace_candidate import TraceCandidate


logger = logging.getLogger(__name__)


# Limit unique imports checked per finding. Deduplicate and cap in reducer
# insertion order before ranking, so model ranking cannot change membership.
# The full admitted list remains in state.trace_candidates; audit decisions
# contain only the selected imports. MAX_PROBE_FETCHES_PER_PASS bounds total I/O.
MAX_CANDIDATES_PER_FINDING: Final[int] = 5

# Python candidates may name a function or class method instead of a module.
# Strip up to two trailing names to find the module (FUP-209); deeper chains
# can remain unresolved. JS/TS relative paths do not use this fallback.
MAX_SUFFIX_STRIP_LEVELS: Final[int] = 2

# Probes and content fetches share one budget to bound GitHub requests.
# Unfunded probes count as missing; an unfunded content fetch leaves its
# decision resolved but adds no file. Warn once per pass. Cross-review rate
# accounting is deferred to FUP-077.
MAX_PROBE_FETCHES_PER_PASS: Final[int] = 1024

# graph._trace_router permits another analysis pass only below this limit
# and when this trace pass fetched new files. Otherwise it routes to synthesize.
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
    github_factory: Callable[[int], Awaitable[InstallationGitHubClient]],
) -> dict[str, object]:
    """Resolve suggested imports and return decisions plus newly fetched files.

    LangGraph deduplicates the two returned lists. The fetched-file count
    describes this invocation alone, so the router can decide whether another
    analysis pass has new work. Ranking uses at most one model call per pass.
    """
    # Retrying the same analysis round reuses its audit phase identity.
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="trace",
        attempt_key=f"trace-pass-{len(state.analysis_rounds)}",
    )

    # Mark the start of this pass.
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

    # Link candidates to findings; duplicate proposal hashes raise.
    join_lookup = _build_proposal_hash_join(state)
    # Relative imports resolve from the finding's file (DECISIONS.md#024).
    finding_file_paths = _build_finding_file_paths(state)

    # Choose candidates for findings that still need a decision.
    ranked_by_finding = await _rank_pending_candidates(
        state=state,
        join_lookup=join_lookup,
        provider=provider,
        trace_model=trace_model,
    )

    # Accumulate decisions and files locally; return state only after success.
    accumulated_decisions: list[TraceDecision] = []
    accumulated_fetched_files: list[TraceFetchedFile] = []

    gh_client = await github_factory(state.pr_context.installation_id)
    head_sha = state.pr_context.head_sha
    pr_file_paths: frozenset[str] = frozenset(cf.path for cf in state.pr_context.changed_files)
    # Already-fetched paths need no new content fetch and must not inflate
    # the count that tells the router whether this pass added new work.
    fetched_target_files: set[str] = {f.path for f in state.trace_fetched_files}

    # Reuse probe results across findings and seed them with known PR content.
    # Deleted files have no head content, so they still require a probe.
    probe_memo: dict[str, bytes | None] = {
        cf.path: cf.content_head.encode("utf-8")
        for cf in state.pr_context.changed_files
        if cf.content_head is not None
    }
    probe_budget = _ProbeBudget(remaining=MAX_PROBE_FETCHES_PER_PASS)

    # A crash can leave audit ahead of state. Recover persisted decisions
    # rather than resolve them again under potentially changed rules.
    # Concurrent conflicts still raise through the persister (spec M7).
    persisted_decisions: dict[UUID, TraceDecisionEvent] = {}
    if ranked_by_finding:
        persisted_decisions = {
            event.source_finding_id: event
            for event in await trace_sink.get_trace_decisions(review_id=state.review_id)
        }

    for finding_id in sorted(ranked_by_finding):
        # Selection already deduplicated and capped this finding's imports.
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
            source_finding_file = finding_file_paths.get(finding_id)
            if source_finding_file is None:
                # An absent file indicates inconsistent finding lookups.
                # The empty anchor makes relative imports unresolved;
                # module-form probes do not use it.
                logger.warning(
                    "trace: finding %s has no file_path in analysis rounds "
                    "(join/rounds divergence?); specifier-form candidates in "
                    "this bucket will land unresolved",
                    finding_id,
                )
                source_finding_file = ""
            probe_outcome = await _resolve_via_probes(
                candidates=bucket,
                source_finding_file=source_finding_file,
                gh_client=gh_client,
                owner=state.pr_context.owner,
                repo=state.pr_context.repo,
                head_sha=head_sha,
                probe_memo=probe_memo,
                probe_budget=probe_budget,
            )

            # Record the selected imports and the paths resolution accepted.
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

            # Persist first; a retry may return an existing canonical event.
            persisted_event = await trace_sink.emit_trace_decision(decision_event)

        # Build state from the event the sink returned, keeping retry/replay
        # consistent even when this attempt proposed different metadata.
        accumulated_decisions.append(
            TraceDecision(
                source_finding_id=persisted_event.source_finding_id,
                target_file=persisted_event.target_file,
                reason=persisted_event.reason,
                resolution_status=persisted_event.resolution_status,
                proposed_import_strings=persisted_event.proposed_import_strings,
                resolved_candidate_paths=persisted_event.resolved_candidate_paths,
                # Retain a persisted trace_path even though new events omit it.
                trace_path=persisted_event.trace_path,
            )
        )

        # Phase 2 fetches content only for resolved paths outside the PR.
        # Deduplicate before I/O so shared targets count as one new file.
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
                probe_budget=probe_budget,
            )
            if fetched_file is not None:
                accumulated_fetched_files.append(fetched_file)
                fetched_target_files.add(fetched_file.path)

    # Earlier failures propagate without an end marker or state delta.
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

    # Always overwrite the per-pass count, including zero, so the router
    # cannot mistake a previous pass's fetched files for new work.
    state_delta: dict[str, object] = {
        "trace_decisions": accumulated_decisions,
        "trace_fetched_files": accumulated_fetched_files,
        "last_trace_pass_fetched_count": len(accumulated_fetched_files),
    }
    return state_delta


async def _rank_pending_candidates(
    *,
    state: ReviewState,
    join_lookup: dict[str, UUID],
    provider: LLMProvider,
    trace_model: str,
) -> dict[UUID, list[TraceCandidate]]:
    """Choose unique imports for each undecided finding, then rank if useful.

    Deduplicate and cap each group in reducer insertion order. Sorting by
    candidate_id would let model-controlled hashes influence selection.
    PR content can still influence analysis emission order; see FOLLOWUPS
    for TraceRankingRejectedEvent attribution.

    Groups flatten in finding-id order. The model may reorder candidates
    within groups, but cannot change membership. Skip ranking when every
    group has one candidate, since there is no within-group order to choose.
    """
    # Do not decide a finding again on graph re-entry (DECISIONS.md#025).
    already_traced: set[UUID] = {d.source_finding_id for d in state.trace_decisions}
    # Rejected-parent candidates remain in state for inspection, but cannot
    # join a finding and therefore produce no decision or fetch (#025).
    candidate_buckets = _bucket_candidates_by_finding(state.trace_candidates, join_lookup)

    pending_buckets = {
        finding_id: bucket
        for finding_id, bucket in candidate_buckets.items()
        if finding_id not in already_traced
    }

    ordered_candidates: tuple[TraceCandidate, ...] = ()
    if pending_buckets:
        # Different reasons can share an import string. Deduplicate first so
        # repeated imports do not consume the cap.
        capped_buckets: dict[UUID, tuple[TraceCandidate, ...]] = {
            finding_id: _dedupe_by_import_string(bucket)[:MAX_CANDIDATES_PER_FINDING]
            for finding_id, bucket in pending_buckets.items()
        }
        flat_candidates = tuple(
            c for finding_id in sorted(capped_buckets) for c in capped_buckets[finding_id]
        )
        # Guard against a group retaining its key after losing all candidates.
        if flat_candidates:
            # Keep the existing order when ranking adds no useful priority.
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

    # Regroup while retaining each finding's candidate priority.
    ranked_by_finding: dict[UUID, list[TraceCandidate]] = {}
    for candidate in ordered_candidates:
        finding_id = join_lookup.get(candidate.source_proposal_hash)
        if finding_id is None:
            # Defensive against mutation during the ranking call.
            continue
        ranked_by_finding.setdefault(finding_id, []).append(candidate)
    return ranked_by_finding


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


def _build_finding_file_paths(state: ReviewState) -> dict[UUID, str]:
    """Map finding IDs to source paths for relative import resolution.

    Read all analysis rounds in order; a repeated ID uses its last recorded path.
    Resolution is importing-file-relative (DECISIONS.md#024, Amended 2026-07-03)."""
    return {
        finding.finding_id: finding.file_path
        for analysis_round in state.analysis_rounds
        for finding in analysis_round.findings
    }


def _bucket_candidates_by_finding(
    candidates: Sequence[TraceCandidate],
    join_lookup: dict[str, UUID],
) -> dict[UUID, list[TraceCandidate]]:
    """Group candidates whose source_proposal_hash joins an admitted finding.

    Unjoined candidates remain in state for inspection but cause no decision or
    fetch (DECISIONS.md#025 point 6). Rejected parents can produce this normal
    forensic path, so it logs at INFO."""
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
    """Use the configured trace model to rank the full selected candidate list.

    A rejected ranking warns and retains input order; TraceRankingRejectedEvent
    attribution remains deferred. The provider records the model call."""

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
    """Keep the first candidate per import, in supplied order.

    Different reasons can produce distinct candidate IDs for one import, while
    TraceDecisionEvent requires unique proposed_import_strings (#024). Selection
    happens before ranking, so duplicate reasons cannot consume the group cap."""
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
    """Remaining GitHub requests, shared by probes and content fetches.

    Findings consume budget in finding-id order; candidates follow ranking
    order. When the budget runs out, that ranking influences which paths
    receive requests. Log exhaustion only once per pass.
    """

    remaining: int
    exhausted_logged: bool = False


async def _resolve_via_probes(
    *,
    candidates: Sequence[TraceCandidate],
    source_finding_file: str,
    gh_client: InstallationGitHubClient,
    owner: str,
    repo: str,
    head_sha: str,
    probe_memo: dict[str, bytes | None],
    probe_budget: _ProbeBudget,
) -> _ProbeOutcome:
    """Phase 1: find a single verified path for this finding (spec M8).

    Python candidates first try the whole dotted name as a module, then
    strip trailing names for function/class fallbacks (FUP-209). A fallback
    must find the first stripped name in a defining context in the content;
    an existing parent package alone does not prove the candidate exists.
    JS/TS relative imports resolve from the finding's file at level zero
    only, using the contained path rules in coordinates (DECISIONS.md#024).

    Stop at the first level with any verified paths across the group. One
    distinct path resolves; several are ambiguous; none at any level is
    unresolved. Deeper fallbacks cannot interfere with a shallower match.

    Validate paths before I/O, reuse known content, and fetch new paths
    concurrently within the shared budget. Invalid, missing, and unfunded
    paths count as negatives. Probe failures (non-404 errors) propagate;
    404 / None content means the candidate did not resolve.
    """
    for strip_level in range(MAX_SUFFIX_STRIP_LEVELS + 1):
        candidate_paths: list[tuple[TraceCandidate, tuple[str, ...]]] = []
        for candidate in candidates:
            validated_paths: list[str] = []
            for candidate_path in _probe_paths_for(
                candidate.import_string, strip_level, source_finding_file
            ):
                try:
                    validated_paths.append(validate_diff_path(candidate_path))
                except CoordinateError:
                    continue
            candidate_paths.append((candidate, tuple(validated_paths)))

        fetch_paths_into_memo = _fetch_paths_into_memo
        probe_paths: list[str] = []
        for _probe_candidate, candidate_probe_paths in candidate_paths:
            for candidate_probe_path in candidate_probe_paths:
                probe_paths.append(candidate_probe_path)
        await fetch_paths_into_memo(
            probe_paths,
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
        unique_real_paths = list(dict.fromkeys(level_real))

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
    """Fetch new validated paths into the shared memo, within the budget.

    Known paths cost no requests. Fund the first unique new paths in order;
    unfunded paths stay absent. Gather completes all funded probes, then
    processes results in path order so the first error is deterministic.
    """
    to_fetch: list[str] = []
    for pending_probe_path in dict.fromkeys(paths):
        if pending_probe_path not in probe_memo:
            to_fetch.append(pending_probe_path)
    if not to_fetch:
        return

    funded = to_fetch[: probe_budget.remaining] if probe_budget.remaining > 0 else []
    if len(funded) < len(to_fetch) and not probe_budget.exhausted_logged:
        logger.warning(
            "trace: pass fetch budget exhausted (MAX_PROBE_FETCHES_PER_PASS=%d); "
            "%d probe path(s) skipped this pass — affected candidates "
            "resolve as unresolved",
            MAX_PROBE_FETCHES_PER_PASS,
            len(to_fetch) - len(funded),
        )
        probe_budget.exhausted_logged = True
    if not funded:
        return
    # Charge every issued request, even if an error stops result processing.
    # Any future transient-error softening (FUP-211) must retain this accounting.
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
    """Fetch one validated path at head SHA; a 404 means no match.

    Other failures propagate instead of hiding unavailable context (spec M8).
    Read status_code without importing vendor exceptions outside github/.
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
    """Check for defining-looking text before accepting a suffix-strip fallback.

    Accept def/class, assignment/annotation, import and bare continuation-line
    patterns. This is a textual heuristic, not AST proof: indentation and text
    inside string literals are not parsed. Wildcard re-exports do not name the
    symbol. Invalid UTF-8 returns False."""
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
    """Map a dotted module name to its Python file and package paths.

    `foo.bar` becomes `foo/bar.py` and `foo/bar/__init__.py`.
    This only constructs paths; the subsequent probe checks existence.
    """
    parts = import_string.split(".")
    base = "/".join(parts)
    return (f"{base}.py", f"{base}/__init__.py")


def _probe_paths_for(
    import_string: str, strip_level: int, source_finding_file: str
) -> tuple[str, ...]:
    """Form-dispatched probe paths per `DECISIONS.md#024` (Amended 2026-07-03).

    Form discrimination delegates to `coordinates.is_relative_specifier_form`
    (the single partition rule). Relative specifiers: level 0 fans out
    via `coordinates.relative_specifier_candidate_paths` against the
    proposing finding's file (importing-file-relative resolution,
    containment + `validate_diff_path` inside `coordinates/`); levels
    >= 1 return () — specifiers are already paths, so no suffix-strip
    ladder exists for them. Everything else is module form and delegates
    to `_tier_paths_for` (the FUP-209 ladder).
    """
    if is_relative_specifier_form(import_string):
        if strip_level > 0:
            return ()
        return relative_specifier_candidate_paths(import_string, source_finding_file)
    return _tier_paths_for(import_string, strip_level)


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
    probe_budget: _ProbeBudget,
) -> TraceFetchedFile | None:
    """Phase 2: fetch a resolved file's UTF-8 content for analysis (spec M8).

    The caller supplies a validated path and persisted finding identity.
    Return None for an exhausted budget, missing content, or invalid UTF-8;
    the audit decision stays resolved. Invalid UTF-8 logs a warning, while
    non-404 fetch failures propagate.
    """
    if probe_budget.remaining <= 0:
        if not probe_budget.exhausted_logged:
            logger.warning(
                "trace: pass fetch budget exhausted (MAX_PROBE_FETCHES_PER_PASS=%d); "
                "Phase 2 content fetch skipped for %s — decision stays resolved, "
                "no fetched file this pass",
                MAX_PROBE_FETCHES_PER_PASS,
                target_file,
            )
            probe_budget.exhausted_logged = True
        return None
    probe_budget.remaining -= 1
    # A second fetch may return no content even after a successful probe.
    content_bytes = await _probe_single_path(
        target_file,
        gh_client=gh_client,
        owner=owner,
        repo=repo,
        head_sha=head_sha,
    )
    if content_bytes is None:
        return None
    try:
        content_head = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Keep invalid text out of analysis state without changing the decision.
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
