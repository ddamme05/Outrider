# Intake node — webhook-seeded ReviewState → PR file metadata + content.
"""Intake enriches `pr_context.changed_files` per `DECISIONS.md#020`.

Sequence per the intake-and-webhook spec:

  1. Emit `ReviewPhaseEvent(marker="start", node_id="intake")` via the
     injected `phase_event_sink`.
  2. `gh = github_factory(state.pr_context.installation_id)`.
  3. Phase 1 (sequential): `gh.rest.pulls.async_list_files(...)` via
     `github.fetch.list_pr_files`. Returns the per-file metadata list.
  4. Whole-PR pre-flight size gate (per `docs/spec.md §6.10`: > 1000 lines
     OR > 30 files → skip). On skip: write `reviews.status='skipped'`
     via `db_factory`, emit phase-end, return `Command(goto=END)`. No
     per-file `FileExaminationEvent` in this branch — the fan-out is
     bypassed.
  5. Phase 2 (parallel under `asyncio.Semaphore(8)`): per-file content
     fetch via `github.fetch.fetch_file_content_at`. Paths are validated
     inside the fetch helper (path-traversal payloads raise
     `CoordinateError` before any githubkit call).
  6. Per-file outcome → emit `FileExaminationEvent`:
       - Clean fetch → `parse_status="clean"`.
       - Per-file content cap exceeded / non-file response →
         `parse_status="skipped"`, `skip_reason=SkipReason.OVERSIZED`.
         File is dropped from the resulting `ChangedFile` tuple.
  7. Construct the `ChangedFile` tuple with status-aware completeness
     per `DECISIONS.md#020`.
  8. Emit `ReviewPhaseEvent(marker="end")`.
  9. Return `Command(update={"pr_context": new_pr_context}, goto="triage")`.

**Failure mode** — fail-loud re-raise: on any exception during intake,
catch at the node boundary, emit phase-end first, write
`reviews.status='failed'` via `db_factory`, then re-raise. The phase-end-
before-status-write ordering preserves the audit story (replay sees node
entry + closing phase-end before terminal status).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from langgraph.graph import END
from langgraph.types import Command
from sqlalchemy import update

from outrider.ast_facts.models import SkipReason
from outrider.audit.events import FileExaminationEvent, ReviewPhaseEvent
from outrider.coordinates.diff_parser import validate_diff_path
from outrider.coordinates.errors import CoordinateError
from outrider.db.models.reviews import Review
from outrider.github.fetch import fetch_file_content_at, list_pr_files
from outrider.schemas.pr_context import ChangedFile, PRContext

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.agent.state import ReviewState
    from outrider.audit.sinks import FileExaminationSink, PhaseEventSink
    from outrider.github import InstallationGitHubClient

__all__ = ["intake"]


logger = logging.getLogger(__name__)


# Per `docs/spec.md §6.10`: skip if > 1000 lines OR > 30 files.
_SIZE_GATE_MAX_LINES = 1000
_SIZE_GATE_MAX_FILES = 30

# `list_pr_files`'s `per_page` MUST be `_SIZE_GATE_MAX_FILES + 1` so the
# size gate fires deterministically: if the API returns `+1` entries, the
# PR is over the threshold regardless of total file count. Encoding this
# structurally (rather than as a constant on each side) catches drift —
# a future bump of `_SIZE_GATE_MAX_FILES` without a matching `per_page`
# change would silently bypass the gate.
_LIST_PR_FILES_PER_PAGE = _SIZE_GATE_MAX_FILES + 1

# Per the spec line 4 in the intake-node sequence: bounded concurrency on
# the per-file content fan-out. 8 is the chosen ceiling — large enough to
# pipeline content fetches over LAN latency, small enough that a 30-file
# PR with worst-case-MB-each content doesn't pin worker memory.
_CONCURRENCY_LIMIT = 8

# Aggregate cap on decoded content held in memory per intake invocation.
# Per-file is 1 MB (`fetch._PER_FILE_CONTENT_CAP_BYTES`); 30 files × 1 MB
# = 30 MB worst case under the file-count gate. 10 MB total is the
# practical ceiling: enough for triage/analyze on a normally-sized PR,
# bounded enough that 100 concurrent webhook deliveries × 10 MB = 1 GB
# memory pressure (manageable on most worker sizes).
_TOTAL_DECODED_BYTES_CAP = 10_000_000


async def intake(
    state: ReviewState,
    *,
    github_factory: Callable[[int], InstallationGitHubClient],
    db_factory: async_sessionmaker[AsyncSession],
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
) -> Command[Literal["triage"]]:
    """Fetch the PR's changed-files metadata + content; enrich state.

    Returns `Command` directing the next graph step:
      - On success: `Command(update={"pr_context": new_pr_context}, goto="triage")`.
      - On size-gate skip: `Command(goto=END)` — END is the langgraph
        sentinel string `"__end__"`. Per LangGraph 1.1.6 docs, the
        `Command[Literal[...]]` annotation lists the NAMED node
        destinations; END routing works at runtime regardless of the
        annotation. mypy may flag the END `goto` at the call site —
        suppress per-line where needed.
      - On failure: re-raises after writing `reviews.status='failed'`
        (no `Command` return — exception propagates through LangGraph).
    """
    phase_id = str(uuid4())
    pr_context = state.pr_context

    # Tracks whether the phase-start event actually persisted. The failure
    # handler only emits a matching phase-end if this is True — otherwise
    # an early phase-start emit failure would produce an end-only marker
    # (durable phase-end with no corresponding phase-start), breaking the
    # phase-events-bound-work invariant in the opposite direction from the
    # orphan-start case the audit_integrity_violation log handles.
    phase_start_persisted = False

    try:
        # Phase-start emission is inside the guarded boundary so a
        # persister failure here triggers the same `status='failed'`
        # cleanup path as any other intake failure. Outside the try,
        # a persister exception would strand the review at 'running'.
        phase_start = ReviewPhaseEvent(
            review_id=state.review_id,
            is_eval=state.is_eval,
            phase_id=phase_id,
            node_id="intake",
            marker="start",
        )
        await phase_event_sink.emit_phase(phase_start)
        phase_start_persisted = True

        gh = github_factory(pr_context.installation_id)

        # Phase 1 (sequential): file list with status / counts / patch.
        files_metadata = await list_pr_files(
            gh,
            owner=pr_context.owner,
            repo=pr_context.repo,
            pull_number=pr_context.pr_number,
            per_page=_LIST_PR_FILES_PER_PAGE,
        )

        # Whole-PR pre-flight size gate per docs/spec.md §6.10.
        total_lines = pr_context.total_additions + pr_context.total_deletions
        if total_lines > _SIZE_GATE_MAX_LINES or len(files_metadata) > _SIZE_GATE_MAX_FILES:
            await _set_review_status(db_factory, state.review_id, "skipped")
            await _emit_phase_end(phase_event_sink, state, phase_id)
            return Command(goto=END)  # type: ignore[arg-type]  # END is a runtime sentinel not in the named-dest Literal

        # Phase 2 (parallel under semaphore): per-file content + emit
        # FileExaminationEvent per file. Shared byte-budget accumulator
        # caps total decoded content at `_TOTAL_DECODED_BYTES_CAP` —
        # protects against a worst-case 30 × 1MB = 30MB pressure under
        # the per-file + file-count gates.
        #
        # `asyncio.TaskGroup` (NOT `asyncio.gather`): with `gather`,
        # the first exception propagates but sibling tasks KEEP
        # RUNNING — a sibling could emit a `FileExaminationEvent`
        # AFTER the failure handler's phase-end marker, violating
        # `phase-events-bound-work` (replay loses the causal barrier).
        # TaskGroup cancels siblings on first failure and propagates
        # as `ExceptionGroup`; we unwrap so the outer handler sees
        # the original cause, not a wrapping group.
        semaphore = asyncio.Semaphore(_CONCURRENCY_LIMIT)
        byte_budget = _ByteBudget(cap=_TOTAL_DECODED_BYTES_CAP)
        per_file_results: list[asyncio.Task[ChangedFile | None]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                for meta in files_metadata:
                    per_file_results.append(
                        tg.create_task(
                            _process_one_file(
                                semaphore=semaphore,
                                gh=gh,
                                pr_context=pr_context,
                                state=state,
                                file_metadata=meta,
                                file_examination_sink=file_examination_sink,
                                byte_budget=byte_budget,
                            )
                        )
                    )
        # `except* BaseException` (not `except* Exception`) so a
        # CancelledError from the OUTER scope (lifespan shutdown,
        # client disconnect) — which since Python 3.8 inherits from
        # BaseException, NOT Exception — matches and gets unwrapped
        # the same way as a real per-file failure. Without this, the
        # BaseExceptionGroup[CancelledError] would propagate raw to the
        # outer `except BaseException` failure handler — cleanup still
        # runs, but operators see a less useful exception trace.
        except* BaseException as eg:
            # Unwrap the ExceptionGroup so the outer `except Exception`
            # handler sees a single exception, matching pre-TaskGroup
            # `gather` semantics for downstream consumers. The TaskGroup
            # has already cancelled and awaited every sibling task
            # before this raise — no per-file task can emit audit
            # events after this point.
            #
            # `eg.exceptions` is ordered by raise-time, so the first
            # entry is typically the originating failure. But TaskGroup
            # injects `CancelledError` into still-running siblings, and
            # under tight timing a sibling's cancellation can finish
            # frame-teardown before the original failure's. Prefer the
            # first non-CancelledError so operators see the root cause,
            # not the cancellation symptom. Fall back to index-0 if all
            # children are CancelledError (cancellation came from
            # OUTSIDE the TaskGroup — e.g., lifespan shutdown).
            #
            # `from eg` explicitly chains the ExceptionGroup as
            # `__cause__` so a debugger / log handler that walks the
            # chain can still reach every sibling failure. Without an
            # explicit `from`, ruff B904 would flag the bare re-raise.
            root_cause = next(
                (exc for exc in eg.exceptions if not isinstance(exc, asyncio.CancelledError)),
                eg.exceptions[0],
            )
            raise root_cause from eg
        changed_file_results = [task.result() for task in per_file_results]
        # `_process_one_file` returns ChangedFile | None — drop the None
        # (skipped) entries; the rest assemble into the immutable tuple.
        changed_files = tuple(cf for cf in changed_file_results if cf is not None)

        new_pr_context = pr_context.model_copy(update={"changed_files": changed_files})

        await _emit_phase_end(phase_event_sink, state, phase_id)
        return Command(update={"pr_context": new_pr_context}, goto="triage")

    except BaseException:
        # Fail-loud re-raise per the intake-failure-mode discipline.
        # Order: emit phase-end → write status='failed' → re-raise.
        # BOTH the phase-end emit AND the status-write are wrapped in
        # try/except so the bare `raise` at the end re-raises the
        # ORIGINAL intake exception, not a SQLAlchemy / persister error
        # from the best-effort cleanup.
        #
        # `except BaseException` (not `except Exception`):
        # `asyncio.CancelledError` inherits from `BaseException`, so
        # catching only `Exception` would let a cancellation (lifespan
        # shutdown, client disconnect, supervisor abort) bypass
        # phase-end emission AND the `status='failed'` write — leaving
        # a durable phase-start with no end and a review row stuck at
        # 'running'. The bare `raise` at the end re-raises the original
        # exception (including CancelledError), so the graph runner's
        # cancellation semantics are preserved.
        # Cleanup runs as a shielded task: a SECOND cancellation arriving
        # mid-cleanup (lifespan abort racing the original failure) would
        # otherwise interrupt `_emit_phase_end` or `_set_review_status`
        # mid-await and leave the review stranded. asyncio.shield protects
        # the cleanup task from cancellation propagating in from the
        # outer await; if the outer await raises CancelledError, we
        # explicitly await the shielded cleanup task to completion before
        # letting the bare `raise` propagate the original failure.

        async def _failure_cleanup() -> None:
            # Only emit phase-end if the matching phase-start actually
            # persisted. Without this guard, a persister failure during
            # the very first emit would produce an end-only marker
            # (durable phase-end with no corresponding phase-start) —
            # the opposite of the orphan-start case below, equally
            # destructive to `phase-events-bound-work` replay semantics.
            if not phase_start_persisted:
                # Distinct log signal so operators querying for "intake
                # phase-start orphans" can distinguish (a) "no phase-start
                # ever attempted" (no log) from (b) "phase-start attempted
                # but persister failed before commit" (this log line).
                # Without it, the gate fires silently and the only signal
                # is the propagating exception in the outer scope.
                logger.warning(
                    "intake: phase-start persistence failed; skipping phase-end "
                    "emit to preserve start↔end pairing invariant",
                    extra={
                        "review_id": str(state.review_id),
                        "phase_id": phase_id,
                        "node_id": "intake",
                    },
                )
            if phase_start_persisted:
                try:
                    await _emit_phase_end(phase_event_sink, state, phase_id)
                except Exception:
                    # AUDIT-INTEGRITY VIOLATION: the failure-path
                    # phase-end event did NOT persist. This leaves a
                    # durable phase-start with no matching phase-end —
                    # replay tools and dashboard projections that bound
                    # work between start/end markers see the phase as
                    # never-completed. `audit_integrity_violation=True`
                    # so the anomaly scanner AND ad-hoc operator greps
                    # can find these.
                    logger.exception(
                        "intake: phase-end emit failed during failure handling; "
                        "proceeding to status='failed' write anyway "
                        "(AUDIT-INTEGRITY: orphan phase-start without phase-end)",
                        extra={
                            "review_id": str(state.review_id),
                            "phase_id": phase_id,
                            "node_id": "intake",
                            "audit_integrity_violation": True,
                        },
                    )
            try:
                await _set_review_status(db_factory, state.review_id, "failed")
            except Exception:
                logger.exception(
                    "intake: status='failed' write failed during failure "
                    "handling; row remains 'running' but original intake "
                    "exception will still re-raise. Operators must rely on "
                    "the audit phase-end marker + the stuck-review sweep "
                    "(future) to recover.",
                    extra={"review_id": str(state.review_id)},
                )

        cleanup_task = asyncio.create_task(_failure_cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            # Outer task is being cancelled, but the shielded cleanup
            # task is still running. Await it before propagating so the
            # phase-end + status='failed' writes complete first.
            await cleanup_task
        raise


class _ByteBudget:
    """Shared accumulator for total decoded bytes per intake invocation.

    Per-file fetches run concurrently under a semaphore; this class
    serializes the cap-check + accumulate step with an asyncio lock so
    the total cannot overshoot. Files that would push the running total
    over the cap are denied (and the caller emits OVERSIZED for that
    file); files that fit are admitted and the total is incremented.

    Second guard at the aggregate level: the per-file 1MB cap + 30-file
    gate alone left a 30MB worst case that this accumulator bounds.
    """

    def __init__(self, *, cap: int) -> None:
        self._cap = cap
        self._used = 0
        self._lock = asyncio.Lock()

    async def try_reserve(self, n_bytes: int) -> bool:
        """Atomic check-and-reserve. Returns True if the reservation
        succeeded (caller can safely decode `n_bytes`), False if the
        reservation would overshoot the cap (caller skips with
        OVERSIZED). On True, the running total is incremented by
        `n_bytes` before the lock releases.
        """
        async with self._lock:
            if self._used + n_bytes > self._cap:
                return False
            self._used += n_bytes
            return True

    async def release(self, n_bytes: int) -> None:
        """Release a previously-reserved chunk back to the budget.

        Used by two-sided fetch paths (modified / renamed) when one
        side classifies clean but the other fails — the file is
        skipped, so the clean side's reservation must return to the
        pool to avoid crowding out later valid files.
        """
        async with self._lock:
            self._used = max(0, self._used - n_bytes)


async def _semaphore_guarded_fetch(
    semaphore: asyncio.Semaphore,
    gh: InstallationGitHubClient,
    *,
    owner: str,
    repo: str,
    path: str,
    ref: str,
) -> bytes | None:
    """Acquire the semaphore for the duration of one GitHub content fetch.

    Pushing the semaphore down to per-fetch (rather than per-file)
    ensures the `_CONCURRENCY_LIMIT` actually caps the number of in-flight
    GitHub requests at the documented value. `modified` / `renamed`
    statuses do two fetches; without this inner bounding, effective
    concurrency would be 2 * limit.
    """
    async with semaphore:
        return await fetch_file_content_at(gh, owner=owner, repo=repo, path=path, ref=ref)


async def _gather_two_fetches(
    base_coro: Coroutine[Any, Any, bytes | None],
    head_coro: Coroutine[Any, Any, bytes | None],
) -> tuple[bytes | None, bytes | None]:
    """Run two per-file fetches concurrently with strict sibling cancellation.

    Uses `asyncio.TaskGroup` rather than `asyncio.gather` so that if
    one fetch fails, the other is CANCELLED and awaited before the
    failure propagates upward. `asyncio.gather` leaves siblings
    running by default, which would let the second fetch finish AFTER
    the outer per-file failure path emits its skip/audit event —
    violating `phase-events-bound-work` at a finer granularity than
    the outer fan-out's TaskGroup.

    ExceptionGroup unwrap mirrors the outer fan-out's logic: prefer
    the first non-CancelledError so operators see the root cause, not
    a sibling's cancellation symptom. `from eg` preserves the
    ExceptionGroup as `__cause__` for traceback walking.
    """
    base_task: asyncio.Task[bytes | None]
    head_task: asyncio.Task[bytes | None]
    try:
        async with asyncio.TaskGroup() as tg:
            base_task = tg.create_task(base_coro)
            head_task = tg.create_task(head_coro)
    # `except* BaseException` mirrors the outer fan-out: outer-scope
    # cancellation propagates as BaseExceptionGroup[CancelledError]
    # (Python 3.8+ CancelledError ∈ BaseException), so `except* Exception`
    # would miss it and the unwrap-the-root-cause logic wouldn't fire.
    except* BaseException as eg:
        root_cause = next(
            (exc for exc in eg.exceptions if not isinstance(exc, asyncio.CancelledError)),
            eg.exceptions[0],
        )
        raise root_cause from eg
    return base_task.result(), head_task.result()


async def _process_one_file(
    *,
    semaphore: asyncio.Semaphore,
    gh: InstallationGitHubClient,
    pr_context: PRContext,
    state: ReviewState,
    file_metadata: Any,
    file_examination_sink: FileExaminationSink,
    byte_budget: _ByteBudget,
) -> ChangedFile | None:
    """Per-file: fetch content per status, emit FileExaminationEvent,
    return ChangedFile or None (skipped)."""
    raw_filename = file_metadata.filename
    status = file_metadata.status
    additions = file_metadata.additions
    deletions = file_metadata.deletions
    patch = getattr(file_metadata, "patch", None)
    raw_previous_filename = getattr(file_metadata, "previous_filename", None)

    # Path validation happens HERE at the top — the validated form is
    # what reaches the GitHub API URL, the FileExaminationEvent
    # `file_path`, and the `ChangedFile.path` / `previous_path` fields.
    # Spec line 71: "`file_path` is the post-`coordinates.validate_diff_path`
    # normalized form (never the raw GitHub-supplied string)."
    #
    # A CoordinateError here is a path-traversal attempt or malformed
    # upstream path. We CANNOT emit a `FileExaminationEvent` for it
    # because the only `file_path` we have is the rejected raw form,
    # and the spec forbids persisting raw GitHub-supplied paths in
    # audit metadata. Persisting `raw_filename` here would leak
    # attacker-controlled bytes into the append-only audit table.
    #
    # Resolution: log the rejection (logs + metrics observe the event),
    # drop the file from `changed_files`, but emit NO audit row. A
    # future audit shape with a safe rejection representation (e.g.,
    # `FileExaminationEvent.parse_status="rejected"` with a content-hash
    # of the raw path + a sanitized excerpt) is FUP-eligible; not added
    # in this spec.
    #
    # Other exceptions inside the fetch block propagate through the
    # surrounding `asyncio.TaskGroup` and trigger the node's failure path.
    try:
        filename = validate_diff_path(raw_filename)
        previous_filename = (
            validate_diff_path(raw_previous_filename) if raw_previous_filename is not None else None
        )
    except CoordinateError as exc:
        # Truncate both the raw bytes AND the exception message so an
        # attacker can't blow up log volume with a 10MB filename.
        # `CoordinateError.__str__` embeds the full rejected path in
        # its message — logging `exc` directly would defeat the
        # `truncated_raw` cap.
        truncated_raw = repr(raw_filename)[:200]
        truncated_reason = repr(str(exc))[:200]
        logger.warning(
            "intake skipping file: path validation rejected upstream filename "
            "(truncated raw: %s; reason: %s)",
            truncated_raw,
            truncated_reason,
        )
        return None

    content_base: str | None = None
    content_head: str | None = None
    try:
        if status == "added":
            bytes_head = await _semaphore_guarded_fetch(
                semaphore,
                gh,
                owner=pr_context.owner,
                repo=pr_context.repo,
                path=filename,
                ref=pr_context.head_sha,
            )
            if bytes_head is None:
                await _emit_skip(
                    file_examination_sink,
                    state=state,
                    file_path=filename,
                    reason=SkipReason.OVERSIZED,
                )
                return None
            content_head, skip_reason = await _classify_or_reserve_decode(bytes_head, byte_budget)
            if skip_reason is not None:
                await _emit_skip(
                    file_examination_sink,
                    state=state,
                    file_path=filename,
                    reason=skip_reason,
                )
                return None
        elif status == "removed":
            bytes_base = await _semaphore_guarded_fetch(
                semaphore,
                gh,
                owner=pr_context.owner,
                repo=pr_context.repo,
                path=filename,
                ref=pr_context.base_sha,
            )
            if bytes_base is None:
                await _emit_skip(
                    file_examination_sink,
                    state=state,
                    file_path=filename,
                    reason=SkipReason.OVERSIZED,
                )
                return None
            content_base, skip_reason = await _classify_or_reserve_decode(bytes_base, byte_budget)
            if skip_reason is not None:
                await _emit_skip(
                    file_examination_sink,
                    state=state,
                    file_path=filename,
                    reason=skip_reason,
                )
                return None
        elif status == "modified":
            # Nested TaskGroup (NOT asyncio.gather) per the same
            # `phase-events-bound-work` discipline as the outer
            # per-file fan-out: if one fetch fails, the other must be
            # cancelled BEFORE the failure propagates upward, otherwise
            # the sibling could keep running past the outer TaskGroup's
            # cancel-point and leak side effects.
            bytes_base, bytes_head = await _gather_two_fetches(
                _semaphore_guarded_fetch(
                    semaphore,
                    gh,
                    owner=pr_context.owner,
                    repo=pr_context.repo,
                    path=filename,
                    ref=pr_context.base_sha,
                ),
                _semaphore_guarded_fetch(
                    semaphore,
                    gh,
                    owner=pr_context.owner,
                    repo=pr_context.repo,
                    path=filename,
                    ref=pr_context.head_sha,
                ),
            )
            if bytes_base is None or bytes_head is None:
                await _emit_skip(
                    file_examination_sink,
                    state=state,
                    file_path=filename,
                    reason=SkipReason.OVERSIZED,
                )
                return None
            content_base, skip_reason_base = await _classify_or_reserve_decode(
                bytes_base, byte_budget
            )
            content_head, skip_reason_head = await _classify_or_reserve_decode(
                bytes_head, byte_budget
            )
            # Skip if either side is binary/malformed. Release the
            # clean side's reservation so it doesn't crowd out later
            # valid files.
            binary_reason = skip_reason_base or skip_reason_head
            if binary_reason is not None:
                if skip_reason_base is None:
                    await byte_budget.release(len(bytes_base))
                if skip_reason_head is None:
                    await byte_budget.release(len(bytes_head))
                await _emit_skip(
                    file_examination_sink,
                    state=state,
                    file_path=filename,
                    reason=binary_reason,
                )
                return None
        elif status == "renamed":
            if previous_filename is None:
                msg = f"GitHub returned status='renamed' for {filename!r} without previous_filename"
                raise ValueError(msg)
            # See modified-branch comment for the TaskGroup rationale.
            bytes_base, bytes_head = await _gather_two_fetches(
                _semaphore_guarded_fetch(
                    semaphore,
                    gh,
                    owner=pr_context.owner,
                    repo=pr_context.repo,
                    path=previous_filename,
                    ref=pr_context.base_sha,
                ),
                _semaphore_guarded_fetch(
                    semaphore,
                    gh,
                    owner=pr_context.owner,
                    repo=pr_context.repo,
                    path=filename,
                    ref=pr_context.head_sha,
                ),
            )
            if bytes_base is None or bytes_head is None:
                await _emit_skip(
                    file_examination_sink,
                    state=state,
                    file_path=filename,
                    reason=SkipReason.OVERSIZED,
                )
                return None
            content_base, skip_reason_base = await _classify_or_reserve_decode(
                bytes_base, byte_budget
            )
            content_head, skip_reason_head = await _classify_or_reserve_decode(
                bytes_head, byte_budget
            )
            binary_reason = skip_reason_base or skip_reason_head
            if binary_reason is not None:
                if skip_reason_base is None:
                    await byte_budget.release(len(bytes_base))
                if skip_reason_head is None:
                    await byte_budget.release(len(bytes_head))
                await _emit_skip(
                    file_examination_sink,
                    state=state,
                    file_path=filename,
                    reason=binary_reason,
                )
                return None
        else:
            # Unknown status from GitHub — log + skip rather than
            # crash the whole intake. Forward-compat for GitHub
            # adding a new file status.
            logger.warning("intake skipping unknown file status %r for %r", status, filename)
            return None
    except CoordinateError:
        # Defense-in-depth: validation happened at the top of this
        # function, but fetch_file_content_at re-validates internally.
        # If somehow validation passed at the top but failed inside
        # fetch (e.g., a future fetch helper changes its inputs), we
        # still emit OVERSIZED + drop rather than fail the whole intake.
        await _emit_skip(
            file_examination_sink,
            state=state,
            file_path=filename,
            reason=SkipReason.OVERSIZED,
        )
        return None

    # Emit clean FileExaminationEvent and build ChangedFile.
    clean_event = FileExaminationEvent(
        review_id=state.review_id,
        is_eval=state.is_eval,
        file_path=filename,
        examination_type="intake_fetch",
        node_id="intake",
        parse_status="clean",
    )
    await file_examination_sink.emit_file_examination(clean_event)

    return ChangedFile(
        path=filename,
        status=status,
        additions=additions,
        deletions=deletions,
        patch=patch,
        content_base=content_base,
        content_head=content_head,
        previous_path=previous_filename if status == "renamed" else None,
    )


async def _emit_skip(
    sink: FileExaminationSink,
    *,
    state: ReviewState,
    file_path: str,
    reason: SkipReason,
) -> None:
    """Emit FileExaminationEvent(parse_status='skipped', skip_reason=reason).

    Single helper for every skip path (per-file cap exceeded, binary
    detection, malformed UTF-8) so the audit-event shape is uniform.
    """
    event = FileExaminationEvent(
        review_id=state.review_id,
        is_eval=state.is_eval,
        file_path=file_path,
        examination_type="intake_fetch",
        node_id="intake",
        parse_status="skipped",
        skip_reason=reason,
    )
    await sink.emit_file_examination(event)


async def _emit_phase_end(
    sink: PhaseEventSink,
    state: ReviewState,
    phase_id: str,
) -> None:
    """Emit `ReviewPhaseEvent(marker='end', node_id='intake')`."""
    event = ReviewPhaseEvent(
        review_id=state.review_id,
        is_eval=state.is_eval,
        phase_id=phase_id,
        node_id="intake",
        marker="end",
    )
    await sink.emit_phase(event)


async def _set_review_status(
    db_factory: async_sessionmaker[AsyncSession],
    review_id: Any,
    status: Literal["skipped", "failed"],
) -> None:
    """Update reviews.status via the injected session factory.

    `status` is narrowed to the two values intake's failure / size-gate
    paths actually write — any other transitions (`running` at INSERT,
    `awaiting_approval` at HITL, `completed` at publish) happen
    elsewhere in the graph and through different paths. Narrow Literal
    catches fat-fingered values (`"skiped"`, `"pending"`) at type-check.
    """
    async with db_factory() as session, session.begin():
        await session.execute(update(Review).where(Review.id == review_id).values(status=status))


async def _classify_or_reserve_decode(
    content_bytes: bytes, byte_budget: _ByteBudget
) -> tuple[str | None, SkipReason | None]:
    """Classify content first, then (only on clean text) reserve the
    aggregate byte budget.

    Outcomes, returned as `(decoded_or_None, skip_reason_or_None)`:

      - Bytes contain a NUL byte (definitive binary marker) →
        `(None, SkipReason.OVERSIZED)`. The binary content is rejected
        WITHOUT consuming the byte budget; subsequent valid text files
        retain their share. (Skip reason routed through OVERSIZED
        pending a canonical amendment to `DECISIONS.md#018` that would
        add a `BINARY` value — see FUP-033. The behavior is preserved:
        binary blobs are skipped, not silently corrupted into
        U+FFFD-filled strings.)
      - Bytes are not valid UTF-8 (truncated text, mixed encoding) →
        `(None, SkipReason.OVERSIZED)`. Same routing as binary; refusal
        to flow corrupted text to the LLM. No budget consumed.
      - Bytes are valid UTF-8 AND aggregate budget allows →
        `(decoded_str, None)`. Budget is reserved AFTER classification
        passes so a failed classification doesn't crowd out later files.
      - Bytes are valid UTF-8 but aggregate budget exhausted →
        `(None, SkipReason.OVERSIZED)`.

    The classify-then-reserve order is the load-bearing invariant:
    binary/malformed bytes must NOT consume the aggregate text budget,
    or a single binary blob in an early file would starve later valid
    files into spurious OVERSIZED skips.
    """
    if b"\x00" in content_bytes:
        return None, SkipReason.OVERSIZED
    try:
        decoded = content_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, SkipReason.OVERSIZED
    admitted = await byte_budget.try_reserve(len(content_bytes))
    if not admitted:
        return None, SkipReason.OVERSIZED
    return decoded, None
