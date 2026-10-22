# Intake node — webhook-seeded ReviewState → PR file metadata + content.
# See DECISIONS.md#065-authorization-is-a-live-github-check-the-local-install-db-is-a-cache
"""Fetch PR metadata and content for triage (DECISIONS.md#020).

Check the persisted review identity and live GitHub authorization before
creating the async installation client (DECISIONS.md#065, DECISIONS.md#070).
Terminal reviews exit. A denied or uncertain authorization result skips the review.
The size gate runs before content fetches and per-file events.

Fetch each file's required refs with bounded concurrency and decoded bytes.
Validate paths before I/O, emit file outcomes, and drop skipped content from
`pr_context.changed_files` before routing to triage.

On failure or cancellation, drain fetch tasks before closing the persisted
phase, attempt to mark the review failed, then re-raise the original error.
Shield cleanup against another cancellation so it can finish before exit.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END
from langgraph.types import Command
from sqlalchemy import and_, select, update

from outrider.agent.nodes.intake_config import IntakeConfig
from outrider.ast_facts.models import SkipReason
from outrider.audit.events import FileExaminationEvent, ReviewPhaseEvent
from outrider.coordinates.diff_parser import validate_diff_path
from outrider.coordinates.errors import CoordinateError
from outrider.db.models.reviews import Review
from outrider.github.fetch import fetch_file_content_at, list_pr_files
from outrider.policy.canonical import compute_phase_id
from outrider.schemas.pr_context import ChangedFile, PRContext

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.agent.state import ReviewState
    from outrider.audit.sinks import FileExaminationSink, PhaseEventSink
    from outrider.github import InstallationAuthorizer, InstallationGitHubClient

__all__ = ["intake"]


logger = logging.getLogger(__name__)


# IntakeConfig sets PR limits (docs/spec.md §6.10). Fetch max_files + 1 entries
# so the first metadata page exposes a file-count overflow.

# Bound concurrent content fetches to keep fan-out memory manageable.
_CONCURRENCY_LIMIT = 8

# Per-file limits still permit multi-file memory growth; cap total decoded
# content retained by one intake invocation.
_TOTAL_DECODED_BYTES_CAP = 10_000_000


async def intake(
    state: ReviewState,
    *,
    github_factory: Callable[[int], Awaitable[InstallationGitHubClient]],
    installation_authorizer: InstallationAuthorizer,
    db_factory: async_sessionmaker[AsyncSession],
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    intake_config: IntakeConfig | None = None,
) -> Command[Literal["triage"]]:
    """Enrich PR context and route to triage, or END on a skip or terminal review.

    Failures propagate after phase and status cleanup. END is a runtime
    sentinel; the return annotation lists named graph destinations.
    """
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="intake",
        attempt_key="intake",
    )
    pr_context = state.pr_context
    # Injected at build_graph; default reads OUTRIDER_INTAKE_* (or the spec defaults).
    limits = intake_config or IntakeConfig()

    # Pair a phase end only with a persisted start (phase-events-bound-work).
    phase_start_persisted = False

    try:
        # A failed start emit also needs status cleanup; keep it inside the try.
        phase_start = ReviewPhaseEvent(
            review_id=state.review_id,
            is_eval=state.is_eval,
            phase_id=phase_id,
            node_id="intake",
            marker="start",
        )
        await phase_event_sink.emit_phase(phase_start)
        phase_start_persisted = True

        # Load persisted identity before live authorization (DECISIONS.md#065).
        # Missing or mismatched identity fails intake; any non-authorized result
        # skips without creating a client. GitHub is the authority, not the cache.
        persisted_installation_id, repo_id, review_status = await _load_review_gate_state(
            db_factory, state.review_id
        )
        if persisted_installation_id != pr_context.installation_id:
            raise _ReviewIdentityError(
                f"reviews.installation_id={persisted_installation_id} does not match "
                f"pr_context.installation_id={pr_context.installation_id} "
                f"for review {state.review_id}"
            )
        if review_status != "running":
            # A replay must not resurrect a terminal review even if auth is restored.
            # Close this invocation's phase without changing the terminal status.
            logger.warning(
                "intake re-entry on non-running review; not resurrecting",
                extra={
                    "review_id": str(state.review_id),
                    "installation_id": pr_context.installation_id,
                    "status": review_status,
                },
            )
            await _emit_phase_end(phase_event_sink, state, phase_id)
            return Command(goto=END)  # type: ignore[arg-type]  # END is a runtime sentinel not in the named-dest Literal
        auth_result = await installation_authorizer(pr_context.installation_id, repo_id)
        if not auth_result.authorized:
            logger.warning(
                "intake live-auth denied; skipping review",
                extra={
                    "review_id": str(state.review_id),
                    "installation_id": pr_context.installation_id,
                    "repo_id": repo_id,
                    "outcome": auth_result.outcome.value,
                    "detail": auth_result.detail,
                },
            )
            await _set_review_status(db_factory, state.review_id, "skipped")
            await _emit_phase_end(phase_event_sink, state, phase_id)
            return Command(goto=END)  # type: ignore[arg-type]  # END is a runtime sentinel not in the named-dest Literal

        github_client = await github_factory(pr_context.installation_id)

        # Phase 1 (sequential): file list with status / counts / patch.
        files_metadata = await list_pr_files(
            github_client,
            owner=pr_context.owner,
            repo=pr_context.repo,
            pull_number=pr_context.pr_number,
            per_page=limits.max_files + 1,
        )

        # Whole-PR pre-flight size gate per docs/spec.md §6.10 (config-driven thresholds).
        total_lines = pr_context.total_additions + pr_context.total_deletions
        if total_lines > limits.max_lines or len(files_metadata) > limits.max_files:
            await _set_review_status(db_factory, state.review_id, "skipped")
            await _emit_phase_end(phase_event_sink, state, phase_id)
            return Command(goto=END)  # type: ignore[arg-type]  # END is a runtime sentinel not in the named-dest Literal

        # Bound concurrent requests and total decoded bytes. TaskGroup cancels
        # and drains siblings before failure propagates, so per-file audit events
        # cannot follow the phase end (phase-events-bound-work).
        semaphore = asyncio.Semaphore(_CONCURRENCY_LIMIT)
        byte_budget = _ByteBudget(cap=_TOTAL_DECODED_BYTES_CAP)
        per_file_results: list[asyncio.Task[ChangedFile | None]] = []
        try:
            async with asyncio.TaskGroup() as fetch_group:
                for file_metadata in files_metadata:
                    per_file_results.append(
                        fetch_group.create_task(
                            _process_one_file(
                                semaphore=semaphore,
                                gh=github_client,
                                pr_context=pr_context,
                                state=state,
                                file_metadata=file_metadata,
                                file_examination_sink=file_examination_sink,
                                byte_budget=byte_budget,
                            )
                        )
                    )
        # Include cancellation in grouped failures: CancelledError is a BaseException.
        except* BaseException as error_group:
            # Sibling tasks have finished. Prefer the first non-cancellation cause,
            # falling back to cancellation if necessary. Chain the group so all
            # sibling failures remain available in the traceback.
            root_cause = next(
                (
                    exc
                    for exc in error_group.exceptions
                    if not isinstance(exc, asyncio.CancelledError)
                ),
                error_group.exceptions[0],
            )
            raise root_cause from error_group
        changed_file_results = [task.result() for task in per_file_results]
        # Only admitted content proceeds to triage.
        changed_files = tuple(
            changed_file for changed_file in changed_file_results if changed_file is not None
        )

        # Revalidate the updated context; model_copy(update=...) bypasses validators.
        new_pr_context = pr_context.__class__.model_validate(
            {**pr_context.model_dump(), "changed_files": changed_files}
        )

        await _emit_phase_end(phase_event_sink, state, phase_id)
        return Command(update={"pr_context": new_pr_context}, goto="triage")

    except BaseException:
        # BaseException includes cancellation; both need phase/status cleanup.
        # Close the persisted phase before attempting failed status, then re-raise.
        # Log ordinary cleanup failures without replacing the original exception.
        # Shield cleanup from another outer cancellation and drain it before
        # propagating the original failure.

        async def _failure_cleanup() -> None:
            # Failed start persistence must not create an end-only marker.
            if not phase_start_persisted:
                # Expose failed start persistence separately from orphaned starts.
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
                    # Flag the orphaned start for scanners and operator logs,
                    # then continue with the failed-status attempt.
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
            # Drain the shielded task before re-raising. Log any BaseException
            # escaping cleanup so it cannot mask the original intake failure.
            try:
                await cleanup_task
            except BaseException:
                logger.exception(
                    "intake: failure-cleanup task did not complete cleanly; "
                    "row may remain at 'running' AND/OR phase-end may be missing "
                    "(AUDIT-INTEGRITY: cascade failure during failure handling)",
                    extra={
                        "review_id": str(state.review_id),
                        "phase_id": phase_id,
                        "node_id": "intake",
                        "audit_integrity_violation": True,
                    },
                )
        raise


class _ByteBudget:
    """Bound total decoded content retained by one intake invocation.

    Serialize reservations and releases under one lock so concurrent files
    cannot exceed the cap.
    """

    def __init__(self, *, cap: int) -> None:
        self._cap = cap
        self._used = 0
        self._lock = asyncio.Lock()

    async def try_reserve(self, n_bytes: int) -> bool:
        """Atomically reserve bytes; return False if they would exceed the cap."""
        async with self._lock:
            if self._used + n_bytes > self._cap:
                return False
            self._used += n_bytes
            return True

    async def release(self, n_bytes: int) -> None:
        """Return a clean side's reservation when its partner makes the file unusable.

        Releasing discarded content keeps later valid files from losing their budget.
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
    """Bound in-flight requests, including both sides of modified or renamed files.

    Holding a permit per file would allow two requests per permit.
    """
    async with semaphore:
        return await fetch_file_content_at(gh, owner=owner, repo=repo, path=path, ref=ref)


async def _gather_two_fetches(
    base_coro: Coroutine[Any, Any, bytes | None],
    head_coro: Coroutine[Any, Any, bytes | None],
) -> tuple[bytes | None, bytes | None]:
    """Fetch both refs and drain siblings before a failure propagates.

    This prevents late fetch effects after the enclosing phase closes
    (phase-events-bound-work). Prefer the first non-cancellation cause and
    chain the group to retain sibling failures in the traceback.
    """
    base_task: asyncio.Task[bytes | None]
    head_task: asyncio.Task[bytes | None]
    try:
        async with asyncio.TaskGroup() as fetch_group:
            base_task = fetch_group.create_task(base_coro)
            head_task = fetch_group.create_task(head_coro)
    # CancelledError belongs to BaseException.
    except* BaseException as error_group:
        root_cause = next(
            (exc for exc in error_group.exceptions if not isinstance(exc, asyncio.CancelledError)),
            error_group.exceptions[0],
        )
        raise root_cause from error_group
    return base_task.result(), head_task.result()


async def _fetch_one_side(
    *,
    semaphore: asyncio.Semaphore,
    gh: InstallationGitHubClient,
    pr_context: PRContext,
    state: ReviewState,
    file_examination_sink: FileExaminationSink,
    byte_budget: _ByteBudget,
    event_path: str,
    fetch_path: str,
    ref: str,
) -> str | None:
    """Fetch and decode one ref; return None only after emitting its skip event."""
    content_bytes = await _semaphore_guarded_fetch(
        semaphore,
        gh,
        owner=pr_context.owner,
        repo=pr_context.repo,
        path=fetch_path,
        ref=ref,
    )
    if content_bytes is None:
        await _emit_skip(
            file_examination_sink,
            state=state,
            file_path=event_path,
            reason=SkipReason.OVERSIZED,
        )
        return None
    content, skip_reason = await _classify_or_reserve_decode(content_bytes, byte_budget)
    if skip_reason is not None:
        await _emit_skip(
            file_examination_sink,
            state=state,
            file_path=event_path,
            reason=skip_reason,
        )
        return None
    return content


async def _fetch_both_sides(
    *,
    semaphore: asyncio.Semaphore,
    gh: InstallationGitHubClient,
    pr_context: PRContext,
    state: ReviewState,
    file_examination_sink: FileExaminationSink,
    byte_budget: _ByteBudget,
    event_path: str,
    base_path: str,
    head_path: str,
) -> tuple[str | None, str | None] | None:
    """Fetch and decode both refs, or emit the skip event and return None.

    Renames use distinct ref paths, but `event_path` stays the head-side name
    GitHub reports. `_gather_two_fetches` drains sibling fetches on failure.
    """
    bytes_base, bytes_head = await _gather_two_fetches(
        _semaphore_guarded_fetch(
            semaphore,
            gh,
            owner=pr_context.owner,
            repo=pr_context.repo,
            path=base_path,
            ref=pr_context.base_sha,
        ),
        _semaphore_guarded_fetch(
            semaphore,
            gh,
            owner=pr_context.owner,
            repo=pr_context.repo,
            path=head_path,
            ref=pr_context.head_sha,
        ),
    )
    if bytes_base is None or bytes_head is None:
        await _emit_skip(
            file_examination_sink,
            state=state,
            file_path=event_path,
            reason=SkipReason.OVERSIZED,
        )
        return None

    content_base, skip_reason_base = await _classify_or_reserve_decode(bytes_base, byte_budget)
    content_head, skip_reason_head = await _classify_or_reserve_decode(bytes_head, byte_budget)

    # Binary admission failures take precedence over budget rejection.
    binary_reason = _merge_skip_reasons(skip_reason_base, skip_reason_head)
    if binary_reason is not None:
        # Release the clean side's reservation so it does not crowd out later
        # valid files. Only a side with no skip reason reserved anything.
        if skip_reason_base is None:
            await byte_budget.release(len(bytes_base))
        if skip_reason_head is None:
            await byte_budget.release(len(bytes_head))
        await _emit_skip(
            file_examination_sink,
            state=state,
            file_path=event_path,
            reason=binary_reason,
        )
        return None
    return content_base, content_head


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
    # GitHubKit uses empty strings for absent optional fields. Normalize them
    # before path validation and model construction.
    patch = getattr(file_metadata, "patch", None) or None
    raw_previous_filename = getattr(file_metadata, "previous_filename", None) or None

    # Only normalized paths may reach fetches, models, and audit rows. A rejected
    # raw path cannot populate FileExaminationEvent, so log and drop it without
    # an audit row. Uncaught fetch errors propagate into intake failure handling.
    try:
        filename = validate_diff_path(raw_filename)
        previous_filename = (
            validate_diff_path(raw_previous_filename) if raw_previous_filename is not None else None
        )
    except CoordinateError as exc:
        # Exception text embeds the raw path; truncate it as well as the filename
        # to bound attacker-controlled log volume.
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
            content_head = await _fetch_one_side(
                semaphore=semaphore,
                gh=gh,
                pr_context=pr_context,
                state=state,
                file_examination_sink=file_examination_sink,
                byte_budget=byte_budget,
                event_path=filename,
                fetch_path=filename,
                ref=pr_context.head_sha,
            )
            if content_head is None:
                return None
        elif status == "removed":
            content_base = await _fetch_one_side(
                semaphore=semaphore,
                gh=gh,
                pr_context=pr_context,
                state=state,
                file_examination_sink=file_examination_sink,
                byte_budget=byte_budget,
                event_path=filename,
                fetch_path=filename,
                ref=pr_context.base_sha,
            )
            if content_base is None:
                return None
        elif status == "modified":
            both_sides = await _fetch_both_sides(
                semaphore=semaphore,
                gh=gh,
                pr_context=pr_context,
                state=state,
                file_examination_sink=file_examination_sink,
                byte_budget=byte_budget,
                event_path=filename,
                base_path=filename,
                head_path=filename,
            )
            if both_sides is None:
                return None
            content_base, content_head = both_sides
        elif status == "renamed":
            if previous_filename is None:
                msg = f"GitHub returned status='renamed' for {filename!r} without previous_filename"
                raise ValueError(msg)
            both_sides = await _fetch_both_sides(
                semaphore=semaphore,
                gh=gh,
                pr_context=pr_context,
                state=state,
                file_examination_sink=file_examination_sink,
                byte_budget=byte_budget,
                event_path=filename,
                base_path=previous_filename,
                head_path=filename,
            )
            if both_sides is None:
                return None
            content_base, content_head = both_sides
        else:
            # An unknown GitHub status should drop this file without failing intake.
            logger.warning("intake skipping unknown file status %r for %r", status, filename)
            return None
    except CoordinateError:
        # If fetch revalidation rejects the normalized path, record an OVERSIZED
        # skip without failing the whole intake.
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
    """Record a skipped-file outcome with the supplied reason."""
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


class _ReviewIdentityError(RuntimeError):
    """Persisted review identity is missing or disagrees with the seed context.

    This fails intake; live authorization denial skips it. A missing row makes
    the failed-status write a no-op, but the error still halts downstream work.
    """


async def _load_review_gate_state(
    db_factory: async_sessionmaker[AsyncSession],
    review_id: Any,
) -> tuple[int, int, str]:
    """Load persisted identity and status before authorization (DECISIONS.md#065).

    A missing row raises `_ReviewIdentityError`. The caller checks status to
    avoid resurrecting a terminal review when authorization has been restored.
    """
    async with db_factory() as session:
        row = (
            await session.execute(
                select(Review.installation_id, Review.repo_id, Review.status).where(
                    Review.id == review_id
                )
            )
        ).first()
    if row is None:
        raise _ReviewIdentityError(f"no reviews row for review_id={review_id}")
    return int(row[0]), int(row[1]), str(row[2])


async def _set_review_status(
    db_factory: async_sessionmaker[AsyncSession],
    review_id: Any,
    status: Literal["skipped", "failed"],
) -> None:
    """Set skipped or failed status only while the review is running.

    The predicate protects concurrent sweep or peer writes. Log a missed
    update so operators can see when another transition has already won.
    """
    async with db_factory() as session, session.begin():
        result = await session.execute(
            update(Review)
            .where(and_(Review.id == review_id, Review.status == "running"))
            .values(status=status)
        )
        rowcount = getattr(result, "rowcount", 0) or 0
        if rowcount == 0:
            logger.warning(
                "intake _set_review_status no-op: review %s is no longer "
                "'running' (concurrent sweep / peer write); refusing to "
                "overwrite to %r",
                review_id,
                status,
            )


def _merge_skip_reasons(base: SkipReason | None, head: SkipReason | None) -> SkipReason | None:
    """Prefer BINARY from either side over a concurrent budget rejection.

    A content-admission failure is more informative than timing pressure.
    Otherwise use the first present reason, base then head; None means both
    decoded cleanly. New priorities require an explicit branch and a
    DECISIONS.md#018 amendment.
    """
    if base == SkipReason.BINARY or head == SkipReason.BINARY:
        return SkipReason.BINARY
    return base or head


async def _classify_or_reserve_decode(
    content_bytes: bytes, byte_budget: _ByteBudget
) -> tuple[str | None, SkipReason | None]:
    """Decode valid text before reserving the aggregate byte budget.

    NUL-containing or invalid UTF-8 bytes return `(None, BINARY)` without
    reserving bytes (DECISIONS.md#018). Clean text returns `(decoded, None)`
    if it fits, otherwise `(None, OVERSIZED)`. Classification precedes
    reservation so rejected content cannot starve later valid files.

    Only decoded strings reach analysis, making its `failed+degraded_llm`
    outcome unreachable alongside the string-to-bytes round-trip there.
    A raw-bytes intake path (FUP-053) would change that contract.
    """
    if b"\x00" in content_bytes:
        return None, SkipReason.BINARY
    try:
        decoded = content_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, SkipReason.BINARY
    admitted = await byte_budget.try_reserve(len(content_bytes))
    if not admitted:
        return None, SkipReason.OVERSIZED
    return decoded, None
