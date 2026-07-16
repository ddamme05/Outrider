# FastAPI router for the GitHub webhook endpoint per intake-and-webhook spec.
"""`POST /webhooks/github` — webhook receiver.

Sequence:

  1. `received_at = datetime.now(UTC)` — capture as early as possible,
     BEFORE any input-validation work. Held in a local; only attached
     to the seed ReviewState once signature/membership/idempotency clear.
  2. Read `X-Hub-Signature-256`; missing → 401 WITHOUT reading the
     request body (defends against unauthenticated multi-GB POST
     buffering pressure).
  2b. Content-Length precheck against `_MAX_WEBHOOK_BODY_BYTES` (1 MiB).
     `Content-Length` exceeding cap → 413 BEFORE `await request.body()`.
     Malformed (non-integer) Content-Length → 400. Post-read length
     guard at step 3 is defense-in-depth for chunked / missing-header
     deliveries. Streaming-HMAC bound remains FUP-034 part 1.
  3. `body = await request.body()` — raw bytes captured BEFORE any
     model binding; rejected with 413 if `len(body)` exceeds the cap
     (catches chunked / lying-Content-Length cases). FastAPI's default
     `Request.body()` caches internally (Starlette `_body`); a second
     call returns the same bytes.
  4. `verify_signature(secret, body, signature_header)` via the
     route-facing module that delegates to `github/webhooks.py`. Returns
     False → 401. Unexpected raises propagate as 5xx (verifier
     programming errors are operator-visible, not auth-failure-shaped).
  5. Parse JSON payload via `PullRequestEventPayload`.
  6. Event/action allowlist (opened/synchronize/reopened) — others 2xx no-op.
  7. Active-membership SELECT — `installations.tombstoned_at IS NULL`
     AND `installation_repositories.removed_at IS NULL` for
     `(installation_id, repo_id)`. Absent/inactive → 4xx.
  8. Idempotency fast-path SELECT on `(repo_id, pr_number, head_sha)`;
     if found → 200 with existing `review_id`.
  9. Single transaction: INSERT review + INSERT AgentTransitionEvent
     (direct SQL, the documented exception to the persister-only rule).
     Commit. On IntegrityError with `uq_review_natural_key` → duplicate;
     return existing review. On other IntegrityError → fall back to
     natural-key SELECT (user-approved defensive pattern; deviates from
     spec's narrow-introspection-only rule), then re-raise only if no
     natural-key row exists.
 10. Construct seed `ReviewState(review_id, pr_context, received_at,
     is_eval=False)`; call `await dispatcher.dispatch(state)`. Dispatch
     failure → mark review failed via a shielded cleanup task (drained
     under `except BaseException` so the original failure isn't masked),
     then re-raise.
 11. Return 202 Accepted with `review_id`.

`X-GitHub-Delivery` is logged for traceability; never persisted.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError as SQLAlchemyIntegrityError

from outrider.api.webhooks.installation_events import (
    handle_installation_event,
    handle_installation_repositories_event,
)
from outrider.api.webhooks.schemas import (
    InstallationEventPayload,
    InstallationRepositoriesEventPayload,
    PullRequestEventPayload,
)
from outrider.api.webhooks.signature import verify_signature
from outrider.db.models.audit_events import AuditEvent as AuditEventRow
from outrider.db.models.installations import (
    Installation,
    InstallationRepository,
    active_repo_membership,
)
from outrider.db.models.reviews import Review
from outrider.github.credentials import GitHubUnconfiguredError

if TYPE_CHECKING:
    from outrider.dispatcher import ReviewDispatcher
    from outrider.github.credentials import GitHubCredentialProvider


__all__ = ["router"]


logger = logging.getLogger(__name__)


router = APIRouter()


# Arc B2 autorun: `ready_for_review` fires when a draft PR becomes ready — the
# trigger to review a PR that opened as a draft. Draft PRs themselves are skipped
# by the draft-check below (a draft `opened`/`synchronize` is a 2xx no-op).
_PULL_REQUEST_ACTION_ALLOWLIST: frozenset[str] = frozenset(
    {"opened", "synchronize", "reopened", "ready_for_review"}
)

# Hard cap on webhook body size at the FastAPI/ASGI boundary. GitHub's
# real `pull_request` payloads are well under 1 MiB (the heaviest cases
# are big PRs with long descriptions; the payload itself does not include
# file content — intake pulls content via the contents API). 1 MiB is
# 5-10x typical worst-case, comfortable headroom without admitting
# attacker-controlled multi-MB / multi-GB bodies into RAM.
#
# Two-layer enforcement:
#   1. Content-Length precheck — reject before `await request.body()`
#      buffers anything. Most well-formed deliveries carry this header.
#   2. Post-read length guard — defense-in-depth for chunked /
#      no-content-length / lying-content-length deliveries. The
#      precheck-only path would still buffer the full body before the
#      schema/signature layer caught oversize.
#
# Full streaming-HMAC bound (no buffering even chunked) is FUP-034 part 1
# remaining work; this commit lands the Content-Length precheck + post-
# read guard, closing the unsigned/signed-but-attacker DoS path with
# Content-Length set.
_MAX_WEBHOOK_BODY_BYTES: int = 1_048_576  # 1 MiB


@router.post(
    "/webhooks/github",
    status_code=status.HTTP_202_ACCEPTED,
)
async def receive_pull_request_webhook(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Annotated[str | None, Header()] = None,
    x_github_event: Annotated[str | None, Header()] = None,
    x_github_delivery: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Receive a GitHub pull_request webhook, seed a review, dispatch the graph."""
    # Step 1: receipt timestamp, held until validation clears.
    received_at = datetime.now(UTC)

    # Log delivery id for traceability; never persisted. Logged BEFORE
    # body-read so missing-signature 401s still carry the correlation id
    # for operators tracing the GitHub-side delivery.
    logger.info(
        "webhook received",
        extra={
            "x_github_delivery": x_github_delivery,
            "x_github_event": x_github_event,
            "received_at": received_at.isoformat(),
        },
    )

    # Step 2: signature header presence — BEFORE body-read.
    #
    # Body-read-after-header-check defends against an unsigned multi-GB
    # POST consuming RAM before failing. Streaming-HMAC cap (full
    # chunked-transfer defense) tracked at FUP-034.
    if x_hub_signature_256 is None:
        logger.warning(
            "webhook rejected: missing X-Hub-Signature-256",
            extra={"x_github_delivery": x_github_delivery},
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing signature")

    # Step 2b: Content-Length precheck BEFORE buffering. A signed but
    # attacker-controlled delivery that sets `Content-Length: 10000000000`
    # would otherwise force `await request.body()` to allocate. The
    # precheck rejects at HTTP-413 before any read; the post-read guard
    # below catches the missing-header / lying-header chunked cases.
    content_length_header = request.headers.get("content-length")
    if content_length_header is not None:
        try:
            content_length = int(content_length_header)
        except ValueError:
            logger.warning(
                "webhook rejected: malformed Content-Length",
                extra={
                    "x_github_delivery": x_github_delivery,
                    "content_length_raw": content_length_header[:64],
                },
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="malformed Content-Length",
            ) from None
        # Negative Content-Length is HTTP-malformed but `int()` accepts
        # "-1" cleanly, and `-1 > cap` is False — without an explicit
        # check, a negative header bypasses both the malformed-400 path
        # AND the 413 cap, deferring rejection until after body buffer.
        if content_length < 0:
            logger.warning(
                "webhook rejected: negative Content-Length",
                extra={
                    "x_github_delivery": x_github_delivery,
                    "content_length_raw": content_length_header[:64],
                },
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="malformed Content-Length",
            )
        if content_length > _MAX_WEBHOOK_BODY_BYTES:
            logger.warning(
                "webhook rejected: Content-Length exceeds cap",
                extra={
                    "x_github_delivery": x_github_delivery,
                    "content_length": content_length,
                    "cap": _MAX_WEBHOOK_BODY_BYTES,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="payload too large",
            )

    # Step 3: capture raw body bytes BEFORE any JSON parsing. Signature
    # verification (next step) requires the raw bytes the sender HMAC'd.
    body = await request.body()
    if len(body) > _MAX_WEBHOOK_BODY_BYTES:
        # Defense-in-depth for chunked / missing / lying Content-Length.
        # Reached after buffering — strictly worse than the precheck,
        # which is why FUP-034 part 1 remains open for streaming-HMAC.
        logger.warning(
            "webhook rejected: body size exceeds cap post-read",
            extra={
                "x_github_delivery": x_github_delivery,
                "body_bytes": len(body),
                "cap": _MAX_WEBHOOK_BODY_BYTES,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="payload too large",
        )

    # Step 4: signature verification BEFORE event-type filtering.
    #
    # `verify_signature` (delegating to `githubkit.webhooks.verify`)
    # returns False for any signature mismatch — malformed digest,
    # wrong-length header, base64 garbage, mismatched HMAC — never
    # raises in those cases. We do NOT wrap this call in
    # `except Exception` → 401: an unexpected verifier exception
    # (programming bug, dependency regression) is a server-side fault
    # that should surface as 5xx, not collapse into a "401 invalid
    # signature" response that hides the actual failure class.
    credential_provider: GitHubCredentialProvider = request.app.state.credential_provider
    try:
        creds = await credential_provider.current()
    except GitHubUnconfiguredError:
        # `database` mode, not yet CONFIGURED → fail closed with 503 (the delivery shows as
        # failed on GitHub; the operator redelivers after setup — GitHub does not auto-retry).
        # The setup-only route gating also returns 503 here; this is the defense-in-depth read
        # at the secret source.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="setup incomplete"
        ) from None
    secret = creds.webhook_secret.get_secret_value()
    signature_ok = verify_signature(secret, body, x_hub_signature_256)
    if not signature_ok:
        logger.warning(
            "webhook rejected: signature mismatch",
            extra={"x_github_delivery": x_github_delivery},
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature")

    # Step 5: dispatch by event type. Install-lifecycle events (`installation`,
    # `installation_repositories`) maintain the local install CACHE via idempotent upserts
    # and 2xx — under #065 the cache is NEVER the authority (GitHub is checked live at
    # intake/publish), so these are a cheap early-out, not a security decision. They run only
    # after signature verification above. `pull_request` proceeds to review seeding below;
    # anything else 2xx no-ops (signed-but-unsupported, so GitHub doesn't retry).
    if x_github_event == "installation":
        return await _dispatch_installation_event(
            body, request.app.state.session_factory, x_github_delivery
        )
    if x_github_event == "installation_repositories":
        return await _dispatch_installation_repositories_event(
            body, request.app.state.session_factory, x_github_delivery
        )
    if x_github_event != "pull_request":
        logger.info(
            "webhook ignored: non-pull_request event (signed, allowed to no-op)",
            extra={"x_github_event": x_github_event, "x_github_delivery": x_github_delivery},
        )
        return {"status": "ignored", "reason": "event_type"}

    # Parse the validated JSON via the shared redacted-400 helper (the raw input carries
    # attacker-controlled strings + LLM-content-filter Tier-1 key collisions — see the helper).
    payload = _parse_webhook_or_reject(PullRequestEventPayload, body, x_github_delivery)

    # Step 6: action allowlist — signed but unsupported actions return 2xx.
    if payload.action not in _PULL_REQUEST_ACTION_ALLOWLIST:
        logger.info(
            "webhook ignored: unsupported action",
            extra={"action": payload.action, "x_github_delivery": x_github_delivery},
        )
        return {"status": "ignored", "reason": "action"}

    # Step 6b (Arc B2 autorun): skip DRAFT PRs. A draft `opened`/`synchronize` is
    # a 2xx no-op — the review runs when `ready_for_review` fires (draft → ready),
    # at which point `pull_request.draft` is False.
    if payload.pull_request.draft:
        logger.info(
            "webhook ignored: draft pull request",
            extra={"action": payload.action, "x_github_delivery": x_github_delivery},
        )
        return {"status": "ignored", "reason": "draft"}

    # Step 7: active-membership SELECT BEFORE the reviews INSERT — otherwise
    # the FK on reviews.installation_id ON DELETE RESTRICT produces an
    # IntegrityError indistinguishable from the natural-key conflict.
    session_factory = request.app.state.session_factory
    installation_id = payload.installation.id
    repo_id = payload.repository.id

    async with session_factory() as session:
        # Active install: exists, not tombstoned (#012), not suspended (Arc B2).
        install = (
            await session.execute(
                select(Installation).where(
                    Installation.installation_id == installation_id,
                    Installation.tombstoned_at.is_(None),
                    Installation.suspended_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        # Arc B2 repository_selection gate: `all` installs authorize at the install
        # level (no per-repo row); `selected` installs require the active per-repo
        # membership row. Absent/tombstoned/suspended install → not authorized.
        # (Per DECISIONS.md#065 the cache is the EARLY-OUT here; the authoritative
        # live GitHub auth check is the intake node's first step — tracked B2 work.)
        authorized = install is not None and (
            install.repository_selection == "all"
            or (
                await session.execute(
                    select(InstallationRepository.id).where(
                        active_repo_membership(installation_id, repo_id)
                    )
                )
            ).first()
            is not None
        )
        if not authorized:
            logger.warning(
                "webhook rejected: unknown or inactive installation+repo membership",
                extra={
                    "installation_id": installation_id,
                    "repo_id": repo_id,
                    "x_github_delivery": x_github_delivery,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="installation or repository not active",
            )

        head_sha = payload.pull_request.head.sha

        # Step 8: idempotency fast-path SELECT.
        existing_row = await session.execute(
            select(Review).where(
                Review.repo_id == repo_id,
                Review.pr_number == payload.pull_request.number,
                Review.head_sha == head_sha,
            )
        )
        existing = existing_row.scalar_one_or_none()
        if existing is not None:
            # Duplicate delivery — return 200 with existing review_id per
            # spec line 8 + 35 ("Duplicate delivery returns 200 with the
            # existing review_id"). Override the route default of 202.
            response.status_code = status.HTTP_200_OK
            return {
                "status": existing.status,
                "review_id": str(existing.id),
            }

    # Step 9: single transaction — INSERT review + INSERT
    # AgentTransitionEvent. Direct SQL on audit_events is the documented
    # exception to the persister-only writer rule (Audit boundary bullet
    # in the spec).
    review_id = uuid4()
    # Pre-mint event_id so the row's event_id column and the payload's
    # event_id field carry the same UUID. Without this, the row would
    # use the DB's `gen_random_uuid()` server-default and the payload
    # would carry Pydantic's `default_factory=uuid4` — replay tools
    # that join payload-event_id to row-event_id would silently mismatch.
    event_id = uuid4()
    pr_context = _build_seed_pr_context(payload)

    try:
        async with session_factory() as session, session.begin():
            # No aggregate-metric seed: those `reviews` columns were dropped per
            # DECISIONS.md#037 — review metrics live in the audit stream.
            session.add(
                Review(
                    id=review_id,
                    installation_id=installation_id,
                    repo_id=repo_id,
                    pr_number=payload.pull_request.number,
                    # Attacker-controlled webhook data (bounded 4096 at the input
                    # boundary), persisted as a parameterized column value. Captured
                    # once at creation; idempotency on (repo_id, pr_number, head_sha)
                    # makes a new head SHA a new row, so the title is never mutated.
                    pr_title=payload.pull_request.title,
                    head_sha=head_sha,
                    status="running",
                    retention_expires_at=(
                        received_at + request.app.state.retention_settings.review_retention_ttl
                    ),
                )
            )
            session.add(
                AuditEventRow(
                    event_id=event_id,
                    review_id=review_id,
                    event_type="agent_transition",
                    timestamp=received_at,
                    is_eval=False,
                    payload=_serialize_webhook_agent_transition(
                        event_id=event_id,
                        review_id=review_id,
                        received_at=received_at,
                    ),
                )
            )
    except SQLAlchemyIntegrityError as exc:
        # Step 9 cont'd: narrow IntegrityError introspection — only
        # uq_review_natural_key violations are duplicate-delivery.
        # Anything else (audit-events PK collision, FK violation,
        # driver without diag) re-raises so the delivery is marked
        # failed on GitHub (redeliverable; GitHub does not auto-retry).
        if _is_reviews_natural_key_conflict(exc):
            # The duplicate row exists. Re-read it (separate session
            # because the failing transaction was rolled back).
            async with session_factory() as session:
                existing_row = await session.execute(
                    select(Review).where(
                        Review.repo_id == repo_id,
                        Review.pr_number == payload.pull_request.number,
                        Review.head_sha == head_sha,
                    )
                )
                existing = existing_row.scalar_one()
            # IntegrityError-path duplicate also returns 200 per spec.
            response.status_code = status.HTTP_200_OK
            return {
                "status": existing.status,
                "review_id": str(existing.id),
            }

        # Defensive fallback. **NOTE: this deviates from the canonical
        # spec's narrow-introspection-only rule** (which mandates
        # re-raise when `_is_reviews_natural_key_conflict` returns
        # False). The deviation is user-approved and documented in the
        # spec's Actual Outcome section.
        #
        # The rationale: primary introspection returned False could
        # mean (a) genuine audit-side conflict / FK violation that must
        # re-raise, OR (b) psycopg / driver shape change made
        # `exc.orig.diag.constraint_name` unreadable. (b) is a
        # silent-misclassification risk: every duplicate delivery
        # surfaces as 5xx → shows failed on GitHub and invites manual
        # redelivery → loss of idempotency under a dependency upgrade.
        #
        # Cheap defense: SELECT the natural-key row. If it exists, the
        # failed INSERT was a duplicate delivery; return 200. If not,
        # the failure was on the audit-side / FK / unrelated constraint
        # and re-raise is correct. The corner case of a concurrent
        # committer also masking a TRUE audit collision is acceptable —
        # the audit-side error surfaces in monitoring; GitHub-side
        # duplicate detection is preserved.
        async with session_factory() as session:
            fallback_row = await session.execute(
                select(Review).where(
                    Review.repo_id == repo_id,
                    Review.pr_number == payload.pull_request.number,
                    Review.head_sha == head_sha,
                )
            )
            fallback_existing = fallback_row.scalar_one_or_none()
        if fallback_existing is not None:
            logger.warning(
                "webhook IntegrityError: natural-key row exists but "
                "constraint-name introspection returned False — falling "
                "back to natural-key SELECT (psycopg diag shape may have "
                "changed). Treating as duplicate delivery.",
                extra={
                    "x_github_delivery": x_github_delivery,
                    "review_id": str(fallback_existing.id),
                },
            )
            response.status_code = status.HTTP_200_OK
            return {
                "status": fallback_existing.status,
                "review_id": str(fallback_existing.id),
            }
        # No natural-key row exists — genuine audit-side conflict, FK
        # violation, or other. Re-raise so GitHub retries.
        raise

    # Step 10: construct seed ReviewState and dispatch.
    from outrider.agent.state import ReviewState as _ReviewState  # noqa: PLC0415 — avoid circular

    seed_state = _ReviewState(
        review_id=review_id,
        pr_context=pr_context,
        received_at=received_at,
        is_eval=False,
    )

    dispatcher = _build_dispatcher(request, background_tasks)
    try:
        await dispatcher.dispatch(seed_state)
    except BaseException:
        # `BaseException` (not `Exception`) so `asyncio.CancelledError`
        # — which inherits from BaseException in Python 3.8+ — doesn't
        # bypass the cleanup. Lifespan shutdown / client disconnect /
        # supervisor abort during dispatch would otherwise strand the
        # review at 'running' with no failure signal.
        #
        # The review row + initial AgentTransitionEvent committed; if
        # dispatch fails (event-loop shutdown, JSON-serialize crash,
        # broker unreachable in V2), the row would sit at 'running'
        # forever and a GitHub retry would short-circuit through the
        # natural-key fast path returning the stranded review. Mark
        # `status='failed'` so the row reflects reality, then re-raise
        # so the handler returns 5xx and operators see the error.
        #
        # Trade-off (acknowledged in spec): GitHub's retry will then
        # see the existing 'failed' row via the natural-key fast path
        # and return 200 — the failed delivery does not auto-retry the
        # graph. The user must push a new commit (new head_sha →
        # different natural key) to get a fresh review. The full
        # durable-recovery story (dispatched_at column with conditional
        # re-dispatch) is FUP-eligible per the spec's idempotency bullet.
        #
        # The failed-status write is wrapped in its own try/except so a
        # DB error during cleanup doesn't mask the original dispatch
        # failure. Bare `raise` re-raises the original exception
        # (including CancelledError); the cleanup-failure log line is
        # distinct from the dispatch-failure log so operators can tell
        # which path went wrong.
        #
        # The cleanup is spawned as a task and awaited via
        # `asyncio.shield` so a SECOND cancellation arriving while the
        # status write is in flight doesn't interrupt the write and
        # strand the review at 'running'. `await asyncio.shield(coro)`
        # alone is insufficient — the outer task still observes the
        # CancelledError; we explicitly catch it and `await cleanup_task`
        # to drain the shielded write before re-raising the original
        # dispatch failure.
        async def _mark_failed() -> None:
            try:
                async with session_factory() as failure_session, failure_session.begin():
                    from sqlalchemy import and_ as _and  # noqa: PLC0415
                    from sqlalchemy import update as _update  # noqa: PLC0415

                    # WHERE-on-status guards against stomping a row that
                    # already advanced past `running` (V2 Celery dispatch
                    # may enqueue successfully then raise on the return
                    # path; an opportunistic worker that picks up the job
                    # AND moves the row to `awaiting_approval` BEFORE
                    # this cleanup task runs would otherwise see its
                    # state overwritten to `failed`). rowcount=0 path
                    # logs explicitly so the dispatch-failure-vs-row-
                    # advancement ambiguity is observable.
                    result = await failure_session.execute(
                        _update(Review)
                        .where(_and(Review.id == review_id, Review.status == "running"))
                        .values(status="failed")
                    )
                    rowcount = getattr(result, "rowcount", 0) or 0
            except Exception:
                logger.exception(
                    "webhook dispatch failed AND failed-status cleanup also failed; "
                    "review remains at 'running' (operator must remediate)",
                    extra={
                        "review_id": str(review_id),
                        "x_github_delivery": x_github_delivery,
                    },
                )
            else:
                if rowcount == 0:
                    logger.error(
                        "webhook dispatch failed but review no longer 'running' "
                        "(concurrent advancement); failed-status write skipped",
                        extra={
                            "review_id": str(review_id),
                            "x_github_delivery": x_github_delivery,
                        },
                    )
                else:
                    logger.error(
                        "webhook dispatch failed after row commit; review marked failed",
                        extra={
                            "review_id": str(review_id),
                            "x_github_delivery": x_github_delivery,
                        },
                    )

        cleanup_task = asyncio.create_task(_mark_failed())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            # Drain the shielded task before re-raising. `_mark_failed`
            # catches Exception, NOT BaseException — so a CancelledError
            # arriving mid-cleanup OR an uncaught system-level exception
            # would propagate from `await cleanup_task` and MASK the
            # original dispatch failure. Catch + log + fall through to
            # the bare `raise` below so the original exception still
            # propagates.
            try:
                await cleanup_task
            except BaseException:
                logger.exception(
                    "webhook dispatch failed AND failed-status cleanup task "
                    "did not complete cleanly; review may remain at 'running'",
                    extra={
                        "review_id": str(review_id),
                        "x_github_delivery": x_github_delivery,
                    },
                )
        raise

    return {"status": "running", "review_id": str(review_id)}


def _parse_webhook_or_reject[PayloadT: BaseModel](
    model_cls: type[PayloadT], body: bytes, x_github_delivery: str | None
) -> PayloadT:
    """Parse the signature-verified `body` into `model_cls`, or raise `HTTPException(400)` with
    a REDACTED error summary (count + first error path only — never the raw input, which carries
    attacker-controlled strings and keys that collide with the LLM-content filter's Tier-1 keys).
    Shared by the pull_request path + the install-event dispatch."""
    try:
        return model_cls.model_validate_json(body)
    except ValidationError as exc:
        errors = exc.errors()
        first_loc = ".".join(str(p) for p in errors[0]["loc"]) if errors else "<unknown>"
        logger.warning(
            "webhook rejected: payload schema validation failed",
            extra={
                "x_github_delivery": x_github_delivery,
                "error_count": len(errors),
                "first_error_loc": first_loc,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid payload shape"
        ) from None


async def _dispatch_installation_event(
    body: bytes, session_factory: Any, x_github_delivery: str | None
) -> dict[str, str]:
    """Parse + handle an `installation` event in one transaction (#065 cache-hint upsert). A
    cheap early-out, never a security decision; returns a 2xx status dict."""
    payload = _parse_webhook_or_reject(InstallationEventPayload, body, x_github_delivery)
    async with session_factory() as session, session.begin():
        result = await handle_installation_event(payload, session)
    logger.info(
        "installation event handled",
        extra={
            "action": payload.action,
            "installation_id": payload.installation.id,
            "x_github_delivery": x_github_delivery,
            "result": result.get("status"),
        },
    )
    return result


async def _dispatch_installation_repositories_event(
    body: bytes, session_factory: Any, x_github_delivery: str | None
) -> dict[str, str]:
    """Parse + handle an `installation_repositories` event in one transaction (#065 cache-hint
    membership upsert); returns a 2xx status dict."""
    payload = _parse_webhook_or_reject(
        InstallationRepositoriesEventPayload, body, x_github_delivery
    )
    async with session_factory() as session, session.begin():
        result = await handle_installation_repositories_event(payload, session)
    logger.info(
        "installation_repositories event handled",
        extra={
            "action": payload.action,
            "installation_id": payload.installation.id,
            "x_github_delivery": x_github_delivery,
            "result": result.get("status"),
        },
    )
    return result


def _build_seed_pr_context(payload: PullRequestEventPayload) -> Any:
    """Construct seed PRContext from the validated payload.

    Per `DECISIONS.md#020`: empty `changed_files=()` — intake fills it.
    """
    # Lazy import avoids the same circular-import shape as ReviewState.
    from outrider.schemas.pr_context import PRContext  # noqa: PLC0415

    # Canonical source per `api/webhooks/schemas.py::RepositoryRef`:
    # `owner.login` for the owner string, `name` for the repo string.
    # NOT `full_name.partition("/")` — `full_name` is informational
    # (used in logs/audit messages); deriving owner/repo from it would
    # bypass the per-field input-boundary validators and risk drift if
    # GitHub ever changes the `full_name` format.
    return PRContext(
        installation_id=payload.installation.id,
        owner=payload.repository.owner.login,
        repo=payload.repository.name,
        pr_number=payload.pull_request.number,
        pr_title=payload.pull_request.title,
        pr_body=payload.pull_request.body,
        base_sha=payload.pull_request.base.sha,
        head_sha=payload.pull_request.head.sha,
        author=payload.pull_request.user.login,
        total_additions=payload.pull_request.additions,
        total_deletions=payload.pull_request.deletions,
        changed_files=(),
    )


def _serialize_webhook_agent_transition(
    *,
    event_id: Any,
    review_id: Any,
    received_at: datetime,
) -> dict[str, Any]:
    """Construct the JSONB payload for the webhook-side AgentTransitionEvent.

    The audit_events row is inserted via direct SQL (not the persister),
    so we construct the payload dict directly. `event_id` is passed in
    (rather than letting Pydantic default-factory generate one) so the
    row's `event_id` column AND the payload's `event_id` field share
    the same UUID — replay tools that join payload-id to row-id depend
    on these matching. Latency is reported as 0 here — the transition
    represents "instantaneous handoff at row creation," and the
    wall-clock latency from webhook receipt to graph start is captured
    by intake's ReviewPhaseEvent(start) timestamp.

    Replay-equivalence: uses `outrider.audit.persister._serialize_event_payload`
    (the same helper the persister's emit methods use) so a future
    field added on `AuditEventBase` automatically produces the same
    JSONB bytes here as it would through the persister.
    """
    from outrider.audit.events import AgentTransitionEvent  # noqa: PLC0415
    from outrider.audit.persister import _serialize_event_payload  # noqa: PLC0415

    event = AgentTransitionEvent(
        event_id=event_id,
        timestamp=received_at,
        review_id=review_id,
        is_eval=False,
        from_node="webhook",
        to_node="intake",
        latency_ms=0,
    )
    return _serialize_event_payload(event)


def _is_reviews_natural_key_conflict(exc: SQLAlchemyIntegrityError) -> bool:
    """Detect whether an `IntegrityError` was raised by the
    `uq_review_natural_key` constraint specifically.

    Uses psycopg3's `exc.orig.diag.constraint_name` per the spec's
    constraint-introspection rule. If `diag` is absent or
    `constraint_name` is `None`/different, returns False.

    **Note on caller behavior:** the spec's load-bearing rule is "False
    → re-raise (audit-side conflicts, FK violations, etc.)." The
    caller adds a defensive fallback: when this returns False, it
    SELECTs the natural-key row and returns 200 if a row exists
    (treating it as a duplicate the introspection couldn't classify,
    e.g., under a psycopg-shape change). Only when the SELECT returns
    no row does the caller re-raise. The deviation is user-approved;
    see the call site (`receive_pull_request_webhook`) for the inline
    rationale.
    """
    orig = exc.orig
    if orig is None:
        return False
    # psycopg3 specific: errors expose .diag with constraint_name when
    # the failing constraint is a UNIQUE constraint.
    diag = getattr(orig, "diag", None)
    if diag is None:
        return False
    return getattr(diag, "constraint_name", None) == "uq_review_natural_key"


def _build_dispatcher(
    request: Request,
    background_tasks: BackgroundTasks,
) -> ReviewDispatcher:
    """Build a per-request `BackgroundTasksDispatcher` from the route's
    `BackgroundTasks` plus the lifespan-bound `run_graph` callable.

    Per the intake-and-webhook spec, V1 dispatcher is per-request (NOT
    lifespan-singleton) because FastAPI's `BackgroundTasks` is
    request-scoped. The lifespan provides `run_graph`; the dispatcher
    wraps it together with the request's `BackgroundTasks`.
    """
    # Lazy import: dispatcher pulls in agent.state which we already
    # lazy-import elsewhere in this module.
    from outrider.dispatcher import BackgroundTasksDispatcher  # noqa: PLC0415

    run_graph = request.app.state.run_graph
    return BackgroundTasksDispatcher(
        background_tasks=background_tasks,
        run_graph=run_graph,
    )
