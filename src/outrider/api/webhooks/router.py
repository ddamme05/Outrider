# FastAPI router for the GitHub webhook endpoint per intake-and-webhook spec.
"""`POST /webhooks/github` — webhook receiver.

Sequence:

  1. `received_at = datetime.now(UTC)` — capture as early as possible,
     BEFORE any input-validation work. Held in a local; only attached
     to the seed ReviewState once signature/membership/idempotency clear.
  2. Read `X-Hub-Signature-256`; missing → 401 WITHOUT reading the
     request body (defends against unauthenticated multi-GB POST
     buffering pressure).
  3. `body = await request.body()` — raw bytes captured BEFORE any
     model binding. FastAPI's default `Request.body()` caches internally
     (Starlette `_body`); a second call returns the same bytes.
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
     is_eval=False)`; call `await dispatcher.dispatch(state)`.
 11. Return 202 Accepted with `review_id`.

`X-GitHub-Delivery` is logged for traceability; never persisted.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, Response, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError as SQLAlchemyIntegrityError

from outrider.api.webhooks.schemas import PullRequestEventPayload
from outrider.api.webhooks.signature import verify_signature
from outrider.db.models.audit_events import AuditEvent as AuditEventRow
from outrider.db.models.installations import Installation, InstallationRepository
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from outrider.dispatcher import ReviewDispatcher
    from outrider.github.config import GitHubAppSettings


__all__ = ["router"]


logger = logging.getLogger(__name__)


router = APIRouter()


_PULL_REQUEST_ACTION_ALLOWLIST: frozenset[str] = frozenset({"opened", "synchronize", "reopened"})


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
    # POST consuming RAM before failing. Full Content-Length /
    # streaming-HMAC cap tracked at FUP-034.
    if x_hub_signature_256 is None:
        logger.warning(
            "webhook rejected: missing X-Hub-Signature-256",
            extra={"x_github_delivery": x_github_delivery},
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing signature")

    # Step 3: capture raw body bytes BEFORE any JSON parsing. Signature
    # verification (next step) requires the raw bytes the sender HMAC'd.
    body = await request.body()

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
    settings: GitHubAppSettings = request.app.state.github_app_settings
    secret = settings.webhook_secret.get_secret_value()
    signature_ok = verify_signature(secret, body, x_hub_signature_256)
    if not signature_ok:
        logger.warning(
            "webhook rejected: signature mismatch",
            extra={"x_github_delivery": x_github_delivery},
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature")

    # Step 5: only pull_request events for V1.
    if x_github_event != "pull_request":
        logger.info(
            "webhook ignored: non-pull_request event (signed, allowed to no-op)",
            extra={"x_github_event": x_github_event, "x_github_delivery": x_github_delivery},
        )
        return {"status": "ignored", "reason": "event_type"}

    # Parse the validated JSON.
    try:
        payload = PullRequestEventPayload.model_validate_json(body)
    except ValidationError as exc:
        # Log a redacted error summary — count + first error path.
        # `exc.errors()` may include `"input"` values from the failing
        # payload, AND key names that collide with the LLM-content
        # logging filter's Tier-1 keys (e.g., `"system"`); both are
        # silently dropped by `RejectLLMContentFilter`. Reducing to a
        # count + path keeps the log line useful for operators without
        # the collision surface.
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

    # Step 6: action allowlist — signed but unsupported actions return 2xx.
    if payload.action not in _PULL_REQUEST_ACTION_ALLOWLIST:
        logger.info(
            "webhook ignored: unsupported action",
            extra={"action": payload.action, "x_github_delivery": x_github_delivery},
        )
        return {"status": "ignored", "reason": "action"}

    # Step 7: active-membership SELECT BEFORE the reviews INSERT — otherwise
    # the FK on reviews.installation_id ON DELETE RESTRICT produces an
    # IntegrityError indistinguishable from the natural-key conflict.
    session_factory = request.app.state.session_factory
    installation_id = payload.installation.id
    repo_id = payload.repository.id

    async with session_factory() as session:
        membership_row = await session.execute(
            select(InstallationRepository, Installation)
            .join(
                Installation,
                Installation.installation_id == InstallationRepository.installation_id,
            )
            .where(
                InstallationRepository.installation_id == installation_id,
                InstallationRepository.repo_id == repo_id,
                InstallationRepository.removed_at.is_(None),
                Installation.tombstoned_at.is_(None),
            )
        )
        if membership_row.first() is None:
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
            session.add(
                Review(
                    id=review_id,
                    installation_id=installation_id,
                    repo_id=repo_id,
                    pr_number=payload.pull_request.number,
                    head_sha=head_sha,
                    status="running",
                    files_examined=0,
                    files_traced_beyond_diff=0,
                    llm_calls_made=0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_cost_usd=Decimal("0"),
                    wall_clock_seconds=Decimal("0"),
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
        # driver without diag) re-raises so GitHub retries.
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
        # surfaces as 5xx → GitHub retries indefinitely → loss of
        # idempotency under a dependency upgrade.
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
        try:
            async with session_factory() as failure_session, failure_session.begin():
                from sqlalchemy import update as _update  # noqa: PLC0415

                await failure_session.execute(
                    _update(Review).where(Review.id == review_id).values(status="failed")
                )
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
            logger.exception(
                "webhook dispatch failed after row commit; review marked failed",
                extra={"review_id": str(review_id), "x_github_delivery": x_github_delivery},
            )
        raise

    return {"status": "running", "review_id": str(review_id)}


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
