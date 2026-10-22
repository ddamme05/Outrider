# See specs/2026-05-26-hitl-node.md Group 8 scheduler-wiring bullet +
# sweep/purge_expired.py + sweep/hitl_expiry.py.
"""Run one maintenance tick: reconcile, recover HITL, purge, then project verdicts.

`run_scheduled_tick` first reconciles installations with GitHub. Hard deletion
of expired installation tombstones is allowed only after this tick confirms
liveness; credential, list, or lock failures defer that deletion while the
other sweeps continue (`DECISIONS.md#012/#065/#067`). The internal lifespan
loop and external schedulers use this entry point.

`run_all_sweeps` runs HITL recovery/expiry, content retention, optional install
purge, and replay verdicts in order. Its caller must hold an explicit transaction
on `conn`, keeping the shared `SWEEP_LOCK_ID` across the time-based sweeps.
This enforces `sweep-jobs-use-advisory-locks`.
Reconcile uses a separate session lock across its GitHub call. Verdict projection
is insert-idempotent and opens separate read sessions; it can still see rows
purged by this tick's uncommitted transaction, which is safe for append-only audit.
"""

from __future__ import annotations

import logging
from datetime import timedelta  # noqa: TC003  (runtime: function-signature annotation)
from typing import TYPE_CHECKING, Any

from outrider.sweep import hitl_expiry, purge_expired, replay_verdict
from outrider.sweep.reconcile_installations import reconcile_installations

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from sqlalchemy.ext.asyncio import (
        AsyncConnection,
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
    )

    from outrider.anomaly.sinks import AnomalySink
    from outrider.audit.persister import AuditPersister
    from outrider.db.sinks import ReviewStatusSink
    from outrider.github.credentials import GitHubCredentialProvider


logger = logging.getLogger(__name__)


async def run_all_sweeps(
    *,
    conn: AsyncConnection,
    session_factory: async_sessionmaker[AsyncSession],
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
    audit_persister: AuditPersister,
    checkpointer: BaseCheckpointSaver[Any],
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    grace_period: timedelta | None = None,
    purge_role: str = "sweep",
    include_install_purge: bool = False,
) -> dict[str, Any]:
    """Run the sweep family in order inside the caller's transaction.

    HITL recovery precedes retention purge; all time-based jobs share the same
    transaction-scoped advisory lock. `include_install_purge` defaults to False
    and may be True only after this tick's reconcile confirms install liveness.
    `run_scheduled_tick` owns that prerequisite; direct callers must preserve it.

    Return HITL, retention, installation-purge, and verdict telemetry. None for
    `grace_period` selects the HITL default; explicit values must be positive.
    """
    # Keep the transaction-scoped lock across all time-based jobs. Reject callers
    # that would otherwise release it between recovery and retention purge.
    if not conn.in_transaction():
        msg = (
            "run_all_sweeps requires `conn` to be in an explicit "
            "transaction so the SWEEP_LOCK_ID advisory lock stays "
            "held across both sub-jobs. Wrap the call in `async with "
            "engine.connect() as conn, conn.begin(): await "
            "run_all_sweeps(conn=conn, ...)`."
        )
        raise RuntimeError(msg)

    # Validate before any job: non-positive grace moves the failure cutoff into
    # the future and can misclassify waiting reviews.
    if grace_period is not None and grace_period <= timedelta(0):
        msg = (
            f"run_all_sweeps: grace_period must be > timedelta(0); "
            f"got {grace_period!r}. A non-positive grace_period makes "
            f"reclaim's grace gate admit every awaiting-approval row "
            f"regardless of age — mass-marks-failed every HITL-in-flight "
            f"review in the system."
        )
        raise ValueError(msg)

    hitl_kwargs: dict[str, Any] = {
        "conn": conn,
        "session_factory": session_factory,
        "anomaly_sink": anomaly_sink,
        "review_status_sink": review_status_sink,
        "audit_persister": audit_persister,
        "checkpointer": checkpointer,
        "compiled_graph": compiled_graph,
    }
    if grace_period is not None:
        hitl_kwargs["grace_period"] = grace_period

    hitl_result = await hitl_expiry.run_once(**hitl_kwargs)
    purge_result = await purge_expired.purge_expired(conn=conn, purge_role=purge_role)
    # Hard-delete expired tombstones only after this tick confirms liveness.
    # Otherwise a restored installation with an uncleared tombstone could be
    # deleted. Keep the distinct default "install-purge" forensic audit role
    # (DECISIONS.md#012); the transaction's SWEEP_LOCK_ID is reentrant here.
    install_purge_result: dict[str, Any]
    if include_install_purge:
        install_purge_result = await purge_expired.purge_expired_installations(conn=conn)
    else:
        install_purge_result = {"skipped": "reconcile_unconfirmed"}
    # Replay-verdict projection runs LAST by ordering convention. It flips no status
    # and is natural-key idempotent, so it needs no advisory lock — it opens its own
    # sessions via `session_factory`, OUTSIDE `conn`'s lock transaction, and so cannot
    # see purge's uncommitted DELETEs above; verdicting a to-be-purged review is
    # harmless (append-only audit row, no FK to reviews) (DECISIONS#039 sibling).
    verdict_result = await replay_verdict.project_replay_verdicts(
        session_factory=session_factory, audit_persister=audit_persister
    )

    return {
        "hitl": hitl_result,
        "purge": purge_result,
        "install_purge": install_purge_result,
        "replay_verdict": verdict_result,
    }


async def run_scheduled_tick(
    *,
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    anomaly_sink: AnomalySink,
    review_status_sink: ReviewStatusSink,
    audit_persister: AuditPersister,
    checkpointer: BaseCheckpointSaver[Any],
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    provider: GitHubCredentialProvider | None,
    grace_period: timedelta | None = None,
    purge_role: str = "sweep",
) -> dict[str, Any]:
    """Reconcile installations, then run the transaction-scoped sweep family.

    Reconcile holds its own session-scoped lock across GitHub's authoritative
    list call. Only a successful, uncontended reconcile enables installation
    hard deletion. Missing/unconfigured credentials and ordinary reconcile
    failures defer that deletion; HITL, TTL, and verdict jobs still run.

    Sweep failures propagate to the scheduler after reconcile has run. Return
    the family telemetry with a `reconcile` entry (`DECISIONS.md#012/#065/#067`).
    """
    # Unconfigured credentials provide no reconcile authority (#070).
    # Credential checks can query the database and raise. Catch them with list
    # failures so unrelated HITL, TTL, and verdict work can still run.
    reconcile_confirmed = False
    reconcile_telemetry: dict[str, Any]
    if provider is None:
        reconcile_telemetry = {"ran": False, "reason": "no_credentials_configured"}
    else:
        try:
            if not await provider.is_configured():
                reconcile_telemetry = {"ran": False, "reason": "no_credentials_configured"}
            else:
                reconcile_result = await reconcile_installations(engine, provider)
                # Confirmed only when THIS tick acquired the lock and completed. A lock-contended
                # tick (another runner reconciling) is NOT this tick's confirmation → skip purge.
                reconcile_confirmed = not reconcile_result.skipped_lock_held
                reconcile_telemetry = {
                    "ran": True,
                    "skipped_lock_held": reconcile_result.skipped_lock_held,
                    "tombstoned": reconcile_result.tombstoned,
                    "restored": reconcile_result.restored,
                }
        except Exception:
            # GitHub unreachable / list raised / config-check raised — logged + skipped.
            # `reconcile_confirmed` stays False so the install hard-delete is skipped this tick; the
            # unrelated sweeps still run below.
            logger.exception("reconcile_tick_failed")
            reconcile_telemetry = {"ran": True, "failed": True}

    # Only confirmed reconcile permits installation deletion. Other sweeps run
    # regardless; their failures propagate after any reconcile writes committed.
    async with engine.connect() as conn, conn.begin():
        sweep_result = await run_all_sweeps(
            conn=conn,
            session_factory=session_factory,
            anomaly_sink=anomaly_sink,
            review_status_sink=review_status_sink,
            audit_persister=audit_persister,
            checkpointer=checkpointer,
            compiled_graph=compiled_graph,
            grace_period=grace_period,
            purge_role=purge_role,
            include_install_purge=reconcile_confirmed,
        )
    sweep_result["reconcile"] = reconcile_telemetry
    return sweep_result


__all__ = ["run_all_sweeps", "run_scheduled_tick"]
