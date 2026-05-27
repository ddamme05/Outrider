"""AnomalyPersister duplicate-emit idempotency — DB-touching contract pin.

Per CR-1 (CodeRabbit, F8 round): `AnomalyPersister.emit_anomaly` must
collapse duplicate `(review_id, rule_name='hitl_timeout')` emissions to
exactly one row via the partial unique index
`uq_anomalies_hitl_timeout_natural_key`. Without explicit
`index_elements=["review_id"]` + `index_where=(Anomaly.rule_name ==
"hitl_timeout")` on `on_conflict_do_nothing()`, PostgreSQL's
conflict-arbiter inference is unreliable when a `WHERE`-clause index
is involved — silent INSERT-duplicate is the failure mode this test
exists to catch.

Tier: integration (real Postgres, real migration, real on-conflict).
Companion to the unit-level `test_anomaly_sink.py` which exercises the
in-memory recording double; this file exercises the durable persister
against the partial unique index from `33f8fe051bec_hitl_node_indexes.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.anomaly.persister import AnomalyPersister
from outrider.anomaly.rule_names import AnomalyRuleName

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
async def anomaly_persister(
    migrated_db: str,
) -> AsyncGenerator[tuple[AnomalyPersister, UUID]]:
    """Build an AnomalyPersister against a freshly-migrated DB + seed a
    review row so the FK constraint is satisfiable."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO installations "
                    "(installation_id, app_slug, account_id, account_login, "
                    " account_type, permissions_at_install) "
                    "VALUES (777, 'test-app', 1, 'octocat', 'User', '{}'::jsonb)"
                )
            )
            result = await conn.execute(
                text(
                    "INSERT INTO reviews ("
                    "  installation_id, repo_id, pr_number, head_sha, status, "
                    "  files_examined, files_traced_beyond_diff, llm_calls_made, "
                    "  total_input_tokens, total_output_tokens, total_cost_usd, "
                    "  wall_clock_seconds, retention_expires_at"
                    ") VALUES ("
                    "  777, 100, 1, 'sha1', 'awaiting_approval', 0, 0, 0, 0, 0, 0, 0, "
                    "  NOW() + INTERVAL '90 days'"
                    ") RETURNING id"
                )
            )
            review_id = UUID(str(result.scalar_one()))
        persister = AnomalyPersister(
            session_factory=async_sessionmaker(engine, expire_on_commit=False),
        )
        yield persister, review_id
    finally:
        await engine.dispose()


async def _count_anomaly_rows(migrated_db: str, review_id: UUID, rule_name: str) -> int:
    """Direct SQL count — bypasses the persister to verify on-conflict
    behavior at the schema layer."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT COUNT(*) FROM anomalies WHERE review_id = :rid AND rule_name = :rn"),
                {"rid": str(review_id), "rn": rule_name},
            )
            return int(result.scalar_one())
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_duplicate_emit_collapses_to_one_row(
    anomaly_persister: tuple[AnomalyPersister, UUID],
    migrated_db: str,
) -> None:
    """Two `emit_anomaly` calls with the SAME `(review_id, HITL_TIMEOUT)`
    pair land EXACTLY ONE row in `anomalies`.

    The partial unique index + the explicit `index_elements` +
    `index_where` on `on_conflict_do_nothing` make the second emit a
    no-op. Without explicit targeting (CR-1's gap), Postgres'
    conflict-arbiter inference could miss the partial index and the
    second INSERT would land — breaking the sweep's anomaly-first
    ordering idempotency claim.
    """
    persister, review_id = anomaly_persister

    await persister.emit_anomaly(
        review_id=review_id,
        rule_name=AnomalyRuleName.HITL_TIMEOUT,
        severity="medium",
        details={"expired_at": "2026-05-26T00:00:00Z"},
    )
    await persister.emit_anomaly(
        review_id=review_id,
        rule_name=AnomalyRuleName.HITL_TIMEOUT,
        severity="medium",
        details={"expired_at": "2026-05-26T00:00:00Z"},
    )

    count = await _count_anomaly_rows(migrated_db, review_id, "hitl_timeout")
    assert count == 1, (
        f"Expected exactly 1 anomaly row after duplicate emit; got {count}. "
        f"Suggests on_conflict_do_nothing failed to target the partial "
        f"unique index `uq_anomalies_hitl_timeout_natural_key` and the "
        f"second INSERT landed as a duplicate."
    )


@pytest.mark.asyncio
async def test_distinct_review_ids_admit_separate_rows(
    migrated_db: str,
) -> None:
    """Distinct `review_id` values produce distinct anomaly rows — the
    partial index is per-`(review_id)`, not global. Verifies the
    `index_elements=["review_id"]` arbiter doesn't over-deduplicate
    across reviews."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO installations "
                    "(installation_id, app_slug, account_id, account_login, "
                    " account_type, permissions_at_install) "
                    "VALUES (888, 'test-app', 1, 'octocat', 'User', '{}'::jsonb)"
                )
            )
            review_ids: list[UUID] = []
            for pr_num in (1, 2):
                result = await conn.execute(
                    text(
                        "INSERT INTO reviews ("
                        "  installation_id, repo_id, pr_number, head_sha, status, "
                        "  files_examined, files_traced_beyond_diff, llm_calls_made, "
                        "  total_input_tokens, total_output_tokens, total_cost_usd, "
                        "  wall_clock_seconds, retention_expires_at"
                        ") VALUES ("
                        "  888, 100, :pr, 'sha1', 'awaiting_approval', 0, 0, 0, 0, 0, 0, 0, "
                        "  NOW() + INTERVAL '90 days'"
                        ") RETURNING id"
                    ),
                    {"pr": pr_num},
                )
                review_ids.append(UUID(str(result.scalar_one())))

        persister = AnomalyPersister(
            session_factory=async_sessionmaker(engine, expire_on_commit=False),
        )
        for rid in review_ids:
            await persister.emit_anomaly(
                review_id=rid,
                rule_name=AnomalyRuleName.HITL_TIMEOUT,
                severity="medium",
                details={"expired_at": "2026-05-26T00:00:00Z"},
            )

        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    "SELECT COUNT(*) FROM anomalies WHERE rule_name = 'hitl_timeout' "
                    "AND review_id IN (:r1, :r2)"
                ),
                {"r1": str(review_ids[0]), "r2": str(review_ids[1])},
            )
            count = int(result.scalar_one())
        assert count == 2, f"Expected 2 anomaly rows (one per distinct review_id); got {count}."
    finally:
        await engine.dispose()


# Forward-compat: a future addition to `AnomalyRuleName` (e.g.,
# `MISSING_TRACE_PATH = "missing_trace_path"`) reaches the persister
# WITHOUT a dispatch update for its own partial unique index. V1 guards
# against this at the Python layer (`raise NotImplementedError`).
# That contract is exercised at the unit tier in `tests/unit/test_anomaly_sink.py`
# because StrEnum construction at the integration tier requires bypassing
# Python's identity guarantees in ways that don't reflect real-world
# extensibility (a real new enum value would be a one-line addition to
# `rule_names.py`, not a hot-patched instance). Integration coverage
# here focuses on the on-conflict idempotency contract that depends on
# real Postgres + the partial unique index.
