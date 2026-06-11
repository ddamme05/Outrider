# Per specs/2026-06-11-file-hash-analyze-cache.md — store + retention contracts.
"""AnalyzeCacheStore against real Postgres: write/lookup round-trip,
ON CONFLICT idempotency, the three no-resurrection layers (lookup-time
expiry, review-purge CASCADE, retention bound at write), and the
reviews-row scope resolution the key's tenant components come from.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.cache import AnalyzeCacheStore, CacheScope

_INSTALLATION_ID = 5151


async def _seed_review(engine, *, retention_days: int = 180):
    """Installation + review row; returns the review id."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, account_type, "
                " permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb) "
                "ON CONFLICT (installation_id) DO NOTHING"
            ),
            {"id": _INSTALLATION_ID},
        )
        result = await conn.execute(
            text(
                "INSERT INTO reviews ("
                "  installation_id, repo_id, pr_number, head_sha, status, "
                "  retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, :sha, 'running', "
                "  NOW() + make_interval(days => :retention_days)"
                ") RETURNING id"
            ),
            {"id": _INSTALLATION_ID, "sha": uuid4().hex, "retention_days": retention_days},
        )
        return result.scalar_one()


def _write_kwargs(cache_key: str, scope: CacheScope, review_id) -> dict:
    return {
        "cache_key": cache_key,
        "scope": scope,
        "source_review_id": review_id,
        "file_path": "src/example.py",
        "payload": {"findings": [{"title": "t"}], "trace_candidates": []},
        "model": "claude-haiku-4-5",
        "prompt_template_version": "analyze-v4",
        "trivial_filter_version": "trivial-filter-v1",
        "query_registry_digest": "a" * 64,
        "active_policy_version": "policy-v1",
        "analyze_parser_version": "analyze-parser-v1",
        "prompt_hash": "b" * 64,
    }


@pytest.mark.asyncio
async def test_resolve_scope_returns_reviews_row_identity(migrated_db: str) -> None:
    """The key's tenant components come from the reviews row — numeric
    installation/repo ids, never PRContext strings."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        scope = await store.resolve_scope(review_id)
        assert scope is not None
        assert scope.installation_id == _INSTALLATION_ID
        assert scope.repo_id == 100
        assert scope.is_eval is False
        assert await store.resolve_scope(uuid4()) is None  # unknown review → None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_write_lookup_roundtrip_and_conflict_noop(migrated_db: str) -> None:
    """First write wins; a concurrent same-key write no-ops; lookup
    returns the live entry with payload intact."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        scope = await store.resolve_scope(review_id)
        assert scope is not None

        key = "c" * 64
        await store.write(**_write_kwargs(key, scope, review_id))
        # Second writer with a DIFFERENT payload must no-op, not overwrite.
        racing = _write_kwargs(key, scope, review_id)
        racing["payload"] = {"findings": [{"title": "RACING"}], "trace_candidates": []}
        await store.write(**racing)

        entry = await store.lookup(key)
        assert entry is not None
        assert entry.payload["findings"][0]["title"] == "t"  # first writer won
        assert entry.source_review_id == review_id
        assert await store.lookup("f" * 64) is None  # unknown key → miss
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_expired_row_is_a_miss(migrated_db: str) -> None:
    """No-resurrection, lookup-time layer: a row past its
    retention_expires_at is a MISS even though it physically exists."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        scope = await store.resolve_scope(review_id)
        assert scope is not None
        key = "d" * 64
        await store.write(**_write_kwargs(key, scope, review_id))
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE analyze_file_cache "
                    "SET retention_expires_at = NOW() - INTERVAL '1 second' "
                    "WHERE cache_key = :key"
                ),
                {"key": key},
            )
            exists = await conn.execute(
                text("SELECT count(*) FROM analyze_file_cache WHERE cache_key = :key"),
                {"key": key},
            )
            assert exists.scalar_one() == 1  # physically present
        assert await store.lookup(key) is None  # but never served
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_review_purge_cascades_cache_rows(migrated_db: str) -> None:
    """No-resurrection, CASCADE layer: deleting the source review takes
    its cache rows with it."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        scope = await store.resolve_scope(review_id)
        assert scope is not None
        key = "e" * 64
        await store.write(**_write_kwargs(key, scope, review_id))
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM reviews WHERE id = :id"), {"id": review_id})
        assert await store.lookup(key) is None
        async with engine.begin() as conn:
            remaining = await conn.execute(
                text("SELECT count(*) FROM analyze_file_cache WHERE cache_key = :key"),
                {"key": key},
            )
            assert remaining.scalar_one() == 0  # row physically gone
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_retention_bound_never_outlives_source_review(migrated_db: str) -> None:
    """Write-time layer: with a source review whose retention is SHORTER
    than the cache TTL, the cache row inherits the shorter bound."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine, retention_days=3)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        scope = await store.resolve_scope(review_id)
        assert scope is not None
        key = "9" * 64
        await store.write(**_write_kwargs(key, scope, review_id))
        async with engine.begin() as conn:
            row = await conn.execute(
                text(
                    "SELECT retention_expires_at <= NOW() + INTERVAL '4 days' AS bounded "
                    "FROM analyze_file_cache WHERE cache_key = :key"
                ),
                {"key": key},
            )
            assert row.scalar_one() is True  # min() took the review's 3 days, not TTL 30
    finally:
        await engine.dispose()
