# Per specs/2026-06-11-file-hash-analyze-cache.md — store + retention contracts.
"""AnalyzeCacheStore against real Postgres: write/lookup round-trip,
conflict semantics (live row wins; an EXPIRED row is refreshed in
place), the three no-resurrection layers (lookup-time expiry on the DB
clock, review-purge CASCADE, retention bound at write), the self-hit
exclusion (a review never reads its own writes as hits), and the
reviews-row scope resolution the key's tenant components come from.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.cache import AnalyzeCacheStore, CacheScope

_INSTALLATION_ID = 5151


async def _seed_review(engine, *, retention_days: int = 180, is_eval: bool = False):
    """Installation + review row; returns the review id. `is_eval` stamps the
    reviews row so `resolve_scope` yields the matching cache partition."""
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
                "  installation_id, repo_id, pr_number, head_sha, status, is_eval, "
                "  retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, :sha, 'running', :is_eval, "
                "  NOW() + make_interval(days => :retention_days)"
                ") RETURNING id"
            ),
            {
                "id": _INSTALLATION_ID,
                "sha": uuid4().hex,
                "retention_days": retention_days,
                "is_eval": is_eval,
            },
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

        entry = await store.lookup(key, is_eval=False)
        assert entry is not None
        assert entry.payload["findings"][0]["title"] == "t"  # first writer won
        assert entry.source_review_id == review_id
        assert await store.lookup("f" * 64, is_eval=False) is None  # unknown key → miss
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
        assert await store.lookup(key, is_eval=False) is None  # but never served
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_lookup_excludes_own_review_writes(migrated_db: str) -> None:
    """Self-hit exclusion: a lookup that names its own review as the
    excluded source gets a MISS on rows that review wrote — a
    crash-resume re-run must not count its own first attempt as a hit —
    while OTHER reviews still see the row as live."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        scope = await store.resolve_scope(review_id)
        assert scope is not None
        key = "1" * 64
        await store.write(**_write_kwargs(key, scope, review_id))

        assert await store.lookup(key, is_eval=False, exclude_source_review_id=review_id) is None
        other = await store.lookup(key, is_eval=False, exclude_source_review_id=uuid4())
        assert other is not None and other.source_review_id == review_id
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_expired_row_is_refreshed_by_a_new_write(migrated_db: str) -> None:
    """Conflict semantics, expired arm: an expired-but-unswept row does
    NOT block re-population — a new same-key write refreshes it in
    place (payload, source review, retention), where a plain DO NOTHING
    would brick the key until the sweep's next tick."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        scope = await store.resolve_scope(review_id)
        assert scope is not None
        key = "2" * 64
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
        assert await store.lookup(key, is_eval=False) is None  # expired = MISS

        fresh_review_id = await _seed_review(engine)
        fresh = _write_kwargs(key, scope, fresh_review_id)
        fresh["payload"] = {"findings": [{"title": "REFRESHED"}], "trace_candidates": []}
        await store.write(**fresh)

        entry = await store.lookup(key, is_eval=False)
        assert entry is not None  # the key is live again
        assert entry.payload["findings"][0]["title"] == "REFRESHED"
        assert entry.source_review_id == fresh_review_id
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
        assert await store.lookup(key, is_eval=False) is None
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


@pytest.mark.asyncio
async def test_lookup_is_eval_predicate_isolates_reads(migrated_db: str) -> None:
    """Read isolation (DECISIONS.md#046): the lookup's required is_eval predicate
    keeps eval reads from seeing production rows. A prod (is_eval=False) row is
    found by an is_eval=False lookup but is INVISIBLE to an is_eval=True lookup —
    the only thing stopping an eval review in a shared DB from reading a production
    row, since the cache_key folds installation/repo but NOT is_eval. The same key
    holds only one partition's row (the write arbiter is ON CONFLICT(cache_key)),
    so this is read isolation, not population coexistence."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        scope = await store.resolve_scope(review_id)
        assert scope is not None and scope.is_eval is False
        key = "a1" * 32  # 64 hex chars
        await store.write(**_write_kwargs(key, scope, review_id))

        assert await store.lookup(key, is_eval=False) is not None  # prod sees prod row
        assert await store.lookup(key, is_eval=True) is None  # eval cannot read it
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_cross_partition_write_arbiter(migrated_db: str) -> None:
    """Write-arbiter cross-scope behavior (DECISIONS.md#046 — the load-bearing half
    of read isolation, distinct from the read predicate above). The conflict key is
    cache_key alone (is_eval is a plain column), so eval/prod rows with the same key
    cannot co-exist: a LIVE prod row BLOCKS an eval write of that key (eval misses,
    never reads prod), while an EXPIRED prod row is REFRESHED in place by an eval
    write, flipping is_eval to the writer's partition (the `is_eval` set_ clause in
    the ON CONFLICT DO UPDATE) so the eval review then hits its own refresh."""
    engine = create_async_engine(migrated_db)
    try:
        prod_review = await _seed_review(engine)
        store = AnalyzeCacheStore(async_sessionmaker(engine, expire_on_commit=False))
        prod_scope = await store.resolve_scope(prod_review)
        assert prod_scope is not None and prod_scope.is_eval is False
        # The eval scope comes from a REAL is_eval=True review (same tenant), matching
        # the production shape where scope + source_review_id share one reviews row.
        eval_review = await _seed_review(engine, is_eval=True)
        eval_scope = await store.resolve_scope(eval_review)
        assert eval_scope is not None and eval_scope.is_eval is True
        key = "b2" * 32  # 64 hex chars

        # A LIVE prod row blocks an eval write of the same key (first-writer-wins).
        await store.write(**_write_kwargs(key, prod_scope, prod_review))
        await store.write(**_write_kwargs(key, eval_scope, eval_review))  # no-op: prod row live
        assert await store.lookup(key, is_eval=True) is None  # eval can't read the live prod row
        assert await store.lookup(key, is_eval=False) is not None  # prod row intact

        # Expire the prod row; an eval write now REFRESHES it in place, flipping is_eval.
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE analyze_file_cache "
                    "SET retention_expires_at = NOW() - INTERVAL '1 second' "
                    "WHERE cache_key = :key"
                ),
                {"key": key},
            )
        await store.write(**_write_kwargs(key, eval_scope, eval_review))
        assert await store.lookup(key, is_eval=True) is not None  # eval hits its own refresh
        assert await store.lookup(key, is_eval=False) is None  # prod can no longer see it (flipped)
    finally:
        await engine.dispose()
