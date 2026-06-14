# Per specs/2026-06-11-file-hash-analyze-cache.md — the cache store.
"""DB-backed store for the file-hash analyze cache.

Three operations, all async (I/O path):

- `resolve_scope(review_id)` — the canonical tenant identity
  `(installation_id, repo_id, is_eval, retention_expires_at)` from the
  `reviews` row. The key's scope components come from HERE, never from
  `PRContext`'s mutable `owner`/`repo` strings.
- `lookup(cache_key, *, is_eval)` — entry or None, scoped to the
  caller's `is_eval` partition (READ isolation, `DECISIONS.md#046`: the
  key folds installation/repo but NOT is_eval, so an eval review must
  never read a production row). An expired row
  (`retention_expires_at <= now()` on the DATABASE clock — the same
  clock the retention sweep deletes with, so the lookup-time and
  sweep-time layers of the no-resurrection rule can never disagree
  under app-host skew) is a MISS by query shape. The optional
  `exclude_source_review_id` filters out rows the current review wrote
  itself — a crash/retry re-execution of analyze must not read its own
  first attempt's writes as hits.
- `write(...)` — `INSERT ON CONFLICT (cache_key) DO UPDATE ... WHERE`
  the existing row is expired: concurrent same-key reviews race
  benignly (first writer wins while the row is LIVE; later writers
  no-op), but an expired-but-unswept row is refreshed in place — a
  plain DO NOTHING would brick the key until the sweep physically
  deletes the stale row. `retention_expires_at = min(now + CACHE_TTL,
  source review retention)` so a cache row never outlives its source.

DB failures raise `CacheStoreError` (typed, per `docs/conventions.md`),
NOT raw SQLAlchemy errors: the analyze node contains that one type to
keep the shadow cache non-fatal without blind `except Exception`.

The store is injected at `build_graph(...)` and closed over
(`nodes-receive-deps-via-closure`); `None` in its place disables the
cache entirely — the eval driver's default for scenarios that don't
exercise it. Eval reviews that DO wire a store use the cache scoped to
is_eval rows (the `lookup` predicate, `DECISIONS.md#046`), not a bypass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from outrider.db.models import AnalyzeFileCache

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Cache TTL per the spec (~30 days); always bounded above by the source
# review's retention_expires_at at write time.
CACHE_TTL_DAYS: Final = 30


class CacheStoreError(RuntimeError):
    """A cache-store DB operation failed (connection, schema, FK, ...).

    The shadow cache is optional infrastructure: callers contain this
    error and degrade to UNCACHED — it must never abort a review.
    """


@dataclass(frozen=True, slots=True)
class CacheScope:
    """The reviews-row identity the key's scope components come from."""

    installation_id: int
    repo_id: int
    is_eval: bool
    retention_expires_at: datetime


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A live (non-expired) cache row, as the serve path will consume it."""

    cache_key: str
    payload: dict[str, Any]
    source_review_id: UUID
    file_path: str
    created_at: datetime


class AnalyzeCacheStore:
    """Async store over `analyze_file_cache` (session-per-call discipline)."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def resolve_scope(self, review_id: UUID) -> CacheScope | None:
        """The canonical `(installation_id, repo_id, is_eval, retention)`
        for `review_id`; None if the review row doesn't exist (caller
        treats as cache-disabled for this review — fail-open to
        UNCACHED, never to cross-scope). DB errors raise
        `CacheStoreError`."""
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    text(
                        "SELECT installation_id, repo_id, is_eval, retention_expires_at "
                        "FROM reviews WHERE id = :review_id"
                    ),
                    {"review_id": review_id},
                )
                row = result.one_or_none()
        except SQLAlchemyError as e:
            raise CacheStoreError(f"analyze-cache resolve_scope failed: {e}") from e
        if row is None:
            return None
        return CacheScope(
            installation_id=row.installation_id,
            repo_id=row.repo_id,
            is_eval=row.is_eval,
            retention_expires_at=row.retention_expires_at,
        )

    async def lookup(
        self,
        cache_key: str,
        *,
        is_eval: bool,
        exclude_source_review_id: UUID | None = None,
    ) -> CacheEntry | None:
        """Live entry or None, scoped to the caller's `is_eval` partition. Expired
        rows are a MISS by query shape — the row may physically exist until the
        sweep prunes it, but it is never served (no-resurrection, lookup-time
        layer; expiry is evaluated on the DB clock, matching the sweep). Rows
        written by `exclude_source_review_id` are also a MISS: a review re-run must
        not count its own prior writes as hits. DB errors raise `CacheStoreError`.

        `is_eval` is a REQUIRED read-isolation predicate (DECISIONS.md#046): the
        `cache_key` folds (installation_id, repo_id) but NOT is_eval, so without
        this filter an eval review in a shared prod+eval DB could read a production
        row. The caller passes the scope's is_eval (the reviews-row value the write
        stamped), so reads and writes share one partition. This is READ isolation
        only — the write arbiter is still `ON CONFLICT (cache_key)`, so eval/prod
        rows with the same key cannot co-exist; in a shared DB a live prod row
        blocks an eval write of that key (the eval review then misses until expiry,
        and never reads the prod content)."""
        conditions = [
            AnalyzeFileCache.cache_key == cache_key,
            AnalyzeFileCache.is_eval == is_eval,
            AnalyzeFileCache.retention_expires_at > func.now(),
        ]
        if exclude_source_review_id is not None:
            conditions.append(AnalyzeFileCache.source_review_id != exclude_source_review_id)
        try:
            async with self._session_factory() as session:
                result = await session.execute(select(AnalyzeFileCache).where(*conditions))
                row = result.scalar_one_or_none()
        except SQLAlchemyError as e:
            raise CacheStoreError(f"analyze-cache lookup failed: {e}") from e
        if row is None:
            return None
        return CacheEntry(
            cache_key=row.cache_key,
            payload=row.payload,
            source_review_id=row.source_review_id,
            file_path=row.file_path,
            created_at=row.created_at,
        )

    async def write(
        self,
        *,
        cache_key: str,
        scope: CacheScope,
        source_review_id: UUID,
        file_path: str,
        payload: dict[str, Any],
        model: str,
        prompt_template_version: str,
        trivial_filter_version: str,
        query_registry_digest: str,
        active_policy_version: str,
        analyze_parser_version: str,
        prompt_hash: str,
    ) -> None:
        """Insert, or refresh an EXPIRED row in place; live rows are
        untouched (first writer wins). The retention bound is computed
        here so no caller can write a row that outlives its source.
        DB errors raise `CacheStoreError`."""
        expires = min(
            datetime.now(UTC) + timedelta(days=CACHE_TTL_DAYS),
            scope.retention_expires_at,
        )
        statement = pg_insert(AnalyzeFileCache).values(
            cache_key=cache_key,
            installation_id=scope.installation_id,
            repo_id=scope.repo_id,
            source_review_id=source_review_id,
            file_path=file_path,
            payload=payload,
            model=model,
            prompt_template_version=prompt_template_version,
            trivial_filter_version=trivial_filter_version,
            query_registry_digest=query_registry_digest,
            active_policy_version=active_policy_version,
            analyze_parser_version=analyze_parser_version,
            prompt_hash=prompt_hash,
            is_eval=scope.is_eval,
            retention_expires_at=expires,
        )
        # Conflict policy: refresh ONLY when the existing row is past its
        # retention bound (DB clock — same predicate shape as lookup and
        # the sweep). A live row wins over every later writer; an expired
        # row would otherwise block re-population for the whole
        # expiry-to-sweep window because lookup treats it as a MISS while
        # its physical PK still swallows inserts.
        statement = statement.on_conflict_do_update(
            index_elements=["cache_key"],
            set_={
                "installation_id": statement.excluded.installation_id,
                "repo_id": statement.excluded.repo_id,
                "source_review_id": statement.excluded.source_review_id,
                "file_path": statement.excluded.file_path,
                "payload": statement.excluded.payload,
                "model": statement.excluded.model,
                "prompt_template_version": statement.excluded.prompt_template_version,
                "trivial_filter_version": statement.excluded.trivial_filter_version,
                "query_registry_digest": statement.excluded.query_registry_digest,
                "active_policy_version": statement.excluded.active_policy_version,
                "analyze_parser_version": statement.excluded.analyze_parser_version,
                "prompt_hash": statement.excluded.prompt_hash,
                "is_eval": statement.excluded.is_eval,
                "retention_expires_at": statement.excluded.retention_expires_at,
                "created_at": func.now(),
            },
            where=(AnalyzeFileCache.retention_expires_at <= func.now()),
        )
        try:
            async with self._session_factory() as session:
                await session.execute(statement)
                await session.commit()
        except SQLAlchemyError as e:
            raise CacheStoreError(f"analyze-cache write failed: {e}") from e
