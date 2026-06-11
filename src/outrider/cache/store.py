# Per specs/2026-06-11-file-hash-analyze-cache.md — the cache store.
"""DB-backed store for the file-hash analyze cache.

Three operations, all async (I/O path):

- `resolve_scope(review_id)` — the canonical tenant identity
  `(installation_id, repo_id, is_eval, retention_expires_at)` from the
  `reviews` row. The key's scope components come from HERE, never from
  `PRContext`'s mutable `owner`/`repo` strings.
- `lookup(cache_key)` — entry or None. An expired row
  (`retention_expires_at <= NOW()`) is a MISS by query shape: the
  lookup-time layer of the no-resurrection rule (the other two layers
  are the `source_review_id` CASCADE and the retention sweep).
- `write(...)` — `INSERT ON CONFLICT (cache_key) DO NOTHING`: concurrent
  same-key reviews race benignly; first writer wins, every later writer
  no-ops. `retention_expires_at = min(now + CACHE_TTL, source review
  retention)` so a cache row never outlives its source content.

The store is injected at `build_graph(...)` and closed over
(`nodes-receive-deps-via-closure`); `None` in its place disables the
cache entirely — the eval driver's default, per the spec's
eval-bypass rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from outrider.db.models import AnalyzeFileCache

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Cache TTL per the spec (~30 days); always bounded above by the source
# review's retention_expires_at at write time.
CACHE_TTL_DAYS: Final = 30


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
        UNCACHED, never to cross-scope)."""
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT installation_id, repo_id, is_eval, retention_expires_at "
                    "FROM reviews WHERE id = :review_id"
                ),
                {"review_id": review_id},
            )
            row = result.one_or_none()
            if row is None:
                return None
            return CacheScope(
                installation_id=row.installation_id,
                repo_id=row.repo_id,
                is_eval=row.is_eval,
                retention_expires_at=row.retention_expires_at,
            )

    async def lookup(self, cache_key: str) -> CacheEntry | None:
        """Live entry or None. Expired rows are a MISS by query shape —
        the row may physically exist until the sweep prunes it, but it
        is never served (no-resurrection, lookup-time layer)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(AnalyzeFileCache).where(
                    AnalyzeFileCache.cache_key == cache_key,
                    AnalyzeFileCache.retention_expires_at > datetime.now(UTC),
                )
            )
            row = result.scalar_one_or_none()
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
        """Insert-or-noop on `cache_key`. The retention bound is computed
        here so no caller can write a row that outlives its source."""
        expires = min(
            datetime.now(UTC) + timedelta(days=CACHE_TTL_DAYS),
            scope.retention_expires_at,
        )
        statement = (
            pg_insert(AnalyzeFileCache)
            .values(
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
            .on_conflict_do_nothing(index_elements=["cache_key"])
        )
        async with self._session_factory() as session:
            await session.execute(statement)
            await session.commit()
