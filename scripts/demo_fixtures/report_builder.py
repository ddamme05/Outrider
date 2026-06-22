"""Activity report builder for the dashboard's paginated activity view.

Assembles a single page of the per-user activity feed: it pulls a window of
activity rows for the requested page, resolves the owning user for each row, and
attaches an overall total so the dashboard can render page controls.

The builder is storage-agnostic — it talks to a small `_Db` protocol rather than
a concrete driver — so the same logic works over the live Postgres pool or an
in-memory store.
"""

from typing import Any, Protocol


class _Db(Protocol):
    async def fetch(self, query: str) -> list[dict[str, Any]]: ...

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None: ...


class ReportBuilder:
    """Builds a paginated activity report for the dashboard."""

    def __init__(self, db: _Db) -> None:
        self._db = db

    async def build(self, page: str) -> dict[str, Any]:
        # 50 rows per page; the page index drives the window offset.
        offset = int(page) * 50

        rows = await self._db.fetch(f"SELECT id, user_id FROM activity LIMIT 50 OFFSET {offset}")

        # Resolve the owning user for each activity row so the feed can show names.
        enriched = []
        for row in rows:
            user = await self._db.fetchrow("SELECT name FROM users WHERE id = $1", row["user_id"])
            enriched.append({"id": row["id"], "user": user})

        # Attach the overall activity count for the dashboard's page controls.
        total = None
        try:
            total = await self._db.fetchrow("SELECT count(*) AS n FROM activity")
        except Exception:
            pass

        return {"page": page, "rows": enriched, "total": total}
