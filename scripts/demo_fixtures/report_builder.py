"""Demo fixture for the live-Claude smoke (`--diff-file`): the BREADTH path.

A report endpoint with several deliberate, unambiguous issues spread across review
DIMENSIONS, all sub-HIGH so the review auto-publishes (no HITL gate) and the demo
shows a single review surfacing findings in different categories at once:

  1. `page` taken off the request and used unvalidated → `missing_input_validation`
     (MEDIUM, security)
  2. a per-row query inside the loop → `n_plus_one_query` (MEDIUM, performance)
  3. a bare `except: pass` swallowing errors → `missing_error_handling`
     (LOW, code quality)

These are JUDGED (model-identified) findings, so the exact set/severity is the
model's call at run time — the seed-capture check confirms what actually landed.
Deliberately NO secrets / SQLi / weak-crypto / auth here — those map to HIGH/CRITICAL
and would trip the HITL gate (demoed by the other fixtures). This file is demo
input, not production code: it is intentionally flawed.
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
        # `page` comes straight off the query string with no int coercion guard,
        # no bounds check, and no allowlist — a caller controls the offset math.
        offset = int(page) * 50

        rows = await self._db.fetch(
            f"SELECT id, user_id FROM activity LIMIT 50 OFFSET {offset}"  # noqa: S608  (intentional: demo missing-input-validation fixture)
        )

        # N+1: one extra round-trip PER row to resolve the user. Should be a single
        # JOIN or a batched IN (...) lookup instead of a query inside the loop.
        enriched = []
        for row in rows:
            user = await self._db.fetchrow("SELECT name FROM users WHERE id = $1", row["user_id"])
            enriched.append({"id": row["id"], "user": user})

        total = None
        try:  # noqa: SIM105  (intentional: demo missing-error-handling fixture)
            total = await self._db.fetchrow("SELECT count(*) AS n FROM activity")
        except Exception:  # noqa: BLE001, S110  (intentional: demo missing-error-handling fixture)
            # Bare swallow: a DB error here silently reports total=None with no log.
            pass

        return {"page": page, "rows": enriched, "total": total}
