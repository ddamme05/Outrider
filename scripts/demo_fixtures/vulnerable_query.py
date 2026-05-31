"""Demo fixture for the live-Claude smoke (`--diff-file`): the CRITICAL path.

An async user-record repository whose lookup builds SQL by f-string-interpolating
a request-supplied id straight into the query text — a textbook SQL injection.
It maps to FindingType.SQL_INJECTION -> CRITICAL, which trips the HITL gate: the
graph calls interrupt() at the hitl node and the review parks in AWAITING_APPROVAL
instead of auto-publishing.

Counterpart to api_request_handler.py (the MEDIUM, auto-publish fixture). This
one exists to demo the human-in-the-loop gate end to end: same plumbing, but the
finding's severity forces a human decision before anything reaches GitHub.

The diff is substantial and unambiguously security-relevant so triage tiers it
DEEP and selects the security dimension. Deliberately a single, clean CRITICAL —
no secrets, auth, or traversal mixed in — so the finding the demo surfaces is
exactly one sql_injection.
"""

from typing import Any, Protocol


class _Connection(Protocol):
    """Minimal async DB connection surface this repository depends on."""

    async def fetchrow(self, query: str) -> dict[str, Any] | None: ...

    async def fetch(self, query: str) -> list[dict[str, Any]]: ...


class UserRepository:
    """Reads user records from the backing store.

    The connection is injected so the repository stays unit-testable; in
    production it is an asyncpg connection acquired from the pool.
    """

    def __init__(self, conn: _Connection) -> None:
        self._conn = conn

    async def get_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Look up a single user by id.

        `user_id` arrives from the request path (e.g. GET /users/{user_id})
        and is interpolated directly into the SQL statement.
        """
        query = f"SELECT id, email, role FROM users WHERE id = {user_id}"  # noqa: S608  (intentional: demo SQL-injection fixture)
        return await self._conn.fetchrow(query)

    async def search_by_email(self, email_fragment: str) -> list[dict[str, Any]]:
        """Find users whose email contains the given fragment.

        `email_fragment` is a raw query-string value spliced into a LIKE clause.
        """
        query = (
            "SELECT id, email, role FROM users "  # noqa: S608  (intentional: demo SQL-injection fixture)
            f"WHERE email LIKE '%{email_fragment}%' ORDER BY email"
        )
        return await self._conn.fetch(query)
