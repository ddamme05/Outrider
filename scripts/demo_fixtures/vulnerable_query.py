"""Async user-record repository backed by the primary users table.

Provides read access to user records for the request-handling layer: a
single-record lookup by id and a substring search over email addresses. The
connection is injected so callers can share a pooled asyncpg connection across
a request lifecycle.
"""

from typing import Any, Protocol


class _Connection(Protocol):
    """Minimal async DB connection surface this repository depends on."""

    async def fetchrow(self, query: str) -> dict[str, Any] | None: ...

    async def fetch(self, query: str) -> list[dict[str, Any]]: ...


class UserRepository:
    """Reads user records from the backing store.

    The connection is injected so callers supply a pooled asyncpg connection
    acquired from the request scope.
    """

    def __init__(self, conn: _Connection) -> None:
        self._conn = conn

    async def get_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Look up a single user by id.

        `user_id` arrives from the request path (e.g. GET /users/{user_id})
        and is interpolated directly into the SQL statement.
        """
        query = f"SELECT id, email, role FROM users WHERE id = {user_id}"
        return await self._conn.fetchrow(query)

    async def search_by_email(self, email_fragment: str) -> list[dict[str, Any]]:
        """Find users whose email contains the given fragment.

        `email_fragment` is a raw query-string value spliced into a LIKE clause.
        """
        query = (
            "SELECT id, email, role FROM users "
            f"WHERE email LIKE '%{email_fragment}%' ORDER BY email"
        )
        return await self._conn.fetch(query)
