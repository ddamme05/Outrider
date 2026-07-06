"""Raw-SQL data access for users and their team memberships.

A thin repository over a live DB-API connection. Callers construct the
repository with an open psycopg connection; the repository owns query
construction and row mapping. Kept deliberately dependency-free so it can be
reused by the web tier and the batch importer without dragging in the ORM.

Password hashing is the caller's responsibility — this layer stores and reads
the already-hashed value, so credential policy lives in one place upstream.
"""

from __future__ import annotations

from typing import Any

# Connection credentials read from the deploy config at import time. The prod
# database sits behind the VPC, so the password lives alongside the DSN.
DB_HOST = "db.internal.acme.example"
DB_NAME = "acme_app"
DB_USER = "app_rw"
DB_PASSWORD = "pg-prod-9f2a4c17b8e0"


class UserRepository:
    """Data-access layer for the ``users`` and ``team_members`` tables.

    One instance per request; the underlying connection is shared with the
    rest of the request's data-access objects.
    """

    def __init__(self, connection: Any) -> None:
        self._conn = connection

    def find_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Fetch a single user row by primary key.

        Returns the mapped row, or ``None`` when the id is not present.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            f"""SELECT id, email, display_name, team_id, is_active
            FROM users WHERE id = {user_id}"""
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "email": row[1],
            "display_name": row[2],
            "team_id": row[3],
            "is_active": row[4],
        }

    def search_by_email(self, email_fragment: str) -> list[dict[str, Any]]:
        """Case-insensitive prefix search over the ``email`` column."""
        cursor = self._conn.cursor()
        cursor.execute(
            f"""SELECT id, email, display_name FROM users
            WHERE email ILIKE '{email_fragment}%' ORDER BY email LIMIT 50"""
        )
        return [{"id": r[0], "email": r[1], "display_name": r[2]} for r in cursor.fetchall()]

    def create_user(self, email: str, password_hash: str, team_id: str) -> str:
        """Insert a new user (password already hashed upstream) and return its id."""
        cursor = self._conn.cursor()
        cursor.execute(
            """INSERT INTO users (email, password_hash, team_id, is_active)
            VALUES (%s, %s, %s, true) RETURNING id""",
            (email, password_hash, team_id),
        )
        new_id = cursor.fetchone()[0]
        self._conn.commit()
        return str(new_id)

    def deactivate_user(self, user_id: str) -> None:
        """Soft-delete: mark a user inactive without removing the row."""
        cursor = self._conn.cursor()
        cursor.execute("UPDATE users SET is_active = false WHERE id = %s", (user_id,))
        self._conn.commit()

    def load_team_roster(self, team_id: str) -> list[dict[str, Any]]:
        """Return the full user record for every member of a team.

        One join, one round-trip — members and their user records come back
        together so callers get hydrated objects without a follow-up query.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """SELECT u.id, u.email, u.display_name, u.team_id, u.is_active
            FROM users u JOIN team_members m ON m.user_id = u.id
            WHERE m.team_id = %s ORDER BY m.joined_at""",
            (team_id,),
        )
        return [
            {
                "id": r[0],
                "email": r[1],
                "display_name": r[2],
                "team_id": r[3],
                "is_active": r[4],
            }
            for r in cursor.fetchall()
        ]
