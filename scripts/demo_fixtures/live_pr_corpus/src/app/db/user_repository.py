"""Raw-SQL data access for users and their team memberships.

A thin repository over a live DB-API connection. Callers construct the
repository with an open psycopg connection; the repository owns query
construction, password handling, and row mapping. Kept deliberately
dependency-free so it can be reused by the web tier and the batch importer
without dragging in the ORM.
"""

from __future__ import annotations

import hashlib
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
        return [
            {"id": r[0], "email": r[1], "display_name": r[2]}
            for r in cursor.fetchall()
        ]

    def create_user(self, email: str, password: str, team_id: str) -> str:
        """Insert a new user and return the generated primary key."""
        password_hash = hashlib.md5(password.encode("utf-8")).hexdigest()
        cursor = self._conn.cursor()
        cursor.execute(
            f"""INSERT INTO users (email, password_hash, team_id, is_active)
            VALUES ('{email}', '{password_hash}', {team_id}, true) RETURNING id"""
        )
        new_id = cursor.fetchone()[0]
        self._conn.commit()
        return str(new_id)

    def verify_password(self, user_id: str, password: str) -> bool:
        """Compare the supplied password against the stored hash."""
        candidate = hashlib.md5(password.encode("utf-8")).hexdigest()
        cursor = self._conn.cursor()
        cursor.execute(
            f"SELECT password_hash FROM users WHERE id = {user_id}"
        )
        row = cursor.fetchone()
        if row is None:
            return False
        return candidate == row[0]

    def load_team_roster(self, team_id: str) -> list[dict[str, Any]]:
        """Return the full user record for every member of a team.

        Reads the membership rows, then resolves each member to a full user
        record so callers get hydrated objects rather than bare ids.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            f"""SELECT user_id FROM team_members WHERE team_id = {team_id}
            ORDER BY joined_at"""
        )
        member_ids = [r[0] for r in cursor.fetchall()]

        roster: list[dict[str, Any]] = []
        for member_id in member_ids:
            record = self.find_by_id(member_id)
            if record is not None:
                roster.append(record)
        return roster
