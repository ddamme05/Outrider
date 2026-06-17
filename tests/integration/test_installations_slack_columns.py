"""Per-install Slack config columns on `installations` (commit 6.2).

After `alembic upgrade head`, the five nullable Slack columns exist and round-trip
— in particular the bytea bot-token ciphertext (DECISIONS.md#051) byte-for-byte.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_installations_slack_columns_round_trip(migrated_db: str) -> None:
    ciphertext = b"gAAAAA-fake-fernet-ciphertext-bytes-\x00\x01\x02"
    engine = create_async_engine(migrated_db)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO installations "
                    "(installation_id, app_slug, account_id, account_login, account_type, "
                    " permissions_at_install, slack_team_id, slack_bot_token_ciphertext, "
                    " slack_channel_id, slack_configured_at, slack_configured_by) "
                    "VALUES (:iid, 'outrider', 1, 'acme', 'Organization', '{}'::jsonb, "
                    " :team, :ct, :chan, NOW(), 'admin')"
                ),
                {"iid": 999001, "team": "T0ABCDE", "ct": ciphertext, "chan": "C0FGHIJ"},
            )
        async with engine.connect() as conn:
            row = await conn.execute(
                text(
                    "SELECT slack_team_id, slack_bot_token_ciphertext, slack_channel_id, "
                    "slack_configured_by FROM installations WHERE installation_id = :iid"
                ),
                {"iid": 999001},
            )
            team, stored_ct, chan, configured_by = row.one()
        assert team == "T0ABCDE"
        assert bytes(stored_ct) == ciphertext  # bytea round-trips byte-for-byte
        assert chan == "C0FGHIJ"
        assert configured_by == "admin"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_installations_slack_columns_default_null(migrated_db: str) -> None:
    """An install that never connects Slack leaves the columns NULL (opt-in)."""
    engine = create_async_engine(migrated_db)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO installations "
                    "(installation_id, app_slug, account_id, account_login, account_type, "
                    " permissions_at_install) "
                    "VALUES (:iid, 'outrider', 2, 'beta', 'User', '{}'::jsonb)"
                ),
                {"iid": 999002},
            )
        async with engine.connect() as conn:
            row = await conn.execute(
                text(
                    "SELECT slack_team_id, slack_bot_token_ciphertext, slack_channel_id, "
                    "slack_configured_at, slack_configured_by "
                    "FROM installations WHERE installation_id = :iid"
                ),
                {"iid": 999002},
            )
            assert all(v is None for v in row.one())
    finally:
        await engine.dispose()
