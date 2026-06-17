"""set_slack_config + get_slack_config persistence helpers (commits 6.3d / 6.4c).

After `alembic upgrade head`: set_slack_config persists the five `slack_*` columns on
an ACTIVE install (returns True, ciphertext round-trips byte-for-byte) and refuses a
tombstoned or absent install (returns False). get_slack_config reads back the channel +
ciphertext for an ACTIVE install, returning None when unconfigured, tombstoned (#012), or
absent. The bot token is stored/returned as opaque ciphertext bytes (DECISIONS.md#051) —
neither helper sees plaintext.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, async_sessionmaker, create_async_engine

from outrider.db.models.installations import get_slack_config, set_slack_config


async def _insert_install(conn: AsyncConnection, iid: int, *, tombstoned: bool = False) -> None:
    # tombstoned_at bound as a parameter (None -> NULL, aware datetime -> timestamptz);
    # no SQL interpolation.
    await conn.execute(
        text(
            "INSERT INTO installations (installation_id, app_slug, account_id, "
            "account_login, account_type, permissions_at_install, tombstoned_at) "
            "VALUES (:iid, 'outrider', 1, 'acme', 'Organization', '{}'::jsonb, :tomb)"
        ),
        {"iid": iid, "tomb": datetime.now(UTC) if tombstoned else None},
    )


@pytest.mark.asyncio
async def test_set_slack_config_active_install(migrated_db: str) -> None:
    ciphertext = b"gAAAAA-fake-fernet-ciphertext-\x00\x01\x02"
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await _insert_install(conn, 990101)
        async with sessionmaker() as session, session.begin():
            updated = await set_slack_config(
                session,
                installation_id=990101,
                team_id="T0AAAAA",
                bot_token_ciphertext=ciphertext,
                channel_id="C0BBBBB",
                configured_by="admin",
            )
        assert updated is True
        async with engine.connect() as conn:
            team, stored_ct, chan, by, at = (
                await conn.execute(
                    text(
                        "SELECT slack_team_id, slack_bot_token_ciphertext, slack_channel_id, "
                        "slack_configured_by, slack_configured_at FROM installations "
                        "WHERE installation_id = :iid"
                    ),
                    {"iid": 990101},
                )
            ).one()
        assert team == "T0AAAAA"
        assert bytes(stored_ct) == ciphertext  # ciphertext round-trips byte-for-byte
        assert chan == "C0BBBBB"
        assert by == "admin"
        assert at is not None  # server-set timestamptz
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_set_slack_config_tombstoned_install_refused(migrated_db: str) -> None:
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await _insert_install(conn, 990102, tombstoned=True)
        async with sessionmaker() as session, session.begin():
            updated = await set_slack_config(
                session,
                installation_id=990102,
                team_id="T0XXXXX",
                bot_token_ciphertext=b"x",
                channel_id="C0XXXXX",
                configured_by="admin",
            )
        assert updated is False
        async with engine.connect() as conn:
            team = (
                await conn.execute(
                    text("SELECT slack_team_id FROM installations WHERE installation_id = :iid"),
                    {"iid": 990102},
                )
            ).scalar_one()
        assert team is None  # a tombstoned-in-grace install stays unconfigured
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_set_slack_config_absent_install_refused(migrated_db: str) -> None:
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session, session.begin():
            updated = await set_slack_config(
                session,
                installation_id=990199,
                team_id="T0NONE0",
                bot_token_ciphertext=b"x",
                channel_id="C0NONE0",
                configured_by="admin",
            )
        assert updated is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_slack_config_round_trips_active_install(migrated_db: str) -> None:
    ciphertext = b"gAAAAA-fake-ct-\x00\x7f"
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await _insert_install(conn, 990201)
        async with sessionmaker() as session, session.begin():
            await set_slack_config(
                session,
                installation_id=990201,
                team_id="T0AAAAA",
                bot_token_ciphertext=ciphertext,
                channel_id="C0RDBCK",
                configured_by="admin",
            )
        async with sessionmaker() as session:
            cfg = await get_slack_config(session, 990201)
        assert cfg is not None
        assert cfg.channel_id == "C0RDBCK"
        assert cfg.bot_token_ciphertext == ciphertext  # bytea round-trips byte-for-byte
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_slack_config_none_when_unconfigured(migrated_db: str) -> None:
    """An active install that never connected Slack (columns NULL) → None."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await _insert_install(conn, 990202)
        async with sessionmaker() as session:
            assert await get_slack_config(session, 990202) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_slack_config_none_when_tombstoned(migrated_db: str) -> None:
    """A configured install that is later tombstoned (pending purge, #012) → None:
    Slack must not post for an install in its purge grace window."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await _insert_install(conn, 990203)
        async with sessionmaker() as session, session.begin():
            await set_slack_config(
                session,
                installation_id=990203,
                team_id="T0AAAAA",
                bot_token_ciphertext=b"ct",
                channel_id="C0RDBCK",
                configured_by="admin",
            )
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE installations SET tombstoned_at = NOW() WHERE installation_id = :iid"),
                {"iid": 990203},
            )
        async with sessionmaker() as session:
            assert await get_slack_config(session, 990203) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_slack_config_none_when_absent(migrated_db: str) -> None:
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session:
            assert await get_slack_config(session, 990299) is None
    finally:
        await engine.dispose()
