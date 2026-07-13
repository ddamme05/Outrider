"""Integration tests for `api/setup/state_machine` against a real Postgres (#070).

Covers what unit tests can't: the CAS transitions, the concurrent-init reject, the atomic
delete-on-consume nonce + replay reject, the single-active-row collision on activate, orphan/reset,
and lazy + startup recovery (stale CONVERTING, expired AWAITING_CALLBACK). The state machine is the
onboarding spine, so its DB-level concurrency + recovery behavior is the load-bearing surface.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

from outrider.api.setup.nonce import new_nonce
from outrider.api.setup.state_machine import (
    SetupConflictError,
    SetupIntegrityError,
    SetupNonceError,
    SetupStateMachine,
)

_ORG = "acme"
_PERMS = {"contents": "read", "pull_requests": "write"}
_EVENTS = ["pull_request", "installation", "installation_repositories"]
_DIGEST = "manifest-digest-abc"


@pytest_asyncio.fixture
async def session_factory(migrated_db: str) -> AsyncGenerator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(migrated_db)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest.fixture
def machine(session_factory: async_sessionmaker[AsyncSession]) -> SetupStateMachine:
    return SetupStateMachine(session_factory)


async def _status(sf: async_sessionmaker[AsyncSession]) -> str | None:
    async with sf() as session:
        return (
            await session.execute(text("SELECT status FROM setup_state WHERE id = 1"))
        ).scalar_one_or_none()


async def _nonce_count(sf: async_sessionmaker[AsyncSession]) -> int:
    async with sf() as session:
        return (await session.execute(text("SELECT count(*) FROM setup_nonce"))).scalar_one()


async def _set_converting(sf: async_sessionmaker[AsyncSession], *, started_at: datetime) -> None:
    async with sf() as session, session.begin():
        await session.execute(
            text("UPDATE setup_state SET status='CONVERTING', conversion_started_at=:t WHERE id=1"),
            {"t": started_at},
        )


async def _begin(machine: SetupStateMachine, nonce_hash: str) -> None:
    await machine.begin_setup(
        expected_org_login=_ORG,
        expected_permissions=_PERMS,
        expected_events=_EVENTS,
        manifest_contract_digest=_DIGEST,
        nonce_hash=nonce_hash,
    )


# ── Start ────────────────────────────────────────────────────────────────────


async def test_begin_setup_starts(machine: SetupStateMachine, session_factory) -> None:
    _, h = new_nonce()
    await _begin(machine, h)
    assert await _status(session_factory) == "AWAITING_CALLBACK"
    assert await _nonce_count(session_factory) == 1
    assert await machine.current_status() == "AWAITING_CALLBACK"


async def test_concurrent_second_init_rejected(machine: SetupStateMachine, session_factory) -> None:
    """A second Start while a live AWAITING_CALLBACK attempt exists loses the CAS (a valid nonce is
    present, so the lazy repair does NOT reset) → SetupConflictError, state unchanged."""
    _, h1 = new_nonce()
    await _begin(machine, h1)
    _, h2 = new_nonce()
    with pytest.raises(SetupConflictError) as exc:
        await _begin(machine, h2)
    assert exc.value.actual == "AWAITING_CALLBACK"
    assert await _nonce_count(session_factory) == 1  # the second nonce was NOT inserted


# ── Callback consume ──────────────────────────────────────────────────────────


async def test_full_happy_path(machine: SetupStateMachine, session_factory) -> None:
    raw, h = new_nonce()
    await _begin(machine, h)
    binding = await machine.consume_callback(raw_nonce=raw)
    assert binding.expected_org_login == _ORG
    assert binding.expected_permissions == _PERMS
    assert binding.expected_events == _EVENTS
    assert binding.manifest_contract_digest == _DIGEST
    assert await _status(session_factory) == "CONVERTING"
    assert await _nonce_count(session_factory) == 0  # consumed

    await machine.mark_configured(
        app_id=4242,
        slug="acme-outrider",
        client_id="Iv1.dead",
        pem_ciphertext=b"pem-ct",
        webhook_secret_ciphertext=b"wh-ct",
    )
    assert await _status(session_factory) == "CONFIGURED"
    async with session_factory() as session:
        row = (
            await session.execute(
                text("SELECT app_id, slug, version, is_active FROM github_app_credentials")
            )
        ).one()
    assert row == (4242, "acme-outrider", 1, True)


async def test_replayed_nonce_rejected(machine: SetupStateMachine, session_factory) -> None:
    raw, h = new_nonce()
    await _begin(machine, h)
    await machine.consume_callback(raw_nonce=raw)  # first consume ok
    with pytest.raises(SetupNonceError):
        await machine.consume_callback(raw_nonce=raw)  # replay → gone


async def test_expired_nonce_rejected(machine: SetupStateMachine, session_factory) -> None:
    raw, h = new_nonce()
    await _begin(machine, h)
    async with session_factory() as session, session.begin():
        await session.execute(
            text("UPDATE setup_nonce SET expires_at = NOW() - INTERVAL '1 minute'")
        )
    with pytest.raises(SetupNonceError):
        await machine.consume_callback(raw_nonce=raw)


async def test_callback_wrong_state_rejected(machine: SetupStateMachine, session_factory) -> None:
    """A live nonce but the machine already in CONVERTING (not AWAITING_CALLBACK): the nonce
    delete-on-consume succeeds but the CAS matches no row → SetupConflictError, whole txn rolls
    back (the nonce is NOT consumed)."""
    raw, h = new_nonce()
    await _begin(machine, h)
    await _set_converting(session_factory, started_at=datetime.now(UTC))  # force CONVERTING
    with pytest.raises(SetupConflictError) as exc:
        await machine.consume_callback(raw_nonce=raw)
    assert exc.value.actual == "CONVERTING"
    assert await _nonce_count(session_factory) == 1  # rolled back — nonce preserved


# ── Activate collision ────────────────────────────────────────────────────────


async def test_mark_configured_single_active_collision(
    machine: SetupStateMachine, session_factory
) -> None:
    """An existing active credential row makes a second activate impossible (unique partial index)
    → SetupConflictError, and the state stays CONVERTING (txn rolled back)."""
    async with session_factory() as session, session.begin():
        await session.execute(
            text(
                "INSERT INTO github_app_credentials "
                "(version, app_id, slug, pem_ciphertext, webhook_secret_ciphertext, is_active) "
                "VALUES (1, 111, 'existing', :p, :w, true)"
            ),
            {"p": b"x", "w": b"y"},
        )
    await _set_converting(session_factory, started_at=datetime.now(UTC))
    with pytest.raises(SetupConflictError):
        await machine.mark_configured(
            app_id=222,
            slug="new",
            client_id=None,
            pem_ciphertext=b"pem",
            webhook_secret_ciphertext=b"wh",
        )
    assert await _status(session_factory) == "CONVERTING"


# ── Orphan / reset ────────────────────────────────────────────────────────────


async def test_orphan_from_converting(machine: SetupStateMachine, session_factory) -> None:
    await _set_converting(session_factory, started_at=datetime.now(UTC))
    assert await machine.orphan() is True
    assert await _status(session_factory) == "ORPHANED"


async def test_orphan_noop_when_not_converting(machine: SetupStateMachine, session_factory) -> None:
    # seeded UNCONFIGURED
    assert await machine.orphan() is False
    assert await _status(session_factory) == "UNCONFIGURED"


async def test_reset_from_orphaned(machine: SetupStateMachine, session_factory) -> None:
    _, h = new_nonce()
    await _begin(machine, h)
    await _set_converting(session_factory, started_at=datetime.now(UTC))
    await machine.orphan()
    await machine.reset()
    assert await _status(session_factory) == "UNCONFIGURED"
    assert await _nonce_count(session_factory) == 0  # leftover nonces cleared
    async with session_factory() as session:
        binding = (
            await session.execute(
                text("SELECT expected_org_login, conversion_started_at FROM setup_state WHERE id=1")
            )
        ).one()
    assert binding == (None, None)  # attempt binding cleared


async def test_reset_rejected_when_not_orphaned(
    machine: SetupStateMachine, session_factory
) -> None:
    with pytest.raises(SetupConflictError) as exc:
        await machine.reset()  # seeded UNCONFIGURED
    assert exc.value.actual == "UNCONFIGURED"


# ── Recovery ──────────────────────────────────────────────────────────────────


async def test_stale_converting_recovered_startup(
    machine: SetupStateMachine, session_factory
) -> None:
    await _set_converting(session_factory, started_at=datetime.now(UTC) - timedelta(minutes=10))
    assert await machine.recover_stale_converting() is True
    assert await _status(session_factory) == "ORPHANED"


async def test_inflight_converting_not_orphaned(
    machine: SetupStateMachine, session_factory
) -> None:
    await _set_converting(session_factory, started_at=datetime.now(UTC))  # fresh, under the timeout
    assert await machine.recover_stale_converting() is False
    assert await _status(session_factory) == "CONVERTING"


async def test_stale_converting_recovered_lazily(
    machine: SetupStateMachine, session_factory
) -> None:
    """The no-restart outage case: a stale CONVERTING is orphaned by the lazy repair in the next
    begin_setup — which then itself rejects (can't Start over ORPHANED; must reset first)."""
    await _set_converting(session_factory, started_at=datetime.now(UTC) - timedelta(minutes=10))
    _, h = new_nonce()
    with pytest.raises(SetupConflictError) as exc:
        await _begin(machine, h)
    assert exc.value.actual == "ORPHANED"
    assert await _status(session_factory) == "ORPHANED"


async def test_expired_awaiting_callback_lazily_repaired(
    machine: SetupStateMachine, session_factory
) -> None:
    """An abandoned AWAITING_CALLBACK whose nonce expired is reset to UNCONFIGURED by the lazy
    repair, and the SAME begin_setup then starts the replacement attempt."""
    _, h_old = new_nonce()
    await _begin(machine, h_old)
    async with session_factory() as session, session.begin():
        await session.execute(
            text("UPDATE setup_nonce SET expires_at = NOW() - INTERVAL '1 minute'")
        )
    _, h_new = new_nonce()
    await _begin(machine, h_new)  # repair resets, then Start succeeds
    assert await _status(session_factory) == "AWAITING_CALLBACK"
    assert await _nonce_count(session_factory) == 1  # only the new nonce (expired one deleted)
    async with session_factory() as session:
        digest = (
            await session.execute(
                text("SELECT manifest_contract_digest FROM setup_state WHERE id=1")
            )
        ).scalar_one()
    assert digest == _DIGEST  # the new attempt's binding


# ── Integrity ─────────────────────────────────────────────────────────────────


async def test_missing_singleton_fails_loud(machine: SetupStateMachine, session_factory) -> None:
    async with session_factory() as session, session.begin():
        await session.execute(text("DELETE FROM setup_state WHERE id = 1"))
    with pytest.raises(SetupIntegrityError):
        await machine.current_status()
    _, h = new_nonce()
    with pytest.raises(SetupIntegrityError):
        await _begin(machine, h)  # CAS matches no row → singleton missing → integrity, not conflict
    # orphan() / recover_stale_converting() must ALSO fail loud, not silently return False (a lost
    # singleton is corruption, not an ordinary no-op).
    with pytest.raises(SetupIntegrityError):
        await machine.orphan()
    with pytest.raises(SetupIntegrityError):
        await machine.recover_stale_converting()


# ── Deterministic CAS/index contention (a held lock forces the overlap, proving serialization) ──
# asyncio.gather alone does NOT prove the two txns overlapped AT the contested operation (one task
# can commit before the other reaches it). Here session A holds the contested lock across an open
# transaction — a barrier that GUARANTEES the machine's op blocks on it — so the "winner"
# outcome is a proof of Postgres serialization, not a lucky interleaving.


async def test_begin_setup_cas_serializes_under_held_lock(
    machine: SetupStateMachine, session_factory
) -> None:
    """Session A holds a `FOR UPDATE` lock on the singleton (a concurrent Start that reached its CAS
    but hasn't committed), so `begin_setup` (task B) provably BLOCKS at its CAS. A then commits a
    complete winning Start (AWAITING_CALLBACK + a live nonce, so B's expired-AWAITING repair can't
    reset it); B unblocks, its CAS re-evaluates `WHERE status='UNCONFIGURED'`, matches nothing."""
    _, h = new_nonce()
    async with session_factory() as sess_a:
        await sess_a.begin()
        await sess_a.execute(text("SELECT 1 FROM setup_state WHERE id = 1 FOR UPDATE"))
        task = asyncio.create_task(_begin(machine, h))
        await asyncio.sleep(0.25)
        assert not task.done(), (
            "begin_setup must BLOCK on the held singleton lock (real contention)"
        )
        await sess_a.execute(text("UPDATE setup_state SET status='AWAITING_CALLBACK' WHERE id=1"))
        await sess_a.execute(
            text(
                "INSERT INTO setup_nonce (nonce_hash, expires_at) "
                "VALUES (:h, NOW() + INTERVAL '30 min')"
            ),
            {"h": "winner-nonce"},
        )
        await sess_a.commit()  # A wins + releases the lock
    with pytest.raises(SetupConflictError):
        await task
    assert await _status(session_factory) == "AWAITING_CALLBACK"
    assert await _nonce_count(session_factory) == 1  # A's winner nonce; B (loser) inserted none


async def test_mark_configured_serializes_on_held_active_index(
    machine: SetupStateMachine, session_factory
) -> None:
    """Session A inserts an is_active credential and holds it uncommitted, so `mark_configured`
    (task B) provably BLOCKS on the single-active unique index. A commits; B unblocks, its insert
    hits the unique violation and loses — exactly one active row, and B's whole transaction (incl.
    its CONVERTING → CONFIGURED CAS) rolls back."""
    await _set_converting(session_factory, started_at=datetime.now(UTC))
    async with session_factory() as sess_a:
        await sess_a.begin()
        await sess_a.execute(
            text(
                "INSERT INTO github_app_credentials "
                "(version, app_id, slug, pem_ciphertext, webhook_secret_ciphertext, is_active) "
                "VALUES (1, 111, 'winner', :p, :w, true)"
            ),
            {"p": b"x", "w": b"y"},
        )
        task = asyncio.create_task(
            machine.mark_configured(
                app_id=222,
                slug="loser",
                client_id=None,
                pem_ciphertext=b"pem",
                webhook_secret_ciphertext=b"wh",
            )
        )
        await asyncio.sleep(0.25)
        assert not task.done(), "mark_configured must BLOCK on the held single-active index"
        await sess_a.commit()  # A's active row lands
    with pytest.raises(SetupConflictError):
        await task
    async with session_factory() as session:
        active = (
            await session.execute(
                text("SELECT count(*) FROM github_app_credentials WHERE is_active")
            )
        ).scalar_one()
    assert active == 1  # only A's row
    assert await _status(session_factory) == "CONVERTING"  # B lost before its CONFIGURED CAS
