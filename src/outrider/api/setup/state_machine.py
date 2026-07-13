# See DECISIONS.md#070 — the durable singleton setup state machine (CAS transitions + recovery).
"""The App-Manifest onboarding state machine (`DECISIONS.md#070`).

A durable, DB-enforced **singleton** (`setup_state` id=1) that owns the onboarding lifecycle:

    UNCONFIGURED → AWAITING_CALLBACK → CONVERTING → CONFIGURED   (+ ORPHANED)

Every transition is a **compare-and-swap on `status`** (`UPDATE ... WHERE status=<from>`), never a
SELECT-then-act — so concurrency is resolved by Postgres row locks, not application checks (spec
§Setup state machine). Recovery is **lazy** (folded into the next `POST /setup`), plus a startup
check for the stale-`CONVERTING` crash case; an abandoned `AWAITING_CALLBACK` clears via the lazy
path only. There is no periodic sweep.

Transition summary:
- `begin_setup` — atomic `UNCONFIGURED → AWAITING_CALLBACK` CAS + nonce insert, preceded by lazy
  repair (expired `AWAITING_CALLBACK → UNCONFIGURED`, stale `CONVERTING → ORPHANED`). A concurrent
  second init loses the CAS → `SetupConflictError`.
- `consume_callback` — atomic delete-on-consume of the nonce + `AWAITING_CALLBACK → CONVERTING`,
  returning the attempt binding for the caller to verify the conversion response against.
- `mark_configured` — credential insert (under the single-active-row index) + `CONVERTING →
  CONFIGURED`, in ONE transaction (F5): a crash leaves neither active-creds-without-CONFIGURED nor
  the inverse.
- `orphan` — `CONVERTING → ORPHANED` on any conversion failure (the App likely already exists).
- `reset` — admin `ORPHANED → UNCONFIGURED` after the operator confirms GitHub-side cleanup.
- `recover_stale_converting` — the standalone startup form of the stale-`CONVERTING` timeout.

Dependencies (the session factory) are injected, not global; state lives only in the DB.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, NoReturn

from sqlalchemy import delete, func, select, update

from outrider.api.setup.nonce import hash_nonce
from outrider.db.models.github_app_credentials import GitHubAppCredential
from outrider.db.models.setup_state import SetupNonce, SetupState

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from sqlalchemy.sql import Update

__all__ = [
    "NONCE_TTL_SECONDS",
    "SetupBinding",
    "SetupConflictError",
    "SetupIntegrityError",
    "SetupNonceError",
    "SetupStateMachine",
    "SetupTransitionError",
]

# Attempt lifetime (seconds): how long AWAITING_CALLBACK is valid — the nonce `expires_at` AND the
# signed-state `exp` both use it, so the token pre-check and the DB consume-gate agree. Generous for
# the operator to register the App on GitHub and get redirected back, within GitHub's ~1h window.
NONCE_TTL_SECONDS = 1800
# A CONVERTING row is stale once its conversion_started_at is older than this — must be LONGER than
# the conversion HTTP timeout (seconds) so a genuinely in-flight conversion is never false-ORPHANED.
_STALE_CONVERTING_AFTER = timedelta(minutes=5)


def _rowcount(result: object) -> int:
    # `AsyncSession.execute` is typed `Result` (no `rowcount`); the runtime object for a DML
    # statement is a `CursorResult` that has it. Mirror `db/models/installations.py`'s access.
    return getattr(result, "rowcount", 0) or 0


class SetupTransitionError(RuntimeError):
    """Base for setup state-machine transition failures. Distinct from `state_token.SetupStateError`
    (the signed-state/secret errors) — this family is about the DB state machine."""


class SetupIntegrityError(SetupTransitionError):
    """The `setup_state` singleton (id=1) is missing — the migration seeds it, so its absence is
    integrity corruption. Fail loud; never treated as a normal transition failure."""


class SetupConflictError(SetupTransitionError):
    """A transition's CAS matched no row: the machine was not in the `expected` state (a concurrent
    init, an already-configured instance, an un-reset ORPHANED, etc.). Routers map this to `409`."""

    def __init__(self, *, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"setup state is {actual!r}, expected {expected!r} for this transition")


class SetupNonceError(SetupTransitionError):
    """The callback nonce did not atomically consume — no matching row, expired, or already spent (a
    replay). Routers map this to a rejection; the nonce is single-use by construction."""


@dataclass(frozen=True)
class SetupBinding:
    """The attempt binding recorded at `begin_setup`, returned by `consume_callback` so the caller
    verifies the conversion response's `owner.login` / `permissions` / `events` against exactly what
    THIS attempt requested (not current constants — the attempt may span a restart)."""

    expected_org_login: str | None
    expected_permissions: dict[str, Any] | None
    expected_events: list[str] | None
    manifest_contract_digest: str | None


@dataclass(frozen=True)
class SetupStateMachine:
    """CAS-guarded transitions over the `setup_state` singleton. The session factory is injected; no
    global state. `stale_converting_after` is overridable for tests (a short timeout drives the
    stale-recovery path deterministically)."""

    session_factory: async_sessionmaker[AsyncSession]
    stale_converting_after: timedelta = _STALE_CONVERTING_AFTER

    def _orphan_stale_converting_stmt(self) -> Update:
        # Shared by the lazy repair (begin_setup) and the standalone startup check: CONVERTING →
        # ORPHANED only when conversion_started_at is older than the (DB-clock) stale threshold.
        return (
            update(SetupState)
            .where(
                SetupState.id == 1,
                SetupState.status == "CONVERTING",
                SetupState.conversion_started_at < func.now() - self.stale_converting_after,
            )
            .values(status="ORPHANED", updated_at=func.now())
        )

    async def _repair(self, session: AsyncSession) -> None:
        # Lazy recovery, run inside begin_setup's transaction BEFORE the start CAS (no periodic
        # sweep). Order matters: delete expired nonces first, so the AWAITING_CALLBACK reset can
        # test "no live nonce remains" as a plain NOT EXISTS.
        await session.execute(delete(SetupNonce).where(SetupNonce.expires_at <= func.now()))
        await session.execute(self._orphan_stale_converting_stmt())
        no_live_nonce = ~select(SetupNonce.id).where(SetupNonce.expires_at > func.now()).exists()
        await session.execute(
            update(SetupState)
            .where(
                SetupState.id == 1,
                SetupState.status == "AWAITING_CALLBACK",
                no_live_nonce,
            )
            .values(
                status="UNCONFIGURED",
                expected_org_login=None,
                expected_permissions=None,
                expected_events=None,
                manifest_contract_digest=None,
                conversion_started_at=None,
                updated_at=func.now(),
            )
        )

    async def _status_or_none(self, session: AsyncSession) -> str | None:
        return (
            await session.execute(select(SetupState.status).where(SetupState.id == 1))
        ).scalar_one_or_none()

    async def _raise_conflict(self, session: AsyncSession, *, expected: str) -> NoReturn:
        # A CAS matched no row: distinguish "wrong state" (409) from "singleton missing".
        actual = await self._status_or_none(session)
        if actual is None:
            raise SetupIntegrityError("setup_state singleton (id=1) is missing")
        raise SetupConflictError(expected=expected, actual=actual)

    async def current_status(self) -> str:
        """The authoritative onboarding status (raises `SetupIntegrityError` if the singleton is
        missing). The source of truth for `/setup/status` gating and setup-only routing."""
        async with self.session_factory() as session:
            status = await self._status_or_none(session)
            if status is None:
                raise SetupIntegrityError("setup_state singleton (id=1) is missing")
            return status

    async def begin_setup(
        self,
        *,
        expected_org_login: str,
        expected_permissions: dict[str, Any],
        expected_events: list[str],
        manifest_contract_digest: str,
        nonce_hash: str,
    ) -> None:
        """Start: lazy repair (its OWN committed transaction), then an atomic CAS `UNCONFIGURED →
        AWAITING_CALLBACK` recording the attempt binding + nonce insert (a second transaction, F3).
        Raises `SetupConflictError` if the machine is not `UNCONFIGURED` after repair (a concurrent
        init, or a CONFIGURED / CONVERTING / ORPHANED instance — the caller maps these to 409).

        The repair is a SEPARATE committed transaction on purpose: a stale `CONVERTING → ORPHANED`
        (or expired `AWAITING_CALLBACK → UNCONFIGURED`) must PERSIST even when the Start below is
        then rejected — if the repair shared the Start's transaction, the raised conflict would roll
        it back and the stale state would never clear via the lazy path."""
        async with self.session_factory() as session, session.begin():
            await self._repair(session)
        async with self.session_factory() as session, session.begin():
            result = await session.execute(
                update(SetupState)
                .where(SetupState.id == 1, SetupState.status == "UNCONFIGURED")
                .values(
                    status="AWAITING_CALLBACK",
                    expected_org_login=expected_org_login,
                    expected_permissions=expected_permissions,
                    expected_events=expected_events,
                    manifest_contract_digest=manifest_contract_digest,
                    conversion_started_at=None,
                    updated_at=func.now(),
                )
            )
            if _rowcount(result) != 1:
                await self._raise_conflict(session, expected="UNCONFIGURED")
            await session.execute(
                delete(SetupNonce).where(SetupNonce.nonce_hash == nonce_hash)
            )  # defensive: a hash collision is unlikely, but never leave a stale twin
            session.add(
                SetupNonce(
                    nonce_hash=nonce_hash,
                    expires_at=func.now() + timedelta(seconds=NONCE_TTL_SECONDS),
                )
            )

    async def consume_callback(self, *, raw_nonce: str) -> SetupBinding:
        """Atomic callback consume: delete-on-consume the nonce, then CAS `AWAITING_CALLBACK →
        CONVERTING`, returning the attempt binding — one transaction (F2). Raises `SetupNonceError`
        if the nonce is missing/expired/replayed, or `SetupConflictError` if the state is not
        `AWAITING_CALLBACK`."""
        async with self.session_factory() as session, session.begin():
            consumed = await session.execute(
                delete(SetupNonce)
                .where(
                    SetupNonce.nonce_hash == hash_nonce(raw_nonce),
                    SetupNonce.expires_at > func.now(),
                )
                .returning(SetupNonce.id)
            )
            if consumed.first() is None:
                raise SetupNonceError("callback nonce is missing, expired, or already consumed")
            advanced = await session.execute(
                update(SetupState)
                .where(SetupState.id == 1, SetupState.status == "AWAITING_CALLBACK")
                .values(
                    status="CONVERTING",
                    conversion_started_at=func.now(),
                    updated_at=func.now(),
                )
                .returning(
                    SetupState.expected_org_login,
                    SetupState.expected_permissions,
                    SetupState.expected_events,
                    SetupState.manifest_contract_digest,
                )
            )
            row = advanced.first()
            if row is None:
                await self._raise_conflict(session, expected="AWAITING_CALLBACK")
            return SetupBinding(
                expected_org_login=row[0],
                expected_permissions=row[1],
                expected_events=row[2],
                manifest_contract_digest=row[3],
            )

    async def mark_configured(
        self,
        *,
        app_id: int,
        slug: str,
        client_id: str | None,
        pem_ciphertext: bytes,
        webhook_secret_ciphertext: bytes,
    ) -> None:
        """Verify-then-activate persist: insert the (already-encrypted) credential as the single
        active row AND CAS `CONVERTING → CONFIGURED`, in ONE transaction (F5). The unique partial
        index prevents two racing activations — the loser's flush raises `SetupConflictError`.
        Raises `SetupConflictError` if the state is not `CONVERTING`."""
        from sqlalchemy.exc import IntegrityError

        async with self.session_factory() as session, session.begin():
            version = (
                await session.execute(
                    select(func.coalesce(func.max(GitHubAppCredential.version), 0) + 1)
                )
            ).scalar_one()
            session.add(
                GitHubAppCredential(
                    version=version,
                    app_id=app_id,
                    slug=slug,
                    client_id=client_id,
                    pem_ciphertext=pem_ciphertext,
                    webhook_secret_ciphertext=webhook_secret_ciphertext,
                    is_active=True,
                )
            )
            try:
                # Surface the single-active-row collision HERE (a racing activate loses the slot).
                # That partial unique index is the only constraint on github_app_credentials, so it
                # is the only reachable IntegrityError; actual=CONFIGURED (an active row exists only
                # post-activate). A future migration adding a constraint must revisit this catch.
                await session.flush()
            except IntegrityError as exc:
                raise SetupConflictError(expected="CONVERTING", actual="CONFIGURED") from exc
            result = await session.execute(
                update(SetupState)
                .where(SetupState.id == 1, SetupState.status == "CONVERTING")
                .values(status="CONFIGURED", updated_at=func.now())
            )
            if _rowcount(result) != 1:
                await self._raise_conflict(session, expected="CONVERTING")

    async def orphan(self) -> bool:
        """CAS `CONVERTING → ORPHANED` on any conversion failure (4xx/timeout/malformed/persist
        crash — the App likely already exists on GitHub). Returns whether it transitioned; tolerant
        of a state that already moved (returns False, no raise)."""
        async with self.session_factory() as session, session.begin():
            result = await session.execute(
                update(SetupState)
                .where(SetupState.id == 1, SetupState.status == "CONVERTING")
                .values(status="ORPHANED", updated_at=func.now())
            )
            return _rowcount(result) == 1

    async def reset(self) -> None:
        """Admin `ORPHANED → UNCONFIGURED` (after the operator confirms the orphaned App was deleted
        on GitHub) — clears the attempt binding and leftover nonces so a fresh Start can proceed.
        Raises `SetupConflictError` if not `ORPHANED` (the caller returns 409)."""
        async with self.session_factory() as session, session.begin():
            result = await session.execute(
                update(SetupState)
                .where(SetupState.id == 1, SetupState.status == "ORPHANED")
                .values(
                    status="UNCONFIGURED",
                    expected_org_login=None,
                    expected_permissions=None,
                    expected_events=None,
                    manifest_contract_digest=None,
                    conversion_started_at=None,
                    updated_at=func.now(),
                )
            )
            if _rowcount(result) != 1:
                await self._raise_conflict(session, expected="ORPHANED")
            await session.execute(delete(SetupNonce))

    async def recover_stale_converting(self) -> bool:
        """Standalone stale-`CONVERTING → ORPHANED` for the startup check (the no-restart outage
        case is covered by the lazy repair in `begin_setup`). Returns if it fired. Timeout-gated
        so a genuinely in-flight conversion under the threshold is never false-ORPHANED."""
        async with self.session_factory() as session, session.begin():
            result = await session.execute(self._orphan_stale_converting_stmt())
            return _rowcount(result) == 1
