# See DECISIONS.md#070 — create-app-time mount of the `/setup` onboarding router.
"""Create-app-time mount for the `/setup` onboarding router (spec F6).

`main.py::_include_routers` calls `mount_setup_router(app)` in `database` credential mode only
(`env` mode has no onboarding; demo mode mounts no side-effecting surface). The router is mounted at
create_app time — BEFORE the SPA catch-all, which is registration-last (`DECISIONS.md#069`) — so the
`/setup/*` routes win over the SPA history-fallback.

The state machine needs the DB `session_factory`, which the lifespan builds at STARTUP (after
create_app returns). `_LazyAppStateSessionmaker` defers that lookup to call time, and the machine is
stashed on `app.state.setup_state_machine` so the lifespan can run its startup stale-`CONVERTING`
repair (`recover_stale_converting`) against the same instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from outrider.api.setup.config import validate_setup_config
from outrider.api.setup.router import build_setup_router
from outrider.api.setup.state_machine import SetupStateMachine
from outrider.github.manifest_conversion import convert_manifest_code

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

__all__ = ["mount_setup_router"]


class _LazyAppStateSessionmaker:
    """A call-compatible stand-in for `app.state.session_factory`, resolved per call.

    The setup router is built at create_app time, but `app.state.session_factory` is built by the
    lifespan at startup. The state machine only ever invokes `session_factory()`, so deferring the
    lookup to call time (by which point the lifespan has run) is sufficient — no session is opened
    at create_app.
    """

    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def __call__(self) -> AsyncSession:
        factory: async_sessionmaker[AsyncSession] = self._app.state.session_factory
        return factory()


def mount_setup_router(app: FastAPI) -> None:
    """Build the `/setup` state machine + router and mount them on `app` (create_app time).

    Stashes the machine on `app.state.setup_state_machine` for the lifespan's startup repair. Only
    called in `database` credential mode; `validate_setup_config()` fails loud at boot if the setup
    config (`OUTRIDER_PUBLIC_BASE_URL`) is missing.
    """
    lazy_factory = cast("async_sessionmaker[AsyncSession]", _LazyAppStateSessionmaker(app))
    machine = SetupStateMachine(lazy_factory)
    app.state.setup_state_machine = machine
    app.include_router(
        build_setup_router(
            machine=machine,
            settings=validate_setup_config(),
            convert=convert_manifest_code,
        )
    )
