# See DECISIONS.md#051-slack-bot-tokens-are-encrypted-at-rest
"""Composition-root resolver: `installation_id -> SlackNotifyTarget | None`.

The graph holds only the `resolve_slack_target` seam (FUP-186); THIS class is the
implementation the lifespan injects. Per `installation_id` it reads the install's
Slack config, decrypts the bot token (`token_crypto`), builds a per-install
orchestrator (a `SlackWebClientNotifier` + the shared audit sink), and caches it
keyed on `(installation_id, ciphertext)` so a token rotation (re-OAuth → new
ciphertext) invalidates the cached notifier. Notifiers are closed at lifespan
teardown.

Lives in `notify/` (not `agent/`) so `cryptography` (via `token_crypto`) and
`slack_sdk` (via `SlackWebClientNotifier`) stay out of the graph — `agent/` imports
neither. The decrypted bot token is read ONLY at the notifier-construction site
below (never logged, stored in state, or placed in an audit event).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from outrider.db.models.installations import get_slack_config
from outrider.notify.orchestrator import SlackNotificationOrchestrator, SlackNotifyTarget
from outrider.notify.slack import SlackWebClientNotifier
from outrider.notify.token_crypto import decrypt_token

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.audit.sinks import SlackEventSink
    from outrider.db.models.installations import InstallSlackConfig

__all__ = ["PerInstallSlackResolver"]

logger = logging.getLogger(__name__)


class PerInstallSlackResolver:
    """Resolve a per-install Slack target, caching constructed orchestrators.

    Constructed once in the lifespan; injected into `build_graph` as
    `resolve_slack_target`. `__call__` IS the `SlackTargetResolver` seam the hitl /
    publish nodes hold (they never see this class). Both nodes wrap the call in a
    no-raise envelope, so a failure here (bad enc key, DB error) degrades to no
    notification rather than breaking the graph.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        sink: SlackEventSink,
        dashboard_base_url: str,
        timeout_seconds: float | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._sink = sink
        self._dashboard_base_url = dashboard_base_url
        self._timeout_seconds = timeout_seconds
        self._cache: dict[tuple[int, bytes], SlackNotifyTarget] = {}
        self._notifiers: list[SlackWebClientNotifier] = []
        self._lock = asyncio.Lock()

    async def __call__(self, installation_id: int) -> SlackNotifyTarget | None:
        async with self._session_factory() as session:
            config = await get_slack_config(session, installation_id)
        if config is None:
            return None
        key = (installation_id, config.bot_token_ciphertext)
        # Fast path: a cached orchestrator for this (install, ciphertext) — re-OAuth
        # produces a new ciphertext, so a rotated token misses and rebuilds.
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        async with self._lock:
            # Re-check under the lock: a concurrent review may have built it meanwhile.
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            target = self._build(config)
            self._cache[key] = target
            return target

    def _build(self, config: InstallSlackConfig) -> SlackNotifyTarget:
        # decrypt_token returns a SecretStr; the plaintext is read ONLY here, at the
        # notifier construction site (never logged, stored, or audit-emitted).
        token = decrypt_token(config.bot_token_ciphertext)
        if self._timeout_seconds is not None:
            notifier = SlackWebClientNotifier(
                token=token.get_secret_value(), timeout_seconds=self._timeout_seconds
            )
        else:
            notifier = SlackWebClientNotifier(token=token.get_secret_value())
        self._notifiers.append(notifier)
        orchestrator = SlackNotificationOrchestrator(
            notifier=notifier,
            sink=self._sink,
            dashboard_base_url=self._dashboard_base_url,
        )
        return SlackNotifyTarget(channel_id=config.channel_id, orchestrator=orchestrator)

    async def aclose(self) -> None:
        """Close every constructed notifier (lifespan teardown). Idempotent; a failing
        aclose is suppressed so one bad notifier can't block the rest of teardown."""
        for notifier in self._notifiers:
            with contextlib.suppress(Exception):
                await notifier.aclose()
        self._notifiers.clear()
        self._cache.clear()
