"""PerInstallSlackResolver — the composition-root Slack target resolver (commit 6.4c).

Pins: None config -> None target; a configured install -> a target carrying the
channel + a real orchestrator (token decrypted via the real token_crypto); caching by
(installation_id, ciphertext); rebuild on token rotation (new ciphertext); idempotent
aclose. `get_slack_config` (the DB read) is monkeypatched — the resolver's DB query has
its own integration coverage; this isolates the decrypt + build + cache logic.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from outrider.db.models.installations import InstallSlackConfig
from outrider.notify.orchestrator import SlackNotificationOrchestrator
from outrider.notify.resolver import PerInstallSlackResolver
from outrider.notify.token_crypto import TOKEN_ENC_KEY_ENV, encrypt_token

# A real (throwaway) Fernet key so the real decrypt_token path runs.
_TEST_FERNET_KEY = "5deJqws1Xhk7vjpx0_pKr7GebqDXYiXLSshLmkI5jy0="  # noqa: S105 (test fixture)


class _FakeSink:
    """Minimal SlackEventSink stand-in; the resolver only stores it on the orchestrator."""

    async def emit_slack_notification(self, _event: Any) -> None:
        return None

    async def query_slack_notification(self, **_kwargs: Any) -> None:
        return None


class _FakeSessionCtx:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def _factory() -> _FakeSessionCtx:
    return _FakeSessionCtx()


def _resolver() -> PerInstallSlackResolver:
    return PerInstallSlackResolver(
        session_factory=_factory,  # type: ignore[arg-type]
        sink=_FakeSink(),  # type: ignore[arg-type]
        dashboard_base_url="https://dash.example",
    )


@pytest.fixture
def enc_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENC_KEY_ENV, _TEST_FERNET_KEY)


@pytest.mark.usefixtures("enc_key")
async def test_resolver_returns_none_when_no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _none(_session: object, _installation_id: int) -> None:
        return None

    monkeypatch.setattr("outrider.notify.resolver.get_slack_config", _none)
    assert await _resolver()(42) is None


@pytest.mark.usefixtures("enc_key")
async def test_resolver_builds_and_caches_target(monkeypatch: pytest.MonkeyPatch) -> None:
    ciphertext = encrypt_token(SecretStr("xoxb-real-bot-token"))

    async def _cfg(_session: object, _installation_id: int) -> InstallSlackConfig:
        return InstallSlackConfig(channel_id="C0XYZ", bot_token_ciphertext=ciphertext)

    monkeypatch.setattr("outrider.notify.resolver.get_slack_config", _cfg)
    resolver = _resolver()

    t1 = await resolver(42)
    assert t1 is not None
    assert t1.channel_id == "C0XYZ"
    assert isinstance(t1.orchestrator, SlackNotificationOrchestrator)

    t2 = await resolver(42)
    assert t2 is t1  # same (installation_id, ciphertext) → cached, not rebuilt

    await resolver.aclose()  # idempotent teardown
    await resolver.aclose()


@pytest.mark.usefixtures("enc_key")
async def test_resolver_rebuilds_on_token_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A re-OAuth produces a new ciphertext; the cache key includes it, so the stale
    notifier is not reused."""
    state = {"ct": encrypt_token(SecretStr("xoxb-old"))}

    async def _cfg(_session: object, _installation_id: int) -> InstallSlackConfig:
        return InstallSlackConfig(channel_id="C0XYZ", bot_token_ciphertext=state["ct"])

    monkeypatch.setattr("outrider.notify.resolver.get_slack_config", _cfg)
    resolver = _resolver()

    t1 = await resolver(42)
    state["ct"] = encrypt_token(SecretStr("xoxb-new"))  # token rotated
    t2 = await resolver(42)
    assert t1 is not None
    assert t2 is not None
    assert t2 is not t1  # rotation invalidates the cache → rebuilt
    await resolver.aclose()
