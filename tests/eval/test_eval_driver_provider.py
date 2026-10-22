"""Provider injection and ownership checks; scripted responses test wiring only."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from outrider.agent import eval_driver, run_review
from outrider.agent.eval_driver import CostProbe, EvalDriverError, _FixtureScriptedProvider
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER

if TYPE_CHECKING:
    from outrider.llm.base import LLMExchangePersister, LLMRequest, LLMResponse

_TRACE_FIXTURE = Path(__file__).parent / "fixtures/mock_github/handler_to_model_trace.json"


class _InjectedScriptedProvider(_FixtureScriptedProvider):
    def __init__(
        self,
        responses: dict[str, list[str]],
        *,
        persister: LLMExchangePersister,
        fail_on_close: bool = False,
    ) -> None:
        super().__init__(responses, persister=persister)
        self.calls: list[LLMRequest] = []
        self.close_calls = 0
        self.fail_on_close = fail_on_close

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return await super().complete(request)

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.fail_on_close:
            raise RuntimeError("provider close failed")


def _fixture_without_responses(tmp_path: Path) -> tuple[Path, dict[str, list[str]]]:
    payload = json.loads(_TRACE_FIXTURE.read_text(encoding="utf-8"))
    responses = payload.pop("llm_responses")
    payload["llm_responses"] = {}
    fixture_path = tmp_path / "provider.json"
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")
    return fixture_path, responses


def test_run_review_injected_provider_drives_trace_and_closes(tmp_path: Path) -> None:
    fixture_path, responses = _fixture_without_responses(tmp_path)
    providers: list[_InjectedScriptedProvider] = []

    def factory(persister: LLMExchangePersister) -> _InjectedScriptedProvider:
        provider = _InjectedScriptedProvider(responses, persister=persister)
        providers.append(provider)
        return provider

    result = run_review(
        fixture_path,
        provider_factory=factory,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )

    assert len(providers) == 1
    assert providers[0].close_calls == 1
    assert any(
        decision.resolution_status == "resolved" and decision.target_file == "app/models.py"
        for decision in result.trace_decisions
    )
    analyze_calls = [call for call in providers[0].calls if call.node_id == "analyze"]
    assert len(analyze_calls) == 2
    assert any(
        "app/models.py" in call.user_prompt and "SELECT * FROM" in call.user_prompt
        for call in analyze_calls
    )
    assert result.published_comments
    assert result.review_metrics is not None
    assert result.review_metrics.llm_calls_made == len(providers[0].calls) > 0
    assert result.review_metrics.total_input_tokens == 100 * len(providers[0].calls)


@pytest.mark.parametrize("failure_stage", ["build", "invoke"])
@pytest.mark.parametrize("fail_on_close", [False, True])
def test_run_review_closes_provider_on_graph_failure(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str, fail_on_close: bool
) -> None:
    providers: list[_InjectedScriptedProvider] = []

    def factory(persister: LLMExchangePersister) -> _InjectedScriptedProvider:
        provider = _InjectedScriptedProvider({}, persister=persister, fail_on_close=fail_on_close)
        providers.append(provider)
        return provider

    failure = RuntimeError("graph failed")
    graph_builder = (
        Mock(side_effect=failure)
        if failure_stage == "build"
        else Mock(return_value=SimpleNamespace(ainvoke=AsyncMock(side_effect=failure)))
    )
    monkeypatch.setattr(eval_driver, "_build_eval_graph", graph_builder)

    with pytest.raises(RuntimeError, match="graph failed") as exc_info:
        run_review(_TRACE_FIXTURE, provider_factory=factory)

    assert exc_info.value is failure
    assert len(providers) == 1
    assert providers[0].close_calls == 1
    assert not providers[0].calls


def test_run_review_disposes_engine_and_drops_database_when_provider_close_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_path, responses = _fixture_without_responses(tmp_path)
    providers: list[_InjectedScriptedProvider] = []
    engines: list[AsyncEngine] = []
    disposed: list[AsyncEngine] = []
    original_dispose = AsyncEngine.dispose

    def factory(persister: LLMExchangePersister) -> _InjectedScriptedProvider:
        provider = _InjectedScriptedProvider(responses, persister=persister, fail_on_close=True)
        providers.append(provider)
        return provider

    def create_driver_engine(*args: Any, **kwargs: Any) -> AsyncEngine:
        engine = create_async_engine(*args, **kwargs)
        engines.append(engine)
        return engine

    async def track_dispose(engine: AsyncEngine, close: bool = True) -> None:
        await original_dispose(engine, close=close)
        disposed.append(engine)

    monkeypatch.setattr(eval_driver, "create_async_engine", create_driver_engine)
    monkeypatch.setattr(AsyncEngine, "dispose", track_dispose)

    with pytest.raises(RuntimeError, match="provider close failed"):
        run_review(
            fixture_path,
            provider_factory=factory,
            import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
        )

    assert len(providers) == len(engines) == 1
    assert providers[0].close_calls == 1
    assert engines[0] in disposed
    database_name = engines[0].url.database
    assert database_name is not None

    async def database_exists() -> bool:
        check_engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
        try:
            async with check_engine.connect() as conn:
                found = await conn.scalar(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": database_name},
                )
                return found is not None
        finally:
            await check_engine.dispose()

    assert not asyncio.run(database_exists())


def test_run_review_rejects_cost_probe_with_provider_before_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    factory = Mock(side_effect=AssertionError("provider constructed"))
    database = Mock(side_effect=AssertionError("database created"))
    monkeypatch.setattr(eval_driver, "ephemeral_database", database)

    with pytest.raises(EvalDriverError, match="CostProbe"):
        run_review(
            tmp_path / "missing.json",
            provider_factory=factory,
            probe=CostProbe(token_estimator=len),
        )

    factory.assert_not_called()
    database.assert_not_called()
