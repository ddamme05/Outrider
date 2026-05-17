# See specs/2026-05-16-audit-persister.md.
"""FastAPI lifespan — durable dependency construction + teardown.

Constructs at startup, in dependency order:

  1. `AsyncEngine` from `DATABASE_URL` env var.
  2. `async_sessionmaker` over the engine (`expire_on_commit=False` so
     post-commit attribute access on returned rows doesn't lazy-refresh).
  3. `RetentionSettings()` — reads `OUTRIDER_AUDIT_*` env vars.
  4. `AuditPersister(session_factory=..., retention_settings=...)`.
  5. `ModelConfig()` — reads `OUTRIDER_MODEL_*` env vars.
  6. `AnthropicProvider(api_key=..., model_config=..., persister=...)`.
  7. `register_filter_on_all_handlers()` — re-applies the log-content
     filter to any handler uvicorn registered between `import outrider`
     and lifespan entry. Per `RejectLLMContentFilter`'s idempotent
     install (see `llm/logging.py`), calling again is safe.

Teardown is `AsyncExitStack` LIFO — every push_async_callback runs even
if a prior callback raises. Closes FUP-006 (filter re-registration) and
FUP-011 (provider aclose).

`build_lifespan(...)` is the test seam: production callers use the
module-level `lifespan` (which calls `build_lifespan()` with defaults);
tests pass factories to inject mocks for engine/provider construction,
or to inject a provider whose `aclose()` raises (the teardown-ordering
test).

V1 has no HTTP routes; the lifespan exists for the persister + provider
construction it owns. A future webhook-receiver spec adds routes that
consume `app.state.persister` and `app.state.provider`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Callable
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from typing import TYPE_CHECKING

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from outrider.audit.config import RetentionSettings
from outrider.audit.persister import AuditPersister
from outrider.llm.anthropic_provider import AnthropicProvider
from outrider.llm.config import ModelConfig
from outrider.llm.logging import register_filter_on_all_handlers

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["build_lifespan", "lifespan"]


_LOGGER = logging.getLogger("outrider.api.lifespan")


# ---------------------------------------------------------------------------
# Test seams: factories that production resolves from env vars; tests can
# replace each via `build_lifespan(engine_factory=..., provider_factory=...)`.
# ---------------------------------------------------------------------------


EngineFactory = Callable[[], AsyncEngine]
ProviderFactory = Callable[[AuditPersister], AnthropicProvider]


def _default_engine_factory() -> AsyncEngine:
    """Production engine factory: read `DATABASE_URL` env, fail loud if missing.

    Construction does not connect to the DB — the engine is lazy. The first
    real query (e.g., from `persister.persist()`) opens the pool.

    `hide_parameters=True` is load-bearing for `DECISIONS.md#016` (logs
    stay metadata-only): SQLAlchemy's default exception string for
    `IntegrityError` / `DataError` / etc. includes the bound parameters
    of the failing statement. Without this setting, a content-INSERT
    failure (e.g., FK violation on installations) would surface raw
    `prompt` / `completion` text in `exc.args[0]`. The real residual
    leak vector this setting defends against is `logger.exception(...)`
    (or any log handler that calls `str(exc)` on the raw SQLAlchemy
    exception BEFORE the wrapper at `anthropic_provider.py::complete()`
    sees it) — those rendered strings bypass `RejectLLMContentFilter`
    (key-based, doesn't pattern-match log record `message` fields, same
    FUP-023 gap). With this set, exception strings render parameter
    values as `?` placeholders, preserving the SQL shape for
    diagnostics without leaking content.

    NOTE on the wrapper-chain leak vector (now closed at the wrapper):
    the round-9 + round-26 hardening of `anthropic_provider.py` no
    longer leaks unknown exception text via the wrapper chain. For ANY
    exception type not in `METADATA_ONLY_EXCEPTION_TYPES`, the wrapper
    renders only `<TypeName>` and raises with `from None`
    (`__suppress_context__=True` blocks traceback walking into the
    original SQLAlchemy exception). So even without
    `hide_parameters=True`, a raw SQLAlchemy exception reaching the
    wrapper renders only its class name. The remaining concern this
    setting addresses is the case where exception text is rendered by
    some other handler BEFORE the wrapper sees it (e.g., SQLAlchemy
    engine-level logging, third-party connection-pool error handlers).
    """
    try:
        database_url = os.environ["DATABASE_URL"]
    except KeyError as exc:
        raise RuntimeError(
            "DATABASE_URL env var is required for the FastAPI lifespan. "
            "See .env.example for the canonical postgres URL shape."
        ) from exc

    # Driver-allowlist gate: SQLAlchemy's `create_async_engine` accepts ANY
    # URL string and constructs lazily (no connection happens at construct
    # time). If an operator copies a SYNC URL from pgAdmin/alembic context
    # (e.g., `postgresql://...` or `postgresql+psycopg2://...`), construction
    # succeeds; the failure surfaces deep in the first `persister.persist()`
    # call as `InvalidRequestError: The asyncio extension requires an async
    # driver` — far from the configuration error. Fail-loud at lifespan
    # startup instead.
    if not (
        database_url.startswith("postgresql+psycopg://")
        or database_url.startswith("postgresql+asyncpg://")
    ):
        raise RuntimeError(
            "DATABASE_URL must use an async driver scheme — "
            "'postgresql+psycopg://' (psycopg3 async) or "
            "'postgresql+asyncpg://'. Bare 'postgresql://' resolves to the "
            "sync psycopg2 driver, which crashes `create_async_engine` on "
            "first use deep inside a request rather than at startup. "
            "See .env.example for the canonical URL shape."
        )

    return create_async_engine(database_url, hide_parameters=True)


def _default_provider_factory(persister: AuditPersister) -> AnthropicProvider:
    """Production provider factory: reads ANTHROPIC_API_KEY env, constructs
    AnthropicProvider with the default ModelConfig (reads OUTRIDER_MODEL_*).

    Privacy startup notice fires inside the provider's `__init__` (per
    DECISIONS#015 point 4), once per lifespan startup. The `persister`
    arg is injected so the wrapper's fail-closed-pre-call gate is
    satisfied at construction.
    """
    try:
        api_key_raw = os.environ["ANTHROPIC_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(
            "ANTHROPIC_API_KEY env var is required for the FastAPI lifespan."
        ) from exc
    return AnthropicProvider(
        api_key=SecretStr(api_key_raw),
        model_config=ModelConfig(),
        persister=persister,
    )


# ---------------------------------------------------------------------------
# Lifespan builder.
# ---------------------------------------------------------------------------


def build_lifespan(
    *,
    engine_factory: EngineFactory = _default_engine_factory,
    provider_factory: ProviderFactory = _default_provider_factory,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Construct a FastAPI lifespan callable with injectable factories.

    Production callers use the module-level `lifespan` constant
    (`build_lifespan()` with defaults). Tests pass `engine_factory=` and
    `provider_factory=` to inject mocks — the teardown-ordering test
    injects a provider whose `aclose()` raises; the filter-re-registration
    test patches the engine factory to return a mock engine; etc.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with AsyncExitStack() as stack:
            # Step 1: construct the engine; register dispose for teardown.
            engine = engine_factory()
            # Push dispose IMMEDIATELY after construction so the engine
            # gets cleaned up even if the validation gate below raises.
            # Without this ordering, a misconfigured engine (one without
            # `hide_parameters=True`) would be constructed (potentially
            # opening a connection pool lazily on first use) and then
            # rejected — but never disposed because the dispose callback
            # would never have been pushed onto the stack.
            stack.push_async_callback(engine.dispose)
            # Production-equivalent gate: the engine factory MUST set
            # `hide_parameters=True` (per `_default_engine_factory`'s
            # contract) so SQLAlchemy exception strings strip bound values.
            # The test seam at `build_lifespan(engine_factory=...)` allows
            # injecting any factory; this check is the safety gate against
            # a test (or future extraction) that returns a real engine
            # without the setting and silently regresses production behavior.
            # MagicMock engines used in lifespan tests satisfy the
            # truthy-check; real misconfigured engines fail loud.
            #
            # Explicit `if not / raise` rather than `assert`: `assert`
            # statements are stripped under `python -O`, making the
            # guard a no-op in optimized production builds. This gate is
            # load-bearing for #016 logs-stay-metadata-only, so it must
            # be a runtime check, not an optimization-strippable assert.
            #
            # The attribute lives on `sync_engine` since AsyncEngine wraps
            # the sync engine; the AsyncEngine type stub doesn't expose it.
            if not engine.sync_engine.hide_parameters:
                raise RuntimeError(
                    "engine_factory returned an engine without hide_parameters=True; "
                    "production engines MUST strip bound parameter values from "
                    "exception strings per DECISIONS#016 logs-stay-metadata-only"
                )

            # Step 2: session factory. `expire_on_commit=False` so callers
            # can access returned ORM attributes after commit without a
            # lazy refresh on a closed session.
            session_factory = async_sessionmaker(engine, expire_on_commit=False)

            # Step 3: retention settings (env-driven; reads
            # `OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL` if set, else default).
            retention_settings = RetentionSettings()

            # Step 4: durable persister.
            persister = AuditPersister(
                session_factory=session_factory,
                retention_settings=retention_settings,
            )

            # Step 5: provider; lifespan teardown awaits aclose.
            provider = provider_factory(persister)
            stack.push_async_callback(provider.aclose)

            # Step 6: re-apply log filter post-handler-registration.
            # uvicorn registers its handlers before the lifespan body
            # runs, so by here all handler chains exist; idempotent
            # `register_filter_on_all_handlers()` adds the filter to any
            # newly-registered handler missing it.
            register_filter_on_all_handlers()

            # Stash deps on app.state so future request handlers (when the
            # webhook spec lands) can resolve them via FastAPI's
            # dependency-injection system.
            app.state.engine = engine
            app.state.session_factory = session_factory
            app.state.retention_settings = retention_settings
            app.state.persister = persister
            app.state.provider = provider

            # Safe: `engine.url.drivername` is the scheme alone (e.g.,
            # "postgresql+psycopg") — never carries credentials. DO NOT log
            # `engine.url` itself or `engine.url.render_as_string(hide_password=False)`
            # — those leak the password. `RejectLLMContentFilter` is
            # key-based and would not catch a `dsn`-keyed extra here, so
            # any future logging change in this block must preserve the
            # scheme-only contract.
            _engine_url_scheme = engine.url.drivername
            _LOGGER.info(
                "outrider.api.lifespan startup complete",
                extra={
                    "engine_url_scheme": _engine_url_scheme,
                    "retention_ttl_seconds": int(
                        retention_settings.llm_content_retention_ttl.total_seconds()
                    ),
                },
            )

            yield

        # AsyncExitStack teardown (LIFO):
        #   - provider.aclose()   (pushed last → runs first)
        #   - engine.dispose()    (pushed first → runs last)
        # Both run even if one raises; the exception propagates AFTER
        # all callbacks have been attempted.
        _LOGGER.info("outrider.api.lifespan teardown complete")

    return _lifespan


# Module-level lifespan for production wiring: `app = FastAPI(lifespan=lifespan)`.
lifespan = build_lifespan()
