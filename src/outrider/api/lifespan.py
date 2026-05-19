# See specs/2026-05-16-audit-persister.md.
"""FastAPI lifespan — durable dependency construction + teardown.

Constructs at startup, in dependency order:

  1. `AsyncEngine` from `DATABASE_URL` env var.
  2. `async_sessionmaker` over the engine (`expire_on_commit=False` so
     post-commit attribute access on returned rows doesn't lazy-refresh).
  3. `RetentionSettings()` — reads `OUTRIDER_AUDIT_*` env vars.
  4. `AuditPersister(session_factory=..., retention_settings=...)`.
  5. `AnthropicProvider(api_key=..., model_config=..., persister=...)`
     (model_config built once, shared with the compiled graph below).
  6. `GitHubAppSettings()` — reads `OUTRIDER_GITHUB_APP_*` env vars.
     Validating the App credentials at startup means missing / malformed
     env surfaces as a friendly RuntimeError at boot, not a deep-stack
     `ValidationError` inside the first intake invocation.
  7. `github_factory = make_installation_client_factory(github_app_settings)`
     — per-installation `GitHub` client factory closing over the
     lifespan-validated settings. Per `DECISIONS.md#020` + the
     `nodes-receive-deps-via-closure` invariant, installation-token
     minting happens at intake call-site, not at webhook receipt.
  8. `compiled_graph = build_graph(...)` — the V1 two-node intake →
     triage graph with all six deps injected at construction time
     (`db_factory`, `github_factory`, `provider`, `model_config`,
     `phase_event_sink=persister`, `file_examination_sink=persister`).
  9. `run_graph` async closure that the V1 `BackgroundTasksDispatcher`
     invokes per request to call `compiled_graph.ainvoke(state)`.
  10. `register_filter_on_all_handlers()` — re-applies the log-content
      filter to any handler uvicorn registered between `import outrider`
      and lifespan entry. Idempotent (see `llm/logging.py`).

Teardown is `AsyncExitStack` LIFO — every push_async_callback runs even
if a prior callback raises. Closes FUP-006 (filter re-registration) and
FUP-011 (provider aclose).

The webhook router (`api/webhooks/router.py`) reads `app.state` bindings
to resolve per-request dependencies: `session_factory`, `retention_settings`,
`github_app_settings` (webhook signature secret), `github_factory`,
`run_graph`. `compiled_graph`, `persister`, `provider`, `engine` are also
stashed for diagnostics / future routes.

`build_lifespan(...)` is the test seam: production callers use the
module-level `lifespan` (which calls `build_lifespan()` with defaults);
tests pass factories to inject mocks for engine/provider construction,
or to inject a provider whose `aclose()` raises (the teardown-ordering
test).
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
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from outrider.agent.graph import build_graph
from outrider.audit.config import RetentionSettings
from outrider.audit.persister import AuditPersister
from outrider.github.auth import make_installation_client_factory
from outrider.github.config import GitHubAppSettings
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
# `ProviderFactory` accepts the lifespan-built `ModelConfig` so the
# provider and the compiled graph share ONE instance. The prior shape
# (`Callable[[AuditPersister], AnthropicProvider]`) silently constructed
# two independent `ModelConfig()` instances — one inside the factory,
# one in the lifespan body for `build_graph(...)`. Env-driven settings
# read once at the lifespan boundary must stay single-instance through
# the whole graph; constructing the same Settings class twice defeats
# the "lifespan-validated once" guarantee.
ProviderFactory = Callable[[AuditPersister, ModelConfig], AnthropicProvider]


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
    # Single allowed scheme per DECISIONS.md#001 (psycopg3 async only;
    # asyncpg is not a project dependency). Bare `postgresql://`
    # resolves to sync psycopg2 and crashes `create_async_engine` deep
    # in the first request; reject at startup.
    if not database_url.startswith("postgresql+psycopg://"):
        raise RuntimeError(
            "DATABASE_URL must use the canonical async driver scheme "
            "'postgresql+psycopg://' (psycopg3 async). Other schemes — "
            "bare 'postgresql://' (sync psycopg2), 'postgresql+psycopg2://' "
            "(sync), 'postgresql+asyncpg://' (driver not installed) — are "
            "rejected at lifespan startup so misconfiguration fails-loud "
            "instead of crashing on first request. See .env.example."
        )

    return create_async_engine(database_url, hide_parameters=True)


def _default_provider_factory(
    persister: AuditPersister,
    model_config: ModelConfig,
) -> AnthropicProvider:
    """Production provider factory: reads ANTHROPIC_API_KEY env, constructs
    AnthropicProvider with the supplied (lifespan-built) `ModelConfig`.

    Privacy startup notice fires inside the provider's `__init__` (per
    DECISIONS#015 point 4), once per lifespan startup. The `persister`
    arg is injected so the wrapper's fail-closed-pre-call gate is
    satisfied at construction. The `model_config` arg is injected so
    the provider and the compiled graph share ONE instance — the prior
    shape constructed an independent `ModelConfig()` here AND a separate
    one in the lifespan body for `build_graph(...)`, defeating the
    single-source guarantee.
    """
    try:
        api_key_raw = os.environ["ANTHROPIC_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(
            "ANTHROPIC_API_KEY env var is required for the FastAPI lifespan."
        ) from exc
    return AnthropicProvider(
        api_key=SecretStr(api_key_raw),
        model_config=model_config,
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
            # MagicMock engines used in lifespan tests MUST explicitly set
            # `mock_engine.sync_engine.hide_parameters = True` (the bool)
            # to pass — MagicMock's default truthy attribute fails the
            # round-39 strict `is not True` check below. Real misconfigured
            # engines fail loud.
            #
            # Explicit `if not / raise` rather than `assert`: `assert`
            # statements are stripped under `python -O`, making the
            # guard a no-op in optimized production builds. This gate is
            # load-bearing for #016 logs-stay-metadata-only, so it must
            # be a runtime check, not an optimization-strippable assert.
            #
            # The attribute lives on `sync_engine` since AsyncEngine wraps
            # the sync engine; the AsyncEngine type stub doesn't expose it.
            #
            # Use strict `is not True` (not bare falsy check) so a
            # test-injected factory returning an engine with
            # `sync_engine.hide_parameters = "true"` (string, falsely
            # truthy) is rejected — SQLAlchemy's exception-string
            # rendering checks the boolean form; non-bool truthy values
            # produce undefined redaction behavior depending on SA
            # version. Production path (`create_async_engine(...,
            # hide_parameters=True)`) always sets a bool, so the
            # production gate is unaffected; the strict check closes a
            # test-injection vector flagged by the round-39 adversarial
            # threat-model.
            if engine.sync_engine.hide_parameters is not True:
                raise RuntimeError(
                    "engine_factory returned an engine without hide_parameters=True; "
                    "production engines MUST strip bound parameter values from "
                    "exception strings per DECISIONS#016 logs-stay-metadata-only "
                    "(strict `is True` check — non-bool truthy values rejected)"
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

            # Step 5: ModelConfig built ONCE here and shared with both
            # the provider (step 5b) AND build_graph (step 8). Reading
            # OUTRIDER_MODEL_* twice would defeat the lifespan's
            # "validated once, reused" guarantee.
            model_config = ModelConfig()

            # Step 5b: provider; lifespan teardown awaits aclose.
            provider = provider_factory(persister, model_config)
            stack.push_async_callback(provider.aclose)

            # Step 6: GitHub App settings (env-driven). Reads
            # OUTRIDER_GITHUB_APP_ID + _APP_PRIVATE_KEY + _WEBHOOK_SECRET.
            # The webhook router reads `webhook_secret.get_secret_value()`
            # at the verify_signature call site (not here at construction).
            github_app_settings = GitHubAppSettings()

            # Step 7: github_factory — per-installation `GitHub` client
            # factory bound to the lifespan-validated `github_app_settings`.
            # Per `DECISIONS.md#020` + `nodes-receive-deps-via-closure`,
            # minting happens at intake call-site, not at webhook receipt.
            # The settings object is closed over once here; each call to
            # `github_factory(iid)` reads `.app_private_key.get_secret_value()`
            # at the call site so the PEM is in plain memory only briefly.
            # The settings-bound factory pattern routes any env-var change
            # through the next lifespan restart — a bare-function binding
            # would re-instantiate `GitHubAppSettings()` per call and
            # defeat the env-validation gate at startup.
            github_factory = make_installation_client_factory(github_app_settings)

            # Step 8: build the compiled graph with all six deps injected
            # at construction time. `db_factory` is the canonical first
            # parameter per `docs/spec.md §9.3`; the order here mirrors
            # the spec's signature. `model_config` is the SAME instance
            # already passed to the provider at step 5b — single-source
            # guarantee.
            compiled_graph = build_graph(
                provider=provider,
                model_config=model_config,
                phase_event_sink=persister,
                file_examination_sink=persister,
                db_factory=session_factory,
                github_factory=github_factory,
            )

            # Step 9: `run_graph` closure for the V1 dispatcher to call
            # from BackgroundTasks. The dispatcher itself is per-request
            # (built via FastAPI Depends in the webhook handler); the
            # graph is lifespan-bound.
            async def run_graph(state: Any) -> Any:
                return await compiled_graph.ainvoke(state)

            # Step 10: re-apply log filter post-handler-registration.
            # uvicorn registers its handlers before the lifespan body
            # runs, so by here all handler chains exist; idempotent
            # `register_filter_on_all_handlers()` adds the filter to any
            # newly-registered handler missing it.
            register_filter_on_all_handlers()

            # Stash deps on app.state so request handlers can resolve
            # them via FastAPI's dependency-injection system.
            app.state.engine = engine
            app.state.session_factory = session_factory
            app.state.retention_settings = retention_settings
            app.state.persister = persister
            app.state.provider = provider
            app.state.github_app_settings = github_app_settings
            app.state.github_factory = github_factory
            app.state.compiled_graph = compiled_graph
            app.state.run_graph = run_graph

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
