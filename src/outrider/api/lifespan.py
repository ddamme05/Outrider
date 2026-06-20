# See specs/2026-05-16-audit-persister.md.
"""FastAPI lifespan — durable dependency construction + teardown.

Constructs at startup, in dependency order:

  1. `AsyncEngine` from `DATABASE_URL` env var.
  1b. Severity-policy fingerprint check: compares the DB row at
     `severity_policies.version=ACTIVE_POLICY_VERSION` to the live
     `SEVERITY_POLICY` mapping; raises `StartupError` on miss or
     mismatch BEFORE the rest of dependency wiring, so a drifted
     policy never initializes downstream consumers (per §0c of
     specs/2026-05-19-analyze-foundation.md).
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
  8. `compiled_graph = build_graph(...)` — the V1 seven-node intake →
     triage → analyze ⇄ trace → synthesize → hitl → publish graph with
     all required deps injected at construction time (`db_factory`,
     `github_factory`, `provider`, `model_config`, eight audit-side sink
     Protocols + one anomaly sink, `publisher`, and
     `import_path_resolver`).
  9. `run_graph` async closure that the V1 `BackgroundTasksDispatcher`
     invokes per request to call `compiled_graph.ainvoke(state)`, wrapped
     by a process-level `asyncio.Semaphore` (`OUTRIDER_MAX_CONCURRENT_REVIEWS`,
     default 8) so a webhook flood can't saturate the Anthropic pool
     (FUP-164 / DECISIONS.md#045).
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

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr, ValidationError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from outrider.agent.graph import build_graph
from outrider.audit.config import RetentionSettings
from outrider.audit.persister import AuditPersister
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER
from outrider.github.auth import make_installation_client_factory
from outrider.github.config import GitHubAppSettings
from outrider.github.publisher import GitHubKitPublisher
from outrider.llm.anthropic_provider import AnthropicProvider
from outrider.llm.base import LLMProvider
from outrider.llm.config import ModelConfig
from outrider.llm.logging import register_filter_on_all_handlers
from outrider.llm.tracing import wrap_provider_if_tracing
from outrider.notify.config import SlackOAuthSettings
from outrider.policy.severity import ACTIVE_POLICY_VERSION, SEVERITY_POLICY
from outrider.policy.versions import (
    UnknownPolicyVersionError,
    load_policy_for_version,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["StartupError", "build_lifespan", "lifespan"]


_LOGGER = logging.getLogger("outrider.api.lifespan")


# OAuth env vars whose PRESENCE (even empty) signals "Slack OAuth intended".
_SLACK_OAUTH_VARS = (
    "OUTRIDER_SLACK_CLIENT_ID",
    "OUTRIDER_SLACK_CLIENT_SECRET",
    "OUTRIDER_SLACK_REDIRECT_URI",
)


def _load_slack_oauth_settings() -> SlackOAuthSettings | None:
    """Construct `SlackOAuthSettings` when any OAuth env var is present, else None.

    Slack is OPTIONAL, so a present-but-invalid OAuth config (partial / empty / short)
    is logged and DISABLED — never fatal. A misconfigured optional integration must not
    block app startup or the core PR-review pipeline; only the `/slack/*` routes go
    dark (their uniform 503). The loud ERROR log (not a silent None) preserves the
    original "don't hide a typo" intent while scoping the failure to Slack. ALL OAuth
    vars unset → None (Slack simply not configured, no log). The state secret +
    token-encryption key are read at their call sites, not here.
    """
    if not any(var in os.environ for var in _SLACK_OAUTH_VARS):
        return None
    try:
        return SlackOAuthSettings()
    except ValidationError as exc:
        _LOGGER.error(
            "Slack OAuth config is present but invalid — the Slack install flow is "
            "DISABLED; the rest of Outrider is unaffected. Fix or unset the "
            "OUTRIDER_SLACK_* OAuth vars. Details: %s",
            exc,
        )
        return None


class StartupError(RuntimeError):
    """Raised when a lifespan startup gate refuses to allow the app to start.

    Subclasses RuntimeError so uvicorn's default startup-failure path
    surfaces it cleanly. The current sole site is the severity-policy
    fingerprint check (§0c per specs/2026-05-19-analyze-foundation.md);
    additional gates may raise this exception as they are added.
    """


async def _verify_severity_policy_fingerprint(engine: AsyncEngine) -> None:
    """Compare DB-stored policy at ACTIVE_POLICY_VERSION to live SEVERITY_POLICY.

    Raises StartupError on miss (no row for ACTIVE_POLICY_VERSION) or
    mismatch (row exists but doesn't equal the live mapping). Closes the
    partial-deploy / drift window per §0c:

      (a) edited SEVERITY_POLICY but forgot to bump ACTIVE_POLICY_VERSION
          and add a migration → DB row at the constant's version differs
          from the live mapping → mismatch raises.
      (b) bumped ACTIVE_POLICY_VERSION but forgot the migration → no row
          exists at the constant's version → UnknownPolicyVersionError
          raises, surfaced as StartupError.
      (c) bumped constant + landed migration but live SEVERITY_POLICY in
          source still has the old mapping → DB row matches the new
          version but differs from the stale live mapping → mismatch
          raises.

    Fails LOUD at lifespan, BEFORE accepting webhooks; without this check
    the divergence would silently surface as wrong severities on findings
    until a replay caught it.
    """
    # READ COMMITTED is sufficient: only the row at ACTIVE_POLICY_VERSION
    # matters; concurrent INSERTs of NEW versions don't affect this check.
    async with engine.connect() as conn:
        try:
            db_policy = await load_policy_for_version(ACTIVE_POLICY_VERSION, conn)
        except UnknownPolicyVersionError as e:
            raise StartupError(
                f"ACTIVE_POLICY_VERSION={ACTIVE_POLICY_VERSION!r} has no row in "
                f"severity_policies. A migration adding this version must run before "
                f"app startup."
            ) from e

    live_policy = dict(SEVERITY_POLICY)
    if db_policy != live_policy:
        # Include a per-key diff so the operator can distinguish case
        # (a) keys differ (forgot the migration / wrong constant) from
        # case (c) values differ (live mapping is stale relative to the
        # seeded row).
        live_keys = set(live_policy)
        db_keys = set(db_policy)
        only_live = live_keys - db_keys
        only_db = db_keys - live_keys
        value_diffs = {
            k: (live_policy[k], db_policy[k])
            for k in live_keys & db_keys
            if live_policy[k] != db_policy[k]
        }
        raise StartupError(
            f"Policy drift detected: ACTIVE_POLICY_VERSION={ACTIVE_POLICY_VERSION!r} "
            f"loads a DB policy that differs from the live SEVERITY_POLICY mapping. "
            f"Diff: only-in-live={sorted(only_live)}, only-in-db={sorted(only_db)}, "
            f"value-mismatches={value_diffs}. "
            f"Either the constant was bumped without a matching migration, or the "
            f"migration ran but the live mapping is stale. Refusing to start."
        )


# ---------------------------------------------------------------------------
# Test seams: factories that production resolves from env vars; tests can
# replace each via `build_lifespan(engine_factory=..., provider_factory=...)`.
# ---------------------------------------------------------------------------


EngineFactory = Callable[[], AsyncEngine]
# `CheckpointerFactory` is the seam for the durable LangGraph
# checkpointer. Production: `AsyncPostgresSaver.from_conn_string(url)`
# returning an async context manager that `AsyncExitStack` enters.
# Tests pass an `InMemorySaver`-yielding factory so no real DB
# connection is attempted. Defaults to the real factory.
CheckpointerFactory = Callable[[], "AbstractAsyncContextManager[Any]"]
# `ProviderFactory` accepts the lifespan-built `ModelConfig` so the
# provider and the compiled graph share ONE instance. The prior shape
# (`Callable[[AuditPersister], AnthropicProvider]`) silently constructed
# two independent `ModelConfig()` instances — one inside the factory,
# one in the lifespan body for `build_graph(...)`. Env-driven settings
# read once at the lifespan boundary must stay single-instance through
# the whole graph; constructing the same Settings class twice defeats
# the "lifespan-validated once" guarantee.
ProviderFactory = Callable[[AuditPersister, ModelConfig], LLMProvider]
# `SeverityPolicyFingerprintCheck` is the injectable seam for §0c's
# fingerprint check. Production runs `_verify_severity_policy_fingerprint`
# (real DB query against severity_policies); lifespan tests that inject a
# MagicMock engine (no DB connection) pass a no-op via this seam. The
# fingerprint-behavior tests (`test_lifespan_startup_fingerprint.py`)
# use a real `migrated_db` engine + the default check.
SeverityPolicyFingerprintCheck = Callable[[AsyncEngine], "Awaitable[None]"]


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
    the + hardening of `anthropic_provider.py` no
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


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    """Teardown helper for the lifespan-scoped sweep task.

    Cancels the task and awaits it with the CancelledError suppressed
    — the asyncio convention for cooperative shutdown. Pushed onto
    `AsyncExitStack` so it runs in LIFO order alongside provider /
    engine teardown.
    """
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _default_checkpointer_factory() -> AbstractAsyncContextManager[Any]:
    """Production checkpointer: AsyncPostgresSaver bound to DATABASE_URL.

    `AsyncPostgresSaver.from_conn_string` expects a bare psycopg URL.
    SQLAlchemy's `postgresql+psycopg://` driver-suffix is rejected by
    psycopg's URL parser, so the suffix is stripped.

    Returns the async context manager unentered — the lifespan body
    pushes it onto its `AsyncExitStack` and calls `.setup()` after
    entry to create checkpoint tables on first boot.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415

    checkpoint_url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    return AsyncPostgresSaver.from_conn_string(checkpoint_url)


def _default_provider_factory(
    persister: AuditPersister,
    model_config: ModelConfig,
) -> LLMProvider:
    """Production provider factory: reads ANTHROPIC_API_KEY env, constructs
    AnthropicProvider with the supplied (lifespan-built) `ModelConfig`, and
    applies LangSmith tracing at this composition-root boundary.

    Privacy startup notice fires inside the provider's `__init__` (per
    DECISIONS#015 point 4), once per lifespan startup. The `persister`
    arg is injected so the wrapper's fail-closed-pre-call gate is
    satisfied at construction. The `model_config` arg is injected so
    the provider and the compiled graph share ONE instance — the prior
    shape constructed an independent `ModelConfig()` here AND a separate
    one in the lifespan body for `build_graph(...)`, defeating the
    single-source guarantee.

    Tracing is applied HERE, not inside the provider (DECISIONS.md#035):
    `wrap_provider_if_tracing` returns the provider wrapped in a
    `TracingLLMProvider` when tracing is enabled, else the provider unchanged.
    The "is tracing on?" decision lives once, at this construction site — the
    concrete provider stays tracing-agnostic. Returns `LLMProvider` because the
    return is the (possibly wrapped) Protocol, not the concrete provider.
    """
    try:
        api_key_raw = os.environ["ANTHROPIC_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(
            "ANTHROPIC_API_KEY env var is required for the FastAPI lifespan."
        ) from exc
    provider = AnthropicProvider(
        api_key=SecretStr(api_key_raw),
        model_config=model_config,
        persister=persister,
    )
    return wrap_provider_if_tracing(provider)


# ---------------------------------------------------------------------------
# Lifespan builder.
# ---------------------------------------------------------------------------


def build_lifespan(
    *,
    engine_factory: EngineFactory = _default_engine_factory,
    provider_factory: ProviderFactory = _default_provider_factory,
    severity_policy_fingerprint_check: SeverityPolicyFingerprintCheck = (
        _verify_severity_policy_fingerprint
    ),
    checkpointer_factory: CheckpointerFactory = _default_checkpointer_factory,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Construct a FastAPI lifespan callable with injectable factories.

    Production callers use the module-level `lifespan` constant
    (`build_lifespan()` with defaults). Tests pass `engine_factory=` and
    `provider_factory=` to inject mocks — the teardown-ordering test
    injects a provider whose `aclose()` raises; the filter-re-registration
    test patches the engine factory to return a mock engine; etc.

    `severity_policy_fingerprint_check` is the §0c seam: defaults to the
    real DB-query check (`_verify_severity_policy_fingerprint`); tests
    that inject a MagicMock engine (no DB connection) pass an async
    no-op. The fingerprint-behavior tests use a real engine + the default.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with AsyncExitStack() as stack:
            # Step 0: validate the truncation-marker HMAC secret is present. It is
            # read LAZILY inside `apply_size_cap` (only when a finding body actually
            # truncates), so a deploy missing OUTRIDER_TRUNCATION_HMAC_SECRET boots
            # clean and reviews short PRs fine, then crashes the whole publish node
            # mid-review the first time any finding body exceeds the size cap (the
            # per-finding routing loop has no recovery wrapper). It is required on the
            # core publish path of EVERY review (not optional like Slack), so fail loud
            # at boot — same posture as DATABASE_URL / ANTHROPIC_API_KEY. Pure env read,
            # no DB dependency, so it runs first, before any resource is allocated.
            from outrider.policy.output_sanitizer import (  # noqa: PLC0415
                require_truncation_secret,
            )

            require_truncation_secret()

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
            # strict `is not True` check below. Real misconfigured
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
            # test-injection vector flagged by the adversarial
            # threat-model.
            if engine.sync_engine.hide_parameters is not True:
                raise RuntimeError(
                    "engine_factory returned an engine without hide_parameters=True; "
                    "production engines MUST strip bound parameter values from "
                    "exception strings per DECISIONS#016 logs-stay-metadata-only "
                    "(strict `is True` check — non-bool truthy values rejected)"
                )

            # Step 1b: severity-policy fingerprint check. Compares the DB
            # row at `severity_policies.version=ACTIVE_POLICY_VERSION` to
            # the live `SEVERITY_POLICY` mapping. Raises `StartupError`
            # on miss or mismatch BEFORE the rest of dependency wiring,
            # so a drifted policy never initializes downstream consumers
            # (provider, build_graph). Per §0c of
            # specs/2026-05-19-analyze-foundation.md.
            #
            # Runs AFTER engine construction because the check requires a
            # live DB connection; runs BEFORE persister/provider so a
            # drifted policy short-circuits before any LLM/audit wiring
            # exists. A refactorer tempted to move this above Step 1
            # thinking "fail loudest first" would break the contract —
            # the engine must exist for the check to run. Injectable via
            # `severity_policy_fingerprint_check=` so MagicMock-engine
            # lifespan tests can bypass with a no-op (§0c devex M-4).
            await severity_policy_fingerprint_check(engine)

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

            # Step 8: build the compiled graph with all deps injected
            # at construction time. `db_factory` is the canonical first
            # parameter per `docs/spec.md §9.3`. `model_config` is the
            # SAME instance already passed to the provider at step 5b —
            # single-source guarantee. The same `persister` implements
            # seven audit-side sink Protocols (phase + file-examination +
            # analyze + publish + trace + hitl + synthesize) plus the
            # `LLMExchangePersister` Protocol, all from one class.
            # `ReviewStatusPersister` and `AnomalyPersister` are
            # separate concrete classes (different table-lifecycle vs
            # audit-event semantics); they're constructed alongside
            # AuditPersister and passed to `build_graph(...)`
            # individually. Routing/eligibility split per
            # DECISIONS.md #023;
            # `import_path_resolver` is the stateless coordinates
            # singleton; `publisher` is the stateless GitHubKitPublisher.
            # HITL dependencies: the durable AuditPersister implements
            # HITLEventSink via emit_hitl_request + emit_hitl_decision;
            # ReviewStatusPersister is a separate concrete class for the
            # reviews-table lifecycle writes; HITLConfig is env-driven
            # via pydantic-settings (OUTRIDER_HITL_TIMEOUT_MINUTES +
            # OUTRIDER_HITL_TIMEOUT_ACTION). Startup raises ValueError
            # if timeout_action is set to anything other than
            # `expire_only` — V1's hitl-gates-high-severity guarantee.
            from outrider.agent.nodes.cache_config import CacheConfig  # noqa: PLC0415
            from outrider.agent.nodes.hitl_config import HITLConfig  # noqa: PLC0415
            from outrider.agent.nodes.patch_config import PatchConfig  # noqa: PLC0415
            from outrider.db.review_status_persister import ReviewStatusPersister  # noqa: PLC0415
            from outrider.dispatcher import (  # noqa: PLC0415
                DispatchConfig,
                concurrency_limited,
            )

            review_status_persister = ReviewStatusPersister(session_factory=session_factory)
            hitl_config = HITLConfig()  # reads env vars; fails loud on AUTO_POST
            # Suggested patches (DECISIONS.md#040): reads OUTRIDER_PATCHES_ENABLED
            # (default on) + the per-review cap. Patch model is ModelConfig.patch_model.
            patch_config = PatchConfig()
            # Analyze-cache read mode (Stage B serve flip): reads
            # OUTRIDER_CACHE_MODE (default `shadow` — behavior-neutral; the flip
            # to `serve` is a deliberate, telemetry-gated config change).
            cache_config = CacheConfig()
            # Concurrent-review ceiling (FUP-164 / DECISIONS.md#045): reads
            # OUTRIDER_MAX_CONCURRENT_REVIEWS (default 8). The semaphore is
            # created HERE (inside the running event loop, so it binds to the
            # right loop) and wraps `run_graph` below so a webhook flood can't
            # saturate the shared Anthropic connection pool. Per-process bound
            # (real ceiling under N workers is N x the limit) — see #045.
            dispatch_config = DispatchConfig()
            review_semaphore = asyncio.Semaphore(dispatch_config.max_concurrent_reviews)
            # Dashboard settings (bearer keys + the public base URL), instantiated
            # ONCE here so the publish-node `dashboard_base_url` injection (build_graph
            # below) and the app.state auth setup (later) share one validated read —
            # the "validated once, reused" discipline (cf. ModelConfig). Fail-loud on a
            # missing OUTRIDER_ADMIN_API_KEY surfaces here at startup, just earlier.
            from outrider.api.dashboard.config import DashboardSettings  # noqa: PLC0415

            _dashboard_settings = DashboardSettings()

            # Step 7b: durable LangGraph checkpointer. HITL `interrupt(...)`
            # writes the suspended state to this checkpointer; the
            # `/decide` endpoint's `Command(resume=...)` reads it back
            # via the same checkpointer in a separate FastAPI handler
            # (potentially a separate process under V2 Celery dispatch).
            # Without a checkpointer the suspended state lives in memory
            # only and dies with the process — the HITL durability
            # contract (specs/2026-05-26-hitl-node.md "interrupt + resume
            # crash-replay windows") collapses.
            #
            # `checkpointer_factory` is the test seam: production default
            # constructs an `AsyncPostgresSaver` against DATABASE_URL;
            # tests inject a factory yielding an `InMemorySaver` to
            # avoid the real DB connection.
            checkpointer_cm = checkpointer_factory()
            checkpointer = await stack.enter_async_context(checkpointer_cm)
            # `.setup()` is the AsyncPostgresSaver bootstrap; the
            # InMemorySaver does not expose it. Call only when present.
            setup = getattr(checkpointer, "setup", None)
            if setup is not None:
                await setup()

            # Construct AnomalyPersister before build_graph: synthesize
            # is the first in-graph anomaly emitter (sweep was the only
            # prior emitter). Same instance is reused by the sweep loop
            # below.
            from outrider.anomaly.persister import (  # noqa: PLC0415
                AnomalyPersister,
            )

            anomaly_persister = AnomalyPersister(session_factory=session_factory)

            # Analyze-cache store (lever #8, Stage B shadow). Store-or-None
            # IS the enable switch per the spec — production wires the
            # store so shadow telemetry (CacheLookupEvent + cache writes)
            # accrues; the eval driver defaults to None unless a cache eval
            # scenario wires its own store. An eval review with a wired store
            # reads/writes scoped to is_eval rows via the lookup's is_eval
            # predicate (DECISIONS.md#046) — isolated from production rows, not
            # bypassed.
            from outrider.cache import AnalyzeCacheStore  # noqa: PLC0415

            analyze_cache_store = AnalyzeCacheStore(session_factory=session_factory)

            # Step 8b: per-install Slack resolver (commit 6.4c). Wired only when token
            # decryption is possible (OUTRIDER_TOKEN_ENC_KEY present) — without it no
            # stored bot token can be decrypted, so Slack posting is impossible and the
            # graph runs with resolve_slack_target=None (no per-review config lookup).
            # The resolver reads each install's Slack config, decrypts the token, and
            # builds a per-install orchestrator on `persister` (the SlackEventSink),
            # caching by (installation_id, ciphertext); registered for LIFO teardown so
            # its notifiers close on shutdown. Keeps cryptography/slack_sdk out of
            # agent/ (the graph holds only the resolver callable, FUP-186).
            from outrider.notify.resolver import PerInstallSlackResolver  # noqa: PLC0415
            from outrider.notify.token_crypto import (  # noqa: PLC0415
                TOKEN_ENC_KEY_ENV,
                TokenCryptoError,
                validate_token_enc_key,
            )

            # Gate on key VALIDITY, not mere presence: a present-but-invalid key (e.g.
            # an uncommented .env.example `replace-me`) would make Slack appear
            # configured while every decrypt fails. Validate once at boot; on failure
            # DISABLE Slack with a loud log rather than crash — Slack is optional and
            # must never block the core app (AUDIT M2).
            slack_resolver: PerInstallSlackResolver | None = None
            slack_token_enc_ok = False
            if TOKEN_ENC_KEY_ENV in os.environ:
                try:
                    validate_token_enc_key()
                    slack_token_enc_ok = True
                except TokenCryptoError as exc:
                    _LOGGER.error(
                        "%s is present but invalid — Slack notifications are DISABLED; "
                        "the rest of Outrider is unaffected. Fix or unset the key. "
                        "Details: %s",
                        TOKEN_ENC_KEY_ENV,
                        exc,
                    )
            if slack_token_enc_ok:
                slack_resolver = PerInstallSlackResolver(
                    session_factory=session_factory,
                    sink=persister,
                    dashboard_base_url=_dashboard_settings.dashboard_base_url or "",
                )
                stack.push_async_callback(slack_resolver.aclose)

            compiled_graph = build_graph(
                provider=provider,
                model_config=model_config,
                phase_event_sink=persister,
                file_examination_sink=persister,
                analyze_event_sink=persister,
                publish_event_sink=persister,
                trace_sink=persister,
                hitl_event_sink=persister,
                synthesize_event_sink=persister,
                review_status_sink=review_status_persister,
                anomaly_sink=anomaly_persister,
                hitl_config=hitl_config,
                patch_config=patch_config,
                checkpointer=checkpointer,
                publisher=GitHubKitPublisher(),
                import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
                db_factory=session_factory,
                github_factory=github_factory,
                analyze_cache_store=analyze_cache_store,
                cache_mode=cache_config.mode,
                dashboard_base_url=_dashboard_settings.dashboard_base_url,
                resolve_slack_target=slack_resolver,
            )

            # Step 9: `run_graph` closure for the V1 dispatcher to call
            # from BackgroundTasks. The dispatcher itself is per-request
            # (built via FastAPI Depends in the webhook handler); the
            # graph is lifespan-bound.
            #
            # `thread_id=str(state.review_id)` is load-bearing: the
            # checkpointer keys per-thread, and HITL resume requires the
            # SAME thread_id to recover the suspended state. The review
            # row's UUID is the canonical identifier and is unique per
            # review.
            async def run_graph(state: Any) -> Any:
                from langchain_core.runnables import (  # noqa: PLC0415, TC002
                    RunnableConfig,
                )

                config: RunnableConfig = {"configurable": {"thread_id": str(state.review_id)}}
                return await compiled_graph.ainvoke(state, config=config)

            # FUP-164 / DECISIONS.md#045: bound concurrent graph executions at
            # the process level so an unbounded webhook flood can't saturate the
            # Anthropic connection pool. The semaphore wraps run_graph HERE (the
            # lifespan-bound closure), NOT in `BackgroundTasksDispatcher` — the
            # dispatcher is request-scoped (one per webhook), so a semaphore held
            # there could not bound across requests. Excess reviews await a free
            # slot as parked coroutines instead of all entering analyze at once.
            bounded_run_graph = concurrency_limited(run_graph, review_semaphore)

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
            # The dispatcher invokes the semaphore-bounded wrapper, not the bare
            # closure, so the FUP-164 concurrency ceiling applies to every
            # webhook-triggered review.
            app.state.run_graph = bounded_run_graph
            # Stash the checkpointer so the HITL-expiry sweep
            # (Group 8) can call `checkpointer.aget(config)` to detect
            # rows in `awaiting_approval` with no pending interrupt
            # (window (c) crash recovery per the HITL spec). The
            # `/decide` endpoint reaches the checkpointer transitively
            # via `compiled_graph.ainvoke(Command(resume=...))`, so it
            # doesn't need this binding — but the sweep does.
            app.state.checkpointer = checkpointer
            # Stash the ReviewStatusReader so the /decide endpoint
            # (Group 7) can preflight the HITLRequest from the
            # `reviews.hitl_request` JSONB cache without holding a
            # graph reference. Same instance as `review_status_sink`
            # because `ReviewStatusPersister` implements both
            # Protocols.
            app.state.review_status_reader = review_status_persister

            # Dashboard auth credentials, from the `_dashboard_settings` read above
            # (FastAPI `Authorization: Bearer ...` HMAC-compared in
            # `api/dashboard/auth.py::require_admin_api_key`; fail-loud on a missing
            # OUTRIDER_ADMIN_API_KEY already fired at the earlier instantiation).
            app.state.admin_api_key = _dashboard_settings.admin_api_key
            # Optional read-only agent token (feature 3 / S2). `None` when
            # `OUTRIDER_AGENT_API_KEY` is unset → the agent-view surface is
            # disabled (require_agent_api_key returns a uniform 401). Admin stays
            # fail-loud above; the agent key tolerates absence.
            app.state.agent_api_key = _dashboard_settings.agent_api_key

            # Slack OAuth install-flow config (commit 6.3c/6.3e). Opt-in + non-fatal:
            # present-but-invalid config disables Slack with a loud log, it does NOT
            # crash startup — Slack is optional, so it can never block the core app.
            # See `_load_slack_oauth_settings`.
            _slack_oauth_settings = _load_slack_oauth_settings()
            # Two-gate reconciliation (AUDIT L6): the OAuth callback ENCRYPTS the bot
            # token before persisting, so the install routes are useless without a valid
            # OUTRIDER_TOKEN_ENC_KEY. If OAuth is configured but the enc key is
            # missing/invalid, disable the routes with a loud log instead of letting the
            # admin walk the whole Slack consent screen only to 500 at token-persist.
            if _slack_oauth_settings is not None and not slack_token_enc_ok:
                _LOGGER.error(
                    "Slack OAuth is configured but %s is missing/invalid — the /slack/* "
                    "install routes are DISABLED (an install cannot persist its token "
                    "without a valid at-rest encryption key). Set %s to enable Slack.",
                    TOKEN_ENC_KEY_ENV,
                    TOKEN_ENC_KEY_ENV,
                )
                _slack_oauth_settings = None
            app.state.slack_oauth_settings = _slack_oauth_settings

            # Stash deps the sweep needs (anomaly_sink, audit_persister)
            # and start the periodic background task. Per
            # docs/spec.md §4.1.6, the HITL-expiry sweep enforces the
            # timeout window on a 5-minute cadence. Without this
            # task, HITL timeout enforcement + window-(c)/(f) crash
            # recovery is inert until an external scheduler invokes
            # `outrider.sweep.runner.run_all_sweeps` manually.
            #
            # APScheduler integration is intentionally out of scope
            # for V1 — a minimal asyncio-based scheduler keeps the
            # dep surface tight + matches the in-process lifespan
            # ownership model. Operators wanting a heavier scheduler
            # (cron, k8s CronJob, APScheduler) can disable this loop
            # via OUTRIDER_SWEEP_DISABLED=1 and run
            # `run_all_sweeps` externally.
            app.state.anomaly_sink = anomaly_persister
            app.state.audit_persister = persister

            sweep_task: asyncio.Task[None] | None = None
            # Accept the common truthy spellings — "1"/"true"/"yes" (any case) — so an
            # operator who writes OUTRIDER_SWEEP_DISABLED=true doesn't silently keep the
            # sweep running. (LANGSMITH_TRACING's literal-"true" parse is intentionally left
            # as-is — it matches LangSmith's own convention.)
            _sweep_disabled = os.environ.get("OUTRIDER_SWEEP_DISABLED", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if not _sweep_disabled:
                from outrider.api.lifespan_sweep_loop import (  # noqa: PLC0415
                    start_periodic_sweep,
                )

                sweep_task = start_periodic_sweep(
                    engine=engine,
                    session_factory=session_factory,
                    anomaly_sink=anomaly_persister,
                    review_status_sink=review_status_persister,
                    audit_persister=persister,
                    checkpointer=checkpointer,
                    compiled_graph=compiled_graph,
                )
                stack.push_async_callback(_cancel_task, sweep_task)
            app.state.sweep_task = sweep_task

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
