# See specs/2026-05-16-audit-persister.md.
# Planned under DECISIONS.md#056: host/profile selector + build_graph identity closure.
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
  6. `build_credential_provider(session_factory=..., env=...)` (DECISIONS.md#070)
     — `env` mode wraps a boot-validated `GitHubAppSettings()` (reads
     `OUTRIDER_GITHUB_APP_*` env vars); `database` mode reads the
     manifest-onboarded row and fails closed until CONFIGURED. `env`-mode
     validation still surfaces missing / malformed env as a friendly
     RuntimeError at boot, not a deep-stack `ValidationError` inside the
     first intake invocation.
  7. `github_factory = make_installation_client_factory(credential_provider)`
     — async per-installation `GitHub` client factory over the provider;
     each `await github_factory(iid)` resolves credentials lazily via
     `await provider.current()`. Per `DECISIONS.md#020` + the
     `nodes-receive-deps-via-closure` invariant, installation-token
     minting happens at intake call-site, not at webhook receipt.
  8. `compiled_graph = build_graph(...)` — the V1 seven-logical-node intake →
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

When `app.state.demo_mode` is set (the keyless public-demo boot), the
lifespan runs steps 1-4 only — engine, fingerprint check, session,
retention, persister — then wires the read-side `app.state`, yields, and
returns. The truncation secret and steps 5-10 (provider, GitHub App, graph,
run_graph, sweeps) are skipped; the demo box serves precomputed reviews
read-only and holds no live LLM/GitHub/Slack credentials. See the
`if demo_mode:` branch after step 4.

Teardown is `AsyncExitStack` LIFO — every push_async_callback runs even
if a prior callback raises. Closes FUP-006 (filter re-registration) and
FUP-011 (provider aclose).

The webhook router (`api/webhooks/router.py`) reads `app.state` bindings
to resolve per-request dependencies: `session_factory`, `retention_settings`,
`credential_provider` (webhook signature secret, resolved per request via
`await current()`), `github_factory`, `run_graph`. `compiled_graph`,
`persister`, `provider`, `engine` are also stashed for diagnostics / future routes.

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
from outrider.github.authz import make_installation_authorizer
from outrider.github.credentials import build_credential_provider
from outrider.github.publisher import GitHubKitPublisher
from outrider.llm.anthropic_provider import AnthropicProvider
from outrider.llm.base import LLMProvider
from outrider.llm.config import ModelConfig
from outrider.llm.host_profiles import (
    ANTHROPIC_PROFILE_ID,
    resolve_host_identity,
    resolve_host_profile,
)
from outrider.llm.logging import register_filter_on_all_handlers
from outrider.llm.openai_compatible_provider import OpenAICompatibleProvider
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
# `host` + `reasoning` are RESOLVED ONCE in the lifespan body (from OUTRIDER_LLM_HOST /
# OUTRIDER_LLM_REASONING) and passed IN — the factory must not re-read the env, or the
# single-authority guarantee breaks (#056, Codex guardrail). The same resolved host feeds
# `ModelConfig.for_host`, the factory's provider selection, and `resolve_host_identity`.
ProviderFactory = Callable[[AuditPersister, ModelConfig, str, bool], LLMProvider]
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


def _resolve_checkpoint_database_url() -> str:
    """Resolve the LangGraph checkpoint connection URL. See DECISIONS.md#068.

    `CHECKPOINT_DATABASE_URL` is OPTIONAL: when present and non-empty it is used;
    when absent it falls back to `DATABASE_URL`. An explicitly-set but empty or
    wrong-scheme value FAILS LOUD at startup — a set-but-broken value is operator
    error, not intent to use the default. Its `postgresql+psycopg://` scheme is
    validated independently, the same check `DATABASE_URL` gets in the engine factory.
    """
    raw = os.environ.get("CHECKPOINT_DATABASE_URL")
    if raw is None:
        # Absent → fall back to DATABASE_URL (its scheme is validated by the engine
        # factory, which runs before this at startup).
        return os.environ["DATABASE_URL"]
    stripped = raw.strip()
    if not stripped:
        raise RuntimeError(
            "CHECKPOINT_DATABASE_URL is set but empty. Unset it to fall back to "
            "DATABASE_URL, or set a valid 'postgresql+psycopg://' URL. Per DECISIONS.md#068 "
            "an explicitly-set-but-broken value fails loud rather than silently falling back."
        )
    if not stripped.startswith("postgresql+psycopg://"):
        raise RuntimeError(
            "CHECKPOINT_DATABASE_URL must use the canonical async driver scheme "
            "'postgresql+psycopg://' (psycopg3 async), matching DATABASE_URL. Unset it to "
            "fall back to DATABASE_URL. See DECISIONS.md#068 / #001."
        )
    return stripped


def _default_checkpointer_factory() -> AbstractAsyncContextManager[Any]:
    """Production checkpointer: AsyncPostgresSaver on the checkpoint DB.

    The connection URL is `CHECKPOINT_DATABASE_URL` when set, else `DATABASE_URL`
    (per DECISIONS.md#068; resolved by `_resolve_checkpoint_database_url`).
    `AsyncPostgresSaver.from_conn_string` expects a bare psycopg URL, so
    SQLAlchemy's `postgresql+psycopg://` driver-suffix is stripped.

    Returns the async context manager unentered — the lifespan body
    pushes it onto its `AsyncExitStack` and calls `.setup()` after
    entry to create checkpoint tables on first boot.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415

    from outrider.agent.checkpoint_serde import build_checkpoint_serde  # noqa: PLC0415

    # See DECISIONS.md#068: CHECKPOINT_DATABASE_URL when present, else DATABASE_URL.
    checkpoint_url = _resolve_checkpoint_database_url().replace(
        "postgresql+psycopg://", "postgresql://", 1
    )
    # FUP-220: register Outrider's state types so resume/replay survive strict-msgpack.
    return AsyncPostgresSaver.from_conn_string(checkpoint_url, serde=build_checkpoint_serde())


def _default_provider_factory(
    persister: AuditPersister,
    model_config: ModelConfig,
    host: str,
    reasoning: bool,
) -> LLMProvider:
    """Production provider factory: selects the provider for `host` (DECISIONS.md#056).

    `host` + `reasoning` are passed IN (resolved once in the lifespan body) — this
    factory never re-reads OUTRIDER_LLM_HOST, so host selection stays single-authority.
    `host == "anthropic"` → `AnthropicProvider` (reads `ANTHROPIC_API_KEY`); any other
    host → `OpenAICompatibleProvider` bound to `resolve_host_profile(host)`, reading the
    profile's declared `api_key_env`. The `model_config` (built via `for_host(host)`) is
    shared with the compiled graph so the provider and graph never diverge.

    Privacy startup notice fires inside the provider's `__init__` (DECISIONS#015 point 4).
    """
    provider: LLMProvider
    if host == ANTHROPIC_PROFILE_ID:
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
            reasoning=reasoning,
        )
    else:
        profile = resolve_host_profile(host)
        try:
            api_key_raw = os.environ[profile.api_key_env]
        except KeyError as exc:
            raise RuntimeError(
                f"{profile.api_key_env} env var is required for OUTRIDER_LLM_HOST={host!r}."
            ) from exc
        # The distinct configured model slugs (for_host filled them) — the provider
        # validates every request.model against this set before any paid call.
        models = tuple(
            sorted(
                {
                    model_config.triage_model,
                    model_config.analyze_model,
                    model_config.standard_analyze_model,
                    model_config.synthesize_model,
                    model_config.trace_model,
                    model_config.patch_model,
                }
            )
        )
        provider = OpenAICompatibleProvider(
            api_key=SecretStr(api_key_raw),
            profile=profile,
            persister=persister,
            models=models,
            reasoning=reasoning,
        )
    return provider


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
            # DEMO_MODE (stashed by create_app): a keyless, read-only boot that
            # serves precomputed seed reviews. It skips the truncation secret + the
            # entire review/write half (provider, GitHub App, graph, Slack, sweeps),
            # so a public demo box holds NO live LLM/GitHub/Slack credentials and
            # cannot call out. The `if demo_mode:` branch after the persister (the
            # last shared step) wires the read-side app.state, yields, and returns —
            # leaving the production path below untouched.
            demo_mode = bool(getattr(app.state, "demo_mode", False))

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

            # Skipped in demo mode: there is no publish/output-generation path that
            # could truncate a finding body, so the secret is never read.
            if not demo_mode:
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

            # ── DEMO_MODE: keyless read-only boot ──────────────────────────────
            # The public demo box holds NO live LLM/GitHub/Slack credentials. It
            # serves precomputed seed reviews through the read-only dashboard
            # allowlist (main.py `_include_routers`) and runs no reviews, so
            # everything below — provider, GitHub App, graph, checkpointer, Slack,
            # sweeps — is skipped. The shared steps above (engine, fingerprint,
            # session, retention, persister) are all the read routers need.
            #
            # Self-contained on purpose: to retire the demo, delete this block, the
            # `demo_mode` read + its truncation guard above, `app.state.demo_mode`
            # in create_app, and the allowlist branch in main.py. The production
            # path below is byte-identical whether or not this block runs.
            if demo_mode:
                from outrider.api.dashboard.config import (  # noqa: PLC0415
                    DashboardSettings,
                )

                _demo_dashboard_settings = DashboardSettings()
                # Read-side deps the dashboard resolves — real.
                app.state.engine = engine
                app.state.session_factory = session_factory
                app.state.retention_settings = retention_settings
                app.state.persister = persister
                app.state.audit_persister = persister
                # Auth: admin is fail-loud (the public read token); the agent token
                # follows its env — unset → None → the agent-view surface is
                # disabled (require_agent_api_key returns a uniform 401).
                app.state.admin_api_key = _demo_dashboard_settings.admin_api_key
                app.state.agent_api_key = _demo_dashboard_settings.agent_api_key
                # Review/write half — absent. Set to None (not unset) so a handler
                # that defensively reads one gets None, not AttributeError; no read
                # route in the demo allowlist consumes any of these.
                app.state.provider = None
                app.state.credential_provider = None
                app.state.github_factory = None
                app.state.compiled_graph = None
                app.state.run_graph = None
                app.state.checkpointer = None
                app.state.review_status_reader = None
                app.state.slack_oauth_settings = None
                app.state.anomaly_sink = None
                app.state.sweep_task = None
                # Re-apply the log-content filter (Step 10 in the full path).
                register_filter_on_all_handlers()
                _LOGGER.info(
                    "DEMO_MODE: keyless boot — read-only dashboard over seed data; "
                    "no provider/GitHub/graph/checkpointer/Slack/sweeps constructed."
                )
                yield
                return
            # ───────────────────────────────────────────────────────────────────

            # Step 5: host selection (DECISIONS.md#056). OUTRIDER_LLM_HOST +
            # OUTRIDER_LLM_REASONING are read ONCE here (single authority) and
            # threaded to ModelConfig.for_host, the provider factory, AND the
            # identity triad — so the model config, the provider's stamp, and
            # build_graph's completion-event closure all derive from one host.
            llm_host = os.environ.get("OUTRIDER_LLM_HOST", ANTHROPIC_PROFILE_ID).strip()
            llm_reasoning = os.environ.get("OUTRIDER_LLM_REASONING", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }

            # ModelConfig built ONCE and shared with both the provider (step 5b)
            # AND build_graph (step 8). for_host applies OUTRIDER_MODEL_* over the
            # host's defaults; reading it twice would defeat the "validated once"
            # guarantee.
            model_config = ModelConfig.for_host(llm_host)

            # Step 5b: provider; lifespan teardown awaits aclose.
            provider = provider_factory(persister, model_config, llm_host, llm_reasoning)
            stack.push_async_callback(provider.aclose)

            # Step 5c: the host-identity triad for build_graph's completion-event
            # closure. resolve_host_identity is the SAME source the provider stamps
            # from, so per-node completion events (zero-LLM cache-serve/skip paths)
            # carry the identical triad as the provider's per-call LLMCallEvents.
            profile_id, reasoning_enabled, profile_contract_digest = resolve_host_identity(
                llm_host, reasoning=llm_reasoning
            )

            # Step 6: the GitHub App CREDENTIAL PROVIDER (DECISIONS.md#070). `env` mode wraps the
            # boot-validated GitHubAppSettings triad (OUTRIDER_GITHUB_APP_ID + _APP_PRIVATE_KEY +
            # _WEBHOOK_SECRET); `database` mode reads the manifest-onboarded row and fails closed
            # (raises GitHubUnconfiguredError) until CONFIGURED — so this boots even before
            # onboarding. Built ONCE here; the four consumers resolve credentials LAZILY
            # per-operation via `await provider.current()`, so a `database`-mode activation takes
            # effect with NO restart.
            credential_provider = build_credential_provider(
                session_factory=session_factory, env=os.environ
            )

            # Step 7: github_factory — per-installation `GitHub` client factory over the provider.
            # Per `DECISIONS.md#020` + `nodes-receive-deps-via-closure`, minting happens at intake
            # call-site. Each `await github_factory(iid)` fetches a fresh credential snapshot and
            # reads `.app_private_key.get_secret_value()` at the call site so the PEM is in plain
            # memory only briefly. The webhook route is 503-gated while not CONFIGURED, so
            # github_factory is never called before credentials exist.
            github_factory = make_installation_client_factory(credential_provider)

            # Step 7b: live-authorization closure for the #065 intake gate, over the provider. Per
            # githubkit's reusing-client guidance (0.15.3), each authorization constructs and `async
            # with`-scopes a FRESH App-JWT client for its GET+POST pair (one shared client, closed
            # on exit) rather than leaking a new per-request client. There is NO long-lived shared
            # client
            # to enter into the AsyncExitStack: githubkit keeps its httpx client in a task-local
            # ContextVar, so a client entered in this lifespan task would be invisible to intake's
            # per-review task anyway. `make_installation_authorizer` yields the githubkit-free
            # `(installation_id, repo_id) -> LiveAuthResult` closure intake calls first.
            installation_authorizer = make_installation_authorizer(credential_provider)

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
            from outrider.agent.nodes.analyze_config import AnalyzeConfig  # noqa: PLC0415
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
            # Per-review analyze token budget (specs/2026-06-17-analyze-cost-fairness.md
            # Stage 0): reads OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS (default 200k).
            # Wired into build_graph below — before this, production silently used
            # the hardcoded DEFAULT_REVIEW_BUDGET_TOKENS because the value was never
            # passed.
            analyze_config = AnalyzeConfig()
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
                total_review_budget_tokens=analyze_config.review_budget_tokens,
                analyze_max_concurrency=analyze_config.max_concurrency,
                checkpointer=checkpointer,
                publisher=GitHubKitPublisher(),
                import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
                db_factory=session_factory,
                github_factory=github_factory,
                installation_authorizer=installation_authorizer,
                analyze_cache_store=analyze_cache_store,
                cache_mode=cache_config.mode,
                dashboard_base_url=_dashboard_settings.dashboard_base_url,
                resolve_slack_target=slack_resolver,
                # Host-identity triad (DECISIONS.md#056 step 4d): closes the triad
                # into the analyze + synthesize completion events and the analyze
                # cache key, so production reviews are host-qualified end-to-end.
                profile_id=profile_id,
                reasoning_enabled=reasoning_enabled,
                profile_contract_digest=profile_contract_digest,
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
            app.state.credential_provider = credential_provider
            # Setup state-machine startup repair (DECISIONS.md#070, spec F6): time out any stale
            # `CONVERTING` attempt a crash left mid-conversion → ORPHANED, so onboarding can be
            # retried. The machine was built + stashed at create_app (`api/setup/mount.py`) in
            # `database` mode ONLY; it resolves `app.state.session_factory` (set just above) lazily.
            # `env` mode / demo have no machine → skip.
            setup_machine = getattr(app.state, "setup_state_machine", None)
            if setup_machine is not None:
                await setup_machine.recover_stale_converting()
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
            # `outrider.sweep.runner.run_scheduled_tick` manually.
            #
            # APScheduler integration is intentionally out of scope
            # for V1 — a minimal asyncio-based scheduler keeps the
            # dep surface tight + matches the in-process lifespan
            # ownership model. Operators wanting a heavier scheduler
            # (cron, k8s CronJob, APScheduler) can disable this loop
            # via OUTRIDER_SWEEP_DISABLED=1 and run
            # `run_scheduled_tick` externally — NOT `run_all_sweeps`
            # directly, which would exclude the reconcile janitor and
            # its liveness-gating of the #012 install hard-delete
            # (#065/#012/#067).
            app.state.anomaly_sink = anomaly_persister
            app.state.audit_persister = persister

            sweep_task: asyncio.Task[None] | None = None
            # Accept the common truthy spellings — "1"/"true"/"yes" (any case) — so an
            # operator who writes OUTRIDER_SWEEP_DISABLED=true doesn't silently keep the
            # sweep running.
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
                    # Reconcile janitor (#065/#012/#067) rides the same loop, gated on the
                    # credential provider being CONFIGURED (#070). `app.state.credential_provider`
                    # is set in BOTH the demo (None) and non-demo branches; demo → None and
                    # `database`-unconfigured → the janitor self-skips.
                    provider=app.state.credential_provider,
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
