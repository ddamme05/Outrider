# See specs/2026-05-16-audit-persister.md.
# See DECISIONS.md#056: host/profile selector + build_graph identity closure.
"""Construct FastAPI dependencies at startup and close them at shutdown.

Startup proceeds through persistence, LLM host/provider (DECISIONS.md#056),
GitHub credentials (DECISIONS.md#070), graph dependencies, graph construction,
dispatch, and background sweeps. Request handlers resolve these dependencies
through app.state. Stage comments explain the required ordering.

Demo mode constructs persistence and read-side state only, then yields and
returns. It skips live review dependencies and the truncation-secret gate.

AsyncExitStack unwinds resources in LIFO order even when cleanup raises,
including provider closure (FUP-011). The log-content filter is reapplied after
handler registration (FUP-006).

Production uses the module-level lifespan. build_lifespan accepts factories
and a fingerprint-check override for tests.
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
    PRODUCTION_UNADMITTED_HOSTS,
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


# Any present OAuth env var, even empty, signals intended Slack configuration.
_SLACK_OAUTH_VARS = (
    "OUTRIDER_SLACK_CLIENT_ID",
    "OUTRIDER_SLACK_CLIENT_SECRET",
    "OUTRIDER_SLACK_REDIRECT_URI",
)


def _load_slack_oauth_settings() -> SlackOAuthSettings | None:
    """Load Slack OAuth settings when any OAuth env var is present.

    Absent config returns None without logging. Invalid config logs an error
    and disables the install routes (503), without blocking the core app.
    The state secret and token-encryption key are read at their call sites.
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
    """Refuse startup when the severity-policy fingerprint check fails.

    See specs/2026-05-19-analyze-foundation.md §0c.
    """


async def _verify_severity_policy_fingerprint(engine: AsyncEngine) -> None:
    """Compare the active DB policy with the live SEVERITY_POLICY mapping.

    A missing version or mismatched mapping raises StartupError before
    accepting webhooks. This catches partial deployments and policy drift
    per specs/2026-05-19-analyze-foundation.md §0c.
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
        # Report missing keys and changed values so operators can locate drift.
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
# Return an unentered context manager; tests can supply an InMemorySaver.
CheckpointerFactory = Callable[[], "AbstractAsyncContextManager[Any]"]
# Share the validated ModelConfig with the graph. Host and reasoning are resolved
# once in the lifespan; factories must not re-read their env vars (DECISIONS.md#056).
ProviderFactory = Callable[[AuditPersister, ModelConfig, str, bool], LLMProvider]
# Mock-engine tests can bypass the DB check; fingerprint tests use the default.
SeverityPolicyFingerprintCheck = Callable[[AsyncEngine], "Awaitable[None]"]


def _default_engine_factory() -> AsyncEngine:
    """Build a lazy async engine from the required DATABASE_URL.

    hide_parameters=True prevents bound content from appearing in SQLAlchemy
    exception strings (DECISIONS.md#016). The key-based RejectLLMContentFilter
    cannot catch content already rendered into a message (FUP-023).
    """
    try:
        database_url = os.environ["DATABASE_URL"]
    except KeyError as exc:
        raise RuntimeError(
            "DATABASE_URL env var is required for the FastAPI lifespan. "
            "See .env.example for the canonical postgres URL shape."
        ) from exc

    # Reject unsupported drivers at startup. psycopg3 is the only project async
    # driver; bare postgresql URLs select sync psycopg2 (DECISIONS.md#001).
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
    """Cancel and await the sweep task, suppressing its CancelledError."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _resolve_checkpoint_database_url() -> str:
    """Resolve the LangGraph checkpoint connection URL. See DECISIONS.md#068.

    Absent CHECKPOINT_DATABASE_URL falls back to DATABASE_URL. An explicit empty
    or wrong-scheme value raises; broken config must not silently use the default.
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
    """Return an unentered AsyncPostgresSaver context on the checkpoint DB.

    Resolve the URL per DECISIONS.md#068 and strip SQLAlchemy's driver suffix
    for psycopg. The lifespan enters the context, then sets up checkpoint tables.
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
    """Build the provider for the resolved host (DECISIONS.md#056).

    Anthropic uses ANTHROPIC_API_KEY; other hosts use their profile's api_key_env.
    Host, reasoning, and ModelConfig come from the lifespan's single validated
    read. Provider construction emits the privacy notice (DECISIONS#015 point 4).
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
        # Restrict paid calls to the distinct configured model slugs.
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
    """Build a lifespan callable with injectable dependency factories.

    Mock-engine tests pass an async no-op fingerprint check. Production uses
    the default DB check (specs/2026-05-19-analyze-foundation.md §0c).
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with AsyncExitStack() as stack:
            # create_app sets demo_mode for read-only access to precomputed reviews.
            demo_mode = bool(getattr(app.state, "demo_mode", False))

            # Truncation secret: validate before allocating resources. apply_size_cap
            # reads it lazily, so a missing secret would otherwise fail mid-publish.
            from outrider.policy.output_sanitizer import (  # noqa: PLC0415
                require_truncation_secret,
            )

            # Demo mode has no publish path that could need the secret.
            if not demo_mode:
                require_truncation_secret()

            # Persistence: register disposal before any engine validation can fail.
            engine = engine_factory()
            stack.push_async_callback(engine.dispose)
            # Injected factories must also hide bound parameters (DECISIONS.md#016).
            # Require the exact bool on the wrapped sync_engine; truthy strings or
            # mocks do not establish redaction. Raise explicitly so python -O cannot
            # remove the gate as it would an assert.
            # Test engines must set mock_engine.sync_engine.hide_parameters = True.
            if engine.sync_engine.hide_parameters is not True:
                raise RuntimeError(
                    "engine_factory returned an engine without hide_parameters=True; "
                    "production engines MUST strip bound parameter values from "
                    "exception strings per DECISIONS#016 logs-stay-metadata-only "
                    "(strict `is True` check — non-bool truthy values rejected)"
                )

            # Check policy drift after constructing its DB engine, before wiring
            # persister/provider/graph (specs/2026-05-19-analyze-foundation.md §0c).
            await severity_policy_fingerprint_check(engine)

            # Keep committed ORM attributes readable without a closed-session refresh.
            session_factory = async_sessionmaker(engine, expire_on_commit=False)

            # Retention settings are validated from the environment.
            retention_settings = RetentionSettings()

            # Durable persister.
            persister = AuditPersister(
                session_factory=session_factory,
                retention_settings=retention_settings,
            )

            # Demo: wire the read-only dashboard, then return before constructing
            # LLM/GitHub/graph/checkpointer/Slack/sweep dependencies. Keep this branch
            # self-contained so the demo can be retired independently of production;
            # its app.state key parity with production is tested.
            if demo_mode:
                from outrider.api.dashboard.config import (  # noqa: PLC0415
                    DashboardSettings,
                )

                _demo_dashboard_settings = DashboardSettings()
                # Persistence dependencies serve the read-only routes.
                app.state.engine = engine
                app.state.session_factory = session_factory
                app.state.retention_settings = retention_settings
                app.state.persister = persister
                app.state.audit_persister = persister
                # Admin auth is required; an absent agent token disables its view (401).
                app.state.admin_api_key = _demo_dashboard_settings.admin_api_key
                app.state.agent_api_key = _demo_dashboard_settings.agent_api_key
                # Explicit None preserves key parity and avoids AttributeError on reads.
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
                # Re-apply the log-content filter (as the full path does at the end).
                register_filter_on_all_handlers()
                _LOGGER.info(
                    "DEMO_MODE: keyless boot — read-only dashboard over seed data; "
                    "no provider/GitHub/graph/checkpointer/Slack/sweeps constructed."
                )
                yield
                return
            # LLM host: resolve once for ModelConfig, provider, and identity triad
            # (DECISIONS.md#056). Normalize casing before the admission gate.
            llm_host = os.environ.get("OUTRIDER_LLM_HOST", ANTHROPIC_PROFILE_ID).strip().lower()
            # openai-native-host: refuse unadmitted hosts before key lookup or
            # provider construction.
            # Eval/probe drivers construct providers outside this production gate.
            if llm_host in PRODUCTION_UNADMITTED_HOSTS:
                raise RuntimeError(
                    f"OUTRIDER_LLM_HOST={llm_host!r} is implemented and evaluable but NOT "
                    "production-admitted: the pre-ship refusal-normalization gate is unmet — "
                    "in json_object mode its refusal probes return completed empty-result "
                    "envelopes indistinguishable from a clean review, so no refusal "
                    "discriminator has been demonstrated. See DECISIONS.md#056. Use 'anthropic' "
                    "(default) or a GLM host (baseten/fireworks)."
                )
            llm_reasoning = os.environ.get("OUTRIDER_LLM_REASONING", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }

            # Share one validated model configuration between provider and graph.
            model_config = ModelConfig.for_host(llm_host)

            # Provider; lifespan teardown awaits aclose.
            provider = provider_factory(persister, model_config, llm_host, llm_reasoning)
            stack.push_async_callback(provider.aclose)

            # Use the provider's identity source for completion events, including
            # zero-call cache-serve and skip paths.
            profile_id, reasoning_enabled, profile_contract_digest = resolve_host_identity(
                llm_host, reasoning=llm_reasoning
            )

            # GitHub credentials (DECISIONS.md#070): env mode validates at boot;
            # database mode permits boot before onboarding but fails closed on use
            # until CONFIGURED. Per-operation current() reads activate changes
            # without a restart.
            credential_provider = build_credential_provider(
                session_factory=session_factory, env=os.environ
            )

            # Mint installation tokens at intake, using a fresh credential snapshot
            # and reading the PEM at the call site (DECISIONS.md#020,
            # nodes-receive-deps-via-closure). Unconfigured webhooks are gated with 503.
            github_factory = make_installation_client_factory(credential_provider)

            # Live authorization (DECISIONS.md#065): each intake call scopes one
            # App-JWT client to its GET+POST pair. githubkit's task-local client
            # cannot be shared from the lifespan task into per-review tasks.
            installation_authorizer = make_installation_authorizer(credential_provider)

            # Graph dependencies: keep review-table lifecycle and anomaly storage
            # separate from audit-event persistence (docs/spec.md §9.3,
            # DECISIONS.md#023). Configuration is validated here, then bound below.
            # HITLConfig requires expire_only to preserve hitl-gates-high-severity.
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
            # Suggested-patch enablement and cap (DECISIONS.md#040).
            patch_config = PatchConfig()
            # Review budget and analyze concurrency
            # (specs/2026-06-17-analyze-cost-fairness.md).
            analyze_config = AnalyzeConfig()
            # Cache reads default to shadow; enabling serve requires a deliberate
            # config change gated by telemetry.
            cache_config = CacheConfig()
            # Create the per-process review semaphore inside its event loop
            # (FUP-164 / DECISIONS.md#045). N workers permit N times this limit.
            dispatch_config = DispatchConfig()
            review_semaphore = asyncio.Semaphore(dispatch_config.max_concurrent_reviews)
            # Share one validated dashboard config between graph links and route auth.
            from outrider.api.dashboard.config import DashboardSettings  # noqa: PLC0415

            _dashboard_settings = DashboardSettings()

            # Checkpointer: persist HITL interrupts so /decide can resume suspended
            # state after a crash (specs/2026-05-26-hitl-node.md). Tests can inject
            # an InMemorySaver; production uses the checkpoint DB (DECISIONS.md#068).
            checkpointer_cm = checkpointer_factory()
            checkpointer = await stack.enter_async_context(checkpointer_cm)
            # Bootstrap Postgres tables; in-memory test savers have no setup method.
            setup = getattr(checkpointer, "setup", None)
            if setup is not None:
                await setup()

            # Share anomaly persistence between the graph and background sweeps.
            from outrider.anomaly.persister import (  # noqa: PLC0415
                AnomalyPersister,
            )

            anomaly_persister = AnomalyPersister(session_factory=session_factory)

            # Cache store presence enables lookups and writes, including shadow
            # telemetry. Wired eval stores isolate is_eval rows (DECISIONS.md#046);
            # eval drivers otherwise default to no store.
            from outrider.cache import AnalyzeCacheStore  # noqa: PLC0415

            analyze_cache_store = AnalyzeCacheStore(session_factory=session_factory)

            # Slack resolver: wire only with a valid token-encryption key. The
            # resolver owns per-install token decryption and notifier cleanup,
            # keeping cryptography/slack_sdk outside agent/ (FUP-186).
            from outrider.notify.resolver import PerInstallSlackResolver  # noqa: PLC0415
            from outrider.notify.token_crypto import (  # noqa: PLC0415
                TOKEN_ENC_KEY_ENV,
                TokenCryptoError,
                validate_token_enc_key,
            )

            # Validate key content, not just presence. Invalid optional Slack config
            # logs an error and disables notifications without blocking the core app.
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

            # Compiled graph: bind the prepared dependencies.
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
                # Qualify completion events and cache keys with the resolved host
                # identity (DECISIONS.md#056 step 4d).
                profile_id=profile_id,
                reasoning_enabled=reasoning_enabled,
                profile_contract_digest=profile_contract_digest,
            )

            # Dispatch: request-scoped dispatchers invoke this lifespan-bound graph.
            # Use the review UUID as thread_id so HITL resume recovers the same state.
            async def run_graph(state: Any) -> Any:
                from langchain_core.runnables import (  # noqa: PLC0415, TC002
                    RunnableConfig,
                )

                config: RunnableConfig = {"configurable": {"thread_id": str(state.review_id)}}
                return await compiled_graph.ainvoke(state, config=config)

            # Bound graph executions across requests to protect the provider pool
            # (FUP-164 / DECISIONS.md#045). A request-scoped dispatcher semaphore
            # cannot enforce this shared limit; excess reviews wait for a slot.
            bounded_run_graph = concurrency_limited(run_graph, review_semaphore)

            # Reapply the filter after uvicorn registers handlers (idempotent, FUP-006).
            register_filter_on_all_handlers()

            # Request handlers resolve the prepared dependencies through app.state.
            app.state.engine = engine
            app.state.session_factory = session_factory
            app.state.retention_settings = retention_settings
            app.state.persister = persister
            app.state.provider = provider
            app.state.credential_provider = credential_provider
            # Repair stale CONVERTING onboarding attempts to ORPHANED for retry
            # (DECISIONS.md#070, spec F6). The database-mode machine from
            # api/setup/mount.py resolves the session_factory bound just above.
            setup_machine = getattr(app.state, "setup_state_machine", None)
            if setup_machine is not None:
                await setup_machine.recover_stale_converting()
            app.state.github_factory = github_factory
            app.state.compiled_graph = compiled_graph
            # Expose the bounded wrapper so every webhook review shares the limit.
            app.state.run_graph = bounded_run_graph
            # Sweeps inspect checkpoints for awaiting_approval rows without a pending
            # interrupt (HITL crash window (c)); /decide reaches them through the graph.
            app.state.checkpointer = checkpointer
            # /decide preflights the cached HITLRequest through the same review-status
            # instance used by the graph, without needing a graph reference.
            app.state.review_status_reader = review_status_persister

            # Route auth uses the dashboard settings already validated above.
            app.state.admin_api_key = _dashboard_settings.admin_api_key
            # An absent optional agent token disables the agent view with a uniform 401.
            app.state.agent_api_key = _dashboard_settings.agent_api_key

            # Slack OAuth: load optional install-flow settings without blocking startup.
            _slack_oauth_settings = _load_slack_oauth_settings()
            # OAuth callbacks encrypt tokens before persistence. Disable install routes
            # without a valid encryption key, before consent can end in a persist failure.
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

            # Background sweeps: enforce HITL timeouts and crash recovery on the
            # five-minute cadence (docs/spec.md §4.1.6). Operators can disable this
            # loop with OUTRIDER_SWEEP_DISABLED and schedule run_scheduled_tick.
            # Do not call run_all_sweeps directly: it omits the reconcile janitor
            # and its install-deletion liveness gate (#065/#012/#067).
            app.state.anomaly_sink = anomaly_persister
            app.state.audit_persister = persister

            sweep_task: asyncio.Task[None] | None = None
            # Accept common truthy spellings regardless of case.
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
                    # Reconciliation self-skips until credentials are CONFIGURED (#070).
                    provider=app.state.credential_provider,
                )
                stack.push_async_callback(_cancel_task, sweep_task)
            app.state.sweep_task = sweep_task

            # Log only the driver scheme; a full URL leaks credentials that the
            # key-based RejectLLMContentFilter would not catch in a dsn field.
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

        # AsyncExitStack attempts every registered cleanup in LIFO order, with
        # engine.dispose last. A cleanup failure propagates after unwinding.
        _LOGGER.info("outrider.api.lifespan teardown complete")

    return _lifespan


# Module-level lifespan for production wiring: `app = FastAPI(lifespan=lifespan)`.
lifespan = build_lifespan()
