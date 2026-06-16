#!/usr/bin/env python3
# ================================================================
#  Outrider — live GitHub demo (real LLM · REAL GitHub · real DB)
# ================================================================
"""Direct-invoke the real 7-node graph against a REAL GitHub PR.

This is step C2 of `docs/live-verification.md`: it swaps the two faked
boundaries of `scripts/live_claude_smoke.py` (stub GitHub client + recording
publisher) for the REAL ones the FastAPI lifespan builds — so intake fetches a
real PR diff via a minted installation token, and publish posts a real review
back to the PR. It bypasses ONLY the webhook receiver (no ingress); the graph,
LLM, GitHub read+write, persistence, and replay are all real.

What it proves (exit 0 = all structural checks hold):
  - a real installation token mints + fetches the real PR diff (intake)
  - real Claude drives triage/analyze/synthesize over that diff
  - a real review posts to GitHub (publish) UNLESS a CRITICAL/HIGH finding
    trips the HITL gate (which pauses before publish — the gate working)
  - the audit stream persists + reconstructs FULL (replay-equivalent)

Guardrails (per the demo constraints + smoke_publish.py precedent):
  - `--repo` is allowlisted; refuses anything but the sandbox repo unless
    `--allow-any-repo` is passed (a visible, deliberate override).
  - CRITICAL/HIGH findings hit the HITL gate (expire_only) — they do NOT
    auto-post. The run reports "PAUSED at HITL" and leaves the review
    awaiting approval rather than commenting on GitHub.

Run (needs ANTHROPIC_API_KEY + the GitHub App env + a reachable Postgres):
  op run --env-file=.env -- uv run python scripts/live_github_demo.py \\
    --owner ddamme05 --repo outrider-smoke-test --pr 1 --installation-id <ID>

Env required (read by GitHubAppSettings + AnthropicProvider + DB):
  OUTRIDER_GITHUB_APP_ID, OUTRIDER_GITHUB_APP_PRIVATE_KEY,
  OUTRIDER_GITHUB_WEBHOOK_SECRET, ANTHROPIC_API_KEY, DATABASE_URL.

Exit codes: 0 = all structural checks passed; 1 = a check failed; 2 = setup/
config error (missing env, DB unreachable, PR not found, or a review already
exists for this head_sha — push a new commit or clear the existing row).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Bare import: running `python scripts/live_github_demo.py` puts scripts/ at
# sys.path[0], so the sibling helper resolves without packaging scripts/.
from _narrate import (
    narrate_audit_stream,
    narrate_db_state,
    narrate_llm_exchanges_from_db,
    narrate_slack_notifications,
)
from _trace_log import TraceTee
from pydantic import SecretStr, ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# --- ensure src/ on path (mirror conftest's pythonpath=["src"]) ---
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from outrider.agent.graph import build_graph  # noqa: E402
from outrider.agent.nodes.hitl_config import HITLConfig  # noqa: E402
from outrider.agent.nodes.patch_config import PatchConfig  # noqa: E402
from outrider.anomaly.persister import AnomalyPersister  # noqa: E402
from outrider.audit.config import RetentionSettings  # noqa: E402
from outrider.audit.persister import AuditPersister  # noqa: E402
from outrider.audit.replay import AuditReplayer, ReplayMode  # noqa: E402
from outrider.cache import AnalyzeCacheStore  # noqa: E402
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER  # noqa: E402
from outrider.db.models.installations import (  # noqa: E402
    get_slack_config,
    set_slack_config,
)
from outrider.db.review_status_persister import ReviewStatusPersister  # noqa: E402
from outrider.github.auth import make_installation_client_factory  # noqa: E402
from outrider.github.config import GitHubAppSettings  # noqa: E402
from outrider.github.publisher import GitHubKitPublisher  # noqa: E402
from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from outrider.llm.config import ModelConfig  # noqa: E402
from outrider.notify.config import SlackSettings  # noqa: E402
from outrider.notify.resolver import PerInstallSlackResolver  # noqa: E402
from outrider.notify.token_crypto import (  # noqa: E402
    TOKEN_ENC_KEY_ENV,
    TokenCryptoError,
    encrypt_token,
)
from outrider.schemas.pr_context import PRContext  # noqa: E402

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_RULE = "=" * 62

# Hard-allowlisted sandbox target (mirrors spikes/publish/smoke_publish.py).
# Posting a real review to anything else requires the explicit
# --allow-any-repo override, which is visible in the command + this code.
_REPO_ALLOWLIST: frozenset[tuple[str, str]] = frozenset({("ddamme05", "outrider-smoke-test")})


class _ReviewAlreadyExistsError(Exception):
    """A reviews row already exists for the natural key (repo_id, pr_number,
    head_sha). C2 seeds a fresh review per run, so a same-head_sha re-run or a
    webhook-created row trips `uq_review_natural_key` — the same idempotency
    shape the webhook handles. Carries the existing row's id + status so the
    runner can point the operator at it.
    """

    def __init__(self, *, existing_review_id: str, status: str) -> None:
        super().__init__(f"review {existing_review_id} already exists (status={status})")
        self.existing_review_id = existing_review_id
        self.status = status


# Full trace tees to scripts/generated/ — shared recipe, scripts/_trace_log.py.
_TRACE: TraceTee | None = None


def _say(msg: str = "") -> None:
    print(msg, flush=True)
    if _TRACE is not None:
        _TRACE.write_line(msg)


class _SayLogHandler(logging.Handler):
    """Route `outrider.*` WARNING+ logs into the trace tee so SWALLOWED failures —
    notably best-effort Slack errors the orchestrator/nodes log-and-ignore — show in
    the terminal AND the scripts/generated/ trace file, not just on stderr. This is
    the "catch every silent failure" hook: a Slack post that fails is never fatal, so
    its only signal is the log line, and this makes that signal impossible to miss."""

    def emit(self, record: logging.LogRecord) -> None:
        for line in self.format(record).splitlines():
            _say(f"  [log:{record.levelname}] {line}")


async def _fetch_pr_seed(
    github_factory: Any,
    *,
    installation_id: int,
    owner: str,
    repo: str,
    pr_number: int,
) -> tuple[PRContext, int]:
    """Mint a real installation token and GET the PR; return (PRContext, repo_id).

    `changed_files=()` — intake enriches it with the real per-file diff via
    `github/fetch.py`, exactly as the webhook path does (DECISIONS.md#020).
    Only the PR-level metadata (title, shas, author, +/- counts) is read here.
    `repo_id` is the real GitHub repository numeric id (`pr.base.repo.id`),
    the same value the webhook reads from `payload.repository.id` — used as the
    reviews-row natural key, NOT a synthesized hash.

    DEMO-ONLY metadata fetch: the direct `gh.rest.pulls.async_get` call lives
    here (not in `github/fetch.py`) because this PR-level metadata read is a
    convenience for the direct-invoke harness only — the webhook path gets the
    same fields from the validated payload, and production per-file fetching
    stays in `github/fetch.py`. `gh` is the same installation-auth client the
    app uses (no raw `import githubkit`). If product code ever needs this
    metadata, promote a `get_pr_metadata` wrapper into `github/fetch.py` with
    tests (decision 2026-05-31).
    """
    gh = github_factory(installation_id)
    resp = await gh.rest.pulls.async_get(owner, repo, pr_number)
    pr = resp.parsed_data
    pr_context = PRContext(
        installation_id=installation_id,
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        pr_title=pr.title,
        pr_body=pr.body,
        base_sha=pr.base.sha,
        head_sha=pr.head.sha,
        author=pr.user.login if pr.user is not None else "unknown",
        total_additions=pr.additions,
        total_deletions=pr.deletions,
        changed_files=(),
    )
    return pr_context, pr.base.repo.id


async def _maybe_wire_slack(
    session_factory: async_sessionmaker[Any],
    persister: AuditPersister,
    pr_context: PRContext,
    *,
    seed_dev_token: bool,
) -> PerInstallSlackResolver | None:
    """Wire the per-install Slack resolver for the smoke (None → no Slack).

    DEFAULT (respecting DECISIONS.md#051): use the install's EXISTING per-install
    Slack config — the one the OAuth flow stored. The dev-bootstrap env token
    (`SlackSettings`/`OUTRIDER_SLACK_BOT_TOKEN`) is explicitly NOT the production
    posting authority, so we do not persist it into the real installations row by
    default. An unconnected install → no Slack (connect it via `/slack/install` first).

    `--seed-dev-slack-token` (`seed_dev_token=True`) is the EXPLICIT dev override: it
    encrypts + stores the dev env token as the install's Slack config so the smoke can
    post without the OAuth dance. It's a knowing deviation from #051 (a dev token
    becoming stored posting authority) — labeled loudly, off by default, and only for
    a sandbox install you control. Decryption-time failures (bad enc key) are caught
    and degrade to no-Slack rather than crashing the run mid-flight.

    Either way the encryption key must be present (the resolver decrypts at post time);
    without it no stored token is usable, so Slack is off.
    """
    if TOKEN_ENC_KEY_ENV not in os.environ:
        _say("  Slack ................ NOT wired (OUTRIDER_TOKEN_ENC_KEY missing — can't decrypt)")
        return None

    resolver = PerInstallSlackResolver(
        session_factory=session_factory,
        sink=persister,
        dashboard_base_url=os.environ.get("OUTRIDER_DASHBOARD_BASE_URL", ""),
    )

    async with session_factory() as session:
        existing = await get_slack_config(session, pr_context.installation_id)
    if existing is not None:
        _say(
            f"  Slack ................ wired → using EXISTING per-install config "
            f"(channel {existing.channel_id})"
        )
        return resolver
    if not seed_dev_token:
        _say(
            "  Slack ................ NOT wired (install not connected; run /slack/install, or "
            "pass --seed-dev-slack-token to seed the dev env token — dev-only, see #051)"
        )
        return None

    # Explicit dev override: persist the env token as this install's Slack config.
    try:
        slack_dev = SlackSettings()
    except ValidationError:
        _say(
            "  Slack ................ NOT wired (--seed-dev-slack-token set but "
            "OUTRIDER_SLACK_BOT_TOKEN / OUTRIDER_SLACK_CHANNEL_ID missing/invalid)"
        )
        return None
    try:
        ciphertext = encrypt_token(slack_dev.bot_token)
    except TokenCryptoError as exc:
        _say(f"  Slack ................ NOT wired (token encryption failed — bad enc key?): {exc}")
        return None
    async with session_factory() as session, session.begin():
        seeded = await set_slack_config(
            session,
            installation_id=pr_context.installation_id,
            team_id="(live-demo)",
            bot_token_ciphertext=ciphertext,
            channel_id=slack_dev.channel_id,
            configured_by="live-github-demo",
        )
    if not seeded:
        _say("  Slack ................ NOT wired (set_slack_config found no active install row)")
        return None
    _say(
        f"  Slack ................ wired → channel {slack_dev.channel_id} (SEEDED the dev env "
        "token as per-install config — dev-only deviation from #051; clear it after)"
    )
    return resolver


async def _run(args: argparse.Namespace) -> int:
    # --- setup gates ---
    for var in ("ANTHROPIC_API_KEY", "DATABASE_URL"):
        value = os.environ.get(var)
        if not value:
            _say(f"  {var} is not set — this runner needs it. Aborting.")
            return 2
        if value.startswith("op://"):
            # Sourcing .env does NOT resolve 1Password references — the literal
            # op:// string passes an is-set check, then fails at first use.
            _say(f"  {var} is a 1Password reference (op://...), not a real value.")
            _say("  Run through op so the references resolve:")
            _say("    op run --env-file=.env -- uv run python scripts/live_github_demo.py ...")
            return 2

    owner, repo = args.owner, args.repo
    if (owner, repo) not in _REPO_ALLOWLIST and not args.allow_any_repo:
        _say(
            f"  Refusing to post to {owner}/{repo}: not in the allowlist "
            f"{sorted(_REPO_ALLOWLIST)}. Pass --allow-any-repo to override "
            "(posts a real review to a real PR)."
        )
        return 2

    # Construct the env-backed settings BEFORE any live resource (engine,
    # provider) so missing/invalid GitHub App config (private key, webhook
    # secret — not just app_id) or a bad retention override fails cleanly with
    # setup exit 2 instead of raising mid-run after resources are built. This
    # validates against the production config surface (GitHubAppSettings /
    # RetentionSettings) rather than re-listing individual OUTRIDER_GITHUB_*
    # env var names, which would drift from the settings model.
    try:
        github_settings = GitHubAppSettings()
    except ValidationError as exc:
        _say(
            "  GitHub App settings invalid or incomplete (needs "
            "OUTRIDER_GITHUB_APP_ID / _APP_PRIVATE_KEY / _WEBHOOK_SECRET). "
            f"Aborting.\n{exc}"
        )
        return 2
    try:
        retention_settings = RetentionSettings()
    except ValidationError as exc:
        _say(f"  Retention settings invalid (OUTRIDER_AUDIT_*_RETENTION_TTL). Aborting.\n{exc}")
        return 2

    _say(_RULE)
    _say("  Outrider — live GitHub demo (real LLM · REAL GitHub · real DB)")
    _say(_RULE)
    _say()

    database_url = os.environ["DATABASE_URL"]
    engine = create_async_engine(database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    review_id = uuid.uuid4()

    # --- real deps, constructed exactly as the lifespan does ---
    model_config = ModelConfig()
    persister = AuditPersister(
        session_factory=session_factory,
        retention_settings=retention_settings,
    )
    provider = AnthropicProvider(
        api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
        model_config=model_config,
        persister=persister,
    )
    github_factory = make_installation_client_factory(github_settings)

    _say(
        f"  Models ............... {model_config.analyze_model} (analyze) + "
        f"{model_config.triage_model} (triage/synthesize)"
    )
    _say(
        f"  Target ............... {owner}/{repo} PR #{args.pr} "
        f"(installation {args.installation_id})"
    )

    try:
        pr_context, repo_id = await _fetch_pr_seed(
            github_factory,
            installation_id=args.installation_id,
            owner=owner,
            repo=repo,
            pr_number=args.pr,
        )
    except Exception as exc:  # noqa: BLE001 — surface any setup failure cleanly
        _say(
            f"  Failed to fetch PR (auth / installation / not found?): {type(exc).__name__}: {exc}"
        )
        await provider.aclose()
        await engine.dispose()
        return 2

    _say(f"  PR ................... {pr_context.pr_title!r} @ {pr_context.head_sha[:8]}")
    _say()

    # --- seed the reviews row (the webhook normally does this) ---
    try:
        await _seed_review_row(
            engine,
            review_id,
            pr_context,
            repo_id=repo_id,
            retention_settings=retention_settings,
        )
    except _ReviewAlreadyExistsError as exc:
        _say(
            f"  A review already exists for {owner}/{repo} PR #{args.pr} @ "
            f"{pr_context.head_sha[:8]} (review_id={exc.existing_review_id}, "
            f"status={exc.status}). `reviews` is UNIQUE(repo_id, pr_number, head_sha), "
            "so C2 can't seed a second row for the same head. To re-run: push a new "
            "commit to the PR (new head_sha), or delete the existing review row."
        )
        await provider.aclose()
        await engine.dispose()
        return 2
    except IntegrityError:
        # Seed-time race backstop: a concurrent run inserted the natural key
        # between our SELECT and INSERT. Same operator guidance as above.
        _say(
            "  Lost a race seeding the review row (uq_review_natural_key); a review for "
            "this head_sha already exists. Re-run after pushing a new commit or clearing "
            "the row."
        )
        await provider.aclose()
        await engine.dispose()
        return 2

    slack_resolver = await _maybe_wire_slack(
        session_factory, persister, pr_context, seed_dev_token=args.seed_dev_slack_token
    )

    graph = build_graph(
        db_factory=session_factory,
        github_factory=github_factory,
        provider=provider,
        model_config=model_config,
        phase_event_sink=persister,
        file_examination_sink=persister,
        analyze_event_sink=persister,
        publish_event_sink=persister,
        trace_sink=persister,
        hitl_event_sink=persister,
        synthesize_event_sink=persister,
        review_status_sink=ReviewStatusPersister(session_factory=session_factory),
        anomaly_sink=AnomalyPersister(session_factory=session_factory),
        hitl_config=HITLConfig(),
        # Required since the suggested-patches arc; OFF to keep the demo's
        # live spend bounded to the review calls themselves.
        patch_config=PatchConfig(patches_enabled=False),
        checkpointer=InMemorySaver(),
        publisher=GitHubKitPublisher(),
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
        # Production-parity shadow wiring (mirrors api/lifespan.py). NOTE:
        # this script targets the REAL configured DATABASE_URL — a demo run
        # writes content-tier cache rows (findings + trace candidates) into
        # that database's analyze_file_cache, exactly as a production review
        # would; they age out under the same 30-day/retention bounds.
        analyze_cache_store=AnalyzeCacheStore(session_factory=session_factory),
        resolve_slack_target=slack_resolver,
    )

    from outrider.agent.state import ReviewState  # noqa: PLC0415

    seed_state = ReviewState(
        review_id=review_id,
        pr_context=pr_context,
        received_at=datetime.now(UTC),
        is_eval=False,
    )

    _say("  Calling real Claude over the real diff ...")
    result = await graph.ainvoke(seed_state, config={"configurable": {"thread_id": str(review_id)}})
    await provider.aclose()

    interrupted = "__interrupt__" in result
    # Full-granularity dumps (same recipe as scripts/smoke_e2e.py). This script
    # targets the REAL DATABASE_URL, so the DB dump is review-scoped — never a
    # whole-table dump of a shared database.
    await narrate_audit_stream(_say, engine, review_id)
    await narrate_llm_exchanges_from_db(_say, engine, review_id)
    await narrate_slack_notifications(_say, engine, review_id)
    await narrate_db_state(_say, engine, review_id=review_id)
    rc = await _report_and_verify(
        engine,
        session_factory,
        review_id,
        result=result,
        interrupted=interrupted,
        allow_empty_publish=args.allow_empty_publish,
        slack_wired=slack_resolver is not None,
    )
    await engine.dispose()
    return rc


async def _seed_review_row(
    engine: AsyncEngine,
    review_id: uuid.UUID,
    pr_context: PRContext,
    *,
    repo_id: int,
    retention_settings: RetentionSettings,
) -> None:
    """Insert the reviews row the graph + findings writer expect.

    The webhook handler normally does this in its single transaction; for the
    direct-invoke path we do the minimal equivalent so the FK targets exist
    (findings.review_id, reviews.installation_id) and replay finds a review row.

    Columns mirror `db/models/reviews.py` + the webhook's own INSERT
    (`api/webhooks/router.py` step 9a). The aggregate-metric columns were
    dropped per DECISIONS.md#037 (review metrics live in the audit stream), so
    this INSERT seeds only the natural-key + status + retention columns.
    `status` / `created_at` / `updated_at` / `is_eval` have server defaults and
    are omitted. There is NO `received_at` column on
    `reviews`. `repo_id` is the real GitHub repository id, matching the natural
    key the webhook uses. `retention_expires_at` is
    `now + retention_settings.review_retention_ttl` — the same operator-
    overridable TTL the webhook reads (NOT a hard-coded 90-day interval), so an
    `OUTRIDER_AUDIT_REVIEW_RETENTION_TTL` override keeps this row consistent with
    the app config and with the content rows the persister writes. synthesize/
    publish overwrite the zeroed metrics with real counts; replay reads
    `audit_events`, not these columns, so the seed values don't affect the
    replay verdict.

    Raises `_ReviewAlreadyExistsError` if a row already exists for the natural
    key (repo_id, pr_number, head_sha) — the caller turns that into a clean
    exit-2 with operator guidance, mirroring the webhook's idempotency path.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations (installation_id, app_slug, account_id, "
                "account_login, account_type, permissions_at_install) "
                "VALUES (:iid, 'outrider-demo', 1, :owner, 'User', CAST('{}' AS jsonb)) "
                "ON CONFLICT (installation_id) DO NOTHING"
            ),
            {"iid": pr_context.installation_id, "owner": pr_context.owner},
        )
        # Natural-key idempotency: `reviews` is UNIQUE(repo_id, pr_number,
        # head_sha) (uq_review_natural_key), NOT just the PK. A fresh review_id
        # per run means an ON CONFLICT (id) guard would never fire; the real
        # collision is a same-head_sha re-run or a webhook-created row. Mirror
        # the webhook's application-level fast path — SELECT first and fail
        # cleanly with guidance rather than raising an opaque IntegrityError.
        # (The IntegrityError backstop at the call site covers the seed-time
        # race between this SELECT and the INSERT.)
        existing = (
            await conn.execute(
                text(
                    "SELECT id, status FROM reviews "
                    "WHERE repo_id = :repo_id AND pr_number = :pr AND head_sha = :sha"
                ),
                {"repo_id": repo_id, "pr": pr_context.pr_number, "sha": pr_context.head_sha},
            )
        ).first()
        if existing is not None:
            raise _ReviewAlreadyExistsError(
                existing_review_id=str(existing[0]), status=str(existing[1])
            )
        retention_expires_at = datetime.now(UTC) + retention_settings.review_retention_ttl
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, "
                "pr_title, status, retention_expires_at) "
                "VALUES (:id, :iid, :repo_id, :pr, :sha, :pr_title, 'running', "
                ":retention_expires_at)"
            ),
            {
                "id": review_id,
                "iid": pr_context.installation_id,
                "repo_id": repo_id,
                "pr": pr_context.pr_number,
                "sha": pr_context.head_sha,
                # Mirror the webhook's persist (direct-invoke bypasses it) so the
                # dashboard shows the real PR title.
                "pr_title": pr_context.pr_title,
                "retention_expires_at": retention_expires_at,
            },
        )


async def _report_and_verify(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[Any],
    review_id: uuid.UUID,
    *,
    result: dict[str, Any],
    interrupted: bool,
    allow_empty_publish: bool,
    slack_wired: bool,
) -> int:
    _say()
    async with engine.begin() as conn:
        findings = (
            await conn.execute(
                text(
                    "SELECT payload->>'finding_type', payload->>'severity' "
                    "FROM audit_events WHERE review_id = :id AND event_type = 'finding' "
                    "ORDER BY sequence_number"
                ),
                {"id": review_id},
            )
        ).all()

    _say("  Real Claude produced:")
    if findings:
        for ft, sev in findings:
            _say(f"    - {ft} ({sev})")
    else:
        _say("    (no findings)")
    _say()

    # --- Outcome: distinguish HITL pause / real post / no-op / failure by
    # inspecting the publish_result, not by assuming "not interrupted == posted".
    publish_result = result.get("publish_result")
    if interrupted:
        _say(
            "  Outcome .............. PAUSED at HITL gate (a CRITICAL/HIGH finding "
            "interrupted before publish — the gate working as designed; no comment posted)"
        )
    elif publish_result is None:
        _say("  Outcome .............. graph ended before publish (no publish_result in state)")
    else:
        # PublishResult.outcome ∈ {success, empty, idempotently_skipped,
        # idempotently_skipped_external_record} (schemas/publish.py). There is
        # no "failed" / "no_op_empty" outcome — publish raises on API failure,
        # and "no_op_empty" is the AUDIT marker for the `empty` outcome.
        outcome = getattr(publish_result, "outcome", None)
        gh_id = getattr(publish_result, "github_review_id", None)
        posted = getattr(publish_result, "comments_posted", 0)
        if outcome == "success":
            _say(
                f"  Outcome .............. REVIEW POSTED to GitHub "
                f"(github_review_id={gh_id}, comments_posted={posted}) — check the PR"
            )
        elif outcome == "empty":
            _say(
                "  Outcome .............. reached publish but posted NOTHING "
                "(empty — no inline-eligible findings; no GitHub review created)"
            )
        elif outcome == "idempotently_skipped_external_record":
            _say(
                f"  Outcome .............. publish skipped as duplicate; a prior run "
                f"already posted GitHub review {gh_id} for this head_sha"
            )
        elif outcome == "idempotently_skipped":
            _say(
                "  Outcome .............. publish skipped as duplicate "
                "(prior run already terminal for this head_sha; no new GitHub write)"
            )
        else:
            _say(f"  Outcome .............. publish outcome={outcome!r} (github_review_id={gh_id})")
    _say()

    # --- structural verdict ---
    checks: list[tuple[str, bool, str]] = []
    async with engine.begin() as conn:
        n_events = (
            await conn.execute(
                text("SELECT COUNT(*) FROM audit_events WHERE review_id = :id"),
                {"id": review_id},
            )
        ).scalar_one()
    checks.append(("audit events persisted", n_events > 0, f"{n_events} rows"))

    # GitHub-write proof. The happy-path C2 demo exists to prove a REAL review
    # reaches GitHub, so by default a non-interrupted run must end with a review
    # row on GitHub. "A GitHub review exists" == `github_review_id is not None`
    # (PublishResult has no `posted_to_github` property — derive it from the id):
    # populated for `success` and `idempotently_skipped_external_record`, None
    # for `empty` and plain `idempotently_skipped`. An `empty` outcome is a
    # structural pass (the pipeline ran) but NOT a write proof — it fails this
    # gate unless --allow-empty-publish is set. A HITL-paused run legitimately
    # never reaches publish, so the gate does not apply there.
    if not interrupted:
        outcome = getattr(publish_result, "outcome", None)
        posted = getattr(publish_result, "github_review_id", None) is not None
        # --allow-empty-publish excuses ONLY an explicit `empty` outcome (the
        # pipeline ran but produced no inline-eligible findings). It must NOT
        # excuse `publish_result is None` ("graph ended before publish") — that
        # is a different, genuine failure shape (the graph never reached the
        # publish node), not a no-findings run, so it always fails this gate.
        excused_empty = allow_empty_publish and outcome == "empty"
        write_ok = posted or excused_empty
        detail = f"outcome={outcome}, github_review_id_present={posted}"
        if excused_empty:
            detail += " (empty publish; accepted by --allow-empty-publish)"
        elif publish_result is None:
            detail = "publish_result is None — graph ended before the publish node (not excusable)"
        checks.append(("GitHub review posted", write_ok, detail))

    replayer = AuditReplayer(session_factory=session_factory)
    try:
        review = await replayer.reconstruct(review_id)
        checks.append(("reconstruct succeeds", True, f"mode={review.mode.value}"))
        await replayer.assert_replay_equivalent(review_id)
        checks.append(("assert_replay_equivalent passes", True, ""))
        # A review WITH findings reconstructs FULL regardless of whether it
        # reached publish or paused at HITL — analyze co-wrote the finding
        # content rows + the provider wrote the LLM content before either
        # terminal node. (Keyed on findings, not publish completion — a
        # HITL-paused review still has full content.)
        if review.findings:
            checks.append(
                (
                    "FULL replay with finding content",
                    review.mode is ReplayMode.FULL,
                    f"mode={review.mode.value}",
                )
            )
    except Exception as exc:  # noqa: BLE001 — surface any replay failure
        checks.append(("replay", False, f"{type(exc).__name__}: {exc}"))

    # Slack write proof (only when Slack was wired). A gated review posts hitl_pending
    # then pauses (interrupted); a clean review posts review_posted after a successful
    # publish. In either case a slack_notification audit row MUST exist — a swallowed
    # post (Slack never gate-breaks) would otherwise let the smoke pass with no message,
    # the exact silent-failure this gate catches. Outcomes that never reach a posting
    # point (empty/skipped publish, graph ended early) are reported, not gated.
    if slack_wired:
        async with engine.begin() as conn:
            n_slack = (
                await conn.execute(
                    text(
                        "SELECT COUNT(*) FROM audit_events "
                        "WHERE review_id = :id AND event_type = 'slack_notification'"
                    ),
                    {"id": review_id},
                )
            ).scalar_one()
        published_ok = (
            publish_result is not None and getattr(publish_result, "outcome", None) == "success"
        )
        if interrupted or published_ok:
            checks.append(
                (
                    "Slack notification posted (wired + reached a posting point)",
                    n_slack >= 1,
                    f"{n_slack} slack_notification row(s)",
                )
            )
        else:
            _say(
                f"  Slack notify check ... skipped ({n_slack} row(s); review reached no Slack "
                "posting point — gated→hitl or successful publish)"
            )

    _say("  Structural checks (exit verdict = all must pass):")
    all_ok = True
    for label, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        _say(f"    [{mark}] {label}  {detail}")
    _say()
    _say(_RULE)
    _say(f"  LIVE GITHUB DEMO {'PASSED' if all_ok else 'FAILED'}")
    _say(_RULE)
    return 0 if all_ok else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct-invoke Outrider over a real GitHub PR.")
    parser.add_argument("--owner", required=True, help="repo owner/org login")
    parser.add_argument("--repo", required=True, help="repo name")
    parser.add_argument("--pr", type=int, required=True, help="pull request number")
    parser.add_argument(
        "--installation-id",
        type=int,
        required=True,
        help="GitHub App installation id (from the install URL)",
    )
    parser.add_argument(
        "--allow-any-repo",
        action="store_true",
        help="override the sandbox-repo allowlist (posts a real review to the given repo)",
    )
    parser.add_argument(
        "--allow-empty-publish",
        action="store_true",
        help=(
            "accept an 'empty' / non-posting publish outcome as a pass. By default a "
            "non-interrupted run must post a real GitHub review (the C2 happy-path proof); "
            "this relaxes that to allow runs where Claude produces no inline-eligible findings"
        ),
    )
    parser.add_argument(
        "--seed-dev-slack-token",
        action="store_true",
        help=(
            "DEV-ONLY: persist the OUTRIDER_SLACK_* env token as this install's per-install "
            "Slack config so the smoke can post without the OAuth dance. Off by default — a "
            "knowing deviation from DECISIONS.md#051 (the env token is not production posting "
            "authority); only for a sandbox install, and clear it afterward. Default behavior "
            "uses the install's existing OAuth-stored config, or skips Slack if unconnected"
        ),
    )
    return parser.parse_args()


def _main_with_log() -> int:
    """Tee the run to scripts/generated/ and name the file on both ends."""
    global _TRACE  # noqa: PLW0603 — single assignment per process, set before any _say
    _TRACE = TraceTee("live_github_demo")
    print(f"  Full trace ........... {_TRACE.path}", flush=True)
    # Route outrider WARNING+ logs (incl. SWALLOWED best-effort Slack failures) into
    # the tee, so a non-fatal Slack error isn't a silent failure — it lands in the
    # terminal + the trace file alongside everything else.
    outrider_logger = logging.getLogger("outrider")
    outrider_logger.setLevel(logging.WARNING)
    log_handler = _SayLogHandler(level=logging.WARNING)
    outrider_logger.addHandler(log_handler)
    try:
        return asyncio.run(_run(_parse_args()))
    except Exception:
        _TRACE.write_current_exception()
        raise
    finally:
        outrider_logger.removeHandler(log_handler)
        print(f"  Full trace ........... {_TRACE.path}", flush=True)
        _TRACE.close()
        _TRACE = None


if __name__ == "__main__":
    sys.exit(_main_with_log())
