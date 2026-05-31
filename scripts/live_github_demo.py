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
config error (missing env, DB unreachable, PR not found).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# --- ensure src/ on path (mirror conftest's pythonpath=["src"]) ---
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from outrider.agent.graph import build_graph  # noqa: E402
from outrider.agent.nodes.hitl_config import HITLConfig  # noqa: E402
from outrider.anomaly.persister import AnomalyPersister  # noqa: E402
from outrider.audit.config import RetentionSettings  # noqa: E402
from outrider.audit.persister import AuditPersister  # noqa: E402
from outrider.audit.replay import AuditReplayer, ReplayMode  # noqa: E402
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER  # noqa: E402
from outrider.db.review_status_persister import ReviewStatusPersister  # noqa: E402
from outrider.github.auth import make_installation_client_factory  # noqa: E402
from outrider.github.config import GitHubAppSettings  # noqa: E402
from outrider.github.publisher import GitHubKitPublisher  # noqa: E402
from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from outrider.llm.config import ModelConfig  # noqa: E402
from outrider.schemas.pr_context import PRContext  # noqa: E402

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_RULE = "=" * 62

# Hard-allowlisted sandbox target (mirrors spikes/publish/smoke_publish.py).
# Posting a real review to anything else requires the explicit
# --allow-any-repo override, which is visible in the command + this code.
_REPO_ALLOWLIST: frozenset[tuple[str, str]] = frozenset({("ddamme05", "outrider-smoke-test")})


def _say(msg: str = "") -> None:
    print(msg, flush=True)


async def _fetch_pr_seed(
    github_factory: Any,
    *,
    installation_id: int,
    owner: str,
    repo: str,
    pr_number: int,
) -> PRContext:
    """Mint a real installation token and GET the PR to seed PRContext.

    `changed_files=()` — intake enriches it with the real per-file diff via
    `github/fetch.py`, exactly as the webhook path does (DECISIONS.md#020).
    Only the PR-level metadata (title, shas, author, +/- counts) is read here.

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
    return PRContext(
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


async def _run(args: argparse.Namespace) -> int:
    # --- setup gates ---
    for var in ("ANTHROPIC_API_KEY", "OUTRIDER_GITHUB_APP_ID", "DATABASE_URL"):
        if not os.environ.get(var):
            _say(f"  {var} is not set — this runner needs it. Aborting.")
            return 2

    owner, repo = args.owner, args.repo
    if (owner, repo) not in _REPO_ALLOWLIST and not args.allow_any_repo:
        _say(
            f"  Refusing to post to {owner}/{repo}: not in the allowlist "
            f"{sorted(_REPO_ALLOWLIST)}. Pass --allow-any-repo to override "
            "(posts a real review to a real PR)."
        )
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
        retention_settings=RetentionSettings(),
    )
    from pydantic import SecretStr  # noqa: PLC0415

    provider = AnthropicProvider(
        api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
        model_config=model_config,
        persister=persister,
    )
    github_factory = make_installation_client_factory(GitHubAppSettings())

    _say(
        f"  Models ............... {model_config.analyze_model} (analyze) + "
        f"{model_config.triage_model} (triage/synthesize)"
    )
    _say(
        f"  Target ............... {owner}/{repo} PR #{args.pr} "
        f"(installation {args.installation_id})"
    )

    try:
        pr_context = await _fetch_pr_seed(
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
    await _seed_review_row(engine, review_id, pr_context, received_at=datetime.now(UTC))

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
        checkpointer=InMemorySaver(),
        publisher=GitHubKitPublisher(),
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
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
    rc = await _report_and_verify(engine, session_factory, review_id, interrupted=interrupted)
    await engine.dispose()
    return rc


async def _seed_review_row(
    engine: AsyncEngine, review_id: uuid.UUID, pr_context: PRContext, *, received_at: datetime
) -> None:
    """Insert the reviews row the graph + findings writer expect.

    The webhook handler normally does this in its single transaction; for the
    direct-invoke path we do the minimal equivalent so the FK targets exist
    (findings.review_id, reviews.installation_id) and replay finds a review row.
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
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, "
                "status, received_at, retention_expires_at) "
                "VALUES (:id, :iid, :repo_id, :pr, :sha, 'running', :rec, "
                ":rec + INTERVAL '90 days') ON CONFLICT (id) DO NOTHING"
            ),
            {
                "id": review_id,
                "iid": pr_context.installation_id,
                "repo_id": abs(hash((pr_context.owner, pr_context.repo))) % (10**9),
                "pr": pr_context.pr_number,
                "sha": pr_context.head_sha,
                "rec": received_at,
            },
        )


async def _report_and_verify(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[Any],
    review_id: uuid.UUID,
    *,
    interrupted: bool,
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

    if interrupted:
        _say(
            "  Outcome .............. PAUSED at HITL gate (a CRITICAL/HIGH finding "
            "interrupted before publish — the gate working as designed; no comment posted)"
        )
    else:
        _say(
            "  Outcome .............. reached publish — a review was posted to the PR "
            "(check GitHub)"
        )
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

    replayer = AuditReplayer(session_factory=session_factory)
    try:
        review = await replayer.reconstruct(review_id)
        checks.append(("reconstruct succeeds", True, f"mode={review.mode.value}"))
        await replayer.assert_replay_equivalent(review_id)
        checks.append(("assert_replay_equivalent passes", True, ""))
        # A completed (non-interrupted) review with a finding should be FULL.
        if not interrupted and review.findings:
            checks.append(
                (
                    "FULL replay with finding content",
                    review.mode is ReplayMode.FULL,
                    f"mode={review.mode.value}",
                )
            )
    except Exception as exc:  # noqa: BLE001 — surface any replay failure
        checks.append(("replay", False, f"{type(exc).__name__}: {exc}"))

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
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(_run(_parse_args())))
