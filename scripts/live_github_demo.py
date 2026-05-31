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
    await _seed_review_row(engine, review_id, pr_context, repo_id=repo_id)

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
    rc = await _report_and_verify(
        engine, session_factory, review_id, result=result, interrupted=interrupted
    )
    await engine.dispose()
    return rc


async def _seed_review_row(
    engine: AsyncEngine, review_id: uuid.UUID, pr_context: PRContext, *, repo_id: int
) -> None:
    """Insert the reviews row the graph + findings writer expect.

    The webhook handler normally does this in its single transaction; for the
    direct-invoke path we do the minimal equivalent so the FK targets exist
    (findings.review_id, reviews.installation_id) and replay finds a review row.

    Columns mirror `db/models/reviews.py` + the webhook's own INSERT
    (`api/webhooks/router.py` step 9a): the NOT-NULL-no-default set is
    `id, installation_id, repo_id, pr_number, head_sha, retention_expires_at`.
    `status` / `created_at` / `is_eval` use their server defaults; the metrics
    columns (`files_examined`, `llm_calls_made`, tokens/cost/wall_clock) are
    nullable and populated by synthesize/publish. There is NO `received_at`
    column on `reviews` (an earlier version of this seed referenced one and
    would have failed the INSERT before the graph ran). `repo_id` is the real
    GitHub repository id, matching the natural key the webhook uses.
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
                "status, retention_expires_at) "
                "VALUES (:id, :iid, :repo_id, :pr, :sha, 'running', "
                "NOW() + INTERVAL '90 days') ON CONFLICT (id) DO NOTHING"
            ),
            {
                "id": review_id,
                "iid": pr_context.installation_id,
                "repo_id": repo_id,
                "pr": pr_context.pr_number,
                "sha": pr_context.head_sha,
            },
        )


async def _report_and_verify(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[Any],
    review_id: uuid.UUID,
    *,
    result: dict[str, Any],
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
        outcome = getattr(publish_result, "outcome", None)
        gh_id = getattr(publish_result, "github_review_id", None)
        posted = getattr(publish_result, "comments_posted", 0)
        if outcome == "success":
            _say(
                f"  Outcome .............. REVIEW POSTED to GitHub "
                f"(github_review_id={gh_id}, comments_posted={posted}) — check the PR"
            )
        elif outcome == "no_op_empty":
            _say(
                "  Outcome .............. reached publish but posted NOTHING "
                "(no_op_empty — no inline-eligible findings; nothing on the PR)"
            )
        elif outcome in ("idempotently_skipped", "idempotently_skipped_external_record"):
            _say(
                f"  Outcome .............. publish skipped as duplicate ({outcome}); "
                "a prior run already posted for this head_sha"
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

    # If the run reached publish (not paused) and a publish_result exists, its
    # outcome must be a non-failure value. `failed` is a hard check failure;
    # `success` / `no_op_empty` / idempotent-skip are all acceptable end states
    # (the demo's job is to prove the path runs, not to force a specific count).
    if not interrupted and publish_result is not None:
        outcome = getattr(publish_result, "outcome", None)
        checks.append(("publish did not fail", outcome != "failed", f"outcome={outcome}"))
        if outcome == "success":
            checks.append(
                (
                    "posted review carries a github_review_id",
                    getattr(publish_result, "github_review_id", None) is not None,
                    f"id={getattr(publish_result, 'github_review_id', None)}",
                )
            )

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
