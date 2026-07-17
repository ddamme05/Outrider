#!/usr/bin/env python3
"""Seed the demo Postgres with the curated review set, then snapshot to demo_seed.sql.

Runs every demo fixture + the #6 "Outrider reviewing Outrider" git-range showcase
through the REAL graph (real Claude) into ONE persistent, isolated demo database,
validates each capture against the spec's acceptance gate, and `pg_dump`s the
result so the deploy re-seeds with ZERO Claude respend.
specs/2026-06-21-demo-deployment.md ("Seed mechanism" + "Acceptance checks").

This is a PAID run (one real review per seed entry). It reuses
scripts/live_claude_smoke.py's proven machinery (the scenario builders + `_run`),
so the seed path and the smoke path cannot drift. Each entry gets a unique
`head_sha` so N reviews of the same (repo, pr_number) don't collide on the
`(repo_id, pr_number, head_sha)` natural key.

The capture gate (the recall safeguard — why a 21-file kitchen sink once shipped a
thin review): for every seeded review assert (a) no COST_BUDGET_EXHAUSTED skip
(except entries with `allow_cost_starvation=True` — the 27-file self-review, whose
largest modules exceed the absolute per-file token cap by design), (b) no
finding_proposal_rejected AND no analyze_response_rejected (a wholesale degraded
analyze), (c) every expected finding type is present, and (d) the expected HITL /
audit rows exist. A review that fails any check is reported and the seed exits
non-zero — a dud is never dumped.

Run it:

    docker compose up -d postgres-test
    export ANTHROPIC_API_KEY=sk-ant-...
    export TEST_DATABASE_URL=postgresql+psycopg://...:5433/outrider_test
    uv run python scripts/seed_demo.py             # seed + validate + dump
    uv run python scripts/seed_demo.py --dry-run   # print the plan; no DB, no Claude

Exit 0 = every entry captured cleanly and demo_seed.sql was written.
1 = a capture check failed (a dud — not dumped). 2 = setup / credentials missing.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import subprocess  # noqa: S404 — local dev tooling; argv lists only, never shell=True
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the proven live-smoke machinery verbatim (single source of truth).
from live_claude_smoke import (  # noqa: E402 — sys.path set above
    _assert_isolated,
    _create_db,
    _drop_db,
    _load_test_db_url,
    _migrate,
    _run,
    _Scenario,
    _scenario_from_file,
    _scenario_from_git_range,
    _swap_db,
)
from sqlalchemy import text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from outrider.audit.config import RetentionSettings  # noqa: E402
from outrider.audit.persister import AuditPersister  # noqa: E402
from outrider.llm.base import LLMRateLimitError, LLMTimeoutError  # noqa: E402
from outrider.sweep.replay_verdict import project_replay_verdicts  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO_ROOT / "scripts" / "demo_fixtures"
_DEMO_DB_NAME = "outrider_test_demo"  # contains 'test' -> passes the 5433 isolation guard
_SEED_SQL = _REPO_ROOT / "scripts" / "demo_fixtures" / "demo_seed.sql"
_GIT_RANGE = "0c70d18^..39c538b"  # the 27-file analyze-cache arc (the #6 showcase)
_SHOWCASE_RANGE = _GIT_RANGE
# The smoke-test repo + PR #9 range back the breadth review (28 files, broad taxonomy).
# Local-only at SEED time — the review content bakes into demo_seed.sql, so the demo box
# never needs this checkout. Override the path with OUTRIDER_DEMO_SMOKE_REPO if it differs.
_SMOKE_REPO = os.environ.get(
    "OUTRIDER_DEMO_SMOKE_REPO", str(Path.home() / "projects" / "outrider-smoke-test")
)
_SMOKE_RANGE = "1f7cd2d..d4e7d6ef"  # PR #9 base..head (28 files); NOT d4e7d6ef^.. (last commit, 9)
# The 27-file showcase tiers ~16 files DEEP/STANDARD; the default 200k analyze
# budget starves ~9 of them at the cost gate. Set a generous budget for the demo
# so the showcase completes cleanly (the spec's "budget tuning for #6"). This is a
# CAP, not spend — actual cost is the analyze input the reviewed files consume.
# `_force_env_at_least` (in main) forces this floor past .env's smaller value while
# honoring a deliberately larger explicit OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS.
_DEMO_ANALYZE_BUDGET_TOKENS = 1_500_000

# A persistent demo box wants its HITL reviews to stay PENDING forever, not inherit
# the 30-min production default — `expires_at = created_at + timeout_minutes`, and the
# dashboard renders `expires_at < now()` as "expired" client-side (dashboard/src/lib/
# format.ts), so a 30-min window makes the two HITL-gated reviews look stale the moment
# the demo is viewed later. A ~100-year timeout bakes a far-future expires_at into BOTH
# the reviews row AND the hitl_request audit event (consistent → replay-safe).
# `_force_env_at_least` (in main) forces this floor past .env's 30-min default while
# honoring a deliberately larger explicit OUTRIDER_HITL_TIMEOUT_MINUTES.
_DEMO_HITL_TIMEOUT_MINUTES = 100 * 365 * 24 * 60  # ~100 years

# Backoff waits (seconds) before each retry of a RATE-LIMITED review. The 27-file
# showcase can exhaust the per-minute token ceiling mid-review; the analyze node has no
# per-call retry yet (FUP-025), so a single 429 is otherwise terminal. On a limit we
# restore the demo DB to the last checkpoint (the prior good reviews — the audit_events
# append-only trigger blocks deleting the partial review in place, so we drop + reload),
# wait for the rolling window to clear, and retry ONLY that review. Retrying it alone
# (not re-running the cheap reviews, which would re-draw the window down) is what lets
# the heavy review land in a fresh token window — the whole-seed retry never could.
_RATE_LIMIT_BACKOFF_SECONDS: tuple[int, ...] = (60, 120)

# A pg_dump of the demo DB after each successful review, restored on a retry so the
# failed review re-runs against the prior good state (not from scratch). Local-only.
_CHECKPOINT = _REPO_ROOT / "scripts" / "generated" / "seed_checkpoint.sql"


def _force_env_at_least(key: str, minimum: int) -> None:
    """Set env `key` to `minimum` UNLESS it already holds a larger integer.

    `setdefault` is wrong for the demo seed: the documented run is
    `op run --env-file=.env`, which injects .env's production values BEFORE this
    process starts, so the key is always present and `setdefault` never fires. This
    forces the demo floor past a smaller .env value (the 30-min HITL default, a tight
    analyze budget) while still honoring a DELIBERATELY larger explicit override.
    """
    current = os.environ.get(key, "").strip()
    if not (current.isdigit() and int(current) >= minimum):
        os.environ[key] = str(minimum)


@dataclass(frozen=True)
class SeedSpec:
    """One seeded review: how to build it + what its capture must contain."""

    key: str
    label: str
    expect_findings: bool  # feeds _run's structural --expect-findings verdict
    expected_finding_types: frozenset[str] = field(default_factory=frozenset)
    # Finding types that MUST land as OBSERVED proofs: evidence_tier='observed' AND a
    # non-null query_match_id (a JUDGED finding of the same type does NOT satisfy this).
    # These are deterministic tree-sitter matches, so requiring them is safe — and this is
    # what actually backs an "N OBSERVED proofs" claim, which expected_finding_types can't.
    required_observed_proofs: frozenset[str] = field(default_factory=frozenset)
    # The advertised ROUTING outcome — asserted at capture, not just the finding
    # types. "hitl": a CRITICAL/HIGH finding must park the review at the gate.
    # "published": the review must auto-publish (NO hitl_request + publish_routing
    # events) — a HIGH/CRITICAL over-flag that parks it is a gate failure, not a pass.
    # "decided": the review must gate AND be pre-decided in-process (hitl_request +
    # hitl_decision + publish_routing all present) — the full gated lifecycle.
    # "any": routing is model-dependent across a real arc; don't assert it.
    expected_outcome: Literal["hitl", "published", "decided", "any"] = "any"
    # Decide the HITL gate in-process after the interrupt (live_claude_smoke
    # resumes with a demo-reviewer decision: approve all gated findings, one
    # truthful severity downgrade). Pair with expected_outcome="decided".
    pre_decide: bool = False
    # Allow COST_BUDGET_EXHAUSTED skips for this entry. Default False: on the
    # single-file / planted-fixture reviews a cost-starved file IS a degraded seed.
    # The 27-file self-review is the exception — its three largest modules
    # (analyze.py, persister.py, events.py) each exceed the ABSOLUTE per-file token
    # cap (MAX_PER_FILE_TOKENS_ABSOLUTE=60_000, deliberately decoupled from the
    # review budget), so the cost gate engaging on them is correct product behavior,
    # not a seed defect — and it makes the showcase demonstrate the cost-control +
    # cost-budget-anomaly features on a genuinely large PR.
    allow_cost_starvation: bool = False
    diff_file: str | None = None  # a fixture under scripts/demo_fixtures/
    git_range: str | None = None  # OR a two-dot git range (the showcase)
    repo_root: str | None = None  # git_range against a DIFFERENT repo (the smoke breadth review)
    pr_title: str | None = None  # override the git-range scenario's default title

    def build_scenario(self, head_sha: str) -> _Scenario:
        if self.git_range is not None:
            root = Path(self.repo_root) if self.repo_root else None
            base = _scenario_from_git_range(self.git_range, root, pr_title=self.pr_title)
        elif self.diff_file is not None:
            base = _scenario_from_file(_FIXTURES / self.diff_file)
        else:
            raise ValueError(f"seed spec {self.key!r} has neither diff_file nor git_range")
        return dataclasses.replace(base, head_sha=head_sha)


# The curated demo review set (specs/2026-06-21-demo-deployment.md "Seeded review list").
SEED_SPECS: tuple[SeedSpec, ...] = (
    SeedSpec(
        key="hitl_gate",
        label="HITL gate (sql_injection -> CRITICAL, parks for approval)",
        diff_file="vulnerable_query.py",
        expect_findings=True,
        expected_finding_types=frozenset({"sql_injection"}),
        expected_outcome="hitl",
    ),
    SeedSpec(
        key="auto_publish",
        label="Auto-publish (sub-HIGH multi-finding)",
        diff_file="api_request_handler.py",
        expect_findings=True,
        expected_finding_types=frozenset({"blocking_call_in_async", "missing_input_validation"}),
        expected_outcome="published",
    ),
    SeedSpec(
        key="observed_proof",
        label="OBSERVED proof (weak_crypto, deterministic query_match_id) + decided gate",
        diff_file="weak_crypto_handler.py",
        expect_findings=True,
        expected_finding_types=frozenset({"weak_crypto"}),
        # weak_crypto is HIGH (severity policy) -> trips the HITL gate; this entry
        # is then PRE-DECIDED in-process so the demo carries one review with the
        # full gated lifecycle (hitl_request -> hitl_decision with a severity
        # override -> publish routing). hitl_gate and smoke_breadth stay parked
        # at AWAITING_APPROVAL for the attention-rail story.
        expected_outcome="decided",
        pre_decide=True,
    ),
    SeedSpec(
        key="breadth",
        label="Breadth across dimensions",
        diff_file="report_builder.py",
        expect_findings=True,
        # Require the two findings the fixture reliably produces — input-validation
        # (security) + n_plus_one (performance), two distinct dimensions = real
        # breadth. The third (missing_error_handling) is a model-dependent JUDGED
        # call on the bare-except; the rewritten file gives it a fair shot but it's
        # not gated on (requiring an uncertain JUDGED type would false-reject).
        expected_finding_types=frozenset({"missing_input_validation", "n_plus_one_query"}),
        # All three planted flaws are sub-HIGH, so breadth must AUTO-PUBLISH. The gate
        # now enforces that: a CRITICAL/HIGH over-flag that parks it at HITL is a
        # failure (caught a false-positive sql_injection on the int(page) OFFSET).
        expected_outcome="published",
    ),
    SeedSpec(
        key="scale_triage",
        label="Scale & triage (27-file self-review)",
        git_range=_SHOWCASE_RANGE,
        expect_findings=False,  # findings model-dependent across a real arc; gate loosely
        expected_outcome="any",  # routing is model-dependent on a real arc; don't assert it
        # The largest modules exceed the absolute per-file token cap, so the cost gate
        # engaging on them is expected — it demonstrates cost control on a big PR.
        allow_cost_starvation=True,
    ),
    SeedSpec(
        key="smoke_breadth",
        label="Breadth showcase (28-file smoke PR — broad taxonomy + OBSERVED proofs)",
        git_range=_SMOKE_RANGE,
        repo_root=_SMOKE_REPO,
        pr_title="Add accounts, payments, reports & client services",
        expect_findings=True,
        # The 6 deterministic OBSERVED security types (tree-sitter matches on planted
        # patterns) must each land as a query-backed proof; the broad JUDGED set is
        # model-dependent, so the gate requires the proofs, not the full taxonomy.
        required_observed_proofs=frozenset(
            {
                "sql_injection",
                "command_injection",
                "unsafe_deserialization",
                "tls_verify_disabled",
                "weak_crypto",
                "blocking_call_in_async",
            }
        ),
        expected_outcome="hitl",  # sql_injection -> CRITICAL parks it at the gate
    ),
)


def _head_sha_for(index: int) -> str:
    """A distinct, valid 40-hex head_sha per seed entry (avoids the natural-key clash)."""
    return format(index + 1, "040x")


@dataclass
class CaptureResult:
    spec_key: str
    ok: bool
    detail: str


async def _validate_capture(db_url: str, review_id: str, spec: SeedSpec) -> CaptureResult:
    """The acceptance gate: no starvation, no proposal rejection, expected types
    present, and the expected HITL/audit rows exist. Queries the persisted review."""
    engine = create_async_engine(db_url, poolclass=NullPool)
    failures: list[str] = []
    try:
        async with engine.begin() as conn:
            starved = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM audit_events WHERE review_id = :id "
                        "AND event_type = 'file_examination' "
                        "AND payload->>'skip_reason' = 'COST_BUDGET_EXHAUSTED'"
                    ),
                    {"id": review_id},
                )
            ).scalar_one()
            if starved and not spec.allow_cost_starvation:
                failures.append(f"{starved} file(s) starved (COST_BUDGET_EXHAUSTED)")

            rejected = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM audit_events WHERE review_id = :id "
                        "AND event_type = 'finding_proposal_rejected'"
                    ),
                    {"id": review_id},
                )
            ).scalar_one()
            if rejected:
                failures.append(f"{rejected} finding_proposal_rejected event(s)")

            # A wholesale analyze-response rejection (invalid JSON, empty, etc.) is a
            # degraded run even with audit rows + no starvation — the loose scale
            # review would otherwise dump it. live_claude_smoke treats it as the
            # semantic-empty-run path; the seed must reject it too.
            resp_rejected = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM audit_events WHERE review_id = :id "
                        "AND event_type = 'analyze_response_rejected'"
                    ),
                    {"id": review_id},
                )
            ).scalar_one()
            if resp_rejected:
                failures.append(f"{resp_rejected} analyze_response_rejected event(s)")

            types = {
                row[0]
                for row in (
                    await conn.execute(
                        text(
                            "SELECT DISTINCT payload->>'finding_type' FROM audit_events "
                            "WHERE review_id = :id AND event_type = 'finding'"
                        ),
                        {"id": review_id},
                    )
                ).all()
            }
            missing = spec.expected_finding_types - types
            if missing:
                failures.append(f"missing expected finding type(s): {sorted(missing)}")

            # OBSERVED-proof gate: each required type must land as a query-backed proof
            # (evidence_tier='observed' AND non-null query_match_id), not merely as a
            # same-named JUDGED finding. This is what enforces the "N OBSERVED proofs" claim.
            if spec.required_observed_proofs:
                observed_proofs = {
                    row[0]
                    for row in (
                        await conn.execute(
                            text(
                                "SELECT DISTINCT payload->>'finding_type' FROM audit_events "
                                "WHERE review_id = :id AND event_type = 'finding' "
                                "AND payload->>'evidence_tier' = 'observed' "
                                "AND payload->>'query_match_id' IS NOT NULL"
                            ),
                            {"id": review_id},
                        )
                    ).all()
                }
                missing_proofs = spec.required_observed_proofs - observed_proofs
                if missing_proofs:
                    failures.append(
                        "missing OBSERVED proof(s) (need evidence_tier=observed + "
                        f"query_match_id): {sorted(missing_proofs)}"
                    )

            n_events = (
                await conn.execute(
                    text("SELECT count(*) FROM audit_events WHERE review_id = :id"),
                    {"id": review_id},
                )
            ).scalar_one()
            if n_events == 0:
                failures.append("no audit events persisted")

            # Routing-outcome gate: the advertised story (HITL vs auto-publish) must
            # match what the review actually did — not just which finding types fired.
            # Without this, a CRITICAL/HIGH over-flag on a supposed auto-publish review
            # parks it at AWAITING_APPROVAL yet still passes on finding types alone,
            # dumping a review that contradicts its own demo label.
            hitl = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM audit_events WHERE review_id = :id "
                        "AND event_type = 'hitl_request'"
                    ),
                    {"id": review_id},
                )
            ).scalar_one()
            if spec.expected_outcome == "hitl":
                if not hitl:
                    failures.append(
                        "expected a HITL request event (CRITICAL/HIGH finding parks the gate), "
                        "found none"
                    )
            elif spec.expected_outcome == "decided":
                if not hitl:
                    failures.append(
                        "expected a HITL request event before the in-process decision, found none"
                    )
                decided = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM audit_events WHERE review_id = :id "
                            "AND event_type = 'hitl_decision'"
                        ),
                        {"id": review_id},
                    )
                ).scalar_one()
                if not decided:
                    failures.append(
                        "expected an in-process hitl_decision event (pre_decide), found none"
                    )
                routed_after_decide = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM audit_events WHERE review_id = :id "
                            "AND event_type = 'publish_routing'"
                        ),
                        {"id": review_id},
                    )
                ).scalar_one()
                if not routed_after_decide:
                    failures.append(
                        "expected publish_routing events after the decided gate, found none"
                    )
            elif spec.expected_outcome == "published":
                if hitl:
                    failures.append(
                        f"expected auto-publish but found {hitl} hitl_request event(s) — a "
                        "CRITICAL/HIGH finding parked the review at the gate, contradicting the "
                        "auto-publish demo"
                    )
                routed = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM audit_events WHERE review_id = :id "
                            "AND event_type = 'publish_routing'"
                        ),
                        {"id": review_id},
                    )
                ).scalar_one()
                if not routed:
                    failures.append(
                        "expected auto-publish routing events (publish_routing), found none"
                    )
            # "any": routing is model-dependent across a real arc; no outcome assertion.
    finally:
        await engine.dispose()

    if failures:
        return CaptureResult(spec.key, ok=False, detail="; ".join(failures))
    n = len(spec.expected_finding_types) + len(spec.required_observed_proofs)
    return CaptureResult(spec.key, ok=True, detail=f"{n} expected type(s)/proof(s) present")


def _pg_dump(db_url: str, out_path: Path) -> bool:
    """pg_dump the seeded demo DB to a portable SQL artifact.

    Runs pg_dump INSIDE the postgres-test container (docker compose exec) so the
    dumper version always matches the server — a host pg_dump older than the server
    refuses to run (the dev box ships v16 against a v18 test container). Output is
    streamed to the host file via stdout; the in-container connection is local
    (localhost:5432, the container's own postgres) with the password via PGPASSWORD.
    """
    url = make_url(db_url)
    exec_argv = [
        "docker",
        "compose",
        "exec",
        "-T",
        "-e",
        f"PGPASSWORD={url.password or ''}",
        "postgres-test",
        "pg_dump",
        "-h",
        "localhost",
        "-U",
        str(url.username),
        "-d",
        str(url.database),
        "--no-owner",
        "--no-privileges",
    ]
    # Dump to a same-directory temp file, then os.replace() into place ONLY on success.
    # A failed pg_dump must never truncate/delete an existing good dump — in append mode
    # out_path IS the known-good seed. os.replace is atomic within one filesystem.
    tmp_path = out_path.with_name(f"{out_path.name}.partial")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(  # noqa: S603 — argv list, no shell; docker on PATH
                exec_argv, stdout=fh, stderr=subprocess.PIPE, text=True, check=False
            )
    except FileNotFoundError:
        print(
            "  pg_dump step needs `docker compose` on PATH (it runs in the "
            "postgres-test container so the version matches the server).",
            flush=True,
        )
        tmp_path.unlink(missing_ok=True)
        return False
    if proc.returncode != 0:
        print(f"  pg_dump failed: {proc.stderr.strip()}", flush=True)
        tmp_path.unlink(missing_ok=True)  # leave the prior out_path untouched
        return False
    os.replace(tmp_path, out_path)
    return True


def _load_sql(db_url: str, in_path: Path) -> bool:
    """Restore a SQL dump INTO the demo DB via in-container psql (version-matched,
    same reasoning as `_pg_dump`). ON_ERROR_STOP so a bad restore fails loud."""
    url = make_url(db_url)
    exec_argv = [
        "docker",
        "compose",
        "exec",
        "-T",
        "-e",
        f"PGPASSWORD={url.password or ''}",
        "postgres-test",
        "psql",
        "-h",
        "localhost",
        "-U",
        str(url.username),
        "-d",
        str(url.database),
        "-v",
        "ON_ERROR_STOP=1",
        "-q",
    ]
    try:
        with in_path.open("r", encoding="utf-8") as fh:
            proc = subprocess.run(  # noqa: S603 — argv list, no shell; docker on PATH
                exec_argv, stdin=fh, stderr=subprocess.PIPE, text=True, check=False
            )
    except FileNotFoundError:
        print("  checkpoint restore needs `docker compose` on PATH.", flush=True)
        return False
    if proc.returncode != 0:
        print(f"  checkpoint restore failed: {proc.stderr.strip()}", flush=True)
        return False
    return True


def _restore_to_checkpoint(admin_url: str, demo_url: str, *, have_checkpoint: bool) -> bool:
    """Reset the demo DB to the last-good state before retrying a failed review.

    Drops + recreates the DB (the only way to clear the partial review — the
    audit_events append-only trigger blocks deleting its rows in place) and reloads the
    checkpoint dump of the prior successful reviews, or a fresh migrated DB if none yet."""
    asyncio.run(_recreate_demo_db(admin_url))
    if have_checkpoint:
        return _load_sql(demo_url, _CHECKPOINT)
    _migrate(demo_url)
    return True


async def _project_replay_verdicts_or_fail(demo_url: str) -> str | None:
    """Project a replay verdict for every completed review, then gate the dump.

    Runs AFTER every graph.ainvoke has returned, so `settle_grace=0` is safe: the
    publish phase-end commit precedes ainvoke's return, and the two-transaction
    race the default 60s grace guards against cannot fire here. The DEFAULT grace
    would silently skip every just-completed review and ship a verdict-less dump
    (the dashboard Replay-% card aggregates persisted verdicts; the demo box
    never runs the projector sweep).

    Postconditions (any failure rejects the dump):
      1. projector `failed == 0` — a swallowed per-review exception is a dud seed;
      2. every completed non-eval review has a verdict (at-most-one is the partial
         unique index's job; at-least-one is asserted via anti-join) — correct on
         both the full and `--only` incremental paths, unlike pinning `projected`;
      3. no verdict on a non-completed review — the parked awaiting_approval
         entries are structurally excluded and must stay verdict-free;
      4. every verdict is replay_equivalent (capture-gate spirit: an inequivalent
         replay must never ship as the public demo's trust story).

    Returns an error string on failure, None on success. Idempotent under the
    checkpoint-restore retry path: emission is natural-key deduped.
    """
    engine = create_async_engine(demo_url, poolclass=NullPool)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        persister = AuditPersister(
            session_factory=session_factory, retention_settings=RetentionSettings()
        )
        result = await project_replay_verdicts(
            session_factory=session_factory,
            audit_persister=persister,
            settle_grace=timedelta(0),
        )
        if result["failed"]:
            return f"replay projection failed for {result['failed']} review(s)"
        async with session_factory() as session:
            missing = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM reviews r"
                        " WHERE r.status = 'completed' AND r.is_eval = false"
                        " AND NOT EXISTS (SELECT 1 FROM audit_events e"
                        "   WHERE e.review_id = r.id"
                        "   AND e.event_type = 'replay_verdict')"
                    )
                )
            ).scalar_one()
            if missing:
                return f"{missing} completed review(s) lack a replay verdict"
            stray = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM audit_events e"
                        " JOIN reviews r ON r.id = e.review_id"
                        " WHERE e.event_type = 'replay_verdict'"
                        " AND r.status <> 'completed'"
                    )
                )
            ).scalar_one()
            if stray:
                return f"{stray} replay verdict(s) on non-completed review(s)"
            inequivalent = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM audit_events e"
                        " WHERE e.event_type = 'replay_verdict'"
                        " AND e.payload->>'replay_equivalent' <> 'true'"
                    )
                )
            ).scalar_one()
            if inequivalent:
                return f"{inequivalent} replay verdict(s) not replay-equivalent"
        return None
    finally:
        await engine.dispose()


def _print_plan() -> None:
    print("  Demo seed plan (specs/2026-06-21-demo-deployment.md):", flush=True)
    for i, spec in enumerate(SEED_SPECS):
        src = spec.git_range or f"demo_fixtures/{spec.diff_file}"
        bits = []
        if spec.expected_finding_types:
            bits.append(", ".join(sorted(spec.expected_finding_types)))
        if spec.required_observed_proofs:
            bits.append(f"OBSERVED proofs: {', '.join(sorted(spec.required_observed_proofs))}")
        exp = "; ".join(bits) or "(model-dependent)"
        outcome = {
            "hitl": " +HITL (parks)",
            "published": " +auto-publish",
            "decided": " +HITL decided in-process",
            "any": "",
        }[spec.expected_outcome]
        print(f"    {i + 1}. {spec.key:<14} head_sha=…{_head_sha_for(i)[-6:]}  {src}", flush=True)
        print(f"        {spec.label} — expect: {exp}{outcome}", flush=True)
    print(
        f"\n  -> one review each into {_DEMO_DB_NAME}, then replay-verdict projection"
        " (grace=0; dump gated on: zero failures, a verdict per completed review,"
        " none on parked reviews, all equivalent),"
        f" then pg_dump -> {_SEED_SQL}",
        flush=True,
    )


async def _recreate_demo_db(admin_url: str) -> None:
    """Drop + recreate the demo DB — fresh every run, so the seed is reproducible."""
    await _drop_db(admin_url, _DEMO_DB_NAME)
    await _create_db(admin_url, _DEMO_DB_NAME)


def _seed_all(admin_url: str, api_key: str, *, only: str | None = None) -> int:
    """SYNC orchestration with per-review CHECKPOINT + retry on a transient LLM limit.

    Each async step gets its own asyncio.run(); _migrate runs in sync context (alembic
    env.py calls asyncio.run() internally and faults inside a running loop). After each
    successful review the demo DB is pg_dump'd to a checkpoint. On a 429/timeout the
    failed review is retried ALONE: restore the DB to that checkpoint (the prior good
    reviews — drop + reload, since the audit_events append-only trigger blocks deleting
    the partial review in place), wait for the per-minute window to clear, and re-run
    only that review. Retrying it against a FRESH token window — without re-running the
    cheap reviews to re-draw the window down — is the thing the whole-seed retry could
    never do, and what lets the heavy 27-file review finally land."""
    demo_url = _swap_db(admin_url, _DEMO_DB_NAME)
    asyncio.run(_recreate_demo_db(admin_url))
    if only is not None:
        # Append mode: restore the existing seed as the baseline (the prior reviews are kept
        # WITHOUT re-running them), then run only `only`. The restored dump also seeds the
        # retry-checkpoint, so a rate-limit retry resets to the baseline, not a fresh DB.
        specs = [(i, s) for i, s in enumerate(SEED_SPECS) if s.key == only]
        if not specs:
            print(f"  --only {only!r}: no spec with that key.", flush=True)
            return 1
        if not _SEED_SQL.exists():
            print(
                f"  --only needs an existing {_SEED_SQL.name} to append to; none found.",
                flush=True,
            )
            return 1
        if not _load_sql(demo_url, _SEED_SQL):
            print(f"  failed to restore {_SEED_SQL.name} for append.", flush=True)
            return 1
        _CHECKPOINT.write_bytes(_SEED_SQL.read_bytes())
        have_checkpoint = True
        print(
            f"  Demo DB .............. {_DEMO_DB_NAME} (recreated + restored {_SEED_SQL.name})",
            flush=True,
        )
        print(
            f"  Append mode .......... only [{only}]; {len(SEED_SPECS) - 1} prior reviews kept\n",
            flush=True,
        )
    else:
        specs = list(enumerate(SEED_SPECS))
        print(f"  Demo DB .............. {_DEMO_DB_NAME} (recreated)", flush=True)
        _migrate(demo_url)
        print("  Migrated ............. alembic upgrade head\n", flush=True)
        _CHECKPOINT.unlink(missing_ok=True)
        have_checkpoint = False

    total_attempts = len(_RATE_LIMIT_BACKOFF_SECONDS) + 1
    results: list[CaptureResult] = []
    for pos, (i, spec) in enumerate(specs):
        print(f"  === seeding [{pos + 1}/{len(specs)}] {spec.key} — {spec.label} ===", flush=True)
        cap = CaptureResult(spec.key, ok=False, detail="(no attempt ran)")  # overwritten below
        for attempt in range(total_attempts):
            if attempt:
                wait = _RATE_LIMIT_BACKOFF_SECONDS[attempt - 1]
                print(
                    f"  Rate limited on {spec.key} — waiting {wait}s, then retrying ONLY this "
                    f"review (attempt {attempt + 1}/{total_attempts}; prior reviews kept from "
                    "checkpoint, not re-run).",
                    flush=True,
                )
                time.sleep(wait)
                if not _restore_to_checkpoint(admin_url, demo_url, have_checkpoint=have_checkpoint):
                    print("  checkpoint restore failed — aborting.", flush=True)
                    return 1
            try:
                scenario = spec.build_scenario(_head_sha_for(i))
                review_id, structural_ok = asyncio.run(
                    _run(
                        demo_url,
                        api_key,
                        scenario,
                        expect_findings=spec.expect_findings,
                        pre_decide=spec.pre_decide,
                    )
                )
                cap = asyncio.run(_validate_capture(demo_url, str(review_id), spec))
                if not structural_ok:
                    cap = CaptureResult(
                        spec.key,
                        ok=False,
                        detail=(cap.detail + "; structural smoke FAILED").strip("; "),
                    )
                break
            except (LLMRateLimitError, LLMTimeoutError):
                if attempt == total_attempts - 1:
                    cap = CaptureResult(
                        spec.key,
                        ok=False,
                        detail=f"rate-limited after {total_attempts} attempts",
                    )
                    break
                continue  # retry this review (restore + wait at the top of the loop)
            except Exception as exc:  # noqa: BLE001 — one bad review must not abort the whole seed
                traceback.print_exc()  # full traceback to the log; the run continues
                cap = CaptureResult(
                    spec.key, ok=False, detail=f"CRASHED: {type(exc).__name__}: {exc}"
                )
                break
        results.append(cap)
        mark = "OK" if cap.ok else "FAIL"
        print(f"  --- capture [{mark}] {spec.key}: {cap.detail}\n", flush=True)
        # Checkpoint the good state so the NEXT review's retry restores here. Skip after
        # the last review — nothing retries past it.
        if cap.ok and pos < len(specs) - 1:
            if not _pg_dump(demo_url, _CHECKPOINT):
                print("  checkpoint dump failed — aborting (cannot retry reliably).", flush=True)
                return 1
            have_checkpoint = True

    _CHECKPOINT.unlink(missing_ok=True)
    print("  Capture summary:", flush=True)
    for r in results:
        print(f"    [{'OK' if r.ok else 'FAIL'}] {r.spec_key}: {r.detail}", flush=True)
    if not all(r.ok for r in results):
        print("\n  SEED REJECTED — a capture check failed; demo_seed.sql NOT written.", flush=True)
        if any(not r.ok and "rate-limited" in r.detail for r in results):
            print(
                "  A review exhausted its rate-limit retries even with checkpoint+resume —\n"
                "  this tier's tokens/min is below what the 27-file showcase needs in one window.\n"
                "  Raise the tier, re-run when quota is fresh, or land FUP-025 (per-call retry).",
                flush=True,
            )
        return 1

    print("  Projecting replay verdicts (grace=0; all graph runs returned) ...", flush=True)
    verdict_error = asyncio.run(_project_replay_verdicts_or_fail(demo_url))
    if verdict_error is not None:
        print(f"\n  SEED REJECTED — {verdict_error}; demo_seed.sql NOT written.", flush=True)
        return 1
    if not _pg_dump(demo_url, _SEED_SQL):
        return 1
    print(
        f"\n  Snapshot ............. {_SEED_SQL} (re-seedable with zero Claude spend)", flush=True
    )
    return 0


class _Tee:
    """Mirror a stream to a logfile so the WHOLE seed run lands in one file."""

    def __init__(self, stream: TextIO, logfile: TextIO) -> None:
        self._stream = stream
        self._logfile = logfile

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._logfile.write(s)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()
        self._logfile.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())


def _run_with_log(admin_url: str, api_key: str, *, only: str | None = None) -> int:
    """Tee all stdout+stderr — the seed's prints, _run's narration, any traceback —
    into ONE clearly-named log, so the run is inspectable after the fact (the live
    output scrolls past the terminal buffer otherwise)."""
    log_dir = _REPO_ROOT / "scripts" / "generated"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed_demo_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.txt"
    logfile = log_path.open("w", encoding="utf-8")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(orig_out, logfile), _Tee(orig_err, logfile)  # type: ignore[assignment]
    try:
        print(f"  Seed log ............. {log_path}\n", flush=True)
        return _seed_all(admin_url, api_key, only=only)
    finally:
        print(f"\n  Seed log ............. {log_path}", flush=True)
        sys.stdout, sys.stderr = orig_out, orig_err
        logfile.close()
        # Echo the path on the REAL stdout so it is unmissable even after the tee.
        print(f"  >> full seed log: {log_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the seed plan (entries + expectations) and exit; no DB, no Claude.",
    )
    parser.add_argument(
        "--only",
        metavar="KEY",
        default=None,
        help=(
            "append mode: restore the existing demo_seed.sql, run ONLY the spec with this "
            "key, re-dump the full seed. Adds one review without re-running the others."
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        return 0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("op://"):
        print("  ANTHROPIC_API_KEY missing or an unresolved op:// reference.", flush=True)
        print("  export a real key (or run via `op run --env-file=.env -- ...`).", flush=True)
        return 2

    # The demo seed needs its OWN budget + HITL timeout, but the documented run is
    # `op run --env-file=.env`, which loads .env's PRODUCTION values (a tight
    # OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS, OUTRIDER_HITL_TIMEOUT_MINUTES=30) into the
    # environment BEFORE this process starts — so `setdefault` would be a silent no-op
    # and the seed would starve the 27-file showcase / bake 30-min HITL expiry. Force the
    # demo floor, while still honoring a DELIBERATELY larger explicit override.
    _force_env_at_least("OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS", _DEMO_ANALYZE_BUDGET_TOKENS)
    _force_env_at_least("OUTRIDER_HITL_TIMEOUT_MINUTES", _DEMO_HITL_TIMEOUT_MINUTES)

    admin_url = _load_test_db_url()
    _assert_isolated(admin_url)
    # _seed_all is SYNC (it calls asyncio.run() per step); the demo DB is left in
    # place after — it backs the dump and is re-creatable by re-running. Wrapped so
    # the whole run is teed to one inspectable log.
    return _run_with_log(admin_url, api_key, only=args.only)


if __name__ == "__main__":
    raise SystemExit(main())
