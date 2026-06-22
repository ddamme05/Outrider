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
thin review): for every seeded review assert (a) no COST_BUDGET_EXHAUSTED skip,
(b) no finding_proposal_rejected AND no analyze_response_rejected (a wholesale
degraded analyze), (c) every expected finding type is present, and (d) the
expected HITL / audit rows exist. A review that fails any check is reported and
the seed exits non-zero — a dud is never dumped.

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
from dataclasses import dataclass, field
from pathlib import Path

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
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO_ROOT / "scripts" / "demo_fixtures"
_DEMO_DB_NAME = "outrider_test_demo"  # contains 'test' -> passes the 5433 isolation guard
_SEED_SQL = _REPO_ROOT / "scripts" / "demo_fixtures" / "demo_seed.sql"
_GIT_RANGE = "0c70d18^..39c538b"  # the 27-file analyze-cache arc (the #6 showcase)
_SHOWCASE_RANGE = _GIT_RANGE


@dataclass(frozen=True)
class SeedSpec:
    """One seeded review: how to build it + what its capture must contain."""

    key: str
    label: str
    expect_findings: bool  # feeds _run's structural --expect-findings verdict
    expected_finding_types: frozenset[str] = field(default_factory=frozenset)
    expect_hitl: bool = False  # a CRITICAL/HIGH finding should park at the gate
    diff_file: str | None = None  # a fixture under scripts/demo_fixtures/
    git_range: str | None = None  # OR a two-dot git range (the showcase)

    def build_scenario(self, head_sha: str) -> _Scenario:
        if self.git_range is not None:
            base = _scenario_from_git_range(self.git_range)
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
        expect_hitl=True,
    ),
    SeedSpec(
        key="auto_publish",
        label="Auto-publish (sub-HIGH multi-finding)",
        diff_file="api_request_handler.py",
        expect_findings=True,
        expected_finding_types=frozenset({"blocking_call_in_async", "missing_input_validation"}),
    ),
    SeedSpec(
        key="observed_proof",
        label="OBSERVED proof (weak_crypto, deterministic query_match_id)",
        diff_file="weak_crypto_handler.py",
        expect_findings=True,
        expected_finding_types=frozenset({"weak_crypto"}),
        # weak_crypto is HIGH (severity policy) -> trips the HITL gate, so this
        # review parks at AWAITING_APPROVAL like hitl_gate. Verify that row exists.
        expect_hitl=True,
    ),
    SeedSpec(
        key="breadth",
        label="Breadth across dimensions",
        diff_file="report_builder.py",
        expect_findings=True,
        # All three dimensions the fixture plants — breadth IS the demo's point, so
        # a review that collapses to one finding is a dud and must be rejected.
        expected_finding_types=frozenset(
            {"missing_input_validation", "n_plus_one_query", "missing_error_handling"}
        ),
    ),
    SeedSpec(
        key="scale_triage",
        label="Scale & triage (27-file self-review)",
        git_range=_SHOWCASE_RANGE,
        expect_findings=False,  # findings model-dependent across a real arc; gate loosely
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
            if starved:
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

            n_events = (
                await conn.execute(
                    text("SELECT count(*) FROM audit_events WHERE review_id = :id"),
                    {"id": review_id},
                )
            ).scalar_one()
            if n_events == 0:
                failures.append("no audit events persisted")

            if spec.expect_hitl:
                hitl = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM audit_events WHERE review_id = :id "
                            "AND event_type = 'hitl_request'"
                        ),
                        {"id": review_id},
                    )
                ).scalar_one()
                if not hitl:
                    failures.append("expected a HITL request event, found none")
    finally:
        await engine.dispose()

    if failures:
        return CaptureResult(spec.key, ok=False, detail="; ".join(failures))
    return CaptureResult(
        spec.key, ok=True, detail=f"{len(spec.expected_finding_types)} expected type(s) present"
    )


def _pg_dump(db_url: str, out_path: Path) -> bool:
    """pg_dump the seeded demo DB to a portable SQL artifact (libpq URL form)."""
    libpq = make_url(db_url).set(drivername="postgresql").render_as_string(hide_password=False)
    argv = ["pg_dump", "--dbname", libpq, "--no-owner", "--no-privileges", "--file", str(out_path)]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        print(f"  pg_dump failed: {proc.stderr.strip()}", flush=True)
        return False
    return True


def _print_plan() -> None:
    print("  Demo seed plan (specs/2026-06-21-demo-deployment.md):", flush=True)
    for i, spec in enumerate(SEED_SPECS):
        src = spec.git_range or f"demo_fixtures/{spec.diff_file}"
        exp = ", ".join(sorted(spec.expected_finding_types)) or "(model-dependent)"
        hitl = " +HITL" if spec.expect_hitl else ""
        print(f"    {i + 1}. {spec.key:<14} head_sha=…{_head_sha_for(i)[-6:]}  {src}", flush=True)
        print(f"        {spec.label} — expect: {exp}{hitl}", flush=True)
    print(f"\n  -> one review each into {_DEMO_DB_NAME}, then pg_dump -> {_SEED_SQL}", flush=True)


async def _seed_all(admin_url: str, api_key: str) -> int:
    demo_url = _swap_db(admin_url, _DEMO_DB_NAME)
    # Fresh DB every run -> reproducible seed.
    await _drop_db(admin_url, _DEMO_DB_NAME)
    await _create_db(admin_url, _DEMO_DB_NAME)
    print(f"  Demo DB .............. {_DEMO_DB_NAME} (recreated)", flush=True)
    _migrate(demo_url)
    print("  Migrated ............. alembic upgrade head\n", flush=True)

    results: list[CaptureResult] = []
    for i, spec in enumerate(SEED_SPECS):
        print(
            f"  === seeding [{i + 1}/{len(SEED_SPECS)}] {spec.key} — {spec.label} ===", flush=True
        )
        scenario = spec.build_scenario(_head_sha_for(i))
        review_id, structural_ok = await _run(
            demo_url, api_key, scenario, expect_findings=spec.expect_findings
        )
        cap = await _validate_capture(demo_url, str(review_id), spec)
        if not structural_ok:
            cap = CaptureResult(
                spec.key, ok=False, detail=(cap.detail + "; structural smoke FAILED").strip("; ")
            )
        results.append(cap)
        mark = "OK" if cap.ok else "FAIL"
        print(f"  --- capture [{mark}] {spec.key}: {cap.detail}\n", flush=True)

    print("  Capture summary:", flush=True)
    for r in results:
        print(f"    [{'OK' if r.ok else 'FAIL'}] {r.spec_key}: {r.detail}", flush=True)
    if not all(r.ok for r in results):
        print("\n  SEED REJECTED — a capture check failed; demo_seed.sql NOT written.", flush=True)
        return 1

    if not _pg_dump(demo_url, _SEED_SQL):
        return 1
    print(
        f"\n  Snapshot ............. {_SEED_SQL} (re-seedable with zero Claude spend)", flush=True
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the seed plan (entries + expectations) and exit; no DB, no Claude.",
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

    admin_url = _load_test_db_url()
    _assert_isolated(admin_url)
    try:
        return asyncio.run(_seed_all(admin_url, api_key))
    finally:
        # The demo DB is intentionally LEFT in place (it backs the dump + lets you
        # inspect it). Re-running drops and recreates it.
        pass


if __name__ == "__main__":
    raise SystemExit(main())
