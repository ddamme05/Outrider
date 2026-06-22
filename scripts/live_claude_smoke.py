#!/usr/bin/env python3
"""Live-Claude smoke — real Anthropic call through analyze/synthesize, no GitHub.

The first real-service step of the live-demo arc (Phase 1, step 1). It answers
ONE question that no green test answers: does a REAL Claude response survive
Outrider's parsing, severity policy, synthesize aggregation, audit emission, and
replay reconstruction?

It is `scripts/smoke_e2e.py` with exactly ONE boundary swapped: the scripted LLM
becomes the real `AnthropicProvider` (real Sonnet analyze + Haiku triage/
synthesize). The other boundary stays faked — a stub GitHub client serves a
LOCAL synthetic diff (no GitHub App, no token, no network to GitHub) and a
recording publisher captures the would-be review without posting. Real Postgres,
real AuditPersister, real replay. So the only new variable vs the green suite is
"the LLM is really Claude."

It reuses the committed test's fakes + scenario verbatim (single source of truth,
no second scenario to drift).

What it proves (hard-asserted): the graph runs with real Claude; analyze produces
at least one AnalysisRound; the audit stream persists; `reconstruct` +
`assert_replay_equivalent` pass over the real stream.
What it reports but does NOT assert (Claude-dependent): how many findings, their
severities, and whether the run reached publish or paused at the HITL gate (a
CRITICAL/HIGH finding legitimately interrupts before publish — that's the gate
working, not a failure).

Run it:

    docker compose up -d postgres-test
    export ANTHROPIC_API_KEY=sk-ant-...          # the only live credential
    export TEST_DATABASE_URL=postgresql+psycopg://...:5433/outrider_test
    uv run python scripts/live_claude_smoke.py

It creates a throwaway DB on the test container, migrates it, runs one review
through the real LLM, prints the findings + audit stream + replay verdict, then
drops the DB. Exit 0 = the structural checks passed (Claude produced a
replayable run); 1 = something broke; 2 = setup/credentials missing.

Cost: one real review — a Haiku triage + Sonnet analyze + Haiku synthesize pass
over a tiny synthetic file. Cents, not dollars. No cost cap is wired here beyond
the synthetic diff's small size; do not point this at large inputs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from collections.abc import Callable

# Bare import: running `python scripts/live_claude_smoke.py` puts scripts/ at
# sys.path[0], so the sibling helper resolves without packaging scripts/.
from _git_range_scenario import (
    FileEntry,
    GitRangeError,
    build_file_entries_from_range,
    summarize_dry_run,
)
from _narrate import (
    narrate_audit_stream,
    narrate_db_state,
    narrate_llm_exchanges_from_db,
    narrate_recorded_publisher,
)
from _trace_log import TraceTee

# Repo root on sys.path so `tests.integration.*` imports resolve when run as a
# plain script (tests/ is a namespace package; `outrider` is an editable install).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool  # noqa: E402

# Reuse the committed e2e-smoke fakes + scenario verbatim (no drift). The default
# (no --diff-file) path runs the exact proven scenario; --diff-file swaps the
# analyzed file via the local Scenario stubs below.
from tests.integration.test_e2e_smoke import (  # noqa: E402
    _RecordingPublisher,
    _seed_installation,
    _seed_review,
    _seed_state,
    _stub_github_factory,
    _StubImportPathResolver,
)

from outrider.agent.graph import build_graph  # noqa: E402
from outrider.agent.nodes.hitl_config import HITLConfig  # noqa: E402
from outrider.agent.nodes.patch_config import PatchConfig  # noqa: E402
from outrider.anomaly.persister import AnomalyPersister  # noqa: E402
from outrider.audit.config import RetentionSettings  # noqa: E402
from outrider.audit.persister import AuditPersister  # noqa: E402
from outrider.audit.replay import AuditReplayer  # noqa: E402
from outrider.cache import AnalyzeCacheStore  # noqa: E402
from outrider.db.review_status_persister import ReviewStatusPersister  # noqa: E402
from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from outrider.llm.config import ModelConfig  # noqa: E402

_RULE = "=" * 62
_INSTALLATION_ID = 12345  # matches tests.integration.test_e2e_smoke._INSTALLATION_ID
_HEAD_SHA = "b" * 40  # matches the seed PRContext head_sha
_BASE_SHA = "a" * 40  # matches the seed PRContext base_sha (factory keys base content on it)


@dataclass(frozen=True)
class _Scenario:
    """One or more files for the live run, as the (stubbed) GitHub client sees them.

    Intake re-fetches content from the stub and rebuilds `ChangedFile`, so the
    analyzed content comes from these `FileEntry` records, NOT the seed state.
    `--diff-file` builds a single ADDED file (every line is a changed line, so any
    finding Claude raises lands inline). `--git-range START..END` builds N real
    modified/added/etc. files reconstructed from local history (the #6 showcase).
    """

    files: tuple[FileEntry, ...]
    pr_title: str
    label: str  # what the run narrates as "analyzing ..."


def _scenario_from_file(diff_path: Path) -> _Scenario:
    content = diff_path.read_text()
    # Serve under `src/` so the synthetic ADDED file lands inline like real
    # production code under review. Triage tiers on the diff content (which it
    # sees in full), not on the directory; the path mainly affects how the
    # finding routes at publish, not whether analyze runs.
    rel = f"src/{diff_path.name}"
    body_lines = content.splitlines()
    # Synthetic all-added unified-diff patch (status="added"): /dev/null -> file.
    # NOTE: this single-file path keeps its proven header-bearing patch shape; the
    # multi-file --git-range path below is wire-faithful (hunks-only) instead.
    patch = (
        f"--- /dev/null\n+++ b/{rel}\n@@ -0,0 +1,{len(body_lines)} @@\n"
        + "\n".join(f"+{line}" for line in body_lines)
        + "\n"
    )
    entry = FileEntry(
        path=rel,
        status="added",
        additions=len(body_lines),
        deletions=0,
        patch=patch,
        content_base=None,
        content_head=content,
        previous_path=None,
    )
    return _Scenario(files=(entry,), pr_title=f"Add {rel}", label=rel)


def _scenario_from_git_range(range_spec: str) -> _Scenario:
    """Reconstruct a faithful N-file PR diff from a local two-dot git range.

    The #6 "Outrider reviewing Outrider" showcase: real hunks/status/base/head per
    file (see scripts/_git_range_scenario.py), so most files carry small changes
    triage SKIM/SKIPs and only the substantive few go DEEP.
    """
    entries = build_file_entries_from_range(range_spec, _REPO_ROOT)
    if not entries:
        raise GitRangeError(f"no changed files in range {range_spec!r}")
    return _Scenario(
        files=tuple(entries),
        pr_title=f"Outrider self-review: {range_spec}",
        label=f"{len(entries)} files from {range_spec}",
    )


def _make_scenario_github_factory(scenario: _Scenario) -> Callable[[int], object]:
    """Stub GitHub serving `scenario`'s files via the real intake fetch path.

    `async_list_files` returns one meta per file (wire-faithful: `previous_filename`
    and `patch` use the empty-string sentinel GitHubKit emits for the
    non-applicable case, which intake collapses to None via `or None`).
    `async_get_content` serves base/head content keyed on (path, ref) — head at
    `_HEAD_SHA`, base at `_BASE_SHA`, and a rename's base at its previous path.
    """

    # (path, ref_sha) -> file text. Built per-status from the FileEntry records so
    # the stub answers exactly the per-status fetches intake makes.
    content_by_path_ref: dict[tuple[str, str], str] = {}
    for f in scenario.files:
        if f.content_head is not None:
            content_by_path_ref[(f.path, _HEAD_SHA)] = f.content_head
        if f.content_base is not None:
            content_by_path_ref[(f.previous_path or f.path, _BASE_SHA)] = f.content_base

    @dataclass
    class _Meta:
        filename: str
        status: str
        additions: int
        deletions: int
        patch: str | None = None
        previous_filename: str | None = None

    @dataclass
    class _ContentFile:
        encoding: str
        content: str

    @dataclass
    class _Resp:
        parsed_data: object

    class _Repos:
        async def async_get_content(self, owner: str, repo: str, path: str, *, ref: str) -> _Resp:
            text_content = content_by_path_ref.get((path, ref))
            if text_content is None:
                raise KeyError(f"stub has no content for {path!r} at ref {ref!r}")
            return _Resp(
                _ContentFile(
                    encoding="base64",
                    # encodebytes -> newline-wrapped base64, matching GitHub's
                    # contents API for content past ~60 base64 chars (real files).
                    content=base64.encodebytes(text_content.encode()).decode("ascii"),
                )
            )

    class _Pulls:
        async def async_list_files(
            self, owner: str, repo: str, pull_number: int, **kwargs: object
        ) -> _Resp:
            return _Resp(
                [
                    _Meta(
                        filename=f.path,
                        status=f.status,
                        additions=f.additions,
                        deletions=f.deletions,
                        patch=f.patch if f.patch is not None else "",
                        previous_filename=f.previous_path or "",
                    )
                    for f in scenario.files
                ]
            )

    class _Rest:
        def __init__(self) -> None:
            self.repos = _Repos()
            self.pulls = _Pulls()

    class _GitHub:
        def __init__(self) -> None:
            self.rest = _Rest()

    def _factory(installation_id: int) -> object:
        if installation_id != _INSTALLATION_ID:
            raise ValueError(f"unexpected installation_id {installation_id}")
        return _GitHub()

    return _factory


def _seed_state_for_scenario(review_id: UUID, scenario: _Scenario) -> object:
    """Seed ReviewState whose PRContext carries the scenario's files.

    Intake rebuilds `changed_files` from the stub fetch, so only the PR
    coordinates (owner/repo/pr_number/shas/installation) are load-bearing here;
    the seed's `changed_files` just has to construct as valid per-status shapes
    (which also catches any reconstruction that violates a status invariant before
    a cent is spent).
    """
    from outrider.schemas.pr_context import ChangedFile, PRContext
    from outrider.schemas.review_state import ReviewState

    changed = tuple(
        ChangedFile(
            path=f.path,
            status=f.status,
            additions=f.additions,
            deletions=f.deletions,
            patch=f.patch,
            content_base=f.content_base,
            content_head=f.content_head,
            previous_path=f.previous_path,
            language="python" if f.is_python else None,
        )
        for f in scenario.files
    )
    return ReviewState(
        review_id=review_id,
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=_INSTALLATION_ID,
            owner="acme",
            repo="widget",
            pr_number=7,
            base_sha=_BASE_SHA,
            head_sha=_HEAD_SHA,
            pr_title=scenario.pr_title,
            pr_body=None,
            author="someone",
            total_additions=sum(f.additions for f in scenario.files),
            total_deletions=sum(f.deletions for f in scenario.files),
            changed_files=changed,
        ),
        is_eval=False,
    )


# Full trace tees to scripts/generated/ — shared recipe, scripts/_trace_log.py.
_TRACE: TraceTee | None = None


def _say(msg: str = "") -> None:
    print(msg, flush=True)
    if _TRACE is not None:
        _TRACE.write_line(msg)


def _redact(url: str) -> str:
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001 — an unparseable URL must still not leak
        return "<unparseable-url>"


# ---------------------------------------------------------------------------
# Ephemeral DB lifecycle (mirrors scripts/smoke_e2e.py / conftest.fresh_db)
# ---------------------------------------------------------------------------


def _load_test_db_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        return url
    env_file = _REPO_ROOT / ".env"
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if line.startswith("TEST_DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        "2: TEST_DATABASE_URL is not set and not found in .env. "
        "Run `set -a && source .env && set +a` and bring up postgres-test "
        "(`docker compose up -d postgres-test`)."
    )


def _assert_isolated(url: str) -> None:
    parsed = make_url(url)
    if parsed.port != 5433:
        raise SystemExit(
            f"2: refusing — TEST_DATABASE_URL must target port 5433 (postgres-test); "
            f"got {_redact(url)}"
        )
    if "test" not in (parsed.database or "").lower():
        raise SystemExit(
            f"2: refusing — TEST_DATABASE_URL db name must contain 'test'; got {_redact(url)}"
        )


def _swap_db(url: str, new_db: str) -> str:
    return make_url(url).set(database=new_db).render_as_string(hide_password=False)


async def _create_db(admin_url: str, db_name: str) -> None:
    eng = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with eng.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    finally:
        await eng.dispose()


async def _drop_db(admin_url: str, db_name: str) -> None:
    eng = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with eng.connect() as conn:
            await conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": db_name},
            )
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
    finally:
        await eng.dispose()


def _migrate(db_url: str) -> None:
    prior = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = db_url
    try:
        cfg = Config(str(_REPO_ROOT / "alembic.ini"), toml_file=str(_REPO_ROOT / "pyproject.toml"))
        command.upgrade(cfg, "head")
    finally:
        if prior is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prior


# ---------------------------------------------------------------------------
# The live run
# ---------------------------------------------------------------------------


async def _run(
    db_url: str, api_key: str, scenario: _Scenario | None, *, expect_findings: bool
) -> bool:
    engine = create_async_engine(db_url, poolclass=NullPool)
    try:
        return await _drive(engine, api_key, scenario, expect_findings=expect_findings)
    finally:
        await engine.dispose()


async def _drive(
    engine: AsyncEngine, api_key: str, scenario: _Scenario | None, *, expect_findings: bool
) -> bool:
    review_id = uuid4()
    await _seed_installation(engine)
    # Persist the same PR title the agent state carries, so the dashboard shows
    # it (the direct-invoke path bypasses the webhook that normally sets it).
    pr_title = scenario.pr_title if scenario is not None else "Add vulnerable handler"
    await _seed_review(engine, review_id, pr_title=pr_title)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    persister = AuditPersister(
        session_factory=session_factory, retention_settings=RetentionSettings()
    )
    publisher = _RecordingPublisher()
    # THE one swap vs scripts/smoke_e2e.py: the real Anthropic provider.
    # model_config reads OUTRIDER_MODEL_* env (defaults to current Claude tiers).
    provider = AnthropicProvider(
        api_key=SecretStr(api_key),
        model_config=ModelConfig(),
        persister=persister,
    )

    # Default (no --diff-file): the exact proven e2e scenario. With --diff-file:
    # local stubs serve the supplied file (intake re-fetches from these).
    if scenario is None:
        github_factory: Callable[[int], object] = _stub_github_factory
        seed_state = _seed_state(review_id)
        analyzed_label = "src/handler.py (built-in synthetic diff)"
    else:
        github_factory = _make_scenario_github_factory(scenario)
        seed_state = _seed_state_for_scenario(review_id, scenario)
        analyzed_label = scenario.label

    graph = build_graph(
        db_factory=session_factory,
        github_factory=github_factory,
        provider=provider,
        model_config=ModelConfig(),
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
        # Required since the suggested-patches arc; OFF here to keep the live
        # spend bounded to the review calls themselves.
        patch_config=PatchConfig(patches_enabled=False),
        checkpointer=InMemorySaver(),
        publisher=publisher,
        import_path_resolver=_StubImportPathResolver(),
        # Production-parity shadow wiring (mirrors api/lifespan.py). Writes go
        # to this script's ephemeral test DB; telemetry is queryable there
        # until the run's DB is dropped.
        analyze_cache_store=AnalyzeCacheStore(session_factory=session_factory),
    )
    _say(
        f"  Models ............... {ModelConfig().analyze_model} (analyze) + "
        f"{ModelConfig().triage_model} (triage/synthesize)"
    )
    _say(f"  Calling real Claude .. analyzing {analyzed_label}")
    _say()

    result = await graph.ainvoke(seed_state, config={"configurable": {"thread_id": str(review_id)}})
    await provider.aclose()

    interrupted = "__interrupt__" in result
    await _report(engine, review_id, result, publisher, interrupted=interrupted)
    # Full-granularity dumps (same recipe as scripts/smoke_e2e.py): the real
    # provider persists every exchange, so the USER prompt + Claude's REAL
    # response come from llm_call_content (system prompt rides as hash +
    # template version per #016 — reconstructable, not retained as text).
    await narrate_audit_stream(_say, engine, review_id)
    await narrate_llm_exchanges_from_db(_say, engine, review_id)
    narrate_recorded_publisher(_say, publisher)
    await narrate_db_state(_say, engine)
    return await _verify(
        engine,
        session_factory,
        review_id,
        publisher=publisher,
        interrupted=interrupted,
        expect_findings=expect_findings,
    )


async def _report(
    engine: AsyncEngine,
    review_id: UUID,
    result: dict,  # type: ignore[type-arg]
    publisher: _RecordingPublisher,
    *,
    interrupted: bool,
) -> None:
    # Findings Claude actually produced (report-only; count/severity vary).
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
        n_events = (
            await conn.execute(
                text("SELECT count(*) FROM audit_events WHERE review_id = :id"),
                {"id": review_id},
            )
        ).scalar_one()
    # Analyze examined/skipped + per-file examination — surfaces WHY analyze did
    # or did not call the LLM (a SKIM/SKIP tier means analyze skips the file, so
    # examined=0 + llm_calls=0). Counters come from AnalyzeCompletedEvent; the
    # triage tiers themselves live in graph state (not the audit stream), so they
    # are read from the returned `result`, not queried from audit_events.
    async with engine.begin() as conn:
        analyze_completed = (
            await conn.execute(
                text(
                    "SELECT payload->>'n_files_analyzed', payload->>'n_files_skipped', "
                    "payload->>'n_llm_calls' "
                    "FROM audit_events WHERE review_id = :id "
                    "AND event_type = 'analyze_completed' ORDER BY sequence_number LIMIT 1"
                ),
                {"id": review_id},
            )
        ).all()
        file_exams = (
            await conn.execute(
                text(
                    "SELECT payload->>'node_id', payload->>'file_path', "
                    "payload->>'parse_status', payload->>'skip_reason' "
                    "FROM audit_events WHERE review_id = :id "
                    "AND event_type = 'file_examination' ORDER BY sequence_number"
                ),
                {"id": review_id},
            )
        ).all()

    _say("  Triage ...............")
    triage_result = result.get("triage_result")
    if triage_result is not None:
        risk = getattr(triage_result, "overall_risk", None)
        risk_name = getattr(risk, "value", risk)
        dims = getattr(triage_result, "relevant_dimensions", ()) or ()
        dim_names = [getattr(d, "value", d) for d in dims]
        # overall_risk + the dimension scope analyze hunts within: if a planted
        # issue's dimension is absent here, analyze was never asked to look for
        # it, which explains a clean result on a file that clearly has a flaw.
        _say(f"    overall_risk={risk_name}  dimensions={dim_names}")
    tiers = getattr(triage_result, "file_tiers", None)
    if tiers:
        for path, tier in tiers.items():
            tier_name = getattr(tier, "value", tier)
            _say(f"    tier {path}: {tier_name}")
    else:
        _say("    (no triage_result in returned state)")
    if analyze_completed:
        examined, skipped, n_llm = analyze_completed[0]
        _say(f"  Analyze .............. examined={examined} skipped={skipped} llm_calls={n_llm}")
    for node_id, path, status, skip in file_exams:
        extra = f" skip_reason={skip}" if skip else ""
        _say(f"    file_examination[{node_id}] {path}: {status}{extra}")
    _say()

    _say("  Real Claude produced:")
    if findings:
        for ft, sev in findings:
            _say(f"    - {ft} ({sev})")
    else:
        _say(
            "    (no findings — see triage/analyze trace above for whether "
            "analyze even ran on the file)"
        )
    _say()
    _say(f"  Audit events ......... {n_events} rows persisted")
    if interrupted:
        _say(
            "  Outcome .............. PAUSED at HITL gate (a CRITICAL/HIGH finding "
            "interrupted before publish — the gate working as designed)"
        )
    elif publisher.create_review_calls:
        n = len(publisher.create_review_calls[0]["comments"])
        _say(
            f"  Outcome .............. reached publish; would post {n} inline comment(s) "
            "(recording publisher — nothing sent to GitHub)"
        )
    else:
        _say("  Outcome .............. reached publish; no inline-eligible findings")
    _say()


async def _verify(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[object],  # type: ignore[type-arg]
    review_id: UUID,
    *,
    publisher: _RecordingPublisher,
    interrupted: bool,
    expect_findings: bool,
) -> bool:
    checks: list[tuple[str, bool, str]] = []

    # Structural (Claude-independent) — these MUST hold for any real run.
    async with engine.begin() as conn:
        n_phase = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM audit_events WHERE review_id = :id "
                    "AND event_type = 'review_phase'"
                ),
                {"id": review_id},
            )
        ).scalar_one()
        analyze_ran = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM audit_events WHERE review_id = :id "
                    "AND event_type = 'analyze_completed'"
                ),
                {"id": review_id},
            )
        ).scalar_one()
        # `n_files_analyzed` summed across analyze passes — the honest signal that
        # analyze actually examined the file (not just that the node ran). A
        # SKIM/SKIP-tiered file yields analyze_completed events with
        # n_files_analyzed=0, which is why counting analyze_completed alone is a
        # vacuous "analyze ran" claim.
        files_analyzed = (
            await conn.execute(
                text(
                    "SELECT COALESCE(SUM((payload->>'n_files_analyzed')::int), 0) "
                    "FROM audit_events WHERE review_id = :id "
                    "AND event_type = 'analyze_completed'"
                ),
                {"id": review_id},
            )
        ).scalar_one()
        n_llm = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM audit_events WHERE review_id = :id "
                    "AND event_type = 'llm_call'"
                ),
                {"id": review_id},
            )
        ).scalar_one()
    checks.append(("graph emitted phase events", n_phase > 0, f"{n_phase} phase events"))
    checks.append(("analyze node ran", analyze_ran > 0, f"{analyze_ran} analyze pass(es)"))
    # Distinct from the above: did analyze actually examine the file? This is the
    # check that catches a file silently tiered out of analysis.
    checks.append(
        (
            "analyze examined the file (real Claude analysis)",
            files_analyzed > 0,
            f"{files_analyzed} file(s) analyzed",
        ),
    )
    checks.append(("real LLMCallEvents persisted", n_llm > 0, f"{n_llm} llm calls"))

    if expect_findings:
        # Semantic gate (opt-in): the run must ADMIT findings, not merely run.
        # Catches the failure mode where the model found issues but its
        # response was rejected wholesale (e.g. invalid JSON) — structurally a
        # pass, semantically empty. See the 2026-06-12 unescaped-quote run.
        async with engine.begin() as conn:
            n_findings = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM audit_events WHERE review_id = :id "
                        "AND event_type = 'finding'"
                    ),
                    {"id": review_id},
                )
            ).scalar_one()
            n_resp_rejected = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM audit_events WHERE review_id = :id "
                        "AND event_type = 'analyze_response_rejected'"
                    ),
                    {"id": review_id},
                )
            ).scalar_one()
        checks.append(
            (
                "--expect-findings: findings were admitted",
                n_findings > 0,
                f"{n_findings} finding(s), {n_resp_rejected} response rejection(s)",
            )
        )
        # Publish side of the same gate: admitted findings must reach the
        # (recording) publisher with at least one comment — UNLESS the HITL
        # gate interrupted, which is the gate working, not a publish failure.
        if interrupted:
            checks.append(
                (
                    "--expect-findings: publish side",
                    True,
                    "deferred at the HITL gate (CRITICAL/HIGH finding) — gate working",
                )
            )
        else:
            calls = publisher.create_review_calls
            n_comments = len(calls[0]["comments"]) if calls else 0
            checks.append(
                (
                    "--expect-findings: publish posted >=1 comment",
                    len(calls) >= 1 and n_comments >= 1,
                    f"{len(calls)} publish call(s), {n_comments} comment(s)",
                )
            )

    # Replay over the real stream — the headline capability.
    replayer = AuditReplayer(session_factory=session_factory)  # type: ignore[arg-type]
    try:
        review = await replayer.reconstruct(review_id)
        checks.append(("reconstruct succeeds", True, f"mode={review.mode.value}"))
        await replayer.assert_replay_equivalent(review_id)
        checks.append(("assert_replay_equivalent passes", True, ""))
    except Exception as exc:  # noqa: BLE001 — surface any replay failure
        checks.append(("replay", False, f"{type(exc).__name__}: {exc}"))

    _say("  Structural checks (exit verdict = all must pass):")
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        tail = f"  {detail}" if detail else ""
        _say(f"    [{mark}] {name}{tail}")
    _say()
    return all(ok for _, ok, _ in checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diff-file",
        type=Path,
        default=None,
        help=(
            "Path to a source file to analyze as an ADDED file (real Claude reviews "
            "its full contents). Try scripts/demo_fixtures/api_request_handler.py — its "
            "diff is substantial and clearly security-relevant so triage tiers it "
            "DEEP/STANDARD (triage reads the diff content, the precondition for analyze "
            "to run), and its body has MEDIUM vulns that publish. Omit to run the "
            "built-in synthetic diff."
        ),
    )
    parser.add_argument(
        "--expect-findings",
        action="store_true",
        help=(
            "fail the run unless findings were ADMITTED and (absent a HITL interrupt) "
            "the publisher captured at least one comment. By default the verdict is "
            "structural — a wholesale response rejection (e.g. the model emitted "
            "invalid JSON) still passes. Pass this when the run is meant to prove the "
            "findings -> admission -> publish flow on a fixture with known vulns; a "
            "HITL pause counts as a pass (the gate firing IS the flow working)."
        ),
    )
    parser.add_argument(
        "--git-range",
        default=None,
        metavar="START..END",
        help=(
            "Reconstruct a real N-file PR diff from a local two-dot git range and "
            "review it live — the #6 'Outrider reviewing Outrider' showcase (e.g. "
            "0c70d18^..39c538b). Mutually exclusive with --diff-file. Pair with "
            "--dry-run to preview the offline file/status/budget summary first."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "With --git-range: print the offline reconstruction summary (files, "
            "status, line deltas, budget bounds), validate the seed PRContext "
            "constructs, then exit WITHOUT touching the DB or calling Claude — the "
            "honest preview before the paid seed run."
        ),
    )
    args = parser.parse_args()

    if args.diff_file is not None and args.git_range is not None:
        _say("  --diff-file and --git-range are mutually exclusive.")
        return 2

    scenario: _Scenario | None = None
    if args.diff_file is not None:
        if not args.diff_file.is_file():
            _say(f"  --diff-file not found: {args.diff_file}")
            return 2
        scenario = _scenario_from_file(args.diff_file)
    elif args.git_range is not None:
        try:
            scenario = _scenario_from_git_range(args.git_range)
        except GitRangeError as exc:
            _say(f"  --git-range error: {exc}")
            return 2

    if args.dry_run:
        if scenario is None:
            _say("  --dry-run requires --git-range (nothing to preview otherwise).")
            return 2
        _say(_RULE)
        _say("  Outrider — git-range dry-run (offline · no DB · no Claude)")
        _say(_RULE)
        _say()
        _say(summarize_dry_run(list(scenario.files), args.git_range))
        _say()
        # Validate the seed PRContext + all N ChangedFile construct — catches a
        # reconstruction that violates a status invariant before the paid run.
        try:
            _seed_state_for_scenario(uuid4(), scenario)
        except Exception as exc:  # noqa: BLE001 — surface any construction failure
            _say(f"  SEED CONSTRUCTION FAILED: {type(exc).__name__}: {exc}")
            return 1
        _say(f"  seed PRContext + {len(scenario.files)} ChangedFile construct OK")
        _say()
        return 0

    _say(_RULE)
    _say("  Outrider — live Claude smoke (real LLM · fake GitHub · real DB)")
    _say(_RULE)
    _say()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _say("  ANTHROPIC_API_KEY is not set — this runner needs a real Claude key.")
        _say("  export ANTHROPIC_API_KEY=sk-ant-...  then re-run.")
        return 2
    if api_key.startswith("op://"):
        # Sourcing .env does NOT resolve 1Password references — the literal
        # op:// string passes an is-set check, then fails auth deep in triage.
        _say("  ANTHROPIC_API_KEY is a 1Password reference (op://...), not a key.")
        _say("  Run through op so the reference resolves:")
        _say("    op run --env-file=.env -- uv run python scripts/live_claude_smoke.py")
        return 2

    admin_url = _load_test_db_url()
    _assert_isolated(admin_url)
    db_name = f"outrider_test_liveclaude_{uuid4().hex[:8]}"
    db_url = _swap_db(admin_url, db_name)

    asyncio.run(_create_db(admin_url, db_name))
    _say(f"  Ephemeral DB ......... {db_name} (created)")
    try:
        _migrate(db_url)
        _say("  Migrated ............. alembic upgrade head")
        _say()
        ok = asyncio.run(_run(db_url, api_key, scenario, expect_findings=args.expect_findings))
    finally:
        asyncio.run(_drop_db(admin_url, db_name))
        _say(f"  Ephemeral DB ......... {db_name} (dropped)")
        _say()

    _say(_RULE)
    _say("  LIVE CLAUDE SMOKE PASSED" if ok else "  LIVE CLAUDE SMOKE FAILED")
    _say(_RULE)
    return 0 if ok else 1


def _main_with_log() -> int:
    """Tee the run to scripts/generated/ and name the file on both ends."""
    global _TRACE  # noqa: PLW0603 — single assignment per process, set before any _say
    _TRACE = TraceTee("live_claude_smoke")
    print(f"  Full trace ........... {_TRACE.path}", flush=True)
    try:
        return main()
    except Exception:
        _TRACE.write_current_exception()
        raise
    finally:
        print(f"  Full trace ........... {_TRACE.path}", flush=True)
        _TRACE.close()
        _TRACE = None


if __name__ == "__main__":
    raise SystemExit(_main_with_log())
