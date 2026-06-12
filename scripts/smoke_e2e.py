#!/usr/bin/env python3
"""Interactive end-to-end smoke runner — watch a review flow through the graph.

The narrating sibling of `tests/integration/test_e2e_smoke.py`. That pytest is
the authoritative CI gate; this script RUNS the same scenario and prints what
happened so you can eyeball the whole pipeline end to end:

    real compiled graph  ->  real AuditPersister/ReviewStatusPersister/AnomalyPersister
    ->  real Postgres  ->  publish  ->  replay-equivalence.

It reuses the test's fakes + scenario verbatim (single source of truth — no second
less-governed scenario to drift), so what you watch here is exactly what CI gates.
The only fakes are the two network boundaries: a scripted LLM (no Anthropic call)
and a fake GitHub (stub fetch + recording publisher).

This is a DEVELOPER smoke / demo harness, not a stable operator runbook: it
imports private helpers from the test module, so its surface tracks the test, not
a public contract. For the authoritative gate, run the pytest.

Crucially, the exit verdict is NOT just "replay passed": the script re-runs the
SAME hard checks the pytest asserts (publish posted inline, review completed,
PublishEvent landed, finding emitted, expected phase coverage, FULL-mode replay)
and FAILS (exit 1) if any of them fail — so a broken pipeline cannot print
green. Each check prints PASS/FAIL individually.

Run it (needs the postgres-test container up, like any integration test):

    docker compose up -d postgres-test
    set -a && source .env && set +a            # provides TEST_DATABASE_URL
    uv run python scripts/smoke_e2e.py

It creates a throwaway DB on the test container, migrates it, runs the review,
prints the trace + audit stream + per-check verdicts, then drops the DB. Exit
code 0 = every check passed, 1 = something broke (or setup failed).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import UUID, uuid4

# Repo root on sys.path so `tests.integration.*` imports resolve when run as a
# plain script (tests/ is a namespace package; `outrider` is an editable install
# already on the path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool  # noqa: E402

# Reuse the committed test's fakes + scenario verbatim (no drift).
from tests.integration.test_e2e_smoke import (  # noqa: E402
    _FILE_PATH,
    _analyze_response,
    _RecordingPublisher,
    _ScriptedLLMProvider,
    _seed_installation,
    _seed_review,
    _seed_state,
    _stub_github_factory,
    _StubImportPathResolver,
    _triage_response,
)

from outrider.agent.graph import build_graph  # noqa: E402
from outrider.agent.nodes.hitl_config import HITLConfig  # noqa: E402
from outrider.agent.nodes.patch_config import PatchConfig  # noqa: E402
from outrider.anomaly.persister import AnomalyPersister  # noqa: E402
from outrider.audit.config import RetentionSettings  # noqa: E402
from outrider.audit.events import CacheLookupEvent, PublishEvent  # noqa: E402
from outrider.audit.persister import AuditPersister  # noqa: E402
from outrider.audit.replay import AuditReplayer, ReplayMode  # noqa: E402
from outrider.cache import AnalyzeCacheStore  # noqa: E402
from outrider.db.review_status_persister import ReviewStatusPersister  # noqa: E402
from outrider.eval_support import (  # noqa: E402
    EvalDBIsolationError,
    assert_test_url_is_isolated,
    create_database,
    drop_database,
    replace_db_name,
    run_alembic_upgrade_head,
)
from outrider.llm.config import ModelConfig  # noqa: E402

_RULE = "=" * 62
# The nodes this fixture actually exercises (trace is skipped: no trace
# candidates; hitl is a pass-through: sub-HIGH finding). Mirrors the phase-
# coverage assertion in tests/integration/test_e2e_smoke.py.
_EXPECTED_NODES = frozenset({"intake", "triage", "analyze", "synthesize", "hitl", "publish"})

# Run 2's head SHA — the re-push shape: a NEW head SHA for the same PR with
# byte-identical file content/patch. head_sha never enters the per-file analyze
# prompt, so run 2 renders the same prompt bytes and the analyze cache must
# report would_hit. A two-run rerun of THIS SCRIPT stays deterministic because
# every invocation gets its own ephemeral scratch DB (empty cache table).
#
# NOTE this proves the cache MECHANISM, not a production hit rate: a fixture
# re-run is ~100% would-hit BY CONSTRUCTION (identical inputs + versions). The
# serve-flip evidence is the dev-DB ledger fed by real reviews.
_RUN2_HEAD_SHA = "c" * 40


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _dump_json(obj: object) -> str:
    """Pretty JSON with non-JSON types (UUID/datetime/Decimal) stringified."""
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def _say_block(prefix: str, body: str) -> None:
    """Print a multi-line body with a per-line prefix so sections stay scannable."""
    for line in body.splitlines() or [""]:
        _say(f"{prefix}{line}")


# ---------------------------------------------------------------------------
# Check accumulator — the exit verdict is the AND of every recorded check
# ---------------------------------------------------------------------------


class _Checks:
    """Collects pass/fail conditions; the smoke fails if ANY fails."""

    def __init__(self) -> None:
        self._results: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self._results.append((name, ok, detail))

    def print_and_verdict(self) -> bool:
        _say("  Checks (exit verdict = all must pass):")
        for name, ok, detail in self._results:
            mark = "PASS" if ok else "FAIL"
            tail = f"  {detail}" if detail else ""
            _say(f"    [{mark}] {name}{tail}")
        _say()
        return all(ok for _, ok, _ in self._results)


# ---------------------------------------------------------------------------
# Ephemeral DB lifecycle (mirrors tests/integration/conftest.py::fresh_db)
# ---------------------------------------------------------------------------


def _load_test_db_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        return url
    # Friendly fallback: pull just TEST_DATABASE_URL out of .env so the script
    # is runnable without the full `set -a && source .env` dance. We read ONLY
    # that one line — we do not source the whole secret-bearing .env.
    env_file = _REPO_ROOT / ".env"
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if line.startswith("TEST_DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        "TEST_DATABASE_URL is not set and not found in .env.\n"
        "Run `set -a && source .env && set +a` first, and make sure the "
        "postgres-test container is up (`docker compose up -d postgres-test`)."
    )


def _assert_isolated(url: str) -> None:
    # Delegate the structural isolation guard (port 5433 + "test" in db name,
    # parsed via make_url, password-redacted errors) to the shared eval_support
    # helper. It raises EvalDBIsolationError; this script's contract is "refuse
    # with a clear message and a non-zero exit", so re-raise as SystemExit to
    # preserve the exit-code behavior the runbook documents.
    try:
        assert_test_url_is_isolated(url)
    except EvalDBIsolationError as exc:
        raise SystemExit(f"refusing: {exc}") from exc


# ---------------------------------------------------------------------------
# The smoke run
# ---------------------------------------------------------------------------


async def _run_smoke(db_url: str) -> bool:
    engine = create_async_engine(db_url, poolclass=NullPool)
    try:
        return await _drive(engine)
    finally:
        await engine.dispose()


async def _drive(engine: AsyncEngine) -> bool:
    checks = _Checks()
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_review(engine, review_id)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    persister = AuditPersister(
        session_factory=session_factory, retention_settings=RetentionSettings()
    )
    publisher = _RecordingPublisher()
    provider = _ScriptedLLMProvider(
        triage_response=_triage_response(), analyze_response=_analyze_response()
    )
    graph = build_graph(
        db_factory=session_factory,
        github_factory=_stub_github_factory,
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
        # Required since the suggested-patches arc; OFF in rehearsal — the
        # scripted provider carries no patch responses.
        patch_config=PatchConfig(patches_enabled=False),
        checkpointer=InMemorySaver(),
        publisher=publisher,
        import_path_resolver=_StubImportPathResolver(),
        # Production-parity shadow wiring (mirrors api/lifespan.py): lookup +
        # CacheLookupEvent telemetry + write-on-miss; the model is always
        # called, nothing is served.
        analyze_cache_store=AnalyzeCacheStore(session_factory=session_factory),
    )
    _say("  Graph built .......... 7-node topology; scripted LLM + fake GitHub")
    _say(f"  Running review ....... synthetic PR ({_FILE_PATH}, +1 MEDIUM finding)")
    _say()

    result = await graph.ainvoke(
        _seed_state(review_id), config={"configurable": {"thread_id": str(review_id)}}
    )

    # The findings content row is written by the production writer during the
    # graph's analyze node (`emit_finding` co-inserts it with the FindingEvent
    # audit row, FUP-111), so the run reconstructs FULL — no raw-SQL seed.
    replayer = AuditReplayer(session_factory=session_factory)
    pre = await replayer.reconstruct(review_id)

    await _narrate_nodes(engine, review_id, publisher)
    await _narrate_audit_stream(engine, review_id)
    _narrate_llm_exchanges(provider)
    _narrate_publisher(publisher)

    # ---- hard checks (mirror tests/integration/test_e2e_smoke.py asserts) ----
    posted = len(publisher.create_review_calls)
    checks.record("publish posted exactly one review", posted == 1, f"create_review calls={posted}")
    if posted == 1:
        call = publisher.create_review_calls[0]
        checks.record("posted >=1 inline comment", len(call["comments"]) >= 1)
        checks.record("review_status is COMMENT", call["review_status"] == "COMMENT")

    pub_result = result.get("publish_result")
    checks.record(
        "terminal PublishResult is success",
        pub_result is not None and pub_result.outcome == "success",
        f"outcome={getattr(pub_result, 'outcome', None)}",
    )

    async with engine.begin() as conn:
        status = (
            await conn.execute(text("SELECT status FROM reviews WHERE id = :id"), {"id": review_id})
        ).scalar_one()
    checks.record("review status is completed", status == "completed", f"status={status}")

    checks.record("exactly one finding emitted", len(pre.findings) == 1)

    started = {p.node_id for p in pre.phases}
    checks.record(
        "expected node phase coverage",
        started >= _EXPECTED_NODES,
        f"ran={sorted(started)}",
    )

    pub_events = [e for e in pre.events if isinstance(e, PublishEvent)]
    checks.record(
        "exactly one PublishEvent in the audit stream",
        len(pub_events) == 1
        and pub_events[0].review_status == "COMMENT"
        and pub_events[0].comments_posted == 1,
        f"count={len(pub_events)}",
    )

    ok_replay = await _narrate_replay(replayer, review_id)
    post = await replayer.reconstruct(review_id)
    checks.record(
        "replay reconstructs FULL", post.mode is ReplayMode.FULL, f"mode={post.mode.value}"
    )
    checks.record("is_eval propagated (False)", post.is_eval is False)
    checks.record("assert_replay_equivalent passes", ok_replay)

    # ---- shadow-cache two-run proof (specs/2026-06-11-file-hash-analyze-cache.md) ----
    # Run 1 above was the cold pass: the analyze lookup must have recorded a
    # MISS and written one cache row. Run 2 models the re-push (same PR, new
    # head SHA, byte-identical content): the lookup must report WOULD_HIT, the
    # model must STILL be called (shadow serves nothing), and no second row
    # may land. Mechanism proof only — see the _RUN2_HEAD_SHA note.
    run1_lookups = [e for e in pre.events if isinstance(e, CacheLookupEvent)]
    checks.record(
        "run 1: one cache lookup, outcome=miss",
        len(run1_lookups) == 1 and run1_lookups[0].outcome == "miss",
        f"outcomes={[e.outcome for e in run1_lookups]}",
    )
    async with engine.begin() as conn:
        rows_after_run1 = (
            await conn.execute(text("SELECT count(*) FROM analyze_file_cache"))
        ).scalar_one()
    checks.record("run 1: one cache row written", rows_after_run1 == 1, f"rows={rows_after_run1}")

    analyze_calls_before = sum(1 for c in provider.calls if c.node_id == "analyze")
    review_id_2 = uuid4()
    await _seed_review(engine, review_id_2, head_sha=_RUN2_HEAD_SHA)
    _say("  Run 2 (re-push) ...... same PR content, new head SHA — expecting would_hit")
    _say()
    await graph.ainvoke(
        _seed_state(review_id_2, head_sha=_RUN2_HEAD_SHA),
        config={"configurable": {"thread_id": str(review_id_2)}},
    )

    await _narrate_audit_stream(engine, review_id_2)

    run2 = await replayer.reconstruct(review_id_2)
    run2_lookups = [e for e in run2.events if isinstance(e, CacheLookupEvent)]
    checks.record(
        "run 2: one cache lookup, outcome=would_hit",
        len(run2_lookups) == 1 and run2_lookups[0].outcome == "would_hit",
        f"outcomes={[e.outcome for e in run2_lookups]}",
    )
    checks.record(
        "run 2: model still called (shadow serves nothing)",
        sum(1 for c in provider.calls if c.node_id == "analyze") == analyze_calls_before + 1,
    )
    async with engine.begin() as conn:
        rows_after_run2 = (
            await conn.execute(text("SELECT count(*) FROM analyze_file_cache"))
        ).scalar_one()
    checks.record(
        "run 2: no second cache row (hit skips the write)",
        rows_after_run2 == 1,
        f"rows={rows_after_run2}",
    )
    checks.record(
        "run 2: keys match across runs (re-push hits the same entry)",
        len(run1_lookups) == 1
        and len(run2_lookups) == 1
        and run1_lookups[0].cache_key == run2_lookups[0].cache_key,
    )

    # Final full-DB dump — both runs' rows, the cache row's complete JSON
    # payload included, so any silent write failure is visible in the output.
    await _narrate_db_state(engine)

    return checks.print_and_verdict()


async def _narrate_nodes(
    engine: AsyncEngine, review_id: UUID, publisher: _RecordingPublisher
) -> None:
    async with engine.begin() as conn:
        phases = [
            r[0]
            for r in (
                await conn.execute(
                    text(
                        "SELECT DISTINCT payload->>'node_id' FROM audit_events "
                        "WHERE review_id = :id AND event_type = 'review_phase'"
                    ),
                    {"id": review_id},
                )
            ).all()
        ]
    order = ["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"]
    ran = set(phases)
    notes = {
        "triage": f"{_FILE_PATH} -> DEEP",
        "analyze": "1 finding: missing_input_validation (MEDIUM)",
        "hitl": "pass-through (no CRITICAL/HIGH)",
    }
    if publisher.create_review_calls:
        call = publisher.create_review_calls[0]
        notes["publish"] = (
            f"posted {len(call['comments'])} inline comment(s) "
            f"(review_status={call['review_status']})"
        )
    _say("  Nodes that ran (from audit phase events):")
    for node in order:
        if node in ran:
            note = f"  {notes[node]}" if node in notes else ""
            _say(f"    {node:<11} ok{note}")
        elif node == "trace":
            _say(f"    {node:<11} -   not exercised (no trace candidates in this fixture)")
    _say()


async def _narrate_audit_stream(engine: AsyncEngine, review_id: UUID) -> None:
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT sequence_number, event_type, payload FROM audit_events "
                    "WHERE review_id = :id ORDER BY sequence_number"
                ),
                {"id": review_id},
            )
        ).all()
    _say(f"  Audit stream ......... {len(rows)} events (append-only), FULL payloads:")
    for seq, et, payload in rows:
        detail = ""
        if et == "review_phase":
            detail = f"{payload.get('node_id')}/{payload.get('marker')}"
        elif et == "finding":
            detail = f"{payload.get('finding_type')} ({payload.get('severity')})"
        elif et == "publish":
            detail = (
                f"status={payload.get('review_status')} posted={payload.get('comments_posted')}"
            )
        elif et == "publish_routing":
            detail = f"-> {payload.get('destination')}"
        _say(f"    {seq:>3}  {et:<20} {detail}")
        _say_block("         ", _dump_json(payload))
    _say()


def _narrate_llm_exchanges(provider: _ScriptedLLMProvider) -> None:
    """Every request the graph sent to the provider, in full — prompts are the
    exact bytes the production AnthropicProvider would send to Claude (the
    graph renders them identically; only the transport is scripted here)."""
    _say(f"  LLM exchanges ........ {len(provider.calls)} request(s), FULL prompts:")
    for i, req in enumerate(provider.calls, 1):
        _say(
            f"    --- request {i}: node={req.node_id} model={req.model} "
            f"max_tokens={req.max_tokens} temp={req.temperature} "
            f"template={req.prompt_template_version} degraded={req.degraded_mode}"
        )
        _say("    SYSTEM PROMPT:")
        _say_block("      | ", req.system_prompt)
        _say("    USER PROMPT:")
        _say_block("      | ", req.user_prompt)
        scripted = {
            "triage": provider.triage_response,
            "analyze": provider.analyze_response,
        }.get(req.node_id, "Smoke test: one input-validation finding on the new function.")
        _say("    SCRIPTED RESPONSE (returned to the graph):")
        _say_block("      | ", scripted)
    _say()


def _narrate_publisher(publisher: _RecordingPublisher) -> None:
    """The exact payload(s) that would have been POSTed to GitHub."""
    _say(f"  GitHub publish ....... {len(publisher.create_review_calls)} call(s), FULL payloads:")
    for i, call in enumerate(publisher.create_review_calls, 1):
        _say(f"    --- create_review call {i}:")
        _say_block("      ", _dump_json(call))
    _say()


# Every content/forensic table in the scratch DB — fixed allowlist, dumped
# whole so silent-write failures (a row that should exist and doesn't, a
# column silently NULL, an is_eval flag gone wrong) are visible in the output.
_DB_DUMP_TABLES = (
    "reviews",
    "findings",
    "analyze_file_cache",
    "llm_call_content",
    "anomalies",
    "purge_audit",
)


async def _narrate_db_state(engine: AsyncEngine) -> None:
    _say("  Database state ....... full dump, every content table (scratch DB):")
    async with engine.begin() as conn:
        for table in _DB_DUMP_TABLES:
            rows = (
                (await conn.execute(text(f"SELECT * FROM {table}")))  # noqa: S608 — fixed allowlist above
                .mappings()
                .all()
            )
            _say(f"    {table}: {len(rows)} row(s)")
            for row in rows:
                _say_block("      ", _dump_json(dict(row)))
    _say()


async def _narrate_replay(replayer: AuditReplayer, review_id: UUID) -> bool:
    _say("  Replay equivalence (the headline capability):")
    review = await replayer.reconstruct(review_id)
    _say(
        f"    reconstruct ........ mode={review.mode.value} · "
        f"{len(review.findings)} finding(s) · {len(review.phases)} phase(s)"
    )
    try:
        await replayer.assert_replay_equivalent(review_id)
    except Exception as exc:  # noqa: BLE001 — surface any replay failure to the operator
        _say(f"    assert_replay_equivalent -> FAILED: {type(exc).__name__}: {exc}")
        _say()
        return False
    _say("    assert_replay_equivalent -> PASS")
    _say()
    return True


def main() -> int:
    _say(_RULE)
    _say("  Outrider — end-to-end smoke (real graph · real Postgres)")
    _say(_RULE)
    _say()

    admin_url = _load_test_db_url()
    _assert_isolated(admin_url)
    db_name = f"outrider_test_smoke_{uuid4().hex[:8]}"
    db_url = replace_db_name(admin_url, db_name)

    asyncio.run(create_database(admin_url, db_name))
    _say(f"  Ephemeral DB ......... {db_name} (created)")
    try:
        # run_alembic_upgrade_head is async (it wraps alembic's sync command in
        # asyncio.to_thread); env.py still reads DATABASE_URL and runs its own
        # asyncio.run, so this must stay OUTSIDE _run_smoke's event loop — hence
        # its own asyncio.run here, preserving the create -> migrate -> run ->
        # drop control flow.
        asyncio.run(run_alembic_upgrade_head(db_url))
        _say("  Migrated ............. alembic upgrade head")
        _say()
        ok = asyncio.run(_run_smoke(db_url))
    finally:
        asyncio.run(drop_database(admin_url, db_name))
        _say(f"  Ephemeral DB ......... {db_name} (dropped)")
        _say()

    _say(_RULE)
    _say("  SMOKE PASSED" if ok else "  SMOKE FAILED")
    _say(_RULE)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
