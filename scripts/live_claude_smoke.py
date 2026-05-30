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

import asyncio
import os
import sys
from pathlib import Path
from uuid import UUID, uuid4

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
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from outrider.agent.graph import build_graph  # noqa: E402
from outrider.agent.nodes.hitl_config import HITLConfig  # noqa: E402
from outrider.anomaly.persister import AnomalyPersister  # noqa: E402
from outrider.audit.config import RetentionSettings  # noqa: E402
from outrider.audit.persister import AuditPersister  # noqa: E402
from outrider.audit.replay import AuditReplayer  # noqa: E402
from outrider.db.review_status_persister import ReviewStatusPersister  # noqa: E402
from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from outrider.llm.config import ModelConfig  # noqa: E402

# Reuse the committed e2e-smoke fakes + scenario verbatim (no drift).
from tests.integration.test_e2e_smoke import (  # noqa: E402
    _FILE_PATH,
    _RecordingPublisher,
    _StubImportPathResolver,
    _seed_installation,
    _seed_review,
    _seed_state,
    _stub_github_factory,
)

_RULE = "=" * 62


def _say(msg: str = "") -> None:
    print(msg, flush=True)


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


async def _run(db_url: str, api_key: str) -> bool:
    engine = create_async_engine(db_url, poolclass=NullPool)
    try:
        return await _drive(engine, api_key)
    finally:
        await engine.dispose()


async def _drive(engine: AsyncEngine, api_key: str) -> bool:
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_review(engine, review_id)

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
        checkpointer=InMemorySaver(),
        publisher=publisher,
        import_path_resolver=_StubImportPathResolver(),
    )
    _say(
        f"  Models ............... {ModelConfig().analyze_model} (analyze) + "
        f"{ModelConfig().triage_model} (triage/synthesize)"
    )
    _say(f"  Calling real Claude .. analyzing synthetic diff ({_FILE_PATH})")
    _say()

    result = await graph.ainvoke(
        _seed_state(review_id), config={"configurable": {"thread_id": str(review_id)}}
    )
    await provider.aclose()

    interrupted = "__interrupt__" in result
    await _report(engine, review_id, result, publisher, interrupted=interrupted)
    return await _verify(engine, session_factory, review_id, interrupted=interrupted)


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
    _say("  Real Claude produced:")
    if findings:
        for ft, sev in findings:
            _say(f"    - {ft} ({sev})")
    else:
        _say("    (no findings on this synthetic diff — valid; Claude's call)")
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
    interrupted: bool,
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
    checks.append(("analyze ran (real Claude call)", analyze_ran > 0, ""))
    checks.append(("real LLMCallEvents persisted", n_llm > 0, f"{n_llm} llm calls"))

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
    _say(_RULE)
    _say("  Outrider — live Claude smoke (real LLM · fake GitHub · real DB)")
    _say(_RULE)
    _say()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _say("  ANTHROPIC_API_KEY is not set — this runner needs a real Claude key.")
        _say("  export ANTHROPIC_API_KEY=sk-ant-...  then re-run.")
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
        ok = asyncio.run(_run(db_url, api_key))
    finally:
        asyncio.run(_drop_db(admin_url, db_name))
        _say(f"  Ephemeral DB ......... {db_name} (dropped)")
        _say()

    _say(_RULE)
    _say("  LIVE CLAUDE SMOKE PASSED" if ok else "  LIVE CLAUDE SMOKE FAILED")
    _say(_RULE)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
