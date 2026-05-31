#!/usr/bin/env python3
# ================================================================
#  Outrider — diagnose what the model actually saw for one review
# ================================================================
"""Read-only forensic dump of a single review's analyze context.

Answers "why did the review miss finding X?" from the audit stream alone —
the observability story the pitch leans on. For the given review_id it prints:

  - per analyze ⇄ trace pass: the `AnalyzeCompletedEvent` aggregate counts
    (files analyzed / skipped, findings emitted, LLM calls)
  - per file: the analyze-side `FileExaminationEvent` parse_status + skip_reason
    (did the file reach a full pass, or get skipped — DECISIONS.md#018)
  - every analyze `LLMCallEvent`'s `context_summary` (the (file_path,
    scope_unit_name) scope window the model's prompt included; empty for
    degraded calls, whose file + hunks live in the prompt text instead)
  - optionally (--show-prompt / --grep) the prompt text from `llm_call_content`,
    so you can grep it for a specific symbol (e.g. `time.sleep`, `async def`,
    `import time`) and see whether the model COULD have made an inference at
    all, vs. had the context and missed it.

Note: triage tier (DEEP/STANDARD/SKIM/SKIP) is NOT printed — it lives on
`TriageResult.file_tiers` in graph state / the LangGraph checkpoint, not in the
audit-events stream this reader queries. Per-file analyze reach is shown instead
via FileExaminationEvent (a file absent from that section was SKIM/SKIP at
triage; analyze only examines DEEP/STANDARD).

This is purely a READER over the append-only audit_events + content tables
(no mutation, no LLM call, no GitHub call) — the same surface `audit/replay.py`
reconstructs from. It does not import the graph; it queries the tables directly.

Run:
  op run --env-file=.env -- uv run python scripts/diagnose_review_context.py \\
    --review-id <uuid> [--file app/handlers.py] [--show-prompt] [--grep "time.sleep"]

If --review-id is omitted, prints the most recent non-eval review's id + PR and
exits, so you don't have to look it up by hand.

Exit codes: 0 = dumped; 2 = setup error (missing DATABASE_URL, review not found).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# --- ensure src/ on path (mirror conftest's pythonpath=["src"]) ---
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from outrider.audit.replay import AuditReplayer  # noqa: E402

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_RULE = "=" * 62


def _say(msg: str = "") -> None:
    print(msg, flush=True)


async def _latest_review(engine: AsyncEngine) -> int:
    """Print the most recent non-eval review (id + repo/PR) and return exit code."""
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT id, repo_id, pr_number, head_sha, status, created_at "
                    "FROM reviews WHERE is_eval = false "
                    "ORDER BY created_at DESC LIMIT 1"
                )
            )
        ).first()
    if row is None:
        _say("  No non-eval reviews found. Run a C2 review first.")
        return 2
    _say("  Most recent non-eval review:")
    _say(f"    review_id ... {row[0]}")
    _say(f"    repo_id ..... {row[1]}  pr #{row[2]}  @ {str(row[3])[:8]}")
    _say(f"    status ...... {row[4]}  created {row[5]}")
    _say()
    _say("  Re-run with --review-id <id> to dump its analyze context.")
    return 0


async def _dump(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[Any],
    review_id: uuid.UUID,
    *,
    file_filter: str | None,
    show_prompt: bool,
    grep: str | None,
) -> int:
    # --- replay-mode banner (reconstruct is the canonical read surface) ---
    replayer = AuditReplayer(session_factory=session_factory)
    try:
        review = await replayer.reconstruct(review_id)
    except Exception as exc:  # noqa: BLE001 — surface any reconstruct failure
        _say(f"  reconstruct({review_id}) failed: {type(exc).__name__}: {exc}")
        return 2

    _say(_RULE)
    _say(f"  Review {review_id}  (replay mode={review.mode.value})")
    _say(_RULE)
    _say()

    # --- did each file get a full analyze pass? ---
    # Two complementary signals, both verified against source:
    #
    #  1. FileExaminationEvent (per file, examination_type='analyze'): the
    #     decisive per-FILE signal. parse_status ∈ {clean, degraded, failed,
    #     skipped}; skip_reason non-None iff skipped (DECISIONS.md#018). A file
    #     with parse_status='skipped' never reached the model — a missed finding
    #     there is a skip decision, not a model recall miss. A file the analyze
    #     node never emitted at all was SKIM/SKIP at triage (analyze only
    #     examines DEEP/STANDARD per `file_tiers`; triage tier itself is not a
    #     standalone audit event — it lives on TriageResult in state/checkpoint).
    #
    #  2. AnalyzeCompletedEvent (per analyze ⇄ trace pass): the aggregate
    #     counts — pass_index, n_files_analyzed, n_files_skipped,
    #     n_findings_emitted. These are COUNTS, not path lists (the event carries
    #     no analyzed/skipped path arrays). Use them to see how many passes ran
    #     and how many files each touched; use signal 1 for per-file detail.
    async with engine.begin() as conn:
        passes = (
            await conn.execute(
                text(
                    "SELECT payload->>'pass_index', payload->>'n_files_analyzed', "
                    "payload->>'n_files_skipped', payload->>'n_findings_emitted', "
                    "payload->>'n_llm_calls' "
                    "FROM audit_events "
                    "WHERE review_id = :rid AND event_type = 'analyze_completed' "
                    "ORDER BY (payload->>'pass_index')::int"
                ),
                {"rid": review_id},
            )
        ).all()
        exam_rows = (
            await conn.execute(
                text(
                    "SELECT payload->>'file_path', payload->>'parse_status', "
                    "payload->>'skip_reason' "
                    "FROM audit_events "
                    "WHERE review_id = :rid AND event_type = 'file_examination' "
                    "AND payload->>'examination_type' = 'analyze' "
                    "ORDER BY payload->>'file_path'"
                ),
                {"rid": review_id},
            )
        ).all()

    _say("  Analyze passes (AnalyzeCompletedEvent — aggregate counts per pass):")
    if not passes:
        _say("    (none — no analyze_completed event; analyze made no pass)")
    for pass_index, n_analyzed, n_skipped, n_findings, n_calls in passes:
        _say(
            f"    pass {pass_index}: files_analyzed={n_analyzed}  files_skipped={n_skipped}  "
            f"findings={n_findings}  llm_calls={n_calls}"
        )
    _say()

    _say("  Per-file analyze examination (FileExaminationEvent — parse_status / skip_reason):")
    if not exam_rows:
        _say("    (none — no file reached an analyze pass; all SKIM/SKIP at triage?)")
    for fpath, parse_status, skip_reason in exam_rows:
        mark = " <--" if file_filter and fpath == file_filter else ""
        skip = f"  skip_reason={skip_reason}" if skip_reason else ""
        _say(f"    parse={parse_status or '?':8s}  {fpath}{skip}{mark}")
    _say()

    # --- analyze LLMCallEvents: what the model saw, from the audit stream ---
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT event_id, sequence_number, payload "
                    "FROM audit_events "
                    "WHERE review_id = :rid AND event_type = 'llm_call' "
                    "AND payload->>'node_id' = 'analyze' "
                    "ORDER BY sequence_number"
                ),
                {"rid": review_id},
            )
        ).all()

    _say(f"  analyze LLMCallEvents: {len(rows)}")
    if not rows:
        _say("    (none — analyze made no LLM calls; check the analyze passes above)")
    _say()

    for event_id, seq, payload in rows:
        ctx = payload.get("context_summary") or []
        ctx_files = sorted({(e.get("file_path"), e.get("scope_unit_name")) for e in ctx})

        # Fetch the prompt up front: it's needed both for output AND for the
        # --file filter. Degraded analyze calls carry an EMPTY context_summary
        # (the _enforce_context_for_scope_nodes special-case) yet their prompt
        # still names the file + its hunks via render_degraded(). Filtering on
        # context_summary alone would skip exactly the degraded prompt the
        # operator asked for, so when context_summary doesn't name the file we
        # fall back to a substring match on the prompt text.
        # Content lives in llm_call_content keyed by event_id (PK) per
        # DECISIONS.md#016; None means purged past retention TTL.
        async with engine.begin() as conn:
            prow = (
                await conn.execute(
                    text("SELECT prompt FROM llm_call_content WHERE event_id = :eid"),
                    {"eid": event_id},
                )
            ).first()
        prompt = prow[0] if prow else None

        if file_filter:
            in_context = any(f == file_filter for f, _ in ctx_files)
            in_prompt = prompt is not None and file_filter in prompt
            if not (in_context or in_prompt):
                continue

        degraded = payload.get("degraded_mode")
        _say(
            f"  [seq {seq}] llm_call analyze  model={payload.get('model')}  "
            f"input_tokens={payload.get('input_tokens')}  "
            f"degraded={degraded}"
        )
        _say("    context_summary (what the model saw — file : scope):")
        if ctx_files:
            for fpath, scope in ctx_files:
                _say(f"      - {fpath} : {scope}")
        elif degraded:
            _say("      (empty — degraded call; file + hunks are in the prompt, not the manifest)")
        else:
            _say("      (empty)")

        if show_prompt or grep:
            if prompt is None:
                _say("    prompt: (content row absent / purged past retention TTL)")
            elif grep:
                hits = [
                    f"{i}: {ln.strip()}"
                    for i, ln in enumerate(prompt.splitlines(), 1)
                    if grep.lower() in ln.lower()
                ]
                _say(f"    prompt grep {grep!r}: {len(hits)} line(s)")
                for h in hits[:20]:
                    _say(f"      {h}")
            else:
                _say(f"    prompt ({len(prompt)} chars):")
                for ln in prompt.splitlines():
                    _say(f"    | {ln}")
        _say()

    return 0


async def _run(args: argparse.Namespace) -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        _say("  DATABASE_URL is not set — this reader needs it. Aborting.")
        return 2

    engine = create_async_engine(database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        if args.review_id is None:
            return await _latest_review(engine)
        try:
            review_id = uuid.UUID(args.review_id)
        except ValueError:
            _say(f"  --review-id is not a valid UUID: {args.review_id!r}")
            return 2
        return await _dump(
            engine,
            session_factory,
            review_id,
            file_filter=args.file,
            show_prompt=args.show_prompt,
            grep=args.grep,
        )
    finally:
        await engine.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only dump of what the model saw for a review (analyze passes + context)."
    )
    parser.add_argument("--review-id", default=None, help="review UUID (omit to list the latest)")
    parser.add_argument("--file", default=None, help="only show context for this file path")
    parser.add_argument(
        "--show-prompt", action="store_true", help="print the full analyze prompt text"
    )
    parser.add_argument(
        "--grep", default=None, help="grep the analyze prompt for a substring (e.g. 'time.sleep')"
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(_run(_parse_args())))
