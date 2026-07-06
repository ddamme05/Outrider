#!/usr/bin/env python3
# ================================================================
#  Outrider — deep inspector: every moving part of one review
# ================================================================
"""Full forensic timeline of a single review — all moving parts, their state,
their results — reconstructed from the append-only audit stream + content tables.

Where `diagnose_review_context.py` answers one focused question ("what did the
model see for analyze"), this is the firehose: the complete ordered event
stream, grouped by graph-node phase, with every event's COMPLETE payload printed
verbatim from JSONB in non-compact mode (no field is curated or hidden —
including the `review_phase` start/end markers, shown both as a header line AND
their full payload), the joined LLM prompt/
completion + finding content, the reviews-row state, and the replay-equivalence
verdict. It is the "open the hood and watch every part move" view.

Sections, in order:
  1. reviews row — the durable record (status, metrics, timestamps, retention)
  2. replay verdict — reconstruct() mode + assert_replay_equivalent() pass/fail
     + orphan detection (findings-rows with no audit event)
  3. phase timeline — every ReviewPhaseEvent start/end pair (grouped under the 7
     logical nodes; analyze emits several keyed pairs per pass since the fan-out),
     and under each, every per-operation event IN SEQUENCE with its full payload
  4. LLM exchanges — per analyze/triage/synthesize/trace call: token/cost/cache
     metadata + the full prompt and completion text (within retention)
  5. findings — per finding: the FindingEvent payload + the content-table row
     (title/description/evidence/suggested_fix + override provenance)
  6. summary — event-type histogram + the accounting that should balance

Pure READER over audit_events + reviews + llm_call_content + findings (no
mutation, no LLM call, no GitHub call). Payloads are rendered generically from
the JSONB column, so this tool does not drift when an event gains a field.

Run:
  op run --env-file=.env -- uv run python scripts/inspect_review.py            # list latest
  op run --env-file=.env -- uv run python scripts/inspect_review.py --review-id <uuid>
  op run --env-file=.env -- uv run python scripts/inspect_review.py --review-id <uuid> --compact
  op run --env-file=.env -- uv run python scripts/inspect_review.py --review-id <uuid> --no-content

Flags:
  --compact      one line per event (seq · type · node · key fields) instead of
                 full payloads; section 3 only. Sections 1/2/5/6 unaffected.
  --no-content   skip prompt/completion + finding-description text (metadata only)
  --phase NODE   restrict the timeline to one node (e.g. --phase analyze — shows ALL
                 of analyze's fan-out phases: plan + every per-file worker + aggregate)
  --phase-key K  restrict to one fan-out phase by phase_key prefix
                 (e.g. --phase-key file:src/ops/deploy_helpers.js isolates one worker)

Exit codes: 0 = dumped; 2 = setup error (missing DATABASE_URL / review not found).
"""

from __future__ import annotations

import argparse
import asyncio
import json
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

_RULE = "=" * 70


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _render_payload(payload: dict[str, Any], indent: str = "      ") -> None:
    """Print every field of an event payload verbatim, sorted, generically.

    No field names are hardcoded — whatever JSONB carries, we print. Long
    string values are shown whole (this is the deep view); nested structures
    are JSON-pretty-printed so list/dict fields (context_summary, decisions,
    sorted_finding_ids, ...) are fully visible.
    """
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, indent=2, default=str)
            first, *rest = rendered.splitlines()
            _say(f"{indent}{key}: {first}")
            for line in rest:
                _say(f"{indent}  {line}")
        else:
            _say(f"{indent}{key}: {value}")


async def _latest_review(engine: AsyncEngine) -> int:
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT id, repo_id, pr_number, head_sha, status, created_at "
                    "FROM reviews ORDER BY created_at DESC LIMIT 1"
                )
            )
        ).first()
    if row is None:
        _say("  No reviews found. Run a review first.")
        return 2
    _say("  Most recent review:")
    _say(f"    review_id ... {row[0]}")
    _say(f"    repo_id ..... {row[1]}  pr #{row[2]}  @ {str(row[3])[:8]}")
    _say(f"    status ...... {row[4]}  created {row[5]}")
    _say()
    _say("  Re-run with --review-id <id> to inspect every moving part.")
    return 0


async def _section_reviews_row(engine: AsyncEngine, review_id: uuid.UUID) -> bool:
    """Section 1: the durable reviews row. Returns False if the row is absent."""
    _say(_RULE)
    _say("  1. REVIEWS ROW — the durable record")
    _say(_RULE)
    async with engine.begin() as conn:
        row = (
            (
                await conn.execute(
                    text("SELECT * FROM reviews WHERE id = :rid"),
                    {"rid": review_id},
                )
            )
            .mappings()
            .first()
        )
    if row is None:
        _say("  (no reviews row — purged, or wrong id)")
        _say()
        return False
    for key in row:
        _say(f"    {key:28s} {row[key]}")
    _say()
    return True


async def _section_replay_verdict(
    session_factory: async_sessionmaker[Any], review_id: uuid.UUID
) -> None:
    """Section 2: reconstruct() + assert_replay_equivalent() — the integrity proof."""
    _say(_RULE)
    _say("  2. REPLAY VERDICT — reconstruct + equivalence + orphan detection")
    _say(_RULE)
    replayer = AuditReplayer(session_factory=session_factory)
    try:
        review = await replayer.reconstruct(review_id)
    except Exception as exc:  # noqa: BLE001 — surface any reconstruct failure
        _say(f"    reconstruct FAILED: {type(exc).__name__}: {exc}")
        _say()
        return
    _say(f"    reconstruct .................. OK (mode={review.mode.value})")
    _say(f"    is_eval ...................... {review.is_eval}")
    _say(f"    events reconstructed ......... {len(review.events)}")
    _say(f"    phases ....................... {len(review.phases)}")
    _say(f"    findings ..................... {len(review.findings)}")
    _say(f"    llm_exchanges ................ {len(review.llm_exchanges)}")
    _say(f"    orphan finding ids ........... {list(review.orphan_finding_ids) or 'none'}")
    try:
        await replayer.assert_replay_equivalent(review_id)
        _say("    assert_replay_equivalent ..... PASS")
    except Exception as exc:  # noqa: BLE001 — surface any equivalence failure
        _say(f"    assert_replay_equivalent ..... FAIL: {type(exc).__name__}: {exc}")
    _say()


def _timeline_passes(
    phase_filter: str | None,
    phase_key_filter: str | None,
    node_for_filter: str | None,
    row_phase_key: str | None,
) -> bool:
    """Whether a timeline row survives the --phase (node_id) and --phase-key (phase_key
    prefix) filters. `node_for_filter` is the phase's node_id for a marker, or the derived
    owner node for a per-operation event. Both filters are AND-ed; either being None passes."""
    node_ok = phase_filter is None or node_for_filter == phase_filter
    key_ok = phase_key_filter is None or (
        row_phase_key is not None and row_phase_key.startswith(phase_key_filter)
    )
    return node_ok and key_ok


async def _section_timeline(
    engine: AsyncEngine,
    review_id: uuid.UUID,
    *,
    compact: bool,
    phase_filter: str | None,
    phase_key_filter: str | None,
) -> None:
    """Section 3: the full ordered event stream, grouped by phase, full payloads."""
    _say(_RULE)
    _say("  3. PHASE TIMELINE — every event in sequence, grouped by graph node")
    if phase_filter:
        _say(f"     (filtered to node_id={phase_filter!r})")
    if phase_key_filter:
        _say(f"     (filtered to phase_key prefix {phase_key_filter!r})")
    _say(_RULE)
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT sequence_number, event_type, phase_key, timestamp, payload "
                    "FROM audit_events WHERE review_id = :rid ORDER BY sequence_number"
                ),
                {"rid": review_id},
            )
        ).all()

    if not rows:
        _say("    (no audit events for this review id)")
        _say()
        return

    # Walk the stream, opening an indented block on each review_phase 'start' and closing on
    # 'end'. The analyze fan-out opens MANY phases under one node_id='analyze' (plan + one
    # worker per file + aggregate), keyed by phase_key, and they INTERLEAVE — so a single
    # scalar "current open node" is wrong: a sibling worker's phase-end would reset it and
    # silently drop every still-open worker's events. Track keyed phases by phase_key->node_id
    # (unique, never reset by a sibling) and un-keyed phases (intake/triage/... legacy
    # adjacency) by a single open node. Attribute every per-operation event by its OWN
    # phase_key, not the ambient bracket.
    key_to_node: dict[str, str | None] = {}
    open_unkeyed_node: str | None = None
    for seq, event_type, phase_key, ts, payload in rows:
        node_id = payload.get("node_id")
        marker = payload.get("marker")

        if event_type == "review_phase" and marker == "start":
            if phase_key is not None:
                key_to_node[phase_key] = node_id
            else:
                open_unkeyed_node = node_id
            if _timeline_passes(phase_filter, phase_key_filter, node_id, phase_key):
                _say()
                _say(f"  ┌─ PHASE start: node={node_id}  phase_key={phase_key}  seq={seq}  {ts}")
                if not compact:
                    _render_payload(payload, indent="  │   ")
            continue
        if event_type == "review_phase" and marker == "end":
            if _timeline_passes(phase_filter, phase_key_filter, node_id, phase_key):
                _say(f"  └─ PHASE end:   node={node_id}  phase_key={phase_key}  seq={seq}  {ts}")
                if not compact:
                    _render_payload(payload, indent="      ")
            # Only an un-keyed phase resets the un-keyed open node; a keyed worker ending must
            # NOT — that reset was the bug that dropped concurrent siblings' events.
            if phase_key is None:
                open_unkeyed_node = None
            continue

        # Per-operation event: attribute by its OWN phase_key (keyed fan-out event) or, if
        # un-keyed, the currently-open un-keyed node.
        owner_node = key_to_node.get(phase_key) if phase_key is not None else open_unkeyed_node
        if not _timeline_passes(phase_filter, phase_key_filter, owner_node, phase_key):
            continue

        if compact:
            # One-liner: generic fields + the fan-out/dual-model/host signals so the
            # interleaved parallel stream is scannable (analyze_model vs standard_analyze_model
            # is the DEEP-Sonnet/STANDARD-Haiku proof; profile_id attributes cost to the host).
            shown = dict(payload)
            if phase_key is not None:
                shown.setdefault("phase_key", phase_key)
            keys = (
                "node_id",
                "phase_key",
                "model",
                "analyze_model",
                "standard_analyze_model",
                "profile_id",
                "file_path",
                "skip_reason",
                "parse_status",
                "finding_type",
                "severity",
                "outcome",
                "cost_usd",
                "to_node",
            )
            extras = "  ".join(f"{k}={shown[k]}" for k in keys if k in shown)
            _say(f"    [seq {seq:>4}] {event_type:22s} {extras}")
        else:
            _say(f"    [seq {seq:>4}] {event_type}   ({ts})")
            _render_payload(payload)
            _say()
    _say()


async def _section_llm_exchanges(
    engine: AsyncEngine,
    review_id: uuid.UUID,
    *,
    show_content: bool,
) -> None:
    """Section 4: per LLM call — metadata from the event + prompt/completion text."""
    _say(_RULE)
    _say("  4. LLM EXCHANGES — token/cost/cache metadata + prompt/completion")
    _say(_RULE)
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT a.event_id, a.sequence_number, a.payload, c.prompt, c.completion "
                    "FROM audit_events a "
                    "LEFT JOIN llm_call_content c ON c.event_id = a.event_id "
                    "WHERE a.review_id = :rid AND a.event_type = 'llm_call' "
                    "ORDER BY a.sequence_number"
                ),
                {"rid": review_id},
            )
        ).all()
    if not rows:
        _say("    (no llm_call events)")
        _say()
        return
    for event_id, seq, payload, prompt, completion in rows:
        node = payload.get("node_id")
        _say(f"  ── [seq {seq}] {node} call  (event_id={event_id})")
        for k in (
            "model",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "cost_usd",
            "latency_ms",
            "cache_hit",
            "degraded_mode",
            "degradation_reason",
            "prompt_template_version",
            "pricing_version",
            # Host-identity triad (#056) — profile_id is the ONLY field attributing a
            # cost/token row to GLM/Fireworks/baseten vs Anthropic (pricing is keyed by
            # (profile_id, model)). All `if k in payload`-guarded, so they no-op on
            # pre-#056 Anthropic-only rows.
            "profile_id",
            "reasoning_enabled",
            "profile_contract_digest",
            "finish_reason",
        ):
            if k in payload:
                _say(f"       {k:24s} {payload[k]}")
        if show_content:
            if prompt is None:
                _say("       prompt:     (purged past retention TTL)")
            else:
                _say(f"       prompt ({len(prompt)} chars):")
                for line in prompt.splitlines():
                    _say(f"       | {line}")
            if completion is None:
                _say("       completion: (purged past retention TTL)")
            else:
                _say(f"       completion ({len(completion)} chars):")
                for line in completion.splitlines():
                    _say(f"       | {line}")
        _say()


async def _section_findings(
    engine: AsyncEngine,
    review_id: uuid.UUID,
    *,
    show_content: bool,
) -> None:
    """Section 5: per finding — the FindingEvent payload + the content-table row."""
    _say(_RULE)
    _say("  5. FINDINGS — audit event + content row (severity is policy-set, not model)")
    _say(_RULE)
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT sequence_number, payload "
                    "FROM audit_events WHERE review_id = :rid AND event_type = 'finding' "
                    "ORDER BY sequence_number"
                ),
                {"rid": review_id},
            )
        ).all()
        content_rows = (
            (
                await conn.execute(
                    text("SELECT * FROM findings WHERE review_id = :rid"),
                    {"rid": review_id},
                )
            )
            .mappings()
            .all()
        )
    by_finding_id = {str(r["finding_id"]): r for r in content_rows if "finding_id" in r}
    if not rows:
        _say("    (no finding events)")
        _say()
        return
    for seq, payload in rows:
        fid = payload.get("finding_id")
        _say(f"  ── [seq {seq}] finding  finding_id={fid}")
        _say("     FindingEvent payload (audit shadow — metadata + content hash):")
        _render_payload(payload, indent="       ")
        content = by_finding_id.get(str(fid))
        if content is None:
            _say("     findings-table row: (none — metadata-only / purged)")
        else:
            _say("     findings-table row (full content):")
            for key in content:
                value = content[key]
                if not show_content and key in (
                    "title",
                    "description",
                    "evidence",
                    "suggested_fix",
                ):
                    value = f"<{len(str(value))} chars, hidden by --no-content>"
                _say(f"       {key:24s} {value}")
        _say()


async def _section_summary(engine: AsyncEngine, review_id: uuid.UUID) -> None:
    """Section 6: event-type histogram — the shape of the whole run at a glance."""
    _say(_RULE)
    _say("  6. SUMMARY — event-type histogram")
    _say(_RULE)
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT event_type, COUNT(*) "
                    "FROM audit_events WHERE review_id = :rid "
                    "GROUP BY event_type ORDER BY event_type"
                ),
                {"rid": review_id},
            )
        ).all()
        total = (
            await conn.execute(
                text("SELECT COUNT(*) FROM audit_events WHERE review_id = :rid"),
                {"rid": review_id},
            )
        ).scalar_one()
    for event_type, count in rows:
        _say(f"    {event_type:28s} {count}")
    _say(f"    {'TOTAL':28s} {total}")
    _say()


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

        _say()
        # Not-found gate (honors the advertised exit-2 contract): a review is
        # "not found" when it has neither a reviews row NOR any audit events.
        # `_section_reviews_row` returns False when the row is absent;
        # `reconstruct()` raises `ReplayReviewNotFoundError` when the audit
        # stream is empty. If BOTH say absent, there is nothing to inspect —
        # return 2 instead of printing empty sections and a misleading exit 0.
        row_present = await _section_reviews_row(engine, review_id)
        async with engine.begin() as conn:
            n_events = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM audit_events WHERE review_id = :rid"),
                    {"rid": review_id},
                )
            ).scalar_one()
        if not row_present and n_events == 0:
            _say(f"  No reviews row and no audit events for {review_id}. Nothing to inspect.")
            return 2
        await _section_replay_verdict(session_factory, review_id)
        await _section_timeline(
            engine,
            review_id,
            compact=args.compact,
            phase_filter=args.phase,
            phase_key_filter=args.phase_key,
        )
        await _section_llm_exchanges(engine, review_id, show_content=not args.no_content)
        await _section_findings(engine, review_id, show_content=not args.no_content)
        await _section_summary(engine, review_id)
        return 0
    finally:
        await engine.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep forensic timeline of one review — every moving part from the audit log."
    )
    parser.add_argument("--review-id", default=None, help="review UUID (omit to list latest)")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="one line per timeline event instead of full payloads",
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="hide prompt/completion + finding text (metadata only)",
    )
    parser.add_argument(
        "--phase", default=None, help="restrict the timeline to one node_id (e.g. analyze)"
    )
    parser.add_argument(
        "--phase-key",
        default=None,
        help="restrict the timeline to one fan-out phase by phase_key prefix "
        "(e.g. 'file:src/ops/deploy_helpers.js' isolates one analyze worker)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(_run(_parse_args())))
