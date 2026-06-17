"""Shared full-trace narration for the rehearsal scripts.

One recipe, three consumers (`smoke_e2e.py`, `live_claude_smoke.py`,
`live_github_demo.py`) — the "prints everything" granularity must not
fork per script. Every helper takes the calling script's `say` so output
rides that script's terminal+file tee (`scripts/_trace_log.py`).

LLM exchanges come from the DATABASE (`llm_call_event` rows joined to
`llm_call_content`), not from a provider spy: the real AnthropicProvider
persists every exchange per DECISIONS.md#016. Retention nuance — the
content row stores the USER prompt + the real completion; the SYSTEM
prompt is deliberately retained as `system_prompt_hash` + template
version on the event (reconstructable from the versioned template
library, spec §8.3), NOT as text. The scripted provider persists
nothing — `smoke_e2e` additionally dumps its recorded requests
provider-side (where both prompts ARE available), and this DB view
prints an explanatory zero there.

`narrate_db_state` dumps whole tables only for the scripts' ephemeral
scratch DBs; pass `review_id` to scope the dump when the target is a
real shared database (`live_github_demo` → DATABASE_URL).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

Say = "Callable[[str], None]"


def dump_json(obj: object) -> str:
    """Pretty JSON with non-JSON types (UUID/datetime/Decimal) stringified."""
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def say_block(say: Callable[..., None], prefix: str, body: str) -> None:
    for line in body.splitlines() or [""]:
        say(f"{prefix}{line}")


async def narrate_audit_stream(
    say: Callable[..., None], engine: AsyncEngine, review_id: UUID
) -> None:
    """Every audit event for the review, in order, with its FULL payload."""
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
    say(f"  Audit stream ......... {len(rows)} events (append-only), FULL payloads:")
    for seq, et, payload in rows:
        detail = ""
        if et == "review_phase":
            detail = f"{payload.get('node_id')}/{payload.get('marker')}"
        elif et == "finding":
            detail = f"{payload.get('finding_type')} ({payload.get('severity')})"
        elif et == "publish":
            # All three eligibility-gated tiers (DECISIONS.md#050): inline comments,
            # review-body "Related concerns", and the dashboard-only aggregate.
            detail = (
                f"status={payload.get('review_status')} "
                f"inline={payload.get('comments_posted')} "
                f"review_body={payload.get('review_body_findings_posted')} "
                f"dashboard_only={payload.get('dashboard_only_findings_surfaced')}"
            )
        elif et == "slack_notification":
            detail = (
                f"{payload.get('kind')} -> {payload.get('channel_id')} "
                f"ts={payload.get('message_ts')}"
            )
        elif et == "publish_routing":
            detail = f"-> {payload.get('destination')}"
        elif et == "cache_lookup":
            detail = f"{payload.get('outcome')} {payload.get('file_path')}"
        say(f"    {seq:>3}  {et:<20} {detail}")
        say_block(say, "         ", dump_json(payload))
    say("")


async def narrate_llm_exchanges_from_db(
    say: Callable[..., None], engine: AsyncEngine, review_id: UUID
) -> None:
    """The persisted LLM exchanges: persisted user prompt + real response.

    Joins each `llm_call` audit row to its `llm_call_content` row (single
    transaction per DECISIONS.md#016). A scripted/stub provider persists
    nothing — zero rows here means no REAL provider call was recorded, not
    that the graph skipped the LLM.
    """
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT ae.sequence_number, ae.payload, c.prompt, c.completion "
                    "FROM audit_events ae "
                    "JOIN llm_call_content c ON c.event_id = ae.event_id "
                    "WHERE ae.review_id = :id AND ae.event_type = 'llm_call' "
                    "ORDER BY ae.sequence_number"
                ),
                {"id": review_id},
            )
        ).all()
    say(
        f"  LLM exchanges (DB) ... {len(rows)} persisted exchange(s) "
        "(user prompt + real response; system prompt as hash/template):"
    )
    if not rows:
        say("    (none persisted — a scripted/stub provider does not write llm_call_content)")
    for seq, payload, prompt, completion in rows:
        say(
            f"    --- exchange seq={seq}: node={payload.get('node_id')} "
            f"model={payload.get('model')} in={payload.get('input_tokens')} "
            f"out={payload.get('output_tokens')} cost=${payload.get('cost_usd')}"
        )
        say(
            f"    system prompt: hash={payload.get('system_prompt_hash')} "
            f"template={payload.get('prompt_template_version')} (text not retained — #016)"
        )
        say("    USER PROMPT (persisted):")
        say_block(say, "      | ", prompt)
        say("    RESPONSE (real):")
        say_block(say, "      | ", completion)
    say("")


# Whole-table dump allowlist — the scripts' scratch DBs only. A review_id
# scopes every table that has one, for runs against a real shared DB.
_DB_DUMP_TABLES = (
    "reviews",
    "findings",
    "analyze_file_cache",
    "llm_call_content",
    "anomalies",
    "purge_audit",
)
_REVIEW_SCOPED_COLUMN = {
    "reviews": "id",
    "findings": "review_id",
    "analyze_file_cache": "source_review_id",
}


async def narrate_db_state(
    say: Callable[..., None], engine: AsyncEngine, *, review_id: UUID | None = None
) -> None:
    """Dump the content tables. Whole-table on a scratch DB (review_id=None);
    review-scoped where possible on a real DB (tables with no review column
    are skipped in scoped mode rather than dumped whole)."""
    scope_label = "scoped to this review" if review_id is not None else "whole scratch DB"
    say(f"  Database state ....... content tables ({scope_label}):")
    async with engine.begin() as conn:
        for table in _DB_DUMP_TABLES:
            scope_col = _REVIEW_SCOPED_COLUMN.get(table)
            if review_id is not None and scope_col is None:
                say(f"    {table}: skipped (no per-review column; real-DB scoped mode)")
                continue
            # Identifiers come from the fixed allowlists above, never input.
            if review_id is not None:
                query = text(f"SELECT * FROM {table} WHERE {scope_col} = :rid")  # noqa: S608
                rows = (await conn.execute(query, {"rid": review_id})).mappings().all()
            else:
                rows = (await conn.execute(text(f"SELECT * FROM {table}"))).mappings().all()  # noqa: S608
            say(f"    {table}: {len(rows)} row(s)")
            for row in rows:
                say_block(say, "      ", dump_json(dict(row)))
    say("")


def narrate_recorded_publisher(say: Callable[..., None], publisher: object) -> None:
    """The exact payload(s) a recording publisher captured (would-be GitHub posts).

    The `body` carries the review-body "Related concerns" section + the aggregate
    dashboard-only note (DECISIONS.md#050) when those tiers have eligible findings —
    dumped in full here alongside the inline `comments`.
    """
    calls = getattr(publisher, "create_review_calls", [])
    say(f"  GitHub publish ....... {len(calls)} call(s), FULL payloads (incl. review body):")
    for i, call in enumerate(calls, 1):
        say(f"    --- create_review call {i}:")
        say_block(say, "      ", dump_json(call))
    say("")


async def narrate_slack_notifications(
    say: Callable[..., None], engine: AsyncEngine, review_id: UUID
) -> None:
    """The Slack notifications actually posted for the review, with FULL payloads.

    A `slack_notification` audit row is written only AFTER a successful
    `chat.postMessage` (metadata only — channel + ts + kind, never the message body
    or the bot token, per DECISIONS.md#051). Zero rows therefore means one of:
      - no Slack resolver wired (no OUTRIDER_TOKEN_ENC_KEY / no per-install config),
      - the OTHER kind fired (a gated review posts `hitl_pending`, a clean review
        posts `review_posted` — mutually exclusive per review),
      - or a post FAILED and was swallowed (Slack is never gate-breaking) — check the
        logs for `notification failed` / `SlackNotifyError` (enable INFO logging to see them).
    """
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT sequence_number, payload FROM audit_events "
                    "WHERE review_id = :id AND event_type = 'slack_notification' "
                    "ORDER BY sequence_number"
                ),
                {"id": review_id},
            )
        ).all()
    say(f"  Slack notifications .. {len(rows)} posted (metadata-only audit rows), FULL payloads:")
    if not rows:
        say("    (none — no resolver wired, the other kind fired, or a swallowed post — see logs)")
    for seq, payload in rows:
        say(
            f"    --- seq={seq}: {payload.get('kind')} -> channel={payload.get('channel_id')} "
            f"ts={payload.get('message_ts')}"
        )
        say_block(say, "      ", dump_json(payload))
    say("")
