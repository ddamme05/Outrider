"""Engine-level hide_parameters contract — bound prompt/completion never
appear in SQLAlchemy exception strings.

Codex sharp-edges finding: `LLMPersisterError(f"{exc!r}")` in
`AnthropicProvider.complete()` wraps the persister's exceptions with
the original exception's repr; SQLAlchemy's default `IntegrityError`
string representation embeds bound parameter values, which for a
failing content INSERT would carry raw `prompt` / `completion` text.
`RejectLLMContentFilter` is key-based (per FUP-023) and would not
strip content from the log record's `message` field.

The defense: `hide_parameters=True` on the engine. SQLAlchemy then
renders bound parameter values as placeholders in exception strings.

This test pins the contract end-to-end: a real `IntegrityError`
triggered by inserting `llm_call_content` with a non-existent
`installation_id` (FK RESTRICT) and embedded prompt/completion text
must NOT carry those values in the exception's string representation
when the engine is constructed with `hide_parameters=True`.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

_SECRET_PROMPT = "extremely_distinctive_prompt_marker_42"  # noqa: S105
_SECRET_COMPLETION = "extremely_distinctive_completion_marker_99"  # noqa: S105


async def test_hide_parameters_strips_bound_values_from_exception_string(
    migrated_db: str,
) -> None:
    """SQLAlchemy IntegrityError on a failing content INSERT does NOT carry
    the bound prompt/completion text when the engine has `hide_parameters=True`.

    Triggers a real FK violation by attempting an `llm_call_content`
    INSERT with a phantom `installation_id` (no matching `installations`
    row). The FK constraint fires `IntegrityError` carrying the failing
    statement's bound parameters; with `hide_parameters=True`, those
    parameter values must not appear in the exception string.
    """
    # Construct the engine with the production-equivalent setting.
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        with pytest.raises(IntegrityError) as exc_info:
            async with engine.begin() as conn:
                # First seed an audit row so the llm_call_content FK to
                # audit_events resolves cleanly; the failure has to come
                # from the installation_id FK, not the audit FK.
                audit_result = await conn.execute(
                    text(
                        "INSERT INTO audit_events (review_id, event_type, payload) "
                        "VALUES (gen_random_uuid(), 'llm_call', '{}'::jsonb) "
                        "RETURNING event_id"
                    )
                )
                event_id = audit_result.scalar_one()

                # Now attempt content INSERT with a non-existent
                # installation_id; embeds the secret strings as bound
                # parameter values for `prompt` and `completion`.
                await conn.execute(
                    text(
                        "INSERT INTO llm_call_content "
                        "(event_id, installation_id, prompt, completion, "
                        " retention_expires_at) "
                        "VALUES (:event_id, 99999, :prompt, :completion, "
                        "NOW() + INTERVAL '90 days')"
                    ),
                    {
                        "event_id": event_id,
                        "prompt": _SECRET_PROMPT,
                        "completion": _SECRET_COMPLETION,
                    },
                )

        # The IntegrityError's string representation must NOT carry the
        # secret prompt/completion values.
        exc_str = str(exc_info.value)
        exc_repr = repr(exc_info.value)
        assert _SECRET_PROMPT not in exc_str, (
            f"SQLAlchemy IntegrityError str() leaked secret prompt; "
            f"hide_parameters=True is not in effect. exc_str={exc_str!r}"
        )
        assert _SECRET_COMPLETION not in exc_str, (
            f"SQLAlchemy IntegrityError str() leaked secret completion. exc_str={exc_str!r}"
        )
        assert _SECRET_PROMPT not in exc_repr
        assert _SECRET_COMPLETION not in exc_repr

        # `args` is what `f"{exc!r}"` ends up rendering through. Also clean.
        for arg in exc_info.value.args:
            arg_str = str(arg)
            assert _SECRET_PROMPT not in arg_str
            assert _SECRET_COMPLETION not in arg_str
    finally:
        await engine.dispose()


async def test_default_engine_without_hide_parameters_does_leak(
    migrated_db: str,
) -> None:
    """Negative control: prove the default engine (without `hide_parameters=True`)
    DOES leak bound parameters. Documents the regression risk: if a future
    refactor removes `hide_parameters=True` from the production engine
    factory, content would leak. This test fails-loud if SQLAlchemy ever
    changes its default behavior (welcome relaxation, surfaces as a test
    failure rather than silent change in security posture).
    """
    # Construct WITHOUT hide_parameters — SQLAlchemy default.
    engine = create_async_engine(migrated_db)
    try:
        with pytest.raises(IntegrityError) as exc_info:
            async with engine.begin() as conn:
                audit_result = await conn.execute(
                    text(
                        "INSERT INTO audit_events (review_id, event_type, payload) "
                        "VALUES (gen_random_uuid(), 'llm_call', '{}'::jsonb) "
                        "RETURNING event_id"
                    )
                )
                event_id = audit_result.scalar_one()
                await conn.execute(
                    text(
                        "INSERT INTO llm_call_content "
                        "(event_id, installation_id, prompt, completion, "
                        " retention_expires_at) "
                        "VALUES (:event_id, 99999, :prompt, :completion, "
                        "NOW() + INTERVAL '90 days')"
                    ),
                    {
                        "event_id": event_id,
                        "prompt": _SECRET_PROMPT,
                        "completion": _SECRET_COMPLETION,
                    },
                )

        # Default behavior DOES expose bound params in exception text.
        # If this assertion ever flips (SQLAlchemy changes the default),
        # the production-engine setting is no longer the gate — surface
        # that as a test failure so we can re-evaluate.
        exc_str = str(exc_info.value)
        assert _SECRET_PROMPT in exc_str, (
            "Default SQLAlchemy engine no longer exposes bound params in "
            "IntegrityError str(); the hide_parameters=True production "
            "setting may have become redundant — verify and remove if so."
        )
    finally:
        await engine.dispose()
