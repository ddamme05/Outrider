"""Unit tests for eval_support's fail-closed guards (no DB).

The URL guard's messages are also pinned by `tests/eval/test_eval_db_isolation_guard.py`
(via the conftest re-export); this file covers the db-name guard on the exported
DDL primitives + the OUTRIDER_IS_EVAL gate, which have no other direct test.
"""

from __future__ import annotations

import pytest

from outrider.eval_support import (
    EvalDBIsolationError,
    EvalModeNotEnabledError,
    create_database,
    drop_database,
    require_eval_mode,
)

# A URL that passes the isolation guard, so create/drop reach the db-name check.
_TEST_URL = "postgresql+psycopg://u:p@127.0.0.1:5433/outrider_test"


@pytest.mark.parametrize("ddl", [create_database, drop_database])
@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"; DROP DATABASE prod; --',  # quote-breaking injection
        "Uppercase",  # [a-z0-9_]+ is lowercase-only
        "has space",
        "has-hyphen",
        "",  # empty
    ],
)
async def test_ddl_primitives_reject_quote_breaking_db_name(ddl, bad_name) -> None:  # type: ignore[no-untyped-def]
    # The db-name guard raises before any engine/connection, so no DB is needed.
    with pytest.raises(ValueError, match=r"must match \[a-z0-9_\]\+"):
        await ddl(_TEST_URL, bad_name)


@pytest.mark.parametrize("ddl", [create_database, drop_database])
async def test_ddl_primitives_reject_non_test_url_before_db_name(ddl) -> None:  # type: ignore[no-untyped-def]
    # The URL isolation guard runs first — a non-test URL is refused even with a
    # valid db_name, so a stray caller can't DDL against a dev/prod database.
    with pytest.raises(EvalDBIsolationError):
        await ddl("postgresql+psycopg://u:p@host:5432/production", "outrider_test_abc123")


def test_ddl_primitives_accept_a_generated_name_shape() -> None:
    # The names every real caller generates must pass the guard. We can't CREATE
    # without a DB, but the guard is pure — assert it does NOT raise on the shape.
    from outrider.eval_support.db_lifecycle import _validate_db_name

    for good in ("outrider_eval_0a1b2c3d", "outrider_test_smoke_deadbeef", "outrider_test_ff00"):
        _validate_db_name(good)  # must not raise


def test_require_eval_mode_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTRIDER_IS_EVAL", raising=False)
    with pytest.raises(EvalModeNotEnabledError):
        require_eval_mode()


def test_require_eval_mode_passes_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_IS_EVAL", "1")
    require_eval_mode()  # must not raise
