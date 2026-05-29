"""The eval-DB isolation guard (`_assert_test_url_is_isolated`) rejects unsafe URLs.

PR-1 fix #7: the guard parses TEST_DATABASE_URL with SQLAlchemy `make_url` and
checks the port + database as structured components, rather than substring
matching that could false-match a port-like or "test"-like sequence embedded in
a password or host. These tests pin the rejection paths — the guard is the
safety mechanism that prevents eval tests from running against the dev DB
(see docs/testing.md "Two-container model").
"""

import pytest

from .conftest import _assert_test_url_is_isolated


def test_guard_accepts_isolated_test_url() -> None:
    """A well-formed postgres-test URL (port 5433, db name containing 'test') passes."""
    _assert_test_url_is_isolated("postgresql+asyncpg://outrider:pw@localhost:5433/outrider_test")


def test_guard_accepts_test_url_with_query_params() -> None:
    """Query params don't fool the structural parse — `.database` excludes them.

    A naive `url.rsplit('/', 1)[-1]` would have read 'outrider_test?sslmode=...'
    as the database segment; make_url returns 'outrider_test'.
    """
    _assert_test_url_is_isolated(
        "postgresql+asyncpg://outrider:pw@localhost:5433/outrider_test?sslmode=disable"
    )


def test_guard_rejects_wrong_port() -> None:
    """A URL on the dev port (5432) is rejected even if the db name contains 'test'."""
    with pytest.raises(RuntimeError, match="must target port 5433"):
        _assert_test_url_is_isolated(
            "postgresql+asyncpg://outrider:pw@localhost:5432/outrider_test"
        )


def test_guard_rejects_wrong_db_name() -> None:
    """A URL on the test port but a non-test database name is rejected."""
    with pytest.raises(RuntimeError, match="database name must contain"):
        _assert_test_url_is_isolated("postgresql+asyncpg://outrider:pw@localhost:5433/outrider")


def test_guard_rejects_non_numeric_port() -> None:
    """A non-numeric port surfaces the guard's RuntimeError, not a raw ValueError.

    make_url raises builtins.ValueError (NOT an ArgumentError subclass) when the
    port component isn't an integer, so the guard must catch ValueError too —
    regression for the `except (ArgumentError, ValueError)` widening.
    """
    with pytest.raises(RuntimeError, match="not a parseable database URL"):
        _assert_test_url_is_isolated(
            "postgresql+asyncpg://outrider:pw@localhost:notaport/outrider_test"
        )
