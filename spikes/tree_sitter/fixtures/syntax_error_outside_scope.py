# Fixture: a syntax error at module level, *outside* any function body.
# Simulates "parse error outside the changed region" per spec §5.5 — the
# changed scope should still be reliable even though the file has errors.

import os


# Stray garbage at module level (not inside any function).
??? this is not python ???


def still_parseable(x: int) -> int:
    return x * 2


def another_ok():
    return os.getcwd()
