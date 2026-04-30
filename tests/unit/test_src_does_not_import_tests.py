"""Regression test: src/ must not import from tests/.

Backs the production/test boundary convention in `docs/conventions.md`
("tests/ is not importable from src/"). The eval-harness spec added
`"."` to `pyproject.toml` `pythonpath` so harness-internal tests in
`tests/unit/` and `tests/integration/` can import factories from
`tests/eval/fixtures/`. That widening means a developer could
accidentally `from tests.X import Y` in production code and the test
suite would pass — production would then break at runtime because
`tests/` isn't on sys.path outside pytest.

This regression test scans `src/outrider/` for forbidden import patterns
and fails if any production file imports from `tests/`. Belt + suspenders
with mypy: if mypy in CI's `types` job catches the import (depending on
namespace resolution), this test catches it explicitly.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "outrider"

_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "from tests.",
    "from tests ",
    "import tests.",
    "import tests\n",
)


def test_src_does_not_import_from_tests() -> None:
    """Scan every .py file under src/outrider/ and fail if any imports tests.*."""
    violations: list[tuple[Path, str]] = []
    for py_file in SRC_ROOT.rglob("*.py"):
        text = py_file.read_text()
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in text:
                violations.append((py_file.relative_to(REPO_ROOT), pattern))

    assert violations == [], (
        "Production code under src/ imports from tests/ — boundary "
        "violation per docs/conventions.md ('tests/ is not importable "
        "from src/'). Pythonpath includes '.' so the import works during "
        "pytest; in production (no pytest), the import will fail at "
        f"runtime. Violations: {violations}"
    )
