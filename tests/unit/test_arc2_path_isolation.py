"""Arc 2 — the spike-import bootstrap must not erode the `pythonpath` guard.

`tests/unit/conftest.py` makes `spikes/openai/arc2/` importable so these tests can
verify the real derivation rather than a re-implementation. `sys.path` is
process-global, so doing that carelessly leaves the repo root on the path for the
whole pytest process and makes top-level `tests` importable — which is precisely
the failure `pythonpath = ["src"]` exists to make loud (`docs/conventions.md`:
"`tests/utils/` is not importable from `src/`").

Measured baseline before the fix: without the conftest the root was absent and
`tests` was NOT importable; with the naive version both flipped. Two copies of the
root were being added (the conftest's own, plus the probe's script bootstrap), so
a single `sys.path.remove()` left one behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def test_repo_root_is_not_left_on_sys_path() -> None:
    """The bootstrap is scoped to the import; the path is clean afterwards."""
    assert _REPO_ROOT not in sys.path, (
        "tests/unit/conftest.py left the repo root on sys.path — the import guard "
        "is eroded for the whole pytest process"
    )


def test_tests_package_is_not_importable() -> None:
    """The property the guard actually protects: production code importing test
    code must FAIL, not silently succeed."""
    import importlib.util

    assert importlib.util.find_spec("tests") is None, (
        "top-level `tests` became importable — a `src/` module doing `import tests.x` "
        "would now succeed instead of failing loudly"
    )


def test_arc2_modules_are_still_importable() -> None:
    """The positive half: the cleanup must not break what it enabled. These
    resolve from `sys.modules`, without any finder consulting `sys.path`."""
    from spikes.openai.arc2.classifier import Verdict
    from spikes.openai.arc2.contracts import STRICT_PROBE_CONTRACT_VERSION
    from spikes.openai.arc2.strict_schema import derive_strict_analyze_schema

    assert Verdict.GO.value == "GO"
    assert STRICT_PROBE_CONTRACT_VERSION.startswith("arc2-strict-schema:")
    assert derive_strict_analyze_schema()["type"] == "object"
