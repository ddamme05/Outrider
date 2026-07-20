"""Make the Arc 2 `spikes/openai/arc2/` probe modules importable — WITHOUT leaving
the repo root on `sys.path`.

`pyproject.toml` sets `pythonpath = ["src"]` only, which keeps `tests/` out of the
import path. An earlier version of this file inserted the repo root and left it
there; `sys.path` is process-global, so that made top-level `tests` importable for
the whole pytest process, eroding the very guard `pythonpath = ["src"]` exists to
provide (a `src/` module doing `import tests.x` would SUCCEED instead of failing
loudly).

The fix is to scope the insertion to the import itself: put the root on the path,
import the modules so they land in `sys.modules`, then remove it. Subsequent
`from spikes.openai.arc2... import X` in test files resolves from `sys.modules`
without any finder consulting `sys.path`, so the path is clean afterwards and
`tests` is not importable.

**Why not the `spikes/`-duplication pattern the repo uses elsewhere.**
`tests/eval/test_openai_scorecard.py` deliberately re-implements probe logic
("independent recomputation is the point — the gate must not trust the manifest's
own conservation_facts block"). That rationale governs GRADING instruments, which
must characterize evidence independently of whatever produced it. It does not
transfer to unit-testing a derivation function: re-implementing
`derive_strict_analyze_schema` would only test the re-implementation.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])

_ARC2_MODULES = (
    "spikes.openai.arc2.classifier",
    "spikes.openai.arc2.contracts",
    "spikes.openai.arc2.attempts",
    "spikes.openai.arc2.verifier",
    "spikes.openai.arc2.strict_schema",
    "spikes.openai.strict_schema_probe",
)

_was_present = _REPO_ROOT in sys.path
if not _was_present:
    sys.path.insert(0, _REPO_ROOT)
try:
    for _name in _ARC2_MODULES:
        importlib.import_module(_name)
finally:
    if not _was_present:
        # Strip EVERY occurrence, not just the first: an imported module may add
        # its own copy (the probe bootstraps the same root to run as a script), and
        # a single `remove()` would leave one behind — which is how the root
        # silently stayed on the path for the whole process.
        while _REPO_ROOT in sys.path:
            sys.path.remove(_REPO_ROOT)

# `tests/unit/test_arc2_path_isolation.py` asserts this cleanup actually held.
