"""Runner for the GitHub App + smee.io spike.

Runs every demo as a subprocess with the project venv's Python. Exits 0
iff every demo exits 0. Prints the number of demos that passed and the
offending demo on first failure, then stops.

Demos import from receiver.py (Q5 only) — the runner adds the spike root
to PYTHONPATH so `from receiver import app` resolves without install.

This is the mechanical verification for NOTES.md — if every claim in NOTES
has a demo here, and run_all.py exits 0, the claims reproduce on the
pinned versions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DEMOS = sorted((ROOT / "demos").glob("demo_q*.py"))


def main() -> int:
    if not DEMOS:
        print("run_all.py: no demos found", file=sys.stderr)
        return 2

    env = os.environ.copy()
    # Make `from receiver import app` work without installing the spike.
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"run_all.py: {len(DEMOS)} demo(s) with {sys.executable}\n")
    passed = 0
    for demo in DEMOS:
        print(f"--- {demo.name} " + "-" * (50 - len(demo.name)))
        result = subprocess.run(
            [sys.executable, str(demo)],
            cwd=ROOT,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"\nrun_all.py: FAILED at {demo.name} (exit {result.returncode}). Stopping.",
                file=sys.stderr,
            )
            return result.returncode
        passed += 1

    print(f"\nrun_all.py: {passed}/{len(DEMOS)} demos passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
