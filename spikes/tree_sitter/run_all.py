"""Run the tree-sitter demos with the current Python interpreter.

Stop on the first nonzero exit and print the demo results. NOTES.md records
the experiment's findings on its pinned dependency versions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DEMOS = sorted((ROOT / "demos").glob("demo_q*.py"))


def main() -> int:
    if not DEMOS:
        print("run_all.py: no demos found", file=sys.stderr)
        return 2

    print(f"run_all.py: {len(DEMOS)} demo(s) with {sys.executable}\n")
    passed = 0
    for demo in DEMOS:
        print(f"--- {demo.name} " + "-" * (50 - len(demo.name)))
        result = subprocess.run(
            [sys.executable, str(demo)],
            cwd=ROOT,
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
