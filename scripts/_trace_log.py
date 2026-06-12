"""Per-run trace tee for the rehearsal scripts.

The scripts' diagnostic dumps (thousands of lines for `smoke_e2e.py`)
truncate in most terminals, so every `_say` line also writes to a
timestamped file under `scripts/generated/` — gitignored (the unanchored
`generated/` pattern), created on demand, flushed per line so even a
killed run leaves a readable prefix of exactly what happened before it
died. One recipe, three consumers (`smoke_e2e.py`, `live_claude_smoke.py`,
`live_github_demo.py`) — the tee must not fork per script.

Imported bare (`import _trace_log` / `from _trace_log import TraceTee`):
when a script under `scripts/` runs as `python scripts/<name>.py`, the
script's own directory is `sys.path[0]`, so the bare import resolves for
all three without packaging `scripts/`.
"""

from __future__ import annotations

import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

_GENERATED_DIR = Path(__file__).resolve().parent / "generated"


class TraceTee:
    """Open-on-construct file sink for a script's full trace."""

    def __init__(self, prefix: str) -> None:
        _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.path = _GENERATED_DIR / f"{prefix}_{stamp}.txt"
        self._fh: TextIO = self.path.open("w", encoding="utf-8")

    def write_line(self, msg: str) -> None:
        self._fh.write(msg + "\n")
        self._fh.flush()

    def write_current_exception(self) -> None:
        """Record the in-flight exception's full traceback in the trace file.

        Crashes propagate outside the scripts' `_say` tee, so without this the
        one thing a failed run most needs diagnosed — the traceback — would be
        the one thing missing from its trace file.
        """
        self.write_line("")
        self.write_line("UNHANDLED EXCEPTION (full traceback):")
        self.write_line(traceback.format_exc())

    def close(self) -> None:
        self._fh.close()
