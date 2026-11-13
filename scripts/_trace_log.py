"""Write smoke-script output to both the terminal and a trace file.

TraceTee creates a timestamped file under ignored scripts/generated/ and flushes
each line, leaving earlier output readable if a run stops. The three smoke/demo
runners share it. They import _trace_log directly because scripts/ is on Python's
import path when a script in that directory is run.
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
