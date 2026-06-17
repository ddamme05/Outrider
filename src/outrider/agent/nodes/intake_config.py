# See docs/spec.md §6.10 — whole-PR pre-flight size gate.
"""IntakeConfig — the whole-PR pre-flight size gate (docs/spec.md §6.10).

The spec defaults — skip a PR with > 1000 changed lines OR > 30 files — are the FIELD
defaults here; `OUTRIDER_INTAKE_MAX_LINES` / `OUTRIDER_INTAKE_MAX_FILES` override them so
operators can tune the limits without a code change. Closure-injected at `build_graph(...)`
per `nodes-receive-deps-via-closure`; intake reads it for the gate AND to size the per-file
list request (`per_page = max_files + 1`, so the API surfaces one extra entry and the gate
fires deterministically).

`max_files` is capped at 99: the gate uses a SINGLE `GET /pulls/{n}/files` call with
`per_page = max_files + 1`, and GitHub caps `per_page` at 100. A larger limit would need
pagination (out of V1 scope), so values > 99 are rejected rather than silently
under-counting files.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["IntakeConfig"]


class IntakeConfig(BaseSettings):
    """Reads `OUTRIDER_INTAKE_MAX_LINES` (default 1000) and `OUTRIDER_INTAKE_MAX_FILES`
    (default 30). Tests construct `IntakeConfig(max_files=...)` directly and inject through
    `build_graph(...)`. `frozen=True`: construction-time-only config."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_INTAKE_",
        extra="forbid",
        frozen=True,
    )

    max_lines: int = Field(default=1000, gt=0)
    max_files: int = Field(default=30, gt=0, le=99)
