# See docs/spec.md §6.10 — whole-PR pre-flight size gate.
"""Startup limits for the whole-PR size gate (docs/spec.md §6.10).

`OUTRIDER_INTAKE_MAX_LINES` and `OUTRIDER_INTAKE_MAX_FILES` override the defaults
of 1000 changed lines and 30 files. build_graph injects these settings into
intake (`nodes-receive-deps-via-closure`).

Intake requests max_files + 1 metadata entries to detect overflow in one page.
GitHub caps a page at 100, so max_files cannot exceed 99 without pagination."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["IntakeConfig"]


class IntakeConfig(BaseSettings):
    """Frozen, positive PR limits; max_files must fit the one-page overflow check."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_INTAKE_",
        extra="forbid",
        frozen=True,
    )

    max_lines: int = Field(default=1000, gt=0)
    max_files: int = Field(default=30, gt=0, le=99)
