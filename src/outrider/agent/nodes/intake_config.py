"""Load size limits for pull requests entering review.

OUTRIDER_INTAKE_MAX_LINES and OUTRIDER_INTAKE_MAX_FILES default to 1000 changed
lines and 30 files. Startup passes them into intake
(nodes-receive-deps-via-closure). The file limit cannot exceed 99 because intake
requests one extra entry to detect overflow within GitHub's 100-item page.

See docs/spec.md §6.10.
"""

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
