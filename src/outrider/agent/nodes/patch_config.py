"""Load the enable flag and per-review limit for suggested fixes.

Startup passes these settings into synthesis (nodes-receive-deps-via-closure).
ModelConfig selects the model. Disabled generation makes no fix request.

See DECISIONS.md#040.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PatchConfig(BaseSettings):
    """Frozen patch settings from OUTRIDER_PATCHES_ENABLED (default True) and
    OUTRIDER_MAX_PATCH_SUGGESTIONS_PER_REVIEW (default 5)."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_",
        extra="forbid",
        frozen=True,
    )

    patches_enabled: bool = True
    # gt=0: a zero cap with patches_enabled=True is a misconfiguration (the pass
    # would run but generate nothing) — reject it loudly. Disable via the flag.
    max_patch_suggestions_per_review: int = Field(default=5, gt=0)
