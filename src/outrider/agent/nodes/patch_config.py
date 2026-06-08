# See DECISIONS.md#040 — suggested-patch generation feature flags.
"""PatchConfig — suggested-patch generation behavior (DECISIONS.md#040).

Closure-injected at `build_graph(...)` per `nodes-receive-deps-via-closure`;
synthesize reads it to decide whether to run the patch pass and how many
suggestions to generate. The patch MODEL lives in `ModelConfig.patch_model`
(the model-string registry, validated there); this config carries the feature
behavior — the on/off flag and the per-review cap.

Mirrors `HITLConfig`: a frozen `BaseSettings` read at startup. When
`patches_enabled` is False the synthesize patch pass is skipped entirely (zero
added LLM cost). `max_patch_suggestions_per_review` bounds worst-case cost +
comment noise: synthesize generates patches for at most this many HIGH/CRITICAL
single-line findings, CRITICAL-first.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PatchConfig(BaseSettings):
    """Reads `OUTRIDER_PATCHES_ENABLED` (default True) and
    `OUTRIDER_MAX_PATCH_SUGGESTIONS_PER_REVIEW` (default 5).

    Tests construct `PatchConfig(patches_enabled=False)` /
    `PatchConfig(max_patch_suggestions_per_review=2)` directly and inject through
    `build_graph(...)`. `frozen=True`: construction-time-only.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_",
        extra="forbid",
        frozen=True,
    )

    patches_enabled: bool = True
    # gt=0: a zero cap with patches_enabled=True is a misconfiguration (the pass
    # would run but generate nothing) — reject it loudly. Disable via the flag.
    max_patch_suggestions_per_review: int = Field(default=5, gt=0)
