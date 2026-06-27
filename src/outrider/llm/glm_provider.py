# Transitional shim — GLMProvider moved to openai_compatible_provider.py
# (DECISIONS.md#056). Re-exports the GLM names so existing importers (the
# wire-equivalence golden, the GLM scorecard) keep resolving across the rename.
# Removed once callers migrate to OpenAICompatibleProvider + host selection (step 4).
"""Backward-compatible re-export of the GLM/Baseten provider surface.

The concrete implementation now lives in `openai_compatible_provider.py`
(`OpenAICompatibleProvider`); `GLMProvider` is the transitional alias bound to
`BASETEN_PROFILE`. This module exists only so the pre-rename import path
`outrider.llm.glm_provider` keeps working until callers migrate.
"""

from __future__ import annotations

from outrider.llm.openai_compatible_provider import (
    BASETEN_BASE_URL,
    GLM_MODEL_ID,
    GLMProvider,
)

__all__ = ["BASETEN_BASE_URL", "GLM_MODEL_ID", "GLMProvider"]
