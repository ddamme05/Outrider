"""Expose the GLM provider configured for Baseten.

openai_compatible_provider.py owns GLMProvider and the shared implementation.
This module exports its Baseten endpoint, model ID and provider class.
See DECISIONS.md#056.
"""

from __future__ import annotations

from outrider.llm.openai_compatible_provider import (
    BASETEN_BASE_URL,
    GLM_MODEL_ID,
    GLMProvider,
)

__all__ = ["BASETEN_BASE_URL", "GLM_MODEL_ID", "GLMProvider"]
