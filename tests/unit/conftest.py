"""Import OpenAI probe modules without leaving the repository root on sys.path.

Tests exercise the real derivation functions. Remove temporary path entries
after import so production modules still cannot import top-level tests."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from spikes.openai.arc2.contracts import RowMeasurement

    from outrider.prompts.analyze import AnalyzePromptParts

_REPO_ROOT = str(Path(__file__).resolve().parents[2])

_ARC2_MODULES = (
    "spikes.openai.arc2.classifier",
    "spikes.openai.arc2.contracts",
    "spikes.openai.arc2.attempts",
    "spikes.openai.arc2.verifier",
    "spikes.openai.arc2.strict_schema",
    "spikes.openai.strict_schema_probe",
)

_was_present = _REPO_ROOT in sys.path
if not _was_present:
    sys.path.insert(0, _REPO_ROOT)
try:
    for _name in _ARC2_MODULES:
        importlib.import_module(_name)
finally:
    if not _was_present:
        # Strip EVERY occurrence, not just the first: an imported module may add
        # its own copy (the probe bootstraps the same root to run as a script), and
        # a single `remove()` would leave one behind — which is how the root
        # silently stayed on the path for the whole process.
        while _REPO_ROOT in sys.path:
            sys.path.remove(_REPO_ROOT)

# `tests/unit/test_arc2_path_isolation.py` asserts this cleanup actually held.


@pytest.fixture
def arc2_reviewed_analyzer(monkeypatch: pytest.MonkeyPatch) -> tuple[RowMeasurement, ...]:
    """Use the November 1 Arc 2 system message and return its recorded measurements."""
    from spikes.openai.arc2 import plan
    from spikes.openai.arc2.contracts import RowMeasurement

    from outrider.prompts import analyze

    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    system_prompt = (fixtures / "arc2-reviewed-v13-system.txt").read_bytes().decode("utf-8")

    def render(**kwargs: Any) -> AnalyzePromptParts:
        return replace(analyze.render(**kwargs), system_prompt=system_prompt)

    monkeypatch.setattr(
        plan,
        "analyze_prompt",
        SimpleNamespace(VERSION="analyze-v13", MAX_TOKENS=8192, TEMPERATURE=0.0, render=render),
    )
    return TypeAdapter(tuple[RowMeasurement, ...]).validate_json(
        (fixtures / "arc2-reviewed-v13-measurements.json").read_bytes()
    )
