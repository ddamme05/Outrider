"""IntakeConfig — the whole-PR size-gate thresholds (docs/spec.md §6.10).

The spec defaults (1000 lines / 30 files) are the field defaults; OUTRIDER_INTAKE_*
env vars override them. max_files is capped at 99 (the gate uses one GET /pulls/files
call; GitHub per_page <= 100).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.agent.nodes.intake_config import IntakeConfig


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTRIDER_INTAKE_MAX_FILES", raising=False)
    monkeypatch.delenv("OUTRIDER_INTAKE_MAX_LINES", raising=False)


def test_defaults_match_spec() -> None:
    cfg = IntakeConfig()
    assert cfg.max_lines == 1000
    assert cfg.max_files == 30


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_INTAKE_MAX_FILES", "60")
    monkeypatch.setenv("OUTRIDER_INTAKE_MAX_LINES", "5000")
    cfg = IntakeConfig()
    assert cfg.max_files == 60
    assert cfg.max_lines == 5000


def test_direct_construction_overrides() -> None:
    cfg = IntakeConfig(max_files=10, max_lines=200)
    assert cfg.max_files == 10
    assert cfg.max_lines == 200


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_rejects_nonpositive(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("OUTRIDER_INTAKE_MAX_FILES", bad)
    with pytest.raises(ValidationError):
        IntakeConfig()


def test_rejects_max_files_over_99(monkeypatch: pytest.MonkeyPatch) -> None:
    # per_page = max_files + 1 must stay <= GitHub's 100 cap; >99 needs pagination.
    monkeypatch.setenv("OUTRIDER_INTAKE_MAX_FILES", "100")
    with pytest.raises(ValidationError):
        IntakeConfig()
