"""Numeric admission checks over synthetic local captures; no model calls."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import TYPE_CHECKING, Any

import pytest

from .test_openai_scorecard import (
    _SOL,
    _require_probe_manifest,
    _rewrite_artifact,
    _write_valid_capture,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    gate_module = sys.modules[_require_probe_manifest.__module__]
    monkeypatch.setattr(gate_module, "_PROBE_MANIFEST", tmp_path / "manifest.json")
    return _write_valid_capture(tmp_path)


def _rewrite_usage(
    capture_dir: Path, capture: dict[str, Any], kind: str, updates: dict[str, object]
) -> None:
    """Change wire usage and its recorded hash together, preserving other evidence."""
    row = capture["manifest"]["results"][f"{_SOL}:{kind}"]
    fixture_path = capture_dir / row["fixture"]
    document = json.loads(fixture_path.read_bytes())
    document["usage"].update(updates)
    payload = json.dumps(document, indent=2).encode("utf-8")
    fixture_path.write_bytes(payload)
    row["sha256"] = hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("kind", ["cold", "warm"])
def test_cache_token_classes_cannot_exceed_prompt(
    tmp_path: Path, capture: dict[str, Any], kind: str
) -> None:
    if kind == "cold":
        counts = {"cached_tokens": 1000, "cache_write_tokens": 1500}
        # Keep billing consistent with the invalid wire residual, so equality
        # alone cannot reject the capture for the wrong reason.
        capture["artifact"]["models"][_SOL]["billed_fresh_input_tokens"] = -500
    else:
        counts = {"cached_tokens": 1500, "cache_write_tokens": 600}
    _rewrite_usage(tmp_path, capture, kind, {"prompt_tokens_details": counts})
    _rewrite_artifact(tmp_path, capture["manifest"], capture["artifact"])

    with pytest.raises(pytest.fail.Exception, match="token|usage|conservation|fresh|negative"):
        _require_probe_manifest()


@pytest.mark.parametrize(
    ("kind", "required_count"), [("cold", "cache_write_tokens"), ("warm", "cached_tokens")]
)
def test_required_cache_counts_reject_boolean(
    tmp_path: Path, capture: dict[str, Any], kind: str, required_count: str
) -> None:
    counts: dict[str, object] = {"cached_tokens": 0, "cache_write_tokens": 0}
    counts[required_count] = True
    if kind == "cold":
        # These billing counts are integers, isolating the boolean wire value.
        binding = capture["artifact"]["models"][_SOL]
        binding["billed_cache_write_tokens"] = 1
        binding["billed_fresh_input_tokens"] = 1999
    _rewrite_usage(tmp_path, capture, kind, {"prompt_tokens_details": counts})
    _rewrite_artifact(tmp_path, capture["manifest"], capture["artifact"])

    with pytest.raises(pytest.fail.Exception, match="integer|token|usage"):
        _require_probe_manifest()


def test_prompt_token_counts_reject_boolean(tmp_path: Path, capture: dict[str, Any]) -> None:
    for kind, counts in (
        ("cold", {"cached_tokens": 0, "cache_write_tokens": 1}),
        ("warm", {"cached_tokens": 1, "cache_write_tokens": 0}),
    ):
        _rewrite_usage(
            tmp_path,
            capture,
            kind,
            {"prompt_tokens": True, "total_tokens": 51, "prompt_tokens_details": counts},
        )
    binding = capture["artifact"]["models"][_SOL]
    binding["billed_cache_write_tokens"] = 1
    binding["billed_fresh_input_tokens"] = 0
    _rewrite_artifact(tmp_path, capture["manifest"], capture["artifact"])

    with pytest.raises(pytest.fail.Exception, match="integer|token|usage"):
        _require_probe_manifest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("billed_fresh_input_tokens", 500.0),
        ("billed_cache_write_tokens", 1500.0),
        ("billed_fresh_input_tokens", -1),
        ("billed_cache_write_tokens", -1),
    ],
)
def test_billed_counts_reject_non_integer_or_negative(
    tmp_path: Path, capture: dict[str, Any], field: str, value: object
) -> None:
    capture["artifact"]["models"][_SOL][field] = value
    _rewrite_artifact(tmp_path, capture["manifest"], capture["artifact"])

    with pytest.raises(pytest.fail.Exception, match="billed|integer|token|usage|negative"):
        _require_probe_manifest()


@pytest.mark.parametrize("kind", ["cold", "warm"])
def test_exact_zero_fresh_tokens_remains_admissible(
    tmp_path: Path, capture: dict[str, Any], kind: str
) -> None:
    if kind == "cold":
        counts = {"cached_tokens": 500, "cache_write_tokens": 1500}
        capture["artifact"]["models"][_SOL]["billed_fresh_input_tokens"] = 0
    else:
        counts = {"cached_tokens": 2000, "cache_write_tokens": 0}
    _rewrite_usage(tmp_path, capture, kind, {"prompt_tokens_details": counts})
    _rewrite_artifact(tmp_path, capture["manifest"], capture["artifact"])

    _require_probe_manifest()
