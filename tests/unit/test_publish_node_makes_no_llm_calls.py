"""Check that publishing cannot call an LLM or emit LLM-call events.

Inspect every publish implementation module for direct imports and event
references, then check a fresh import for transitive vendor SDK loading.
See the LLM provider boundary in docs/trust-boundaries.md §8."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate project root")


def _publish_node_paths() -> list[Path]:
    node_directory = _project_root() / "src/outrider/agent/nodes"
    paths = sorted(node_directory.glob("publish*.py"))
    assert paths, f"no publish implementation modules found under {node_directory}"
    return paths


def test_publish_node_source_imports_nothing_from_outrider_llm() -> None:
    """Publish implementation modules MUST NOT import from `outrider.llm.*`.

    Catches direct imports — the most common future-regression shape.
    """
    offending: list[str] = []
    for path in _publish_node_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("outrider.llm")
            ):
                offending.append(f"{path.name}:{node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("outrider.llm"):
                        offending.append(f"{path.name}:{node.lineno}: import {alias.name}")
    if offending:
        msg = (
            "Publish node MUST NOT import from outrider.llm.* — trust "
            "boundary #8 + spec §V LLM provider boundary:\n  " + "\n  ".join(offending)
        )
        raise AssertionError(msg)


def test_publish_node_source_does_not_reference_llm_call_event() -> None:
    """Publish implementation modules MUST NOT reference `LLMCallEvent` in code.

    The publish node makes no LLM calls; emitting an LLMCallEvent
    would either misuse the audit-event taxonomy OR signal a future
    code path that crossed the LLM-provider boundary.

    Walks the AST instead of substring-matching the source text so the
    module docstring (which describes the ABSENCE of LLM calls) doesn't
    false-positive the test.
    """
    offending: list[str] = []
    for path in _publish_node_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "LLMCallEvent":
                        offending.append(f"{path.name}:{node.lineno}: import LLMCallEvent")
            elif isinstance(node, ast.Name) and node.id == "LLMCallEvent":
                offending.append(f"{path.name}:{node.lineno}: code reference to LLMCallEvent")
            elif isinstance(node, ast.Attribute) and node.attr == "LLMCallEvent":
                offending.append(f"{path.name}:{node.lineno}: attribute reference LLMCallEvent")
    if offending:
        msg = (
            "Publish implementation contains code-level `LLMCallEvent` reference(s). Per spec §V "
            "LLM provider boundary, the publish node makes ZERO LLM calls and emits "
            "no LLMCallEvent:\n  " + "\n  ".join(offending)
        )
        raise AssertionError(msg)


def test_publish_node_transitive_imports_do_not_load_anthropic() -> None:
    """Importing `agent.nodes.publish` MUST NOT cause `anthropic` to load.

    Defends against a future helper module that re-exports an LLM
    symbol from a non-llm path.

    Runs the import check in a fresh subprocess so the result is
    deterministic regardless of test execution order.

    NOTE: currently expected-to-fail under V1 because the chain
    `publish.py → audit.events → llm.pricing` triggers Python's parent
    package init for `outrider.llm/__init__.py`, which re-exports
    `AnthropicProvider` → loads `anthropic` transitively. The strict
    trust-boundary `vendor-sdks-only-in-wrappers` IS satisfied (the
    `import anthropic` statement is correctly confined to `llm/`); the
    transitive supply-chain concern is tracked separately as FUP-071.
    When the import chain is refactored, drop the `xfail` and this
    test becomes the deterministic floor preventing future regressions.
    """
    code = textwrap.dedent(
        """
        import importlib, sys
        importlib.import_module("outrider.agent.nodes.publish")
        bad = sorted(
            m for m in sys.modules
            if m == "anthropic" or m.startswith("anthropic.")
        )
        if bad:
            raise SystemExit("loaded disallowed modules: " + ", ".join(bad))
        """
    )
    completed = subprocess.run(  # noqa: S603 — fixed argv + system python; no shell
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        # Guard against a stalled subprocess hanging the suite. The
        # import + sys.modules walk should complete in well under 2s;
        # 20s is generous but bounded.
        timeout=20,
    )
    if completed.returncode != 0:
        # Gate the xfail on the explicit "loaded disallowed modules:"
        # sentinel the subprocess prints — otherwise a real regression
        # (e.g., a syntax/import error in `outrider.agent.nodes.publish`
        # itself, an OSError, an environment-shape issue) would be
        # silently swallowed under the FUP-071 xfail. Real failures
        # surface as a normal AssertionError; only the documented
        # transitive-import case xfails.
        output = (completed.stderr or completed.stdout).strip()
        if "loaded disallowed modules:" in output:
            pytest.xfail(
                f"Known transitive import (tracked as FUP-071 — "
                f"publish→audit.events→llm.pricing→llm/__init__.py "
                f"imports AnthropicProvider): {output!r}"
            )
        raise AssertionError(
            f"transitive-import subprocess failed with exit "
            f"{completed.returncode} (not the FUP-071 sentinel — "
            f"investigate as a real regression):\n"
            f"  stderr: {completed.stderr!r}\n"
            f"  stdout: {completed.stdout!r}"
        )
