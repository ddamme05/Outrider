# Trust-boundary #8 floor test for the publish node per spec §V LLM provider boundary.
"""Pin the contract that the publish node makes ZERO LLM calls.

Per the publish-node spec §V LLM provider boundary:

    Publish makes ZERO LLM calls. Verified by absence of `LLMCallEvent`
    emission and import-graph unit test.

This is that import-graph unit test. Three layers of verification:

  1. The `agent.nodes.publish` module does NOT import any symbol from
     `outrider.llm.*`. (Direct-import check.)
  2. Importing `agent.nodes.publish` does NOT transitively cause
     `anthropic` (or any other LLM vendor SDK) to land in `sys.modules`.
     (Transitive-import check — defends against a future helper that
     re-exports an LLM symbol from a non-llm path.)
  3. The publish node's source contains no `LLMCallEvent` reference.
     (Emission check — defends against future code paths that mint
     an LLMCallEvent and pass it to a non-LLM sink.)

Trust boundary #8 (LLM provider boundary): vendor SDK imports are
confined to `outrider.llm.*`. The publish node is not an LLM consumer;
this test is the structural floor that pins the property even when a
contributor adds a "just one quick" import that crosses the boundary.
"""

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


_PUBLISH_NODE_PATH = "src/outrider/agent/nodes/publish.py"


def test_publish_node_source_imports_nothing_from_outrider_llm() -> None:
    """`agent.nodes.publish` MUST NOT import any symbol from `outrider.llm.*`.

    Catches direct imports — the most common future-regression shape.
    """
    source = (_project_root() / _PUBLISH_NODE_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offending: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("outrider.llm")
        ):
            offending.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("outrider.llm"):
                    offending.append(f"line {node.lineno}: import {alias.name}")
    if offending:
        msg = (
            "Publish node MUST NOT import from outrider.llm.* — trust "
            "boundary #8 + spec §V LLM provider boundary:\n  " + "\n  ".join(offending)
        )
        raise AssertionError(msg)


def test_publish_node_source_does_not_reference_llm_call_event() -> None:
    """`agent.nodes.publish` MUST NOT reference `LLMCallEvent` in code.

    The publish node makes no LLM calls; emitting an LLMCallEvent
    would either misuse the audit-event taxonomy OR signal a future
    code path that crossed the LLM-provider boundary.

    Walks the AST instead of substring-matching the source text so the
    module docstring (which describes the ABSENCE of LLM calls) doesn't
    false-positive the test.
    """
    source = (_project_root() / _PUBLISH_NODE_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offending: list[str] = []
    for node in ast.walk(tree):
        # ImportFrom: `from outrider.audit.events import LLMCallEvent`
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "LLMCallEvent":
                    offending.append(f"line {node.lineno}: import LLMCallEvent")
        # Name reference: `LLMCallEvent(...)` or `x: LLMCallEvent` or
        # any other code-level use of the bare name.
        elif isinstance(node, ast.Name) and node.id == "LLMCallEvent":
            offending.append(f"line {node.lineno}: code reference to LLMCallEvent")
        # Attribute reference: `events.LLMCallEvent(...)`.
        elif isinstance(node, ast.Attribute) and node.attr == "LLMCallEvent":
            offending.append(f"line {node.lineno}: attribute reference LLMCallEvent")
    if offending:
        msg = (
            "publish.py contains code-level `LLMCallEvent` reference(s). Per spec §V "
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
