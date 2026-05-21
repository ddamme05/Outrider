"""Prompt templates and rendering for agent nodes.

V1 ships one prompt per node:
- `prompts.triage` — fast Haiku triage classifier (DEEP/STANDARD/SKIM
  per file + overall PR risk + relevant dimensions).
- `prompts.analyze` — Sonnet analyze pass (per-file finding emission
  with proof-tier enforcement at the parser layer).

`prompts.trace` and `prompts.synthesize` land with their respective
node specs. Multi-version registry + A/B harness deferred to V1.5+ per
the triage-node spec's non-goal #5.

Convention (locked 2026-05-20): each prompt module exposes module-level
constants (`SYSTEM_PROMPT*`, `USER_TEMPLATE`, `VERSION`, `MAX_TOKENS`,
`TEMPERATURE`), a frozen `XxxPromptParts` dataclass result type, and a
pure `render(...)` function. NOT a `PromptRegistry` class — direct
module import (e.g., `from outrider.prompts import analyze as
analyze_prompt`) is the supported access pattern. The analyze-node
spec §5 originally described a class; the shipped convention is
followed instead and the spec divergence is recorded in that spec's
Actual Outcome.
"""
