"""Prompt templates and rendering for agent nodes.

V1 ships one prompt per node:
- `prompts.triage` — fast Haiku triage classifier (DEEP/STANDARD/SKIM
  per file + overall PR risk + relevant dimensions).
- `prompts.analyze` — Sonnet analyze pass (per-file finding emission
  with proof-tier enforcement at the parser layer).

`prompts.trace` and `prompts.synthesize` land with their respective
node specs. Multi-version registry + A/B harness deferred to V1.5+.

Convention: each prompt module exposes module-level constants
(`SYSTEM_PROMPT*`, `USER_TEMPLATE`, `VERSION`, `MAX_TOKENS`,
`TEMPERATURE`), a frozen `XxxPromptParts` dataclass result type, and a
pure `render(...)` function. Direct module import is the supported
access pattern (`from outrider.prompts import analyze as
analyze_prompt`).
"""
