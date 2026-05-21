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


def safe_code_fence(body: str, *, lang: str = "") -> str:
    """Wrap `body` in a markdown code fence longer than any backtick run
    inside it.

    Markdown closes a fence when the same-or-longer backtick run appears.
    PR-controlled `body` containing ` ``` ` (e.g., a docstring with
    embedded markdown examples, a diff line that quotes another file's
    fence) would close a fixed `` ``` `` fence and let attacker content
    escape the renderer's structure — exactly the
    `webhook-strings-are-data-not-format-strings` invariant. Returns
    `f"{fence}{lang}\\n{body}\\n{fence}"` where `fence` is the shortest
    string of ≥3 backticks not appearing in `body`.

    Use for every PR-controlled body the analyze/triage prompts wrap in
    a fence (scope-unit bodies, diff patches, bounded hunks). Producer-
    internal strings (e.g., `query_match_id` from the registry, enum
    values) do not need this helper.
    """
    fence = "```"
    while fence in body:
        fence += "`"
    return f"{fence}{lang}\n{body}\n{fence}"


__all__ = ["safe_code_fence"]
