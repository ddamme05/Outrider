"""Versioned prompt templates and renderers for review model calls.

Triage selects review tiers; analyze proposes findings; trace ranks imports;
patch proposes replacement lines; synthesize summarizes admitted findings.
Each module keeps static instructions separate from request-specific data and
returns named fields from `render()`. Nodes import their prompt module directly."""


def safe_code_fence(body: str, *, lang: str = "") -> str:
    """Wrap data in a fence longer than any backtick run it contains.

    PR-controlled source, patches and model-derived text can contain markdown
    fences. A dynamic fence keeps that text inside the renderer's data block
    (`webhook-strings-are-data-not-format-strings`). Registry IDs and enum values
    do not need fencing."""
    fence = "```"
    while fence in body:
        fence += "`"
    return f"{fence}{lang}\n{body}\n{fence}"


__all__ = ["safe_code_fence"]
