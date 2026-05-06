"""Outrider — agentic PR review.

This module's import-time side effect is **wiring the LLM-content
logging filter** onto every reachable handler in the logger chain
(`RejectLLMContentFilter`, per
`specs/2026-05-05-llm-provider-wrapper.md` AC#23 + DECISIONS#016 point 4).

The filter is the defense-in-depth backup to the schema-level
default-redaction on `LLMRequest`/`LLMResponse`/`LLMMessage`. Wiring it
at `import outrider` time means the filter is active before any agent /
LLM / audit code runs — production code paths import `outrider` at
process startup before any review-handling code.

Re-invocable: call `register_filter_on_all_handlers()` again after
later handler registration (e.g., FastAPI/uvicorn registering theirs at
app startup). The function is idempotent.
"""

from outrider.llm.logging import register_filter_on_all_handlers

# Side-effect import: install the LLM content filter on all reachable
# handlers immediately. Production code that does `import outrider` (or
# any submodule, which transitively imports this) gets the filter wired
# before any review work begins.
register_filter_on_all_handlers()


def main() -> None:
    print("Hello from outrider!")


if __name__ == "__main__":
    main()
