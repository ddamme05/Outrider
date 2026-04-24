# Outrider

Outrider is an agentic pull request review tool: a FastAPI webhook receives a GitHub PR event, dispatches a 7-node LangGraph review flow, uses tree-sitter structure to focus analysis, gates high-severity findings through HITL approval, and records every meaningful step in an append-only Postgres audit trail.

## Setup

```bash
uv sync --dev
cp .env.example .env
.venv/bin/pre-commit install
```

Fill in `.env` with local database, GitHub App, webhook, and Anthropic credentials. See `docs/deployment.md` for the required variables and permission model.

Run `uv sync --dev` before `pre-commit install`; the hooks call tools from `.venv/bin`.

## Checks

```bash
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/mypy src scripts tests
.venv/bin/python scripts/extract_invariants.py --check
.venv/bin/python scripts/test_extract_invariants.py
.venv/bin/python scripts/check_decision_refs.py
.venv/bin/python scripts/test_check_decision_refs.py
```

`pytest` is configured, but the current scaffold has no collected pytest tests yet; the smoke tests above are script entry points and are also wired into pre-commit.

## Docs

- `docs/architecture.md` explains the review graph and subsystem layout.
- `docs/trust-boundaries.md` captures the rules that protect the security and replay story.
- `docs/invariants.md` is generated from tags in `docs/spec.md`; do not edit it by hand.
- `.claude/skills/outrider-navigator/SKILL.md` is the Claude Code skill that surfaces applicable invariants before Outrider code work.
