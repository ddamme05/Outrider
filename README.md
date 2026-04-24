# Outrider

Outrider is an agentic pull request review tool: a FastAPI webhook receives a GitHub PR event, dispatches a 7-node LangGraph review flow, uses tree-sitter structure to focus analysis, gates high-severity findings through HITL approval, and records every meaningful step in an append-only Postgres audit trail.

V1 is self-hostable: you run Outrider in your own infrastructure, connect it to a GitHub App installed on your repos, and point it at your LLM provider. Code and PR text reach one third party — the LLM provider (Anthropic in V1). Under Anthropic's default terms, inputs are retained for 30 days; content flagged for policy violations is retained up to 2 years and classification scores up to 7 years; no data is used for training without permission. If your organization has a zero-data-retention (ZDR) arrangement with Anthropic, set `ANTHROPIC_ZDR_ENABLED=true` — Outrider uses the flag to adjust its startup notice and privacy statement; it does not enable ZDR on its own (contact Anthropic sales to arrange). Outrider itself adds no retention beyond the TTLs you configure. Outrider does not currently support HIPAA-subject workloads. See [`DECISIONS.md#011`](DECISIONS.md#011-self-hosted-is-canonical-v1-saas-is-v15) for the deployment-model commitment, [`DECISIONS.md#012`](DECISIONS.md#012-data-retention-ttls-configurable-purge-on-installationdeleted) + [`DECISIONS.md#014`](DECISIONS.md#014-audit-events-are-metadata-only-content-purge-targets-reviews-and-findings) for retention and purge behavior, and [`DECISIONS.md#013`](DECISIONS.md#013-llmprivacy-contract-anthropic-egress-retention-zdr) + [`DECISIONS.md#015`](DECISIONS.md#015-zdr-is-an-operator-attestation-not-a-runtime-opt-in) for the full LLM privacy contract.

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
- `DECISIONS.md` is the public record of architectural choices. Code and comments cite entries by slug (e.g., `# See DECISIONS.md#011-self-hosted-is-canonical-v1`).
