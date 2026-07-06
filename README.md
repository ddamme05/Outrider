# Outrider

Outrider is an agentic pull request review tool: a FastAPI webhook receives a GitHub PR event, dispatches a LangGraph review flow of seven logical nodes (the analyze stage fans out to parallel per-file workers), uses tree-sitter structure to focus analysis, gates high-severity findings through HITL approval, and records every meaningful step in an append-only Postgres audit trail.

V1 is self-hostable: you run Outrider in your own infrastructure, connect it to a GitHub App installed on your repos, and point it at your LLM provider. Code and PR text reach one third party — the LLM provider (Anthropic in V1). Under Anthropic's default terms, inputs are retained for 30 days; content flagged for policy violations is retained up to 2 years and classification scores up to 7 years; no data is used for training without permission. If your organization has a zero-data-retention (ZDR) arrangement with Anthropic, set `ANTHROPIC_ZDR_ENABLED=true` — Outrider uses the flag to adjust its startup notice and privacy statement; it does not enable ZDR on its own (contact Anthropic sales to arrange). Outrider stores LLM request and response content in your local database under configured retention TTL (default values in operator configuration; purged on `installation.deleted` along with reviews and findings, per [`DECISIONS.md#012`](DECISIONS.md#012-data-retention-ttls-configurable-purge-on-installationdeleted) + [`DECISIONS.md#014`](DECISIONS.md#014-audit-events-are-metadata-only-content-purge-targets-reviews-and-findings)). Outrider does not transmit stored LLM content to any third party other than the configured LLM provider at request time per [`DECISIONS.md#013`](DECISIONS.md#013-llmprivacy-contract-anthropic-egress-retention-zdr) + [`DECISIONS.md#015`](DECISIONS.md#015-zdr-is-an-operator-attestation-not-a-runtime-opt-in) egress rules. Logs stay metadata-only; LLM content lives in the database, not in log streams. Outrider does not currently support HIPAA-subject workloads. See [`DECISIONS.md#011`](DECISIONS.md#011-self-hosted-is-canonical-v1-saas-is-v15) for the deployment-model commitment, and [`DECISIONS.md#016`](DECISIONS.md#016-llm-exchanges-stored-locally-under-retention-logs-stay-metadata-only) for the storage-vs-logging surface split.

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

The script-style smoke tests above are also wired into pre-commit. The pytest suite covers the implemented modules:

```bash
.venv/bin/pytest tests/unit                    # fast, no DB
.venv/bin/pytest tests/integration             # needs the postgres-test container (see docs/testing.md)
.venv/bin/pytest tests/eval --is-eval          # --is-eval is mandatory (conftest fails-loud without it)
```

See `docs/testing.md` for the three-tier test strategy and the two-container Postgres model.

## Docs

- `docs/architecture.md` explains the review graph and subsystem layout.
- `docs/trust-boundaries.md` captures the rules that protect the security and replay story.
- `docs/invariants.md` is generated from tags in `docs/spec.md`; do not edit it by hand.
- `.claude/skills/outrider-navigator/SKILL.md` is the Claude Code skill that surfaces applicable invariants before Outrider code work.
- `DECISIONS.md` is the public record of architectural choices. Code and comments cite entries by slug (e.g., `# See DECISIONS.md#011-self-hosted-is-canonical-v1`).
