# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/ddamme05/Outrider/security/advisories/new)
for this repository. Do not open a public issue for anything you believe is exploitable.

Include what you can: the affected component, a reproduction path, and the impact you see.
You can expect an acknowledgment within a week. This is a solo-maintained project, so triage
and fixes are best-effort rather than an SLA.

## Scope

In scope: anything under `src/outrider/`, the dashboard under `dashboard/`, database
migrations under `db/`, the deployment stack under `deploy/`, CI workflows under `.github/`,
packaging and dependency configuration (`pyproject.toml`, `uv.lock`), and the operational
scripts under `scripts/`.

Out of scope:

- The files under `scripts/demo_fixtures/`. They are deliberately vulnerable review targets
  for exercising the product and are never imported by the application.
- Reports whose entire impact is that an operator published a documented secret (for example,
  posting the dashboard admin key somewhere public). If a product weakness makes such a
  mistake easier to cause or worse in effect, that weakness is in scope.
- Vulnerabilities in upstream dependencies with no Outrider-specific exposure. Report those
  upstream, though a heads-up is welcome if Outrider's usage makes one exploitable.

## Supported versions

Outrider is pre-1.0. Only the current `main` branch receives fixes.

## What Outrider itself sends and stores

The README's [Security and privacy](README.md#security-and-privacy) section documents the
data flow: what reaches the LLM provider, what is stored locally, and the retention model.
Outrider has not been independently audited or penetration tested.
