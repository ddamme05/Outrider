# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/ddamme05/Outrider/security/advisories/new)
for this repository. Do not open a public issue for anything you believe is exploitable.

Include what you can: the affected component, steps to reproduce the problem, and its impact.
You can expect an acknowledgment within a week. One person maintains this project, so
assessing reports and making fixes depend on available time, with no guaranteed deadline.

## Scope

In scope: anything under `src/outrider/`, the dashboard under `dashboard/`, database
migrations under `db/`, deployment files under `deploy/`, automated workflows under `.github/`,
packaging and dependency configuration (`pyproject.toml`, `uv.lock`), and the operations
scripts under `scripts/`.

Out of scope:

- The files under `scripts/demo_fixtures/`. They deliberately contain vulnerabilities
  for testing reviews and are never imported by the application.
- Reports whose entire impact is that an operator published a documented secret (for example,
  posting the dashboard admin key somewhere public). If a product weakness makes such a
  mistake easier to cause or worse in effect, that weakness is in scope.
- Vulnerabilities in dependencies with no Outrider-specific exposure. Report those to the
  dependency's maintainers, though a heads-up is welcome if Outrider's use makes one exploitable.

## Supported versions

Outrider is pre-1.0. Only the current `main` branch receives fixes.

## What Outrider itself sends and stores

The README's [Security and privacy](README.md#security-and-privacy) section documents the
data flow: what reaches the model provider, what is stored locally, and how long it is kept.
Outrider has not been independently audited or penetration tested.
