# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/ddamme05/Outrider/security/advisories/new)
for this repository. Do not open a public issue for anything you believe is exploitable.

Include what you can: the affected component, a reproduction path, and the impact you see.
You can expect an acknowledgment within a week. This is a solo-maintained project, so triage
and fixes are best-effort rather than an SLA.

## Scope

In scope: anything under `src/outrider/`, the deployment stack under `deploy/`, and the
dashboard under `dashboard/`.

Out of scope:

- The files under `scripts/demo_fixtures/`. They are deliberately vulnerable review targets
  for exercising the product and are never imported by the application.
- Findings that require the operator to have already misconfigured a documented secret
  (for example, publishing the dashboard admin key).
- Vulnerabilities in upstream dependencies with no Outrider-specific exposure. Report those
  upstream, though a heads-up is welcome if Outrider's usage makes one exploitable.

## Supported versions

Outrider is pre-1.0. Only the current `main` branch receives fixes.

## What Outrider itself sends and stores

The README's [Security and privacy](README.md#security-and-privacy) section documents the
data flow: what reaches the LLM provider, what is stored locally, and the retention model.
Outrider has not been independently audited or penetration tested.
