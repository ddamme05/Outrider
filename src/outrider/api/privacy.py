# See DECISIONS.md#013-llmprivacy-contract-anthropic-egress-retention-zdr
# See DECISIONS.md#015-zdr-is-an-operator-attestation-not-a-runtime-opt-in
# See DECISIONS.md#016-llm-exchanges-stored-locally-under-retention-logs-stay-metadata-only
"""Public privacy-disclosure surface (Arc B3).

Renders the MANDATED user-facing privacy statement — fixed by `DECISIONS.md#013`
point 6 → `#015` point 5 → `#016` point 6 — on a public, unauthenticated page
(`GET /privacy`): the GitHub App listing's privacy-policy URL target, readable while
a user is deciding whether to install, AND the dashboard footer target.

The statement is CONFIG-AWARE — it matches the deployment's real egress + attestation
so a public disclosure cannot overstate what a deployment actually does (`#056` host
selection + `#015` ZDR attestation):

  - Provider-NEUTRAL clauses (self-hosted egress shape, local storage + TTL/purge,
    metadata-only logs, HIPAA) ALWAYS render — they hold regardless of host.
  - ANTHROPIC-specific clauses (Anthropic retention terms + the ZDR clause) render
    ONLY when the configured host is Anthropic, and their wording flips on
    `ANTHROPIC_ZDR_ENABLED`, read through `resolve_zdr_attestation` — the SAME
    truthy/falsy/warn-once parser `AnthropicProvider` uses — so the rendered statement
    cannot drift from runtime attestation (`#015`).
  - A configured NON-Anthropic OpenAI-compatible host (`OUTRIDER_LLM_HOST`) SUPPRESSES
    every Anthropic-specific claim and renders that host's `HostPrivacy` provenance
    (`#056`) as the authoritative retention source.
  - An UNRECOGNIZED host NEVER falls back to the Anthropic disclosure — it renders the
    neutral clauses plus an explicit "host not recognized" notice (fail-loud, not
    fail-Anthropic). A running app crashes at startup on an unknown host
    (`build_graph` → `resolve_host_identity`), so this branch is defense-in-depth for
    demo mode / misconfiguration.

The mandated FACTS are canonical (mirrored in README.md:5); changing a mandated clause
is a supersession decision (`#016` point 6's own change rule), not an edit here. This
module RENDERS; it authors no privacy facts
(`memory/feedback_no_unverified_external_citations`).

No trust boundaries: a read-only render of existing canonical text + existing
`HostPrivacy` provenance + the shared ZDR parser. Mounted ALWAYS (including demo mode)
because it must be reachable before any install exists.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from outrider.llm.anthropic_provider import resolve_zdr_attestation
from outrider.llm.host_profiles import (
    ANTHROPIC_PROFILE_ID,
    HostPrivacy,
    resolve_host_profile,
)

# Provider-NEUTRAL clauses, as (heading, body). True regardless of the configured host,
# so they render for Anthropic, a GLM host, and the unrecognized-host case alike. Wording
# mirrors README.md:5 (single source of the FACTS; this is the rendered surface).
_NEUTRAL_CLAUSES: tuple[tuple[str, str], ...] = (
    (
        "What leaves your infrastructure",
        "Outrider is self-hosted: it runs in your own infrastructure. Code and PR "
        "text (changed-file contents, PR title and body, commit messages, branch "
        "names, author login, and extracted scope/evidence snippets) reach exactly "
        "one third party — the configured LLM provider. No secret material (GitHub "
        "App key, webhook secret, installation tokens) is ever sent.",
    ),
    (
        "Local storage of LLM content",
        "Outrider stores LLM request and response content in your local database "
        "under a configured retention TTL (default values in operator configuration; "
        "purged on installation.deleted along with reviews and findings, per "
        "DECISIONS.md#012 + #014). Outrider does not transmit stored LLM content to "
        "any third party other than the configured LLM provider at request time.",
    ),
    (
        "Logs stay metadata-only",
        "Structured logs carry only metadata (token counts, model, cost, latency, "
        "finish reason) — never prompt or completion text. LLM content lives in the "
        "database, not in log streams.",
    ),
    (
        "HIPAA",
        "Outrider does not currently support HIPAA-subject workloads. Do not install "
        "it on repositories containing PHI.",
    ),
)


def _anthropic_clauses(zdr_attested: bool) -> tuple[tuple[str, str], ...]:
    """The Anthropic-specific retention + ZDR clauses — rendered ONLY when the
    configured host is Anthropic. Wording flips on `zdr_attested` (resolved via
    `resolve_zdr_attestation`, the provider's own parser) so the disclosed retention
    posture matches runtime attestation (`#015`). Under ZDR the policy-violation
    exception (2y / 7y) still holds — surfaced explicitly, not dropped."""
    if zdr_attested:
        retention = (
            "Zero-data-retention is attested for this deployment "
            "(ANTHROPIC_ZDR_ENABLED=true): under Anthropic's ZDR terms, inputs and "
            "outputs are not retained. Content flagged for policy violations is still "
            "retained up to 2 years and classification scores up to 7 years even under "
            "ZDR; no data is used for training without permission."
        )
        zdr = (
            "This deployment has attested a zero-data-retention arrangement with "
            "Anthropic (ANTHROPIC_ZDR_ENABLED=true). Outrider surfaces the attestation "
            "to adjust its startup notice and this statement; it does not enforce ZDR "
            "on its own — the arrangement is between your organization and Anthropic. "
            "Policy-violation retention still applies even under ZDR."
        )
    else:
        retention = (
            "Under Anthropic's default terms, inputs and outputs are retained for 30 "
            "days; content flagged for policy violations is retained up to 2 years and "
            "classification scores up to 7 years; no data is used for training without "
            "permission. Zero-data-retention is not attested for this deployment "
            "(ANTHROPIC_ZDR_ENABLED is not set)."
        )
        zdr = (
            "If your organization has a zero-data-retention arrangement with Anthropic, "
            "set ANTHROPIC_ZDR_ENABLED=true. Outrider uses the flag only to adjust its "
            "startup notice and this statement — it does not enable ZDR on its own "
            "(contact Anthropic sales to arrange). Policy-violation retention still "
            "applies even under ZDR."
        )
    return (
        ("Provider retention (Anthropic)", retention),
        ("Zero-data-retention (ZDR)", zdr),
    )


@dataclass(frozen=True)
class PrivacyContext:
    """Resolved deployment privacy config that drives the render.

    `is_anthropic` and `host_privacy` are mutually exclusive. The three reachable
    shapes: Anthropic (`is_anthropic=True`, `host_privacy=None`); a recognized
    non-Anthropic host (`is_anthropic=False`, `host_privacy` set); an UNRECOGNIZED host
    (`is_anthropic=False`, `host_privacy=None`) → neutral clauses + notice, NEVER the
    Anthropic disclosure.
    """

    configured_host: str
    is_anthropic: bool
    zdr_attested: bool
    host_privacy: HostPrivacy | None


def resolve_privacy_context() -> PrivacyContext:
    """Resolve `OUTRIDER_LLM_HOST` (+ `ANTHROPIC_ZDR_ENABLED` on the Anthropic path)
    into the render context.

    Plain env reads (like `main._demo_mode_from_env`) so the page renders in demo mode
    without constructing the LLM provider. Host match is case-sensitive, mirroring
    `resolve_host_identity`. An unrecognized host does NOT fall back to Anthropic — the
    caller renders a fail-loud notice instead (a running app would already have crashed
    at startup on an unknown host).
    """
    host_id = os.environ.get("OUTRIDER_LLM_HOST", ANTHROPIC_PROFILE_ID).strip()
    if not host_id or host_id == ANTHROPIC_PROFILE_ID:
        return PrivacyContext(
            configured_host=ANTHROPIC_PROFILE_ID,
            is_anthropic=True,
            zdr_attested=resolve_zdr_attestation(None),
            host_privacy=None,
        )
    try:
        profile = resolve_host_profile(host_id)
    except ValueError:
        # Unknown host id: neutral clauses + explicit notice. NEVER Anthropic terms.
        return PrivacyContext(
            configured_host=host_id,
            is_anthropic=False,
            zdr_attested=False,
            host_privacy=None,
        )
    return PrivacyContext(
        configured_host=host_id,
        is_anthropic=False,
        zdr_attested=False,
        host_privacy=profile.privacy,
    )


def _clause_section(heading: str, body: str) -> str:
    """One `<section>` for a (heading, body) clause; both escaped."""
    return (
        f"    <section>\n      <h2>{html.escape(heading)}</h2>\n"
        f"      <p>{html.escape(body)}</p>\n    </section>"
    )


def _host_privacy_section(privacy: HostPrivacy) -> str:
    """Render a recognized non-Anthropic host's provenance block (developer-defined
    data, escaped defensively) — the authoritative retention source for that host."""
    rows = (
        ("Egress host", privacy.egress_host),
        ("Model origin", privacy.model_origin),
        ("Directly hosted", "yes" if privacy.direct_hosted else "no"),
        ("Trains on inputs", "yes" if privacy.trains_on_inputs else "no"),
        ("Retention", privacy.retention),
        ("Source", privacy.source_url),
        ("Verified", privacy.verified_date),
    )
    items = "\n".join(
        f"      <dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>" for label, value in rows
    )
    return (
        '    <section class="host">\n'
        "      <h2>Configured LLM host</h2>\n"
        "      <p>This deployment is configured for a non-Anthropic OpenAI-compatible "
        "host; Anthropic's terms above do not apply. Its published privacy posture:</p>\n"
        "      <dl>\n"
        f"{items}\n"
        "      </dl>\n"
        "    </section>"
    )


def _unrecognized_host_section(host_id: str) -> str:
    """Rendered when `OUTRIDER_LLM_HOST` names a host Outrider does not recognize.
    Provider-specific retention cannot be shown — and Anthropic's terms are NOT
    substituted (fail-loud). A running deployment fails startup on an unknown host;
    this notice is for pre-start / misconfiguration."""
    return (
        '    <section class="host">\n'
        "      <h2>Configured LLM host</h2>\n"
        f"      <p>This deployment is configured for LLM host "
        f"<code>{html.escape(host_id)}</code>, which Outrider does not recognize. "
        "Provider-specific retention terms cannot be shown for it, and Anthropic's "
        "terms do not apply — resolve OUTRIDER_LLM_HOST to a supported host to restore "
        "the provider disclosure.</p>\n"
        "    </section>"
    )


def render_privacy_html(ctx: PrivacyContext) -> str:
    """Render the full privacy page from `ctx`: the neutral clauses always, plus exactly
    one provider surface — the Anthropic clauses (host=anthropic, ZDR-aware), the
    configured host's `HostPrivacy` block, or an unrecognized-host notice."""
    clause_pairs = list(_NEUTRAL_CLAUSES)
    if ctx.is_anthropic:
        clause_pairs.extend(_anthropic_clauses(ctx.zdr_attested))
    clauses = "\n".join(_clause_section(heading, body) for heading, body in clause_pairs)

    if ctx.host_privacy is not None:
        provider_block = f"\n{_host_privacy_section(ctx.host_privacy)}"
    elif not ctx.is_anthropic:
        provider_block = f"\n{_unrecognized_host_section(ctx.configured_host)}"
    else:
        provider_block = ""

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "  <title>Outrider — Privacy &amp; data handling</title>\n"
        "  <style>\n"
        "    body { max-width: 46rem; margin: 3rem auto; padding: 0 1.25rem;\n"
        "      font: 16px/1.6 system-ui, sans-serif; color: #1a1a1a; }\n"
        "    h1 { font-size: 1.6rem; } h2 { font-size: 1.1rem; margin-top: 1.75rem; }\n"
        "    dl { display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; }\n"
        "    dt { font-weight: 600; } dd { margin: 0; }\n"
        "    .lead { color: #444; }\n"
        "    @media (prefers-color-scheme: dark) {\n"
        "      body { background: #111; color: #e8e8e8; } .lead { color: #b8b8b8; } }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <main>\n"
        "    <h1>Privacy &amp; data handling</h1>\n"
        '    <p class="lead">Outrider is self-hosted: your code and data stay in your '
        "infrastructure, and the only egress is the LLM call described below.</p>\n"
        f"{clauses}{provider_block}\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


router = APIRouter()


@router.get("/privacy", response_class=HTMLResponse)
async def privacy_page() -> HTMLResponse:
    """PUBLIC, unauthenticated privacy disclosure (the App-listing privacy URL + the
    dashboard footer target). Renders the mandated #013/#015/#016 statement, config-aware
    per `OUTRIDER_LLM_HOST` + `ANTHROPIC_ZDR_ENABLED`."""
    return HTMLResponse(render_privacy_html(resolve_privacy_context()))
