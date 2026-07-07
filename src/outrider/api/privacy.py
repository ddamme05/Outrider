# See DECISIONS.md#013-llmprivacy-contract-anthropic-egress-retention-zdr
# See DECISIONS.md#015-zdr-is-an-operator-attestation-not-a-runtime-opt-in
# See DECISIONS.md#016-llm-exchanges-stored-locally-under-retention-logs-stay-metadata-only
"""Public privacy-disclosure surface (Arc B3).

Renders the MANDATED user-facing privacy statement — the one fixed by
`DECISIONS.md#013` point 6 → `#015` point 5 → `#016` point 6 — on a public,
unauthenticated page (`GET /privacy`). This is the consent surface: it is the
target of the GitHub App listing's privacy-policy URL, readable while a user is
deciding whether to install, AND the destination of the dashboard footer link.

The statement is CANONICAL, not free text: the clauses below mirror the same
facts the README ships (per #015 point 5 / #016 point 6). They may be branded /
wrapped but not reworded away — changing a mandated clause is a supersession
decision (#016 point 6's own change rule), not an edit here. This module
RENDERS; it authors no privacy facts (`memory/feedback_no_unverified_external_citations`).

Host-qualification (#056): the fixed clauses describe the V1 default host
(Anthropic). When the deployment is configured for a non-Anthropic
OpenAI-compatible host (`OUTRIDER_LLM_HOST`), the page additionally shows that
host's `HostPrivacy` provenance (egress host, retention, training stance, source
+ verified date) so the disclosure reflects where code actually egresses.

No trust boundaries: this is a read-only render of existing canonical text +
existing `HostPrivacy` provenance. It is mounted ALWAYS (including demo mode)
because it must be reachable before any install exists.
"""

from __future__ import annotations

import html
import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from outrider.llm.host_profiles import (
    ANTHROPIC_PROFILE_ID,
    HostPrivacy,
    resolve_host_profile,
)

# The mandated statement, as (heading, body) clauses. Each clause is one of the
# disclosures #013/#015/#016 require; the unit test asserts every clause is
# present so a future edit cannot silently drop one. Wording mirrors README's
# canonical statement (single source of the FACTS; this is the rendered surface).
PRIVACY_CLAUSES: tuple[tuple[str, str], ...] = (
    (
        "What leaves your infrastructure",
        "Outrider is self-hosted: it runs in your own infrastructure. Code and PR "
        "text (changed-file contents, PR title and body, commit messages, branch "
        "names, author login, and extracted scope/evidence snippets) reach exactly "
        "one third party — the configured LLM provider (Anthropic in V1). No secret "
        "material (GitHub App key, webhook secret, installation tokens) is ever sent.",
    ),
    (
        "Provider retention",
        "Under Anthropic's default terms, inputs and outputs are retained for 30 "
        "days; content flagged for policy violations is retained up to 2 years and "
        "classification scores up to 7 years; no data is used for training without "
        "permission.",
    ),
    (
        "Zero-data-retention (ZDR)",
        "If your organization has a zero-data-retention arrangement with Anthropic, "
        "set ANTHROPIC_ZDR_ENABLED=true. Outrider uses the flag only to adjust its "
        "startup notice and this statement — it does not enable ZDR on its own "
        "(contact Anthropic sales to arrange). Policy-violation retention still "
        "applies even under ZDR.",
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


def resolve_configured_host_privacy() -> HostPrivacy | None:
    """Return the configured host's `HostPrivacy`, or None for the Anthropic default.

    Reads `OUTRIDER_LLM_HOST` directly (a plain env read, like `main._demo_mode_from_env`)
    so the privacy page never depends on the LLM provider being constructed — it must
    render in demo mode too. Anthropic (the default) has no `HostProfile` (it is the
    native SDK path), so it returns None and only the fixed clauses render. An unknown
    host id also returns None (the fixed clauses are the safe default) rather than
    raising on a public page.
    """
    host_id = os.environ.get("OUTRIDER_LLM_HOST", ANTHROPIC_PROFILE_ID).strip()
    if host_id == ANTHROPIC_PROFILE_ID or not host_id:
        return None
    try:
        return resolve_host_profile(host_id).privacy
    except ValueError:
        # `resolve_host_profile` raises ValueError on an unknown host id; on a
        # PUBLIC page fall back to the fixed Anthropic clauses rather than 500.
        return None


def _host_privacy_section(privacy: HostPrivacy) -> str:
    """Render the non-Anthropic host's provenance block (developer-defined data,
    escaped defensively)."""
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
        "host; its published privacy posture:</p>\n"
        "      <dl>\n"
        f"{items}\n"
        "      </dl>\n"
        "    </section>"
    )


def render_privacy_html(host_privacy: HostPrivacy | None = None) -> str:
    """Render the full privacy page from the mandated clauses (+ optional host block)."""
    clauses = "\n".join(
        f"    <section>\n      <h2>{html.escape(heading)}</h2>\n"
        f"      <p>{html.escape(body)}</p>\n    </section>"
        for heading, body in PRIVACY_CLAUSES
    )
    host_block = f"\n{_host_privacy_section(host_privacy)}" if host_privacy is not None else ""
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
        f"{clauses}{host_block}\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


router = APIRouter()


@router.get("/privacy", response_class=HTMLResponse)
async def privacy_page() -> HTMLResponse:
    """PUBLIC, unauthenticated privacy disclosure (the App-listing privacy URL +
    the dashboard footer target). Renders the mandated #013/#015/#016 statement,
    host-qualified for a non-Anthropic configured host."""
    return HTMLResponse(render_privacy_html(resolve_configured_host_privacy()))
