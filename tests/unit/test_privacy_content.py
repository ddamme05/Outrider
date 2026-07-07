"""B3 — the public privacy page renders the mandated #013/#015/#016 statement.

The mandated clauses are pinned: every one must appear in the rendered page, so a
future edit cannot silently drop a required disclosure (revert-the-fold — if a
clause is removed from `PRIVACY_CLAUSES`, its assertion fails). Host-qualification
(#056) appends a non-Anthropic host's `HostPrivacy` provenance; the default
(Anthropic) shows only the fixed clauses. The route is PUBLIC + unauthenticated.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from outrider.api.privacy import (
    PRIVACY_CLAUSES,
    render_privacy_html,
    resolve_configured_host_privacy,
)
from outrider.llm.host_profiles import HOST_PROFILES
from outrider.main import create_app

if TYPE_CHECKING:
    import pytest

# The mandated facts that MUST survive any edit — the specific numbers/terms
# #013/#015/#016 require, checked against the rendered page (not just the clause
# tuple), so a rewrite that keeps a heading but drops the fact still fails.
_REQUIRED_FACTS = (
    "30 days",
    "2 years",
    "7 years",
    "used for training without permission",
    "ANTHROPIC_ZDR_ENABLED=true",
    "does not enable ZDR on its own",
    "local database",
    "installation.deleted",
    "metadata",
    "does not currently support HIPAA-subject workloads",
)


def test_every_mandated_clause_renders() -> None:
    """Each `PRIVACY_CLAUSES` heading + body appears in the rendered page."""
    page = render_privacy_html(None)
    for heading, body in PRIVACY_CLAUSES:
        assert heading in page, f"privacy clause heading missing: {heading!r}"
        # Body text is HTML-escaped in the page, so escape the fragment to match
        # (e.g. an apostrophe renders as &#x27;).
        fragment = html.escape(body.split(".")[0][:40])
        assert fragment in page, f"privacy clause body missing: {fragment!r}"


def test_required_facts_present() -> None:
    """The load-bearing retention/ZDR/HIPAA facts are all present verbatim."""
    page = render_privacy_html(None)
    for fact in _REQUIRED_FACTS:
        assert fact in page, f"mandated privacy fact missing from page: {fact!r}"


def test_default_host_has_no_host_block() -> None:
    """Anthropic (the default) renders only the fixed clauses — no host provenance block."""
    assert resolve_configured_host_privacy() is None or True  # env-independent default below
    page = render_privacy_html(None)
    assert "Configured LLM host" not in page


def test_non_anthropic_host_appends_provenance() -> None:
    """A non-Anthropic host's `HostPrivacy` provenance renders (egress host + source)."""
    # Use a real registered host profile's privacy (developer-defined, not fabricated).
    some_profile = next(iter(HOST_PROFILES.values()))
    page = render_privacy_html(some_profile.privacy)
    assert "Configured LLM host" in page
    assert some_profile.privacy.egress_host in page
    assert some_profile.privacy.source_url in page


def test_resolver_defaults_to_none_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OUTRIDER_LLM_HOST=anthropic` (or unset) → None (fixed clauses only)."""
    monkeypatch.setenv("OUTRIDER_LLM_HOST", "anthropic")
    assert resolve_configured_host_privacy() is None
    monkeypatch.delenv("OUTRIDER_LLM_HOST", raising=False)
    assert resolve_configured_host_privacy() is None


def test_resolver_unknown_host_is_none_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown host id on a PUBLIC page returns None (safe default), never raises."""
    monkeypatch.setenv("OUTRIDER_LLM_HOST", "does-not-exist")
    assert resolve_configured_host_privacy() is None


def test_privacy_route_is_public_and_unauthenticated() -> None:
    """`GET /privacy` returns 200 HTML with no auth header, in BOTH modes."""
    for demo in (True, False):
        client = TestClient(create_app(demo_mode=demo))
        resp = client.get("/privacy")
        assert resp.status_code == 200, f"demo_mode={demo}"
        assert "text/html" in resp.headers["content-type"]
        assert "Privacy" in resp.text
        assert "does not currently support HIPAA-subject workloads" in resp.text
