"""B3 — the public /privacy page renders the mandated #013/#015/#016 statement,
CONFIG-AWARE so the disclosure matches runtime egress + attestation.

Drop-protection note: the guard against silently deleting a mandated disclosure is the
hand-maintained fact lists below (`_NEUTRAL_FACTS`, `_ANTHROPIC_DEFAULT_FACTS`,
`_ANTHROPIC_ZDR_FACTS`), matched against the RENDERED page per matrix cell. They are a
SEPARATE source from the clause tuples the renderer reads, so deleting or rewording-away
a clause fails a fact assertion (unlike a test that loops the same tuple it renders).
`_README_FACTS` additionally binds the mandated facts to README.md:5 so the two canonical
copies cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

import outrider.api.privacy as privacy_mod
import outrider.llm.anthropic_provider as anthropic_mod
from outrider.api.privacy import (
    render_privacy_html,
    resolve_privacy_context,
)
from outrider.llm.host_profiles import FIREWORKS_PROFILE
from outrider.main import create_app

if TYPE_CHECKING:
    import pytest

# --- Marker strings that are Anthropic-SPECIFIC. For a non-Anthropic / unrecognized
# host, NONE of these may appear (constraint: suppress every Anthropic-specific claim). ---
_ANTHROPIC_MARKERS = (
    "Under Anthropic's default terms",
    "Provider retention (Anthropic)",
    "Zero-data-retention (ZDR)",
    "ANTHROPIC_ZDR_ENABLED",
)

# --- Provider-neutral facts: MUST appear for every host (Anthropic, GLM host, unknown). ---
_NEUTRAL_FACTS = (
    "local database",
    "installation.deleted",
    "never prompt or completion text",  # strong pin (was the too-generic "metadata")
    "does not currently support HIPAA-subject workloads",
)

# --- Anthropic + ZDR NOT attested: the default-terms retention facts. ---
_ANTHROPIC_DEFAULT_FACTS = (
    "retained for 30 days",
    "2 years",
    "7 years",
    "used for training without permission",
    "ANTHROPIC_ZDR_ENABLED=true",
    "does not enable ZDR on its own",
    "not attested for this deployment",
)

# --- Anthropic + ZDR attested: retention posture flips; policy-violation exception stays. ---
_ANTHROPIC_ZDR_FACTS = (
    "Zero-data-retention is attested for this deployment",
    "not retained",
    "2 years",  # policy-violation exception survives ZDR
    "7 years",
    "Policy-violation retention still applies even under ZDR",
    "used for training without permission",
)

# --- Facts that must appear VERBATIM in BOTH the default-Anthropic page and README.md:5,
# so the two canonical copies of the mandated statement cannot drift. ---
_README_FACTS = (
    "30 days",
    "2 years",
    "7 years",
    "used for training without permission",
    "ANTHROPIC_ZDR_ENABLED=true",
    "does not enable ZDR on its own",
    "installation.deleted",
    "does not currently support HIPAA-subject workloads",
    "local database",
)


def _anthropic_page(monkeypatch: pytest.MonkeyPatch, *, zdr: bool) -> str:
    monkeypatch.delenv("OUTRIDER_LLM_HOST", raising=False)
    if zdr:
        monkeypatch.setenv("ANTHROPIC_ZDR_ENABLED", "true")
    else:
        monkeypatch.delenv("ANTHROPIC_ZDR_ENABLED", raising=False)
    return render_privacy_html(resolve_privacy_context())


# ---------------------------------------------------------------------------
# Neutral clauses render for every host shape.
# ---------------------------------------------------------------------------
def test_neutral_facts_render_for_every_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider-neutral facts appear regardless of configured host."""
    monkeypatch.delenv("ANTHROPIC_ZDR_ENABLED", raising=False)
    for host in ("anthropic", "fireworks", "does-not-exist"):
        monkeypatch.setenv("OUTRIDER_LLM_HOST", host)
        page = render_privacy_html(resolve_privacy_context())
        for fact in _NEUTRAL_FACTS:
            assert fact in page, f"neutral fact {fact!r} missing for host={host!r}"


# ---------------------------------------------------------------------------
# Anthropic host, ZDR off — default retention terms.
# ---------------------------------------------------------------------------
def test_anthropic_default_retention_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _anthropic_page(monkeypatch, zdr=False)
    for fact in (*_NEUTRAL_FACTS, *_ANTHROPIC_DEFAULT_FACTS):
        assert fact in page, f"default-Anthropic fact missing: {fact!r}"


# ---------------------------------------------------------------------------
# Constraint #7: ZDR flag (read through the provider's own parser) flips the wording.
# ---------------------------------------------------------------------------
def test_anthropic_zdr_attested_flips_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    """ANTHROPIC_ZDR_ENABLED=true changes the rendered retention posture — proving
    /privacy actually reads the flag (previously it did not)."""
    page = _anthropic_page(monkeypatch, zdr=True)
    for fact in _ANTHROPIC_ZDR_FACTS:
        assert fact in page, f"ZDR-attested fact missing: {fact!r}"
    # The default 30-day retention claim must NOT appear once ZDR is attested.
    assert "retained for 30 days" not in page, "30-day retention shown despite ZDR attested"


def test_privacy_reuses_provider_zdr_resolver() -> None:
    """Constraint #7 (structural): /privacy parses ZDR through the SAME function the
    provider uses — not a reimplementation that could drift."""
    assert privacy_mod.resolve_zdr_attestation is anthropic_mod.resolve_zdr_attestation


# ---------------------------------------------------------------------------
# Non-Anthropic recognized host: suppress Anthropic claims, show that host's provenance.
# ---------------------------------------------------------------------------
def test_non_anthropic_host_suppresses_anthropic_and_shows_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTRIDER_LLM_HOST", FIREWORKS_PROFILE.host_id)
    monkeypatch.delenv("ANTHROPIC_ZDR_ENABLED", raising=False)
    page = render_privacy_html(resolve_privacy_context())
    # That host's provenance is the authoritative retention source.
    assert "Configured LLM host" in page
    assert FIREWORKS_PROFILE.privacy.egress_host in page
    assert FIREWORKS_PROFILE.privacy.source_url in page
    # Every Anthropic-specific claim is suppressed.
    for marker in _ANTHROPIC_MARKERS:
        assert marker not in page, f"Anthropic marker {marker!r} leaked onto non-Anthropic page"


# ---------------------------------------------------------------------------
# Constraint #6: an unrecognized host NEVER falls back to the Anthropic disclosure.
# ---------------------------------------------------------------------------
def test_unrecognized_host_never_falls_back_to_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTRIDER_LLM_HOST", "totally-made-up-host")
    monkeypatch.delenv("ANTHROPIC_ZDR_ENABLED", raising=False)
    page = render_privacy_html(resolve_privacy_context())
    assert "does not recognize" in page
    assert "totally-made-up-host" in page
    for marker in _ANTHROPIC_MARKERS:
        assert marker not in page, f"unrecognized host fell back to Anthropic ({marker!r})"
    # No HostPrivacy provenance block (there is no profile for an unknown host).
    assert "Egress host" not in page
    # Neutral clauses still hold.
    for fact in _NEUTRAL_FACTS:
        assert fact in page

    # A present-but-blank host is likewise unrecognized (mirrors runtime), never Anthropic.
    monkeypatch.setenv("OUTRIDER_LLM_HOST", "   ")
    blank_page = render_privacy_html(resolve_privacy_context())
    assert "does not recognize" in blank_page
    assert "(blank)" in blank_page
    for marker in _ANTHROPIC_MARKERS:
        assert marker not in blank_page, f"blank host fell back to Anthropic ({marker!r})"


# ---------------------------------------------------------------------------
# Resolver context matrix.
# ---------------------------------------------------------------------------
def test_resolve_privacy_context_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_ZDR_ENABLED", raising=False)

    monkeypatch.delenv("OUTRIDER_LLM_HOST", raising=False)
    unset = resolve_privacy_context()
    assert unset.is_anthropic and unset.host_privacy is None

    # Present-but-blank is NOT Anthropic — it mirrors runtime, which passes "" through and
    # rejects it as unknown (lifespan.py:621). Only an UNSET var defaults to Anthropic.
    monkeypatch.setenv("OUTRIDER_LLM_HOST", "  ")
    blank = resolve_privacy_context()
    assert not blank.is_anthropic and blank.host_privacy is None

    monkeypatch.setenv("OUTRIDER_LLM_HOST", "anthropic")
    assert resolve_privacy_context().is_anthropic

    monkeypatch.setenv("OUTRIDER_LLM_HOST", FIREWORKS_PROFILE.host_id)
    fw = resolve_privacy_context()
    assert not fw.is_anthropic
    assert fw.host_privacy == FIREWORKS_PROFILE.privacy

    monkeypatch.setenv("OUTRIDER_LLM_HOST", "no-such-host")
    unknown = resolve_privacy_context()
    assert not unknown.is_anthropic
    assert unknown.host_privacy is None  # #6: not Anthropic, not a profile → notice branch


def test_resolver_reads_zdr_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTRIDER_LLM_HOST", raising=False)
    monkeypatch.setenv("ANTHROPIC_ZDR_ENABLED", "true")
    assert resolve_privacy_context().zdr_attested is True
    monkeypatch.setenv("ANTHROPIC_ZDR_ENABLED", "false")
    assert resolve_privacy_context().zdr_attested is False
    monkeypatch.delenv("ANTHROPIC_ZDR_ENABLED", raising=False)
    assert resolve_privacy_context().zdr_attested is False


# ---------------------------------------------------------------------------
# The mandated facts stay mirrored between the page and README.md:5 (drift guard).
# ---------------------------------------------------------------------------
def test_mandated_facts_mirrored_in_readme(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each mandated fact appears in BOTH the rendered default-Anthropic page AND
    README.md — so editing one canonical copy without the other fails here."""
    page = _anthropic_page(monkeypatch, zdr=False)
    readme = Path("README.md").read_text(encoding="utf-8")
    for fact in _README_FACTS:
        assert fact in page, f"mandated fact missing from page: {fact!r}"
        assert fact in readme, f"mandated fact missing from README.md: {fact!r}"


# ---------------------------------------------------------------------------
# Route is public + unauthenticated in both modes.
# ---------------------------------------------------------------------------
def test_privacy_route_is_public_and_unauthenticated() -> None:
    """`GET /privacy` returns 200 HTML with no auth header, in BOTH modes."""
    for demo in (True, False):
        client = TestClient(create_app(demo_mode=demo))
        resp = client.get("/privacy")
        assert resp.status_code == 200, f"demo_mode={demo}"
        assert "text/html" in resp.headers["content-type"]
        assert "Privacy" in resp.text
        assert "does not currently support HIPAA-subject workloads" in resp.text
