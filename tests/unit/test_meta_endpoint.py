"""GET /api/meta returns the deployment-shape `demo_mode` flag.

The demo allowlist test pins that the ROUTE exists; this pins its VALUE in both
modes — the flag drives the SPA's read-only banner + disabled HITL controls, so
a regression that hardcoded it would silently disable the whole affordance chain
while every route-presence test still passed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from outrider.main import create_app


def test_meta_reports_demo_mode_true_in_demo_app() -> None:
    client = TestClient(create_app(demo_mode=True))
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    assert resp.json() == {"demo_mode": True}


def test_meta_reports_demo_mode_false_in_production_app() -> None:
    client = TestClient(create_app(demo_mode=False))
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    assert resp.json() == {"demo_mode": False}


def test_meta_needs_no_auth() -> None:
    # Unauthenticated by design (same posture as /health): the banner must render
    # before any token is entered. No Authorization header here.
    client = TestClient(create_app(demo_mode=True))
    assert client.get("/api/meta").status_code == 200
