"""Q5 — FastAPI receiver route end-to-end.

Drive ``receiver.app`` through ``httpx.AsyncClient`` + ASGI transport.
Assert:

1. A request with a **valid** signature + known event → 202 with the
   shape-summary dict populated from the parsed event.
2. A request with a **wrong** signature → 401.
3. A request with **no** signature header → 401.
4. A request with an **unknown event name** → 400.
5. The ACK latency is well under a second — matching spec §6.3's "webhook
   must ACK within 10 seconds, ideally well under a second." Anything
   approaching 10s on this code path is a sign dispatcher work leaked into
   the request handler.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx
from githubkit.webhooks import sign as gh_sign

FIXTURES = Path(__file__).parent.parent / "fixtures"
SECRET = "outrider-spike-webhook-secret"


async def run_q5() -> None:
    os.environ["OUTRIDER_SPIKE_WEBHOOK_SECRET"] = SECRET
    # Import after env var is set — receiver reads it lazily at request
    # time, but importing under a hermetic env keeps the dependency explicit.
    from receiver import app  # type: ignore[import-not-found]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://spike"
    ) as client:
        body = (FIXTURES / "sample_pull_request_opened.json").read_bytes()
        sig = gh_sign(SECRET, body, method="sha256")

        # Warmup: githubkit.webhooks.parse lazy-loads the full tagged-union
        # model on first call (~1s). Production pays this once per worker
        # process; the spike's contract is the warm path. See NOTES.md
        # "Cold-start parser import" finding for the startup-hook mitigation.
        warmup = await client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "warmup",
                "Content-Type": "application/json",
            },
        )
        assert warmup.status_code == 202, (
            f"Q5 FAIL: warmup request → {warmup.status_code}"
        )

        # Case 1: good signature (warm path).
        t0 = time.perf_counter()
        resp = await client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "test-delivery-001",
                "Content-Type": "application/json",
            },
        )
        elapsed_good = time.perf_counter() - t0

        assert resp.status_code == 202, (
            f"Q5 FAIL: good-signature request → {resp.status_code}, "
            f"body={resp.text!r}"
        )
        payload = resp.json()
        assert payload["event"] == "pull_request"
        assert payload["action"] == "opened"
        assert payload["delivery_id"] == "test-delivery-001"
        assert isinstance(payload["correlation_id"], str) and payload["correlation_id"]
        assert payload["pr_number"] > 0
        assert len(payload["pr_head_sha"]) == 40
        assert "/" in payload["repo_full_name"]
        assert isinstance(payload["installation_id"], int)

        # ACK latency on the warm path is ~10ms observed. 100ms is a
        # generous ceiling that still flags dispatcher leakage (anything
        # doing non-trivial work on the request path blows past this).
        assert elapsed_good < 0.1, (
            f"Q5 FAIL: warm good-signature handler took {elapsed_good:.3f}s "
            "— cold parser was already warmed; something else on the "
            "request path is slow."
        )

        # Case 2: wrong signature (same length, different content).
        wrong_sig = "sha256=" + ("0" * 64)
        resp = await client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-Hub-Signature-256": wrong_sig,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 401, (
            f"Q5 FAIL: wrong-signature request → {resp.status_code}"
        )

        # Case 3: no signature header at all.
        resp = await client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 401, (
            f"Q5 FAIL: no-signature request → {resp.status_code} "
            "(should 401 before any parse work)"
        )

        # Case 4: unknown event name (signature valid; event name invalid).
        resp = await client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "not_a_real_event",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 400, (
            f"Q5 FAIL: unknown-event request → {resp.status_code} "
            "(should 400, not 500 — parse errors are client errors here)"
        )

        # Case 5: missing X-GitHub-Event header (signature valid, header
        # absent). Receiver rejects before parse work.
        resp = await client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 400, (
            f"Q5 FAIL: missing X-GitHub-Event → {resp.status_code} "
            "(should 400 before any parse work)"
        )

        # Case 6: installation.created event drives the installation branch
        # of the receiver. Q4 parsed the payload; Q5 proves the route wires
        # parsing → summary correctly for the second event type we care about.
        inst_body = (FIXTURES / "sample_installation_created.json").read_bytes()
        inst_sig = gh_sign(SECRET, inst_body, method="sha256")
        resp = await client.post(
            "/webhooks/github",
            content=inst_body,
            headers={
                "X-Hub-Signature-256": inst_sig,
                "X-GitHub-Event": "installation",
                "X-GitHub-Delivery": "test-inst-001",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 202, (
            f"Q5 FAIL: installation.created → {resp.status_code}, {resp.text!r}"
        )
        inst_payload = resp.json()
        assert inst_payload["event"] == "installation"
        assert inst_payload["action"] == "created"
        assert isinstance(inst_payload["installation_id"], int)
        assert isinstance(inst_payload["app_slug"], str) and inst_payload["app_slug"]
        assert "account_login" in inst_payload

    print(
        f"Q5 OK: /webhooks/github — good=202 ({elapsed_good*1000:.1f} ms), "
        "wrong-sig=401, no-sig=401, bad-event=400, no-event-header=400, "
        f"installation.created=202 (installation_id={inst_payload['installation_id']})."
    )


def main() -> None:
    asyncio.run(run_q5())


if __name__ == "__main__":
    main()
