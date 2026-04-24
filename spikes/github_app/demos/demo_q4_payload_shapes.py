"""Q4 — webhook payload shapes.

For each of the three events V1 cares about, parse the fixture via
``githubkit.webhooks.parse`` and assert the fields the future PRContext
builder will need are populated and correctly typed.

If this demo fails, it means either the octokit sample has drifted from
GitHub's current schema OR githubkit's pinned schema has drifted. Either
way, ``api/webhooks/schemas.py`` needs to be revisited. NOTES.md will carry
the exact field list that was confirmed on the date the spike ran.
"""

from __future__ import annotations

from pathlib import Path

from githubkit.webhooks import parse

FIXTURES = Path(__file__).parent.parent / "fixtures"


def q4_pull_request_opened() -> None:
    body = (FIXTURES / "sample_pull_request_opened.json").read_bytes()
    event = parse("pull_request", body)

    assert event.action == "opened", f"expected opened, got {event.action!r}"

    pr = event.pull_request
    repo = event.repository
    installation = event.installation

    # The fields PRContext needs (spec §4.1.1 intake).
    assert isinstance(pr.number, int) and pr.number > 0
    assert isinstance(pr.head.sha, str) and len(pr.head.sha) == 40
    assert isinstance(pr.base.sha, str) and len(pr.base.sha) == 40
    assert isinstance(repo.id, int)
    assert isinstance(repo.full_name, str) and "/" in repo.full_name
    assert isinstance(pr.diff_url, str) and pr.diff_url.startswith("https://")
    assert isinstance(pr.patch_url, str) and pr.patch_url.startswith("https://")

    # installation may be None on webhooks not delivered via App; for our
    # App-only pipeline it must be present.
    assert installation is not None, (
        "Q4 FAIL: pull_request event with no installation reference"
    )
    assert isinstance(installation.id, int) and installation.id > 0

    print(
        f"Q4 pull_request.opened OK: PR #{pr.number} on {repo.full_name}, "
        f"head={pr.head.sha[:7]}, installation={installation.id}."
    )


def q4_pull_request_synchronize() -> None:
    body = (FIXTURES / "sample_pull_request_synchronize.json").read_bytes()
    event = parse("pull_request", body)

    assert event.action == "synchronize", (
        f"expected synchronize, got {event.action!r}"
    )
    # synchronize carries 'before' and 'after' at the event top level —
    # useful for knowing which commits are new in this push.
    assert event.before is not None, "synchronize payload missing 'before'"
    assert event.after is not None, "synchronize payload missing 'after'"
    assert event.before != event.after, (
        "Q4 FAIL: synchronize before/after SHAs are identical"
    )
    # The new head SHA should equal the event's 'after'.
    assert event.pull_request.head.sha == event.after, (
        f"Q4 FAIL: pr.head.sha={event.pull_request.head.sha} differs from "
        f"event.after={event.after}. PRContext update logic needs to use "
        "pr.head.sha as authoritative."
    )
    print(
        f"Q4 pull_request.synchronize OK: {event.before[:7]} -> "
        f"{event.after[:7]} on PR #{event.pull_request.number}. "
        "pr.head.sha is authoritative."
    )


def q4_installation_created() -> None:
    body = (FIXTURES / "sample_installation_created.json").read_bytes()
    event = parse("installation", body)

    assert event.action == "created", f"expected created, got {event.action!r}"
    inst = event.installation
    assert isinstance(inst.id, int) and inst.id > 0
    assert isinstance(inst.app_slug, str) and inst.app_slug, (
        "Q4 FAIL: app_slug missing — octokit sample was stale without patch"
    )
    assert inst.account is not None, (
        "Q4 FAIL: installation.account is None on a 'created' event"
    )

    # Installation events deliver the list of repositories the App was
    # installed on. Needed by spec §6.2 (installation event handling).
    assert event.repositories is not None and len(event.repositories) >= 1, (
        "Q4 FAIL: installation.created with no repositories list"
    )

    print(
        f"Q4 installation.created OK: installation_id={inst.id}, "
        f"app_slug={inst.app_slug!r}, "
        f"{len(event.repositories)} repo(s) authorized."
    )


def main() -> None:
    q4_pull_request_opened()
    q4_pull_request_synchronize()
    q4_installation_created()


if __name__ == "__main__":
    main()
