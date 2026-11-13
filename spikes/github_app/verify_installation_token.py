"""Make a live GitHub request with the experiment's App credentials.

Configure GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY_PATH and GITHUB_INSTALLATION_ID
as described in runbook.md (DECISIONS.md#006). The script switches to installation
authentication. With TEST_REPO=owner/name it reads that repository; otherwise it
calls the authenticated-App endpoint. Both paths contact GitHub.

The printed success details are a manual experiment record for NOTES.md.
This is separate from Outrider's production credential setup.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from githubkit import AppAuthStrategy, GitHub


def env_or_die(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(
            f"ERROR: {name} is not set. See module docstring for required env vars.",
            file=sys.stderr,
        )
        sys.exit(2)
    return value


async def main() -> None:
    app_id = env_or_die("GITHUB_APP_ID")
    private_key_path = env_or_die("GITHUB_APP_PRIVATE_KEY_PATH")
    installation_id = int(env_or_die("GITHUB_INSTALLATION_ID"))
    test_repo = os.environ.get("TEST_REPO")  # optional

    pem_path = Path(private_key_path)
    if not pem_path.is_file():
        print(
            f"ERROR: GITHUB_APP_PRIVATE_KEY_PATH={private_key_path!r} is not a file.",
            file=sys.stderr,
        )
        sys.exit(2)
    private_key = pem_path.read_text()

    # Step 1: construct an App-auth client. This signs JWTs internally per call.
    app_github = GitHub(AppAuthStrategy(app_id, private_key))

    # Step 2: switch to installation-scoped auth. githubkit mints an
    # installation token under the hood on the next API call (per
    # aegis-docs::githubkit/pr-review-bot.md and the Q2 docs in NOTES.md).
    installation_github = app_github.with_auth(app_github.auth.as_installation(installation_id))

    # Step 3: make a real API call to force token minting + verify the
    # token is accepted by GitHub.
    if test_repo:
        owner, _, repo = test_repo.partition("/")
        if not owner or not repo:
            print(
                f"ERROR: TEST_REPO={test_repo!r} must be 'owner/name' shape.",
                file=sys.stderr,
            )
            sys.exit(2)
        resp = await installation_github.rest.repos.async_get(owner, repo)
        repo_data = resp.parsed_data
        print(
            f"q2_verified installation_id={installation_id} "
            f"repo={test_repo!r} repo_id={repo_data.id} "
            f"private={repo_data.private} default_branch={repo_data.default_branch!r}"
        )
    else:
        # Without TEST_REPO, hit a lightweight installation-self endpoint.
        # This still exercises the token-mint path because GitHub validates
        # the bearer token before answering.
        resp = await installation_github.rest.apps.async_get_authenticated()
        app_data = resp.parsed_data
        print(
            f"q2_verified installation_id={installation_id} "
            f"app_authenticated_as={app_data.slug!r} app_id={app_data.id} "
            "(set TEST_REPO=owner/name for a repo-scoped API call)"
        )

    print(
        "Token mint succeeded. Capture this line into "
        "spikes/github_app/NOTES.md under Q2 as evidence the installation "
        "token round-trip works against real GitHub."
    )


if __name__ == "__main__":
    asyncio.run(main())
