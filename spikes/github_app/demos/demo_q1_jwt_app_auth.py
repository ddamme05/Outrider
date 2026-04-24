"""Q1 — GitHub App JWT authentication.

Two checkpoints:

Q1a (primitive): sign a JWT with `pyjwt[crypto]` using the RS256 algorithm,
decode it with the public key, assert the canonical claims are present and
correctly typed. If this fails, we don't trust any higher-level library.

Q1b (canonical): instantiate `githubkit.AppAuthStrategy` with the same key
and confirm the resulting GitHub client constructs without error. We do
NOT make a live API call here — Q2 (live runbook) covers that. The
offline check is: does the SDK accept the key shape we'll generate from
pydantic-settings in V1?
"""

from __future__ import annotations

import time
from pathlib import Path

import jwt
from githubkit import AppAuthStrategy, GitHub

FIXTURES = Path(__file__).parent.parent / "fixtures"
PRIVATE_KEY = (FIXTURES / "test_private_key.pem").read_bytes()
TEST_APP_ID = "123456"  # numeric GitHub App ID as string


def q1a_primitive() -> None:
    now = int(time.time())
    # GitHub App JWT claims per docs:
    # https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-json-web-token-jwt-for-a-github-app
    # - iat: issued-at (int unix seconds; GitHub tolerates 60s of drift)
    # - exp: expiration (max 10 min after iat; GitHub's server-side clamp)
    # - iss: app ID (numeric) OR client ID (string starting with Iv or Iv1.)
    payload = {
        "iat": now - 10,  # small backdate to tolerate clock skew, per docs
        "exp": now + 9 * 60,  # 9 min, under the 10-min hard cap
        "iss": TEST_APP_ID,
    }
    token = jwt.encode(payload, PRIVATE_KEY, algorithm="RS256")
    assert isinstance(token, str) and token.count(".") == 2, (
        "Q1a FAIL: RS256 JWT should be three dot-separated segments, got "
        f"{token!r}"
    )

    # Derive public key from the private key and decode.
    from cryptography.hazmat.primitives import serialization

    private_key = serialization.load_pem_private_key(PRIVATE_KEY, password=None)
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    decoded = jwt.decode(token, public_pem, algorithms=["RS256"])
    assert decoded["iss"] == TEST_APP_ID, (
        f"Q1a FAIL: iss round-trip got {decoded['iss']!r}"
    )
    assert decoded["exp"] - decoded["iat"] <= 10 * 60, (
        "Q1a FAIL: GitHub enforces exp <= iat + 10min; our payload exceeded it"
    )

    # Negative case: tampered token must not decode.
    tampered = token[:-4] + "AAAA"
    try:
        jwt.decode(tampered, public_pem, algorithms=["RS256"])
    except jwt.InvalidSignatureError:
        pass
    else:
        raise AssertionError("Q1a FAIL: tampered JWT decoded successfully")

    print(
        "Q1a OK: pyjwt RS256 sign/verify round-trip works; claims preserved; "
        "GitHub's 10-min exp cap respected; tamper detection fires."
    )


def q1b_canonical() -> None:
    # AppAuthStrategy constructs a GitHub client that signs JWTs internally
    # on each request. We don't fire a request (no real App) — we only
    # check the construction path accepts the key shape.
    strategy = AppAuthStrategy(TEST_APP_ID, PRIVATE_KEY.decode("utf-8"))
    github = GitHub(strategy)
    assert github is not None, "Q1b FAIL: GitHub() returned None"
    # Also check the installation-context derivation path compiles.
    installation_client = github.with_auth(
        github.auth.as_installation(installation_id=1)
    )
    assert installation_client is not None, (
        "Q1b FAIL: with_auth(as_installation(...)) returned None"
    )
    print(
        "Q1b OK: AppAuthStrategy + GitHub() construct cleanly; "
        "with_auth(as_installation(id)) returns a usable client. "
        "Live JWT/installation-token round-trip covered by runbook.md."
    )


def main() -> None:
    q1a_primitive()
    q1b_canonical()


if __name__ == "__main__":
    main()
