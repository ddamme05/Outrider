"""Q7 — githubkit surface audit.

Confirm the names and import paths the V1 `src/outrider/github/` wrapper will
use are present and callable on the pinned 0.15.3. This is a lightweight
"smoke test" of the SDK surface, not a behavior test — each item's behavior
is exercised by another demo.

If githubkit is upgraded and this demo fails, the wrapper's imports are the
first place to check for breaking changes.
"""

from __future__ import annotations

import importlib.metadata
import inspect

PINNED_VERSION = "0.15.3"  # must match pyproject.toml and requirements.txt


def main() -> None:
    # Pinned version assertion: a spike finding only holds at the version
    # it was captured under. If the installed wheel drifts from the pin,
    # the rest of the surface checks below apply to a different SDK than
    # NOTES.md documents.
    installed = importlib.metadata.version("githubkit")
    assert installed == PINNED_VERSION, (
        f"Q7 FAIL: githubkit {installed!r} installed but spike pinned to "
        f"{PINNED_VERSION!r}. Either update requirements.txt + rerun the "
        "spike, or revert the environment to the pinned version."
    )
    # Auth strategies — confirmed present per aegis-docs::githubkit/usage/
    # getting-started/authentication.md.
    import githubkit
    from githubkit import (
        AppAuthStrategy,
        AppInstallationAuthStrategy,
        GitHub,
        TokenAuthStrategy,
        UnauthAuthStrategy,
    )

    assert callable(GitHub)
    assert callable(AppAuthStrategy)
    assert callable(AppInstallationAuthStrategy)
    assert callable(TokenAuthStrategy)
    assert callable(UnauthAuthStrategy)

    # Webhook namespace — confirmed present.
    from githubkit.webhooks import (
        parse,
        parse_obj,
        parse_obj_without_name,
        parse_without_name,
        sign,
        verify,
    )

    for fn in (parse, parse_obj, parse_obj_without_name, parse_without_name, sign, verify):
        assert callable(fn), f"Q7 FAIL: {fn.__name__!r} not callable"

    # Raw-request escape hatch per aegis-docs::githubkit/pr-review-bot.md:
    # "Do not claim an exact generated method name ... unless the local
    # generated docs or installed package confirms it. arequest() is the
    # stable documented fallback."
    gh = GitHub()
    assert hasattr(gh, "arequest") and callable(gh.arequest), (
        "Q7 FAIL: GitHub.arequest missing — wrapper cannot fall back to raw "
        "async REST"
    )
    assert hasattr(gh, "request") and callable(gh.request), (
        "Q7 FAIL: GitHub.request missing — wrapper cannot fall back to raw "
        "sync REST"
    )

    # Signatures for the three verbs the webhook path uses. If these shift,
    # receiver.py breaks.
    assert list(inspect.signature(verify).parameters) == [
        "secret",
        "payload",
        "signature",
    ], f"verify signature changed: {inspect.signature(verify)}"
    assert list(inspect.signature(sign).parameters) == [
        "secret",
        "payload",
        "method",
    ], f"sign signature changed: {inspect.signature(sign)}"
    assert list(inspect.signature(parse).parameters) == [
        "name",
        "payload",
    ], f"parse signature changed: {inspect.signature(parse)}"

    # Version module — confirms the generated-client tree is present.
    assert hasattr(githubkit, "versions"), "githubkit.versions missing"

    print(
        f"Q7 OK: githubkit=={installed} (pinned); all imports resolve "
        "(auth strategies, webhook verbs, raw request escape hatch); "
        "verify/sign/parse signatures match what receiver.py + demo_q3 "
        "+ demo_q4 assume."
    )


if __name__ == "__main__":
    main()
