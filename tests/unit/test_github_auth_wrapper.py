# Tests for the github/auth.py vendor wrapper.
"""Confirm `make_installation_client_factory` produces per-installation
client factories and is the only call site of
`githubkit.AppInstallationAuthStrategy` in the codebase (per
`vendor-sdks-only-in-wrappers`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from outrider.github.auth import make_installation_client_factory
from outrider.github.config import GitHubAppSettings

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.github import InstallationGitHubClient

# PEM + env-var injection live in the top-level `tests/conftest.py` per
# round-31 fold (DevEx audit, HIGH): one shared `TEST_GITHUB_APP_PRIVATE_KEY_PEM`
# constant + one `github_app_env` fixture across all auth/lifespan/filter
# tests so a PEM rotation or env-var rename touches one place.


@pytest.fixture(autouse=True)
def _activate_github_app_env(github_app_env: None) -> None:  # noqa: ARG001 — fixture activates env
    """Module-local autouse wrapper around the shared `github_app_env` fixture.

    Module tests all need GitHubAppSettings() to succeed; activating the
    shared opt-in fixture for every test here saves per-test argument
    plumbing.
    """


def _bound_factory() -> Callable[[int], InstallationGitHubClient]:
    """Helper: load settings once + build the bound factory.

    Mirrors the lifespan composition shape (settings constructed at
    startup; factory bound to those settings; per-installation calls
    happen later). Return type uses the wrapper alias
    `InstallationGitHubClient` so this test file complies with
    `vendor-sdks-only-in-wrappers` (no direct `from githubkit import`).
    """
    return make_installation_client_factory(GitHubAppSettings())


def test_returns_github_client() -> None:
    """The bound factory returns a per-installation client whose runtime
    class is the `githubkit.GitHub` shape (duck-typed via the `rest` +
    `auth` attributes — see `outrider.github.auth.InstallationGitHubClient`).

    Asserting against the SDK class directly would require importing
    `GitHub` here, which the `vendor-sdks-only-in-wrappers` invariant
    bans in test code (per `docs/invariants.md` "rg src tests"). The
    structural check is sufficient: a wrong return type (e.g., None
    from a buggy lexical-capture factory) lacks both attributes.
    """
    factory = _bound_factory()
    client = factory(42)
    assert hasattr(client, "rest"), "factory should return a GitHub-shaped client"
    assert hasattr(client, "auth"), "factory should return a GitHub-shaped client"


def test_distinct_installation_ids_yield_distinct_auth_contexts() -> None:
    """Calling the factory with two installation IDs yields clients
    whose underlying auth strategies are bound to distinct installations.

    Defends the bound factory against the lexical-capture variant — a
    bug where the factory returns one cached client across installations
    would silently use one installation's token for cross-tenant PRs.
    The auth strategy carries `installation_id`; reading it asserts the
    binding actually happened.
    """
    factory = _bound_factory()
    client_a = factory(42)
    client_b = factory(43)

    # Different installation IDs on the auth strategies. Accessing the
    # private attribute is acceptable in a test that exists specifically
    # to guard the installation_id binding; the alternative (introspecting
    # mint behavior) would require a real GitHub API call.
    auth_a = client_a.auth
    auth_b = client_b.auth
    assert auth_a.installation_id == 42
    assert auth_b.installation_id == 43
    # The two clients are different objects (not the same cached client).
    assert client_a is not client_b


def test_installation_auth_strategy_only_call_site() -> None:
    """`github/auth.py` is the ONLY file importing
    `githubkit.AppInstallationAuthStrategy`.

    Same intent as `test_github_webhooks_wrapper.py`'s call-site grep —
    enforces `vendor-sdks-only-in-wrappers` at CI time. A future refactor
    that adds the import elsewhere would silently bypass the wrapper.
    """
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src" / "outrider"

    # Match the import statement, not bare docstring mentions of the symbol.
    # A docstring reference like "wraps `githubkit.AppInstallationAuthStrategy`"
    # is fine; what we're banning is `from githubkit import
    # AppInstallationAuthStrategy` or `import githubkit.AppInstallationAuthStrategy`.
    import_pattern = (
        r"^from\s+githubkit\s+import\s+[^\n]*AppInstallationAuthStrategy"
        r"|^import\s+githubkit\.AppInstallationAuthStrategy"
    )
    rg = shutil.which("rg")
    if rg is not None:
        result = subprocess.run(  # noqa: S603 — fixed args, absolute rg path
            [
                rg,
                "--type",
                "py",
                "-l",
                "-U",  # multiline / regex anchors
                import_pattern,
                str(src_root),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        hits = [line for line in result.stdout.splitlines() if line.strip()]
    else:
        import re as _re

        compiled = _re.compile(import_pattern, _re.MULTILINE)
        hits = []
        for py_file in src_root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if compiled.search(text):
                hits.append(str(py_file))

    # Allowed import sites: `github/auth.py` (uses the SDK class to
    # construct clients) and `github/__init__.py` (re-exports the type
    # alias `InstallationGitHubClient` for callers outside `github/`).
    # Both are inside the wrapper folder; the rule is "only inside
    # src/outrider/github/" — not "only in one specific file."
    github_dir = (src_root / "github").resolve()
    assert hits, "Expected at least one import of `AppInstallationAuthStrategy`."
    for hit in hits:
        hit_path = Path(hit).resolve()
        # `is_relative_to` is the correct ancestor check: rejects
        # sibling directories like `github_extra/` that would pass a
        # naive `startswith(str(github_dir))`.
        assert hit_path.is_relative_to(github_dir), (
            f"`AppInstallationAuthStrategy` imported from {hit!r}; "
            f"all imports must be inside {github_dir}. "
            f"Move the call into the wrapper."
        )
