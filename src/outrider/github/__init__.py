# GitHub vendor SDK wrapper folder.
"""GitHub-related wrappers around `githubkit`.

Re-exports the type alias `InstallationGitHubClient` for consumers outside
the wrapper folder that need to type-annotate parameters but must not
import `githubkit` directly per `vendor-sdks-only-in-wrappers`.


This folder is the ONLY place in the codebase allowed to `import githubkit`
per `docs/conventions.md` "Imports" and `CLAUDE.md` rule 3 (the
`vendor-sdks-only-in-wrappers` invariant). Consumers in `api/`, `agent/`,
and elsewhere reach githubkit only through the thin helpers exported
here:

  - `outrider.github.auth.make_installation_client_factory(settings)` —
    binds a `GitHubAppSettings` once at lifespan startup; returns a
    `Callable[[int], GitHub]` that mints per-installation clients
    wrapping `githubkit.AppInstallationAuthStrategy`.
  - `outrider.github.webhooks.verify_webhook_signature` — wraps
    `githubkit.webhooks.verify`.
  - `outrider.github.fetch` (later spec) — per-file content fetch helpers.

Settings live in `outrider.github.config.GitHubAppSettings`.
"""

# Re-export the type alias from auth.py. The alias's definition
# requires importing `githubkit.AppInstallationAuthStrategy`; per the
# spec ("github/auth.py is the only file importing
# `githubkit.AppInstallationAuthStrategy`"), the alias is defined in
# auth.py and re-exported here for ergonomic import from
# `outrider.github`.
from outrider.github.auth import InstallationGitHubClient

__all__ = ["InstallationGitHubClient"]
