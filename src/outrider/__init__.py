"""Outrider: review GitHub pull requests through the agent pipeline.

Import installs the LLM-content logging filter before review work
(specs/2026-05-05-llm-provider-wrapper.md AC#23; DECISIONS.md#016 point 4).
The lifespan reapplies it after later handler registration. Forced policy imports
also run the finding/dimension lockstep and subsumption guards at startup
(specs/2026-05-19-analyze-foundation.md §6; DECISIONS.md#055).
"""

from outrider.llm.logging import register_filter_on_all_handlers

# Side-effect import: install the LLM content filter on all reachable
# handlers immediately. Production code that does `import outrider` (or
# any submodule, which transitively imports this) gets the filter wired
# before any review work begins.
register_filter_on_all_handlers()

# Forced-import side effect per §6 of
# `specs/2026-05-19-analyze-foundation.md`: `outrider.policy.dimensions`
# runs a module-load lockstep assertion across `FindingType`,
# `SEVERITY_POLICY`, and `FINDING_TYPE_TO_DIMENSION`. Importing it here
# guarantees the assertion fires at app startup / test collection even
# when no analyze code is on the import path — the deterministic floor
# below CI's set-equality unit test (which may be bypassed via
# `git commit --no-verify`).
import outrider.policy.dimensions  # noqa: F401, E402 — forced import for lockstep guard

# Forced-import side effect per DECISIONS.md#055: `outrider.policy.subsumption`
# runs a module-load well-formedness assertion over the `SUBSUMES` cross-type
# relation (enum-membership / irreflexivity / single-hop acyclicity / severity-
# monotonicity). Same deterministic-floor rationale as the lockstep guard above.
import outrider.policy.subsumption  # noqa: F401, E402 — forced import for SUBSUMES guard


def main() -> None:
    print("Hello from outrider!")


if __name__ == "__main__":
    main()
