"""Presentation helpers — turn a canonical `ReviewFinding` into structured display
sections shared across the GitHub, Slack, and dashboard renderers.

Trust-boundary placement: this subsystem is a pure PRESENTATION layer. It consumes
`ReviewFinding` + policy enums and produces plain semantic data — never markdown/mrkdwn/JSX,
never pre-escaped, and never a raw string shared across channels (each renderer escapes with
its own primitive). It imports no vendor SDK, does no I/O, and does NO severity/tier
derivation (`effective_severity` is an input). It lives OUTSIDE llm/, github/, notify/,
ast_facts/, coordinates/, and policy/ by design (see specs/2026-07-06-finding-presentation.md).
"""

from outrider.presentation.finding_sections import (
    FindingSections,
    build_finding_sections,
)

__all__ = ["FindingSections", "build_finding_sections"]
