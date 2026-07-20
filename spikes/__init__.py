"""Research spikes. Not production code, not importable from `src/outrider/`.

Carries `__init__.py` only so type checkers can resolve `spikes.<area>.<module>`
unambiguously — without it a checker sees e.g. `arc2.classifier` and
`spikes.openai.arc2.classifier` as two names for one file and refuses to proceed.
Spikes remain outside the CI `mypy --strict src` gate.
"""
