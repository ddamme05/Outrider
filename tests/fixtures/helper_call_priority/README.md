# Recorded helper-selection inputs

These fixtures retain the native Outrider inputs and analyze exchanges for two
public python-trio/trio pull requests, captured on November 12, 2026.

- [Trio #2938](https://github.com/python-trio/trio/pull/2938) introduces the exception-chain defect. Its base is `7cb15ea5bf9a27342e7c409fd336777add89809e` and its head is `03fee2612de1636c28e935125f89c5de6f770133`.
- [Trio #3394](https://github.com/python-trio/trio/pull/3394) repairs that defect. Its base is `6deaf2a94d3d54437606ba031a06be79362c82f6` and its head is `48aadaf9c7f9eb1a958476357cc4e4fbbc71675d`.

The graph states retain both complete changed HEAD files, BASE source, patches,
triage and recorded workflow fields. The test clears completed analysis and
report fields in memory. It first reproduces the original requests exactly,
then enables the existing optional helper context. Raw files remain unchanged.

The saved native answers are empty. Reusing them is a scripted source-delivery
check, with no fresh generation or review-quality result. The regression checks
that the factory called by the changed method reaches its own worker. It does
not expect the context-manager class or claim that the model detects the bug.

Trio source is copyright Contributors to the Trio project. The original
`LICENSE`, `LICENSE.MIT` and `LICENSE.APACHE2` files accompany the substantial
source text retained here. Those license files match both captured HEAD trees.
`provenance.json` records the exact raw fixture hashes and sizes.
