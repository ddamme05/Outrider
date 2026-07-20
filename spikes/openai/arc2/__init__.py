"""Arc 2 strict-schema feasibility probe internals.

See `specs/2026-07-20-arc2-strict-schema-feasibility.md`.

Probe-local by decision: nothing here is a production export. The strict schema
derivation and the tier->proof-field mapping stay in `spikes/` until a post-GO
adoption spec promotes them into shared schema code.

Split out of `strict_schema_probe.py` purely so the offline units (schema
derivation, verdict classifier, manifest binding) are unit-testable without
importing the CLI entrypoint.
"""
