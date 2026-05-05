"""CoordinateError — single failure-mode exception for the coordinates module.

Lives in its own module so both `translator.py` and `diff_parser.py` can
import it without creating a circular dependency.

Per docs/spec.md §5.6 — coordinates' contract is that any patch-parse
failure, span-out-of-hunk, span-out-of-bounds, or path-validation rejection
surfaces as `CoordinateError`, never as an underlying `unidiff` parse
exception, `IndexError`, or path-library leak.
"""


class CoordinateError(Exception):
    """Raised when coordinate translation cannot produce a reviewable result.

    Single failure mode for the coordinates module per docs/spec.md §5.6.
    Catchable as `Exception`; the specific intermediate base in the MRO is
    an implementation detail.
    """
