# Public review comparison development cases

Dataset: `review-comparison-development-v1`. These are exposed historical public
PRs, selected before generating any model answer for this set. They are not
held-out evidence, synthetic programs, or certifications that a whole PR is clean.
Keep each repository and bug/fix family together in future data splits.
The pairs are authentic PR heads, not edited one-hunk counterfactuals; unrelated
changes and historical versions differ between the introducing and fixing PRs.

| Captured PR head | Upstream head commit date (UTC) | Named mechanism |
| --- | --- | --- |
| [Starlette #933](https://github.com/Kludex/starlette/pull/933) | 2020-05-06 | A spooled-file write crosses its memory threshold and rolls to disk on the event-loop thread. |
| [Starlette #2962](https://github.com/Kludex/starlette/pull/2962) | 2025-07-20 | The corresponding threshold-crossing operation runs in a worker thread. |
| [Flask-Security #1183](https://github.com/pallets-eco/flask-security/pull/1183) | 2026-03-05 | A valid OAuth identity from another account refreshes the current session. |
| [Flask-Security #1216](https://github.com/pallets-eco/flask-security/pull/1216) | 2026-05-17 | The callback requires the OAuth identity to match the current user before refreshing. |

Capture time is **2026-11-01 UTC** (October 31 in the working session's timezone),
not the date of the upstream code. Each immutable head SHA, original PR creation/
update/merge dates, commit date and selected-file last-touch commit/date is saved
in the receipt or `provenance/`. The original importer captured every changed
UTF-8 file, base/head contents, original hunks, parsed API metadata and upstream
license. It checked inventory and line totals, Git blob identities, immutability
during capture and exact patch reconstruction. Its checks were not weakened.
Raw files and receipts are frozen; labels live separately in `manifest.json`.

The full changed-file inventory is retained: 3, 2, 21 and 3 files respectively.
No unchanged repository files were captured. The model input therefore cannot
assume it has all callers, tests, configuration or dependency versions. The
selected `analysis_file` is diagnostic metadata; a full-workflow comparison can
review all captured files. Do not silently truncate a direct-model input to fit.

Original titles, bodies, tests, comments and security/advisory references remain
authentic and can reveal the defect or fix. The official advisories are
[Starlette GHSA-2c2j-9gv5-cj73](https://github.com/Kludex/starlette/security/advisories/GHSA-2c2j-9gv5-cj73)
and [Flask-Security GHSA-97r5-pg8x-p63p](https://github.com/pallets-eco/flask-security/security/advisories/GHSA-97r5-pg8x-p63p).
Public examples may overlap model training. Outrider's fixed severity labels are
policy metadata; in particular, its `auth_bypass=critical` does not claim the
upstream advisory's Moderate rating was Critical.

`tests/eval/test_review_case_semantics.py` supplies executable source oracles:

- Upload tests execute the complete captured datastructures module, real Python
  `SpooledTemporaryFile`, and the installed Starlette/AnyIO thread-pool helper.
  They observe the actual rollover thread, using a small threshold rather than a
  large upload. They do not establish HTTP throughput or a denial-of-service rate.
- OAuth tests execute the exact captured callback and identity-resolver bodies with
  explicit OAuth transport/identity, datastore, session, current-user, clock and
  redirect/config facades. Same-user, different-user
  and unknown-user cases isolate freshness mutation. They do not execute Flask
  middleware, CSRF, a datastore, OAuth network exchanges or protected account routes.

These labels were derived from actual source behavior, independently of model
output. Fixed counterparts certify only the named repaired mechanism; another
warning requires independent adjudication. Runtime/dependency versions and raw
oracle results are retained in local comparison evidence.

Run without provider keys or an application database:

```bash
uv run pytest tests/eval/test_review_case_semantics.py --is-eval -q
```
