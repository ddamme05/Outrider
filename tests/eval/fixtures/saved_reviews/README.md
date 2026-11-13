# Saved review regressions

These three normalized responses were captured on October 31, 2026. They are
copied verbatim from the local paid-run exchanges named in each record, with
the exchange hash, original prompt identity and source-fixture hash. Raw provider
wire payloads were not retained.

[The replay tests](../../test_saved_review_cases.py) use them to check current
admission and scoring. The original token counts and timing belong to the paid
call; replay makes no new provider request and measures no model quality.

- The two Haiku responses use the same SSRF input. One has a valid range; the
  other declares line 5 through line 2 and must be rejected.
- The Sonnet response quotes shipping logic on lines 8–13 but declares 9–14.
  Location accuracy is separate from issue detection. The input does not
  establish whether tests exist, so the missing-test expectation stays unscored.

Keep these responses unchanged. Add a new capture when investigating a new
failure; establish its expected result from the source and supported contract.
