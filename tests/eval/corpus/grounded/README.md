# Grounded development examples

These ten synthetic examples form five bug/fix pairs: request SQL, JSON request
decoding, JSON decoder-depth recovery, account access and async scheduling. They
are exposed development data, not held-out evidence or real repository findings.
Keep each pair in the same data split.
The two JSON pairs share a parsing boundary; keep them together in any future split.

Both members of a pair use the same inputs and public API; the relevant mechanism
changes. Functions include the imports and request/response behavior they need.
There are no scripted model responses. Metadata and inspiration references live
in `grounded_cases.py`, outside model input.

`test_grounded_cases.py` executes each program independently of model output:
SQLite injection versus binding, malformed JSON at a request boundary, denied
credentials accessing an account, and whether another task can run during an async
job's pause. The scheduling check asserts event order rather than elapsed time.
The depth pair uses well-formed deeply nested JSON that exhausts the decoder
recursion guard; its response-boundary oracle distinguishes escaped RecursionError
from a 400 response. Unsupported body types are rejected before decoding.
These oracles support the named issue only;
they do not prove that every possible emitted finding would be false.

The async pair can also be covered by Outrider's structural detector. Keep that
product coverage separate from credit for model-only discovery.

Run `pytest tests/eval/test_grounded_cases.py --is-eval` without provider keys.
Future held-out cases need different repository/behavior families.

Dataset v3 appends the decoder-depth pair; the original eight fixture programs
remain byte-identical. It extends development coverage after the narrower JSON
decode check left this failure path untested. Historical reports retain their
original dataset version and input hashes.
