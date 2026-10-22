"""Review phases and per-file workers for the LangGraph review graph.

Async bodies receive ReviewState plus closure-injected dependencies. They return
state deltas; intake and the analyze planner also return routing Commands.
`agent/graph.py` wires the seven review phases and analyze's fan-out workers."""
