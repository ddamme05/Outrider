import { render } from "@testing-library/react";
import { expect, test } from "vitest";

import type { components } from "../api/schema";
import { AuditFeed } from "./AuditFeed";

type AuditEvent = components["schemas"]["ReviewEventsResponse"]["events"][number];

// event_type drives the family class under test; `fields` carries whatever
// summarize() reads for that type so the row renders against the real wire shape
// (llm_call reads cost_usd.toFixed, etc.) rather than a shape-blind stub.
function ev(event_type: string, seq: number, fields: Record<string, unknown> = {}): AuditEvent {
  return {
    event_type,
    sequence_number: seq,
    timestamp: "2026-06-01T00:00:00Z",
    ...fields,
  } as unknown as AuditEvent;
}

test("each row carries its --ev-* family class; unknown event types fall back to framing", () => {
  const { container } = render(
    <AuditFeed
      events={[
        ev("llm_call", 1, {
          model: "claude-sonnet-4-6",
          cost_usd: 0.12,
          input_tokens: 1000,
          output_tokens: 200,
          node_id: "analyze",
        }),
        ev("finding", 2, { finding_type: "SQL_INJECTION", severity: "high", file_path: "a.py" }),
        ev("finding_proposal_rejected", 3),
        ev("file_examination", 4, { file_path: "b.py" }),
        ev("trace_decision", 5, { resolution_status: "resolved" }),
        ev("hitl_decision", 6),
        ev("publish_routing", 7),
        ev("review_phase", 8, { marker: "start" }),
        ev("synthesize_completed", 9),
        ev("something_unmapped", 10),
      ]}
    />,
  );
  const fam = [...container.querySelectorAll(".afrow")].map(
    (r) => [...r.classList].find((c) => c.startsWith("ev-c-")) ?? "",
  );
  expect(fam).toEqual([
    "ev-c-model",
    "ev-c-finding",
    "ev-c-rejected",
    "ev-c-file",
    "ev-c-trace",
    "ev-c-human",
    "ev-c-output",
    "ev-c-framing",
    "ev-c-complete",
    "ev-c-framing", // unmapped → neutral fallback, never uncolored
  ]);
});
