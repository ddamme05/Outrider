import { render } from "@testing-library/react";
import { expect, test } from "vitest";

import type { components } from "../api/schema";
import { AuditFeed } from "./AuditFeed";

type AuditEvent = components["schemas"]["ReviewEventsResponse"]["events"][number];

// Supply the fields read by summarizeEvent; event_type selects the family class.
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
        ev("file_examination", 4, { file_path: "b.py", examination_type: "analyze", parse_status: "clean" }),
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

test("file rows show the recorded intake and parse outcomes", () => {
  const rows = [
    { examination_type: "intake_fetch", parse_status: "skipped", skip_reason: "BINARY" },
    { examination_type: "analyze", parse_status: "clean" },
    { examination_type: "analyze", parse_status: "degraded" },
    { examination_type: "analyze", parse_status: "failed" },
  ];
  const { container } = render(
    <AuditFeed events={rows.map((row, i) => ev("file_examination", i + 1, { ...row, file_path: "a.py" }))} />,
  );
  expect([...container.querySelectorAll(".af-summary")].map((row) => row.textContent)).toEqual([
    "Skipped at intake · binary · a.py",
    "Examined a.py",
    "Analysis · parse degraded · a.py",
    "Analysis · parse failed · a.py",
  ]);
});

test("rows number 1-based for the review, not the raw global sequence_number", () => {
  // A review whose events start at global sequence 108 (as they do in practice).
  // The displayed # must be 1, 2, 3… — the Nth event in THIS review — not 108/109.
  const { container } = render(
    <AuditFeed
      events={[
        ev("review_phase", 108, { marker: "start" }),
        ev("review_phase", 109, { marker: "end" }),
        ev("publish", 110),
      ]}
    />,
  );
  const seqs = [...container.querySelectorAll(".af-seq")].map((s) => s.textContent);
  expect(seqs).toEqual(["1", "2", "3"]);
  // The raw audit sequence is preserved on hover for traceability.
  expect(container.querySelector(".af-seq")?.getAttribute("title")).toBe("audit sequence 108");
});
