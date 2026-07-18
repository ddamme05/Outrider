// Nullable-cost rendering (openai-native-host spec): an unpriceable completed
// call carries cost_usd=null + a typed reason — summaries render the FACT,
// never a fabricated $0.00 and never a crash.
import { expect, test } from "vitest";

import type { AuditEvent } from "./auditEvent";
import { summarizeEvent } from "./auditEvent";

function llmCall(cost: number | null): AuditEvent {
  return {
    event_type: "llm_call",
    model: "gpt-5.6-sol",
    cost_usd: cost,
    input_tokens: 100,
    output_tokens: 50,
  } as unknown as AuditEvent;
}

test("priced call renders the exact figure", () => {
  expect(summarizeEvent(llmCall(0.0234))).toContain("$0.02");
});

test("unpriced call renders 'unpriced' — no toFixed crash, no $0.00", () => {
  const summary = summarizeEvent(llmCall(null));
  expect(summary).toContain("unpriced");
  expect(summary).not.toContain("$0.00");
});
