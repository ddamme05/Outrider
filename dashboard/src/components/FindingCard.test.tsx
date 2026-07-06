import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import type { components } from "../api/schema";
import { FindingCard } from "./FindingCard";

type FindingView = components["schemas"]["FindingView"];

// Minimal FindingView against the real wire shape (lowercase StrEnum values). Overrides let each
// test vary the field under check without restating the whole row.
function finding(overrides: Partial<FindingView> = {}): FindingView {
  return {
    finding_id: "f1",
    finding_type: "sql_injection",
    dimension: "security",
    severity: "critical",
    evidence_tier: "judged",
    file_path: "app/db/users.py",
    line_start: 40,
    line_end: 43,
    content_redacted: false,
    title: "Unparameterized query in the login path",
    description: "User input is interpolated straight into the SQL string.",
    evidence: 'cur.execute(f"select * from users where token = {token}")',
    suggested_fix: "Use a parameterized query.",
    query_match_id: null,
    trace_path: null,
    publish_destination: "inline_comment",
    eligibility: "withheld",
    eligibility_reason: "hitl_required_node_absent",
    hitl_decision: null,
    redaction_sweep_at: null,
    ...overrides,
  };
}

test("head badges render humanized labels, not raw enum wire values", () => {
  render(<FindingCard finding={finding()} />);
  // severity pill: "Critical" (not "critical"); type + dimension chip humanized; tier acronym.
  expect(document.querySelector(".sev-pill")).toHaveTextContent("Critical");
  const tag = document.querySelector(".ft-tag");
  expect(tag).toHaveTextContent("SQL injection");
  expect(tag).toHaveTextContent("Security");
  expect(document.querySelector(".tier")).toHaveTextContent("JUDGED");
  // publish destination humanized ("Inline comment", not "INLINE_COMMENT"); eligibility humanized.
  expect(screen.getByText("Inline comment")).toBeInTheDocument();
  expect(document.querySelector(".f-elig")).toHaveTextContent("Withheld · HITL gate not reached");
  // No raw slug leaks into the card.
  const body = document.querySelector(".finding")?.textContent ?? "";
  expect(body).not.toContain("sql_injection");
  expect(body).not.toContain("inline_comment");
});

test("title is promoted above the description", () => {
  render(<FindingCard finding={finding()} />);
  const title = document.querySelector(".f-title");
  const desc = document.querySelector(".f-desc");
  expect(title).toHaveTextContent("Unparameterized query in the login path");
  expect(desc).toHaveTextContent("User input is interpolated straight into the SQL string.");
});

test("evidence renders in a CodeBlock <pre> with a language chip from the path", () => {
  render(<FindingCard finding={finding()} />);
  const pre = document.querySelector(".codeblock-pre");
  expect(pre?.tagName).toBe("PRE");
  expect(pre).toHaveTextContent('cur.execute(f"select * from users where token = {token}")');
  // .py → python language chip.
  expect(document.querySelector(".codeblock-lang")).toHaveTextContent("python");
});

test("adversarial evidence markup stays inert (React text node, never a real element)", () => {
  render(
    <FindingCard finding={finding({ evidence: "<img src=x onerror=alert(1)> </pre></div>" })} />,
  );
  // The markup is rendered as text inside the <code>, not parsed into DOM nodes.
  expect(document.querySelector(".codeblock-pre code")).toHaveTextContent(
    "<img src=x onerror=alert(1)> </pre></div>",
  );
  expect(document.querySelector("img")).toBeNull(); // no injected element
});

test("a redacted finding shows the retention stub and hides content", () => {
  render(
    <FindingCard
      finding={finding({
        content_redacted: true,
        title: null,
        description: null,
        evidence: null,
        suggested_fix: null,
        redaction_sweep_at: "2026-05-20T00:00:00Z",
      })}
    />,
  );
  expect(screen.getByText(/Content redacted/)).toBeInTheDocument();
  expect(document.querySelector(".codeblock")).toBeNull(); // no evidence block
  // Metadata (humanized) still renders — it's permanent.
  expect(document.querySelector(".sev-pill")).toHaveTextContent("Critical");
});
