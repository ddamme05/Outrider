import { render } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { renderInlineCode } from "./renderInlineCode";

describe("renderInlineCode", () => {
  test("backtick spans become <code>; surrounding text stays plain", () => {
    const { container } = render(<div>{renderInlineCode("bind `status` as a parameter")}</div>);
    const codes = container.querySelectorAll("code");
    expect(codes).toHaveLength(1);
    expect(codes[0]?.textContent).toBe("status");
    expect(container.textContent).toBe("bind status as a parameter");
  });

  test("multiple code spans, including one containing quotes/braces", () => {
    const { container } = render(
      <div>{renderInlineCode("`status` in `f\"WHERE status = '{status}'\"`")}</div>,
    );
    const codes = [...container.querySelectorAll("code")].map((c) => c.textContent);
    expect(codes).toEqual(["status", "f\"WHERE status = '{status}'\""]);
  });

  test("plain text with no backticks renders unchanged (no <code>)", () => {
    const { container } = render(<div>{renderInlineCode("no code here")}</div>);
    expect(container.querySelector("code")).toBeNull();
    expect(container.textContent).toBe("no code here");
  });

  test("XSS-safe: markup inside a code span is a text node, never parsed", () => {
    const { container } = render(
      <div>{renderInlineCode("`<img src=x onerror=alert(1)>`")}</div>,
    );
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("code")?.textContent).toBe("<img src=x onerror=alert(1)>");
  });
});
