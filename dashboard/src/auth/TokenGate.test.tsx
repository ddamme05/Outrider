import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test } from "vitest";

import { useTokenStore } from "./token";
import { TokenGate } from "./TokenGate";

beforeEach(() => {
  sessionStorage.clear();
  useTokenStore.setState({ token: null });
});

afterEach(() => {
  sessionStorage.clear();
  useTokenStore.setState({ token: null });
});

test("gates on the admin key, then renders children + persists once entered", async () => {
  render(
    <TokenGate>
      <div>protected content</div>
    </TokenGate>,
  );

  // No token -> gated: children hidden, the key input shown.
  expect(screen.queryByText("protected content")).not.toBeInTheDocument();
  const input = screen.getByLabelText("admin API key");

  await userEvent.type(input, "test-admin-key");
  await userEvent.click(screen.getByRole("button", { name: "Enter" }));

  // Entered -> children render, key persisted to sessionStorage.
  expect(screen.getByText("protected content")).toBeInTheDocument();
  expect(sessionStorage.getItem("outrider.adminToken")).toBe("test-admin-key");
});
