import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { expect, test } from "vitest";

import { Sidebar } from "./Sidebar";

// B3: the sidebar footer carries a persistent privacy link → the public /privacy
// page (same-origin absolute href served by the FastAPI app, NOT a SPA route).
test("sidebar footer links to the public privacy page", () => {
  render(
    <MemoryRouter>
      <Sidebar />
    </MemoryRouter>,
  );
  const link = screen.getByRole("link", { name: /privacy/i });
  expect(link).toHaveAttribute("href", "/privacy");
});
