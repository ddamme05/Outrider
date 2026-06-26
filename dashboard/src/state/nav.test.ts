import { beforeEach, expect, test } from "vitest";

import { useNav } from "./nav";

beforeEach(() => useNav.setState({ open: false }));

test("the drawer starts closed", () => {
  expect(useNav.getState().open).toBe(false);
});

test("setOpen(true) opens the drawer", () => {
  useNav.getState().setOpen(true);
  expect(useNav.getState().open).toBe(true);
});

test("setOpen(false) closes the drawer", () => {
  useNav.setState({ open: true });
  useNav.getState().setOpen(false);
  expect(useNav.getState().open).toBe(false);
});

test("toggle() flips the open state", () => {
  expect(useNav.getState().open).toBe(false);
  useNav.getState().toggle();
  expect(useNav.getState().open).toBe(true);
  useNav.getState().toggle();
  expect(useNav.getState().open).toBe(false);
});
