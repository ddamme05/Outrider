import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import { SetupGitHubApp } from "./SetupGitHubApp";

const STATUS = "http://localhost/setup/status";
const START = "http://localhost/setup";
const RESET = "http://localhost/setup/reset";
const GH_TARGET = "https://github.com/organizations/acme/settings/apps/new?state=signed-state";
const MANIFEST = { name: "outrider", public: false, url: "https://ci.example.com" };

beforeEach(() => {
  useTokenStore.setState({ token: "admin-key" });
});

afterEach(() => {
  vi.restoreAllMocks();
  useTokenStore.setState({ token: null });
});

type StatusBody = { status: string; configured: boolean; install_known: boolean };

function mockStatus(body: StatusBody | Record<string, never>, httpStatus = 200): void {
  server.use(http.get(STATUS, () => HttpResponse.json(body, { status: httpStatus })));
}

const startButton = () => screen.queryByRole("button", { name: /^set up github app$/i });

const clickSetUp = async (): Promise<void> => {
  await userEvent.type(await screen.findByPlaceholderText("acme-inc"), "acme");
  await userEvent.click(screen.getByRole("button", { name: /^set up github app$/i }));
};

test("UNCONFIGURED → clicking submits the manifest to github.com with the admin token", async () => {
  mockStatus({ status: "UNCONFIGURED", configured: false, install_known: false });
  let seenAuth: string | null = null;
  server.use(
    http.post(START, ({ request }) => {
      seenAuth = request.headers.get("Authorization");
      return HttpResponse.json({ target_url: GH_TARGET, manifest: MANIFEST });
    }),
  );
  let capturedForm: HTMLFormElement | null = null;
  const submitSpy = vi
    .spyOn(HTMLFormElement.prototype, "submit")
    .mockImplementation(function (this: HTMLFormElement) {
      capturedForm = this;
    });

  render(<SetupGitHubApp />);
  await clickSetUp();

  await waitFor(() => expect(submitSpy).toHaveBeenCalledTimes(1));
  const form = capturedForm as unknown as HTMLFormElement;
  expect(form.action).toBe(GH_TARGET);
  expect(form.method.toLowerCase()).toBe("post");
  const field = form.querySelector<HTMLInputElement>('input[name="manifest"]');
  expect(field).not.toBeNull();
  expect(JSON.parse(field!.value)).toEqual(MANIFEST);
  expect(seenAuth).toBe("Bearer admin-key");
});

test("SECURITY: refuses to submit the manifest to a non-github.com origin", async () => {
  mockStatus({ status: "UNCONFIGURED", configured: false, install_known: false });
  server.use(
    http.post(START, () =>
      HttpResponse.json({
        target_url: "https://evil.example.com/apps/new?state=s",
        manifest: MANIFEST,
      }),
    ),
  );
  const submitSpy = vi.spyOn(HTMLFormElement.prototype, "submit").mockImplementation(() => {});

  render(<SetupGitHubApp />);
  await clickSetUp();

  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/non-github origin/i));
  expect(submitSpy).not.toHaveBeenCalled();
});

test("env-credential instance (404 status) shows no onboarding form", async () => {
  mockStatus({}, 404);
  render(<SetupGitHubApp />);
  expect(await screen.findByText(/environment credentials/i)).toBeInTheDocument();
  expect(startButton()).toBeNull();
});

test("CONFIGURED + installed → fully set up, no form", async () => {
  mockStatus({ status: "CONFIGURED", configured: true, install_known: true });
  render(<SetupGitHubApp />);
  expect(await screen.findByText(/fully set up/i)).toBeInTheDocument();
  expect(startButton()).toBeNull();
});

test("CONFIGURED but NOT installed → finish-installing guidance, no form", async () => {
  mockStatus({ status: "CONFIGURED", configured: true, install_known: false });
  render(<SetupGitHubApp />);
  expect(await screen.findByText(/isn.t installed on any repositories/i)).toBeInTheDocument();
  expect(startButton()).toBeNull();
});

test("ORPHANED → Reset is gated on confirming App deletion, then recovers to the Start form", async () => {
  let statusBody: StatusBody = { status: "ORPHANED", configured: false, install_known: false };
  server.use(http.get(STATUS, () => HttpResponse.json(statusBody)));
  server.use(
    http.post(RESET, () => {
      statusBody = { status: "UNCONFIGURED", configured: false, install_known: false };
      return HttpResponse.json(statusBody);
    }),
  );

  render(<SetupGitHubApp />);
  const reset = await screen.findByRole("button", { name: /reset and start over/i });
  expect(startButton()).toBeNull();
  // Gated (spec F4): reset stays disabled until the operator confirms they deleted the orphaned App.
  expect(reset).toBeDisabled();

  await userEvent.click(screen.getByRole("checkbox", { name: /deleted the orphaned app/i }));
  expect(reset).toBeEnabled();
  await userEvent.click(reset);

  // After reset → refresh → UNCONFIGURED → the Start form appears.
  expect(await screen.findByPlaceholderText("acme-inc")).toBeInTheDocument();
});

test("in-flight (CONVERTING) → Retry re-POSTs /setup (the repair path), not a dead-end refresh", async () => {
  mockStatus({ status: "CONVERTING", configured: false, install_known: false });
  render(<SetupGitHubApp />);
  // A Retry form (POST /setup is the lazy-repair path) — NOT a status-only Refresh, which would
  // never clear an abandoned attempt.
  expect(await screen.findByRole("button", { name: /retry setup/i })).toBeInTheDocument();
  expect(await screen.findByPlaceholderText("acme-inc")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /refresh status/i })).toBeNull();
});

test("stale CONVERTING Retry is rejected → status re-syncs to ORPHANED's cleanup flow", async () => {
  // begin_setup commits stale CONVERTING → ORPHANED, THEN 409s the Start. The UI must refresh so the
  // operator sees the ORPHANED reset/cleanup flow — not a stale in-progress screen implying success.
  let statusBody: StatusBody = { status: "CONVERTING", configured: false, install_known: false };
  server.use(http.get(STATUS, () => HttpResponse.json(statusBody)));
  server.use(
    http.post(START, () => {
      statusBody = { status: "ORPHANED", configured: false, install_known: false };
      return HttpResponse.json({ detail: "the instance is ORPHANED" }, { status: 409 });
    }),
  );

  render(<SetupGitHubApp />);
  await userEvent.type(await screen.findByPlaceholderText("acme-inc"), "acme");
  await userEvent.click(screen.getByRole("button", { name: /retry setup/i }));

  // Refreshed → ORPHANED → the delete-confirmation + reset flow; the Retry form is gone.
  expect(
    await screen.findByRole("checkbox", { name: /deleted the orphaned app/i }),
  ).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /retry setup/i })).toBeNull();
});

test("expired AWAITING_CALLBACK → Retry is repaired by the backend and proceeds to GitHub", async () => {
  // The success case the fix must NOT regress: begin_setup resets an expired AWAITING_CALLBACK and
  // returns a fresh target, so Retry submits the manifest to GitHub.
  mockStatus({ status: "AWAITING_CALLBACK", configured: false, install_known: false });
  server.use(http.post(START, () => HttpResponse.json({ target_url: GH_TARGET, manifest: MANIFEST })));
  const submitSpy = vi.spyOn(HTMLFormElement.prototype, "submit").mockImplementation(() => {});

  render(<SetupGitHubApp />);
  await userEvent.type(await screen.findByPlaceholderText("acme-inc"), "acme");
  await userEvent.click(screen.getByRole("button", { name: /retry setup/i }));

  await waitFor(() => expect(submitSpy).toHaveBeenCalledTimes(1));
});
