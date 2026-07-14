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

test("ORPHANED → Reset recovers to the UNCONFIGURED Start form", async () => {
  let statusBody: StatusBody = { status: "ORPHANED", configured: false, install_known: false };
  server.use(http.get(STATUS, () => HttpResponse.json(statusBody)));
  server.use(
    http.post(RESET, () => {
      statusBody = { status: "UNCONFIGURED", configured: false, install_known: false };
      return HttpResponse.json(statusBody);
    }),
  );

  render(<SetupGitHubApp />);
  // ORPHANED shows Reset, not the Start form.
  const reset = await screen.findByRole("button", { name: /reset and start over/i });
  expect(startButton()).toBeNull();

  await userEvent.click(reset);

  // After reset → refresh → UNCONFIGURED → the Start form appears.
  expect(await screen.findByPlaceholderText("acme-inc")).toBeInTheDocument();
});

test("in-flight (CONVERTING) → Refresh, not a Start form that would 409", async () => {
  mockStatus({ status: "CONVERTING", configured: false, install_known: false });
  render(<SetupGitHubApp />);
  expect(await screen.findByRole("button", { name: /refresh status/i })).toBeInTheDocument();
  expect(startButton()).toBeNull();
});
