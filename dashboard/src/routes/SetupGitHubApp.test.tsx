import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useTokenStore } from "../auth/token";
import { server } from "../test/server";
import { SetupGitHubApp } from "./SetupGitHubApp";

const STATUS = "http://localhost/setup/status";
const START = "http://localhost/setup";
const GH_TARGET = "https://github.com/organizations/acme/settings/apps/new?state=signed-state";
const MANIFEST = { name: "outrider", public: false, url: "https://ci.example.com" };

beforeEach(() => {
  useTokenStore.setState({ token: "admin-key" });
});

afterEach(() => {
  vi.restoreAllMocks();
  useTokenStore.setState({ token: null });
});

function mockStatus(body: Record<string, unknown>, status = 200): void {
  server.use(http.get(STATUS, () => HttpResponse.json(body, { status })));
}

const clickSetUp = async (): Promise<void> => {
  await userEvent.type(await screen.findByPlaceholderText("acme-inc"), "acme");
  await userEvent.click(screen.getByRole("button", { name: /set up github app/i }));
};

test("unconfigured → clicking submits the manifest to github.com with the admin token", async () => {
  mockStatus({ status: "UNCONFIGURED", configured: false });
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
  // The form POSTs to github.com (never elsewhere) and carries the manifest as a hidden field.
  const form = capturedForm as unknown as HTMLFormElement;
  expect(form.action).toBe(GH_TARGET);
  expect(form.method.toLowerCase()).toBe("post");
  const field = form.querySelector<HTMLInputElement>('input[name="manifest"]');
  expect(field).not.toBeNull();
  expect(JSON.parse(field!.value)).toEqual(MANIFEST);
  // The admin token authed the /setup call — and rode the Authorization header, NOT the GitHub form.
  expect(seenAuth).toBe("Bearer admin-key");
});

test("SECURITY: refuses to submit the manifest to a non-github.com origin", async () => {
  mockStatus({ status: "UNCONFIGURED", configured: false });
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
  expect(screen.queryByRole("button", { name: /set up github app/i })).toBeNull();
});

test("already-configured instance shows no onboarding form", async () => {
  mockStatus({ status: "CONFIGURED", configured: true });
  render(<SetupGitHubApp />);
  expect(await screen.findByText(/already configured/i)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /set up github app/i })).toBeNull();
});
