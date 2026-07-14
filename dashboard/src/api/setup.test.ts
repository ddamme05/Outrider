import { http, HttpResponse } from "msw";
import { expect, test } from "vitest";

import { server } from "../test/server";
import {
  SetupError,
  SetupProtocolError,
  SetupUnreachableError,
  fetchSetupStatus,
  startSetup,
} from "./setup";

// API-BOUNDARY tests (FUP-230). The component tests prove the operator SEES the right copy; these
// prove the client REJECTS the wrong payload in the first place — the guarantee the UI rests on.
// `fetchSetupStatus`/`startSetup` previously ended in `as SetupStatus` / `as SetupStartResponse`,
// casts that TypeScript erases at compile time and which therefore validate nothing at runtime.

const STATUS = "http://localhost/setup/status";
const START = "http://localhost/setup";

const VALID = { status: "UNCONFIGURED", configured: false, install_known: false };

test("a well-formed status payload is returned as-is", async () => {
  server.use(http.get(STATUS, () => HttpResponse.json(VALID)));
  await expect(fetchSetupStatus()).resolves.toEqual(VALID);
});

test("404 means env-credential mode, not an error — the accepted production contract", async () => {
  server.use(http.get(STATUS, () => new HttpResponse(null, { status: 404 })));
  await expect(fetchSetupStatus()).resolves.toBeNull();
});

test("HTML instead of JSON rejects as a protocol error", async () => {
  server.use(
    http.get(STATUS, () =>
      new HttpResponse("<!doctype html><html></html>", {
        headers: { "content-type": "text/html" },
      }),
    ),
  );
  await expect(fetchSetupStatus()).rejects.toBeInstanceOf(SetupProtocolError);
});

test("unparseable JSON rejects as a protocol error", async () => {
  server.use(
    http.get(STATUS, () =>
      new HttpResponse("{ not json", { headers: { "content-type": "application/json" } }),
    ),
  );
  await expect(fetchSetupStatus()).rejects.toBeInstanceOf(SetupProtocolError);
});

test("valid JSON of the wrong shape rejects — the `as` cast could not catch this", async () => {
  server.use(http.get(STATUS, () => HttpResponse.json({ unexpected: "payload" })));
  await expect(fetchSetupStatus()).rejects.toBeInstanceOf(SetupProtocolError);
});

test.each([
  ["a status outside the closed vocabulary", { ...VALID, status: "TOTALLY_NEW_STATE" }],
  ["an empty status", { ...VALID, status: "" }],
  ["a non-boolean configured", { ...VALID, configured: "yes" }],
  ["a missing install_known", { status: "UNCONFIGURED", configured: false }],
])("%s rejects as a protocol error", async (_label, body) => {
  // `status` is a CLOSED vocabulary (SETUP_STATUSES, DB CHECK-enforced). An unknown value is not
  // forward-compatibility — the UI's affordances are defined only over the five known states, so a
  // sixth must fail loudly here rather than render a blank, actionless page.
  server.use(http.get(STATUS, () => HttpResponse.json(body)));
  await expect(fetchSetupStatus()).rejects.toBeInstanceOf(SetupProtocolError);
});

test("a transport failure rejects as unreachable, NOT as a protocol error", async () => {
  // The two must stay distinct: unreachable can recover on retry, a non-API peer cannot.
  server.use(http.get(STATUS, () => HttpResponse.error()));
  const err = await fetchSetupStatus().catch((e: unknown) => e);
  expect(err).toBeInstanceOf(SetupUnreachableError);
  expect(err).not.toBeInstanceOf(SetupProtocolError);
});

test("a non-404 HTTP failure rejects as a plain backend error, not a protocol error", async () => {
  server.use(http.get(STATUS, () => HttpResponse.json({ detail: "boom" }, { status: 503 })));
  const err = await fetchSetupStatus().catch((e: unknown) => e);
  expect(err).toBeInstanceOf(SetupError);
  expect(err).not.toBeInstanceOf(SetupProtocolError);
  expect(err).not.toBeInstanceOf(SetupUnreachableError);
});

test("startSetup rejects a response missing target_url before it can reach the GitHub form POST", async () => {
  // Defense in depth for submitManifestToGitHub's origin check: a shape guard here means a
  // malformed target never reaches `new URL(...)` at the form-submit site.
  server.use(http.post(START, () => HttpResponse.json({ manifest: { name: "x" } })));
  await expect(startSetup("acme")).rejects.toBeInstanceOf(SetupProtocolError);
});

test("startSetup rejects a non-object manifest", async () => {
  server.use(http.post(START, () => HttpResponse.json({ target_url: "https://github.com/x", manifest: [] })));
  await expect(startSetup("acme")).rejects.toBeInstanceOf(SetupProtocolError);
});
