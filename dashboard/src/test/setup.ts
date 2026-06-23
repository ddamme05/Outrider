import "@testing-library/jest-dom/vitest";

import { afterAll, afterEach, beforeAll } from "vitest";

import { server } from "./server";

// jsdom doesn't implement scrollIntoView; the replay auto-follow (ReplayReconstruct) calls it.
// Stub it so the passive effect is a no-op under test (production uses the real browser API).
Element.prototype.scrollIntoView = () => {};

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
