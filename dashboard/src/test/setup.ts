import "@testing-library/jest-dom/vitest";

import { afterAll, afterEach, beforeAll } from "vitest";

import { server } from "./server";

// Replay auto-follow calls scrollIntoView, which jsdom does not implement.
Element.prototype.scrollIntoView = () => {};

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
