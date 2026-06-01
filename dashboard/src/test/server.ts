import { setupServer } from "msw/node";

// Shared MSW server; per-test handlers via `server.use(...)`.
export const server = setupServer();
