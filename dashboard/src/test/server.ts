import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

// Shared MSW server with production metadata for useDemoStatus callers.
// Demo tests override it with server.use; unexpected requests still fail.
export const server = setupServer(
  http.get("http://localhost/api/meta", () => HttpResponse.json({ demo_mode: false })),
);
