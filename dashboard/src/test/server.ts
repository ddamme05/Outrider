import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

// Shared MSW server; per-test handlers via `server.use(...)`.
//
// Default: GET /api/meta → production shape. `useDemoStatus` (DemoBanner + ReviewDetail)
// fires this on every mount; a shared default keeps unrelated tests off the
// onUnhandledRequest:"error" tripwire. Demo-mode tests override with `server.use`.
export const server = setupServer(
  http.get("http://localhost/api/meta", () => HttpResponse.json({ demo_mode: false })),
);
