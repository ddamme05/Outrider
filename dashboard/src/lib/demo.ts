import { $api } from "../api/client";

export type DemoStatus = "loading" | "demo" | "production";

// Deployment-shape discovery via GET /api/meta (unauthenticated, static per boot).
// Tri-state so each caller picks its own fail direction — neither may collapse to
// "production," which is what let a live-looking Submit reach the demo box's dead
// /decide route:
//   - the read-only banner shows ONLY on a confirmed "demo" (fails to no-banner
//     while loading/erroring, so a production box never flashes a demo strip);
//   - the HITL mutation gate enables Submit ONLY on a confirmed "production"
//     (fails CLOSED while loading/erroring, so a demo box — or a box whose meta
//     hasn't resolved — never shows a live Submit).
//
// Bounded retry + backoff rides out the demo box's documented boot-502 window
// (deploy/README.md) since the app-wide QueryClient sets retry:false. staleTime
// Infinity: the flag never changes without a server restart.
export function useDemoStatus(): DemoStatus {
  const meta = $api.useQuery(
    "get",
    "/api/meta",
    {},
    {
      staleTime: Infinity,
      retry: 5,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
    },
  );
  // STRICT: only an exact boolean resolves the status. openapi-fetch types the
  // response at compile time but does not validate at runtime, so a malformed 200
  // ({}, {demo_mode: null}, text/HTML) must stay "loading" (unknown) rather than
  // fall through to "production" — otherwise a garbled meta response fails open
  // and re-enables the live Submit this gate exists to withhold.
  if (meta.data?.demo_mode === true) return "demo";
  if (meta.data?.demo_mode === false) return "production";
  return "loading";
}
