import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dev backend (uvicorn). 127.0.0.1, never 0.0.0.0 (WSL2 exposure — see the
// project SESSION_RETRO). Override with VITE_API_BASE_URL in production where
// the SPA and API may be different origins.
const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // The read-API — queue / detail / findings / replay are all under /api.
      "/api": { target: BACKEND, changeOrigin: true },
      // The one legacy write path. A SPECIFIC RegExp rule (Vite treats a key
      // starting with ^ as a regex) so it matches ONLY POST /reviews/{id}/decide
      // and never swallows the SPA's own /reviews + /reviews/:id client routes.
      "^/reviews/[^/]+/decide$": { target: BACKEND, changeOrigin: true },
      // The public privacy page is backend-served HTML (FastAPI GET /privacy), NOT an SPA
      // route. Without this, the Sidebar footer link resolves against the Vite dev server
      // and React Router 404s. An exact ^…$ regex so it can't shadow a future SPA route.
      "^/privacy$": { target: BACKEND, changeOrigin: true },
      // `/setup*` is DELIBERATELY ABSENT — do not add it. Two explicit topologies (FUP-230):
      //
      //   1. THIS dev server: component development only. NOT a live onboarding flow.
      //   2. FastAPI serving the built SPA (OUTRIDER_SERVE_SPA=1, as deploy/Dockerfile.prod bakes):
      //      the ONLY supported end-to-end onboarding topology.
      //
      // Why a proxy rule cannot bridge the gap: production splits /setup by METHOD in ONE origin —
      // FastAPI owns POST /setup, while the exact GET /setup falls through to the app shell via
      // api/spa.py's RESERVED_DESCENDANT_PREFIXES, and /setup/<sub> 404s. Vite's proxy keys match on
      // PATH ONLY, so any rule here would have to re-implement that method split in a `bypass` hook
      // — a second copy of a rule the backend already owns, free to drift such that dev goes green
      // while prod stays untested. (Note a naive "/setup" key would also REGRESS today's behavior:
      // it would proxy GET /setup to a backend that has no GET /setup, and Vite's non-regex matcher
      // is `url.startsWith(context)`, so it would swallow /setupwizard too.)
      //
      // The full flow additionally needs GET /setup/callback (GitHub's redirect, which mints the
      // App private key) and POST /webhooks/github to hit the real backend — under this server they
      // land on the SPA shell and the manifest `code` is silently dropped, stranding a real App on
      // GitHub. Onboarding hitting a non-API peer is surfaced loudly by api/setup.ts's
      // SetupProtocolError rather than papered over here.
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    // Absolute base so the openapi-fetch client issues absolute URLs that
    // node's fetch + MSW can resolve/intercept (relative URLs fail in node).
    env: { VITE_API_BASE_URL: "http://localhost" },
  },
});
