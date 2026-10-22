import { $api } from "../api/client";

export type DemoStatus = "loading" | "demo" | "production";

// Read the public deployment flag, cached for this client session. Startup
// failures get bounded retries with backoff (deploy/README.md).
// Only confirmed demo mode shows the banner; only confirmed production mode
// enables approval submission. Unknown or malformed flags remain "loading".
export function useDemoStatus(): DemoStatus {
  const metadataQuery = $api.useQuery(
    "get",
    "/api/meta",
    {},
    {
      staleTime: Infinity,
      retry: 5,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
    },
  );
  // Compile-time API types do not validate responses. Require an exact boolean
  // before enabling a production action.
  if (metadataQuery.data?.demo_mode === true) return "demo";
  if (metadataQuery.data?.demo_mode === false) return "production";
  return "loading";
}
