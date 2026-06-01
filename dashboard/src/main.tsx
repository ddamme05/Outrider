import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "react-router/dom";

import { TokenGate } from "./auth/TokenGate";
import { router } from "./router";
import "./theme.css";

// No query retries: every read view already polls/refetches (2s), so a failed
// read should render its "unavailable" state once rather than amplify a 5xx into
// 3 retries × every poll (e.g. a replay 500 flooding the server log). Mutations
// default to no retry already — a HITL decide must never be re-fired casually.
const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false } },
});

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("#root element not found");
}

createRoot(rootElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <TokenGate>
        <RouterProvider router={router} />
      </TokenGate>
    </QueryClientProvider>
  </StrictMode>,
);
