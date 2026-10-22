import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "react-router/dom";

import { adoptTokenFromUrlFragment } from "./auth/token";
import { TokenGate } from "./auth/TokenGate";
import { DemoBanner } from "./components/DemoBanner";
import { router } from "./router";
import "./theme.css";

// Adopt and remove a shared-link token before rendering the auth gate.
adoptTokenFromUrlFragment();

// Queries default to no automatic retries; live views schedule their own
// refetches. useDemoStatus overrides this with bounded boot-time retries.
// Mutations keep their default no-retry policy; decision retries use the
// accepted payload retained by ReviewDetail.
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
      <DemoBanner />
      <TokenGate>
        <RouterProvider router={router} />
      </TokenGate>
    </QueryClientProvider>
  </StrictMode>,
);
