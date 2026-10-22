import { useEffect } from "react";

import { useDemoStatus } from "../lib/demo";

// main.tsx places this above the token gate so demo mode is visible before auth.
// Unknown deployment status shows no banner; only confirmed demo mode shows it.
export function DemoBanner() {
  const demoStatus = useDemoStatus();
  // Set the tab title on confirmed demo discovery; the status query is cached.
  useEffect(() => {
    if (demoStatus === "demo") {
      document.title = "Outrider — Read-Only Demo";
    }
  }, [demoStatus]);
  if (demoStatus !== "demo") {
    return null;
  }
  return (
    <div className="demo-banner" role="note">
      Read-only demo — seeded snapshot data; HITL decisions are disabled.
    </div>
  );
}
