import { useDemoStatus } from "../lib/demo";

// Full-width read-only-demo strip, rendered ABOVE the token gate (main.tsx) so a
// demo viewer sees it before entering the admin key — the gate would otherwise
// hide it until after auth. Shows ONLY on a confirmed demo deployment: it fails to
// no-banner while discovery is loading or errored, so a production box never
// flashes a demo strip.
export function DemoBanner() {
  const status = useDemoStatus();
  if (status !== "demo") {
    return null;
  }
  return (
    <div className="demo-banner" role="note">
      Read-only demo — seeded snapshot data; HITL decisions are disabled.
    </div>
  );
}
