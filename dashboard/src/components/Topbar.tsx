import { useLocation } from "react-router";

import { $api } from "../api/client";
import { useDemoStatus } from "../lib/demo";
import { replayRate } from "../lib/metrics";
import { useFilters } from "../state/filters";
import { useNav } from "../state/nav";

// Route title, drawer control, shared review filter and replay health.
function screenTitle(pathname: string): string {
  if (pathname === "/") return "Overview";
  if (pathname === "/reviews") return "Reviews";
  if (pathname.startsWith("/reviews/")) return "Review";
  return "Outrider";
}

export function Topbar() {
  const { pathname } = useLocation();
  const search = useFilters((filterState) => filterState.search);
  const setSearch = useFilters((filterState) => filterState.setSearch);
  const navOpen = useNav((navState) => navState.open);
  const toggleNav = useNav((navState) => navState.toggle);
  // Match the banner: only confirmed demo discovery gets a label.
  const isDemo = useDemoStatus() === "demo";

  // Poll global 30d replay health. Absent data or zero verdicts produce no rate;
  // a refetch can retain previously loaded data.
  const replayQuery = $api.useQuery(
    "get",
    "/api/metrics/replay",
    { params: { query: { window: "30d" } } },
    { refetchInterval: 30000 },
  );
  const replayCounts = replayQuery.data?.deltas.current;
  const replayPercent = replayCounts
    ? replayRate(replayCounts.equivalent, replayCounts.total)
    : null;
  // Name the recorded window end when metrics use an anchored snapshot.
  const replayWindowLabel = replayQuery.data?.anchored
    ? `30d ending ${new Date(replayQuery.data.window_end).toLocaleDateString()}`
    : "30d";

  return (
    <header className="topbar">
      {/* CSS shows the drawer control on mobile; aria-expanded follows store state. */}
      <button
        type="button"
        className="nav-toggle"
        aria-label="Open navigation"
        aria-expanded={navOpen}
        onClick={toggleNav}
      >
        <svg viewBox="0 0 18 18" fill="none" aria-hidden="true">
          <path
            d="M2.5 4.5h13M2.5 9h13M2.5 13.5h13"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      </button>
      <div className="tb-title">
        <span className="tb-kicker">
          <span className="tb-tick" aria-hidden="true" />
          OUTRIDER
          {isDemo ? <span className="tb-demo">read-only demo</span> : null}
        </span>
        <h1>{screenTitle(pathname)}</h1>
      </div>
      <div className="topbar-right">
        {replayCounts && replayPercent !== null ? (
          <span
            className="pill replay-pill"
            title={`replay-equivalence over ${replayWindowLabel} — ${replayCounts.equivalent}/${replayCounts.total} verified`}
          >
            {replayPercent.toFixed(0)}% replay
          </span>
        ) : null}
        <div className="cmdbar" role="search">
          <span className="prompt" aria-hidden="true">
            ›
          </span>
          <input
            type="text"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            aria-label="Filter reviews"
            placeholder="filter reviews…"
          />
        </div>
      </div>
    </header>
  );
}
