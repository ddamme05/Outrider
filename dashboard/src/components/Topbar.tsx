import { useLocation } from "react-router";

import { $api } from "../api/client";
import { replayRate } from "../lib/metrics";
import { useFilters } from "../state/filters";
import { useNav } from "../state/nav";

// Signal topbar: a mono kicker + tick, the screen title (derived from the route),
// and a client-side search field (no backend search in V1). The eval toggle moved
// to the sidebar footer; no avatar, no theme switcher.
function screenTitle(pathname: string): string {
  if (pathname === "/") return "Overview";
  if (pathname === "/reviews") return "Reviews";
  if (pathname.startsWith("/reviews/")) return "Review";
  return "Outrider";
}

export function Topbar() {
  const { pathname } = useLocation();
  const search = useFilters((s) => s.search);
  const setSearch = useFilters((s) => s.setSearch);
  const navOpen = useNav((s) => s.open);
  const toggleNav = useNav((s) => s.toggle);

  // Global replay-equivalence health pill: 30d production scope, slow poll (it's on
  // every screen and the verdict stream changes slowly). Fails CLOSED — no pill while
  // loading, on error, or when no reviews are verdicted yet (replayRate → null), never
  // a fabricated 0%.
  const replay = $api.useQuery(
    "get",
    "/api/metrics/replay",
    { params: { query: { window: "30d" } } },
    { refetchInterval: 30000 },
  );
  const rc = replay.data?.deltas.current;
  const rate = rc ? replayRate(rc.equivalent, rc.total) : null;
  // Demo-snapshot anchoring: the title never implies live recency on a demo box.
  const pillWindow = replay.data?.anchored
    ? `30d ending ${new Date(replay.data.window_end).toLocaleDateString()}`
    : "30d";

  return (
    <header className="topbar">
      {/* Mobile-only drawer toggle (hidden on desktop via CSS). aria-expanded
          reflects the live `useNav` state so assistive tech tracks the drawer. */}
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
        </span>
        <h1>{screenTitle(pathname)}</h1>
      </div>
      <div className="topbar-right">
        {rc && rate !== null ? (
          <span
            className="pill replay-pill"
            title={`replay-equivalence over ${pillWindow} — ${rc.equivalent}/${rc.total} verified`}
          >
            {rate.toFixed(0)}% replay
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
