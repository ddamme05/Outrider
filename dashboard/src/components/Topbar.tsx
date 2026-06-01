import { useLocation } from "react-router";

import { useFilters } from "../state/filters";

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

  return (
    <header className="topbar">
      <div className="tb-title">
        <span className="tb-kicker">
          <span className="tb-tick" aria-hidden="true" />
          OUTRIDER
        </span>
        <h1>{screenTitle(pathname)}</h1>
      </div>
      <div className="topbar-right">
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
