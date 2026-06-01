import { useFilters } from "../state/filters";

// Quiet chrome. The cmdbar is a client-side row filter (no backend search in
// V1). The Eval toggle drives the `is_eval` query param. The mockup's
// appearance/accent switcher is intentionally NOT replicated — one theme.
export function Topbar() {
  const search = useFilters((s) => s.search);
  const setSearch = useFilters((s) => s.setSearch);
  const includeEval = useFilters((s) => s.includeEval);
  const setIncludeEval = useFilters((s) => s.setIncludeEval);

  return (
    <header className="topbar">
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
      <div className="topbar-right">
        <button
          type="button"
          className="toggle"
          aria-pressed={includeEval}
          onClick={() => setIncludeEval(!includeEval)}
        >
          <span className="switch" aria-hidden="true" />
          <span>Eval</span>
        </button>
        <div className="avatar" title="operator">
          op
        </div>
      </div>
    </header>
  );
}
