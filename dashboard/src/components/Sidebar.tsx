import { NavLink } from "react-router";

import { useFilters } from "../state/filters";
import { useNav } from "../state/nav";

// Signal sidebar: static org/brand mark (no org-switcher — single-tenant admin
// scope, so a dropdown would imply org-switching we don't have), icon nav, and a
// footer eval-data toggle wired to the `includeEval` filter. Nav is Overview +
// Reviews only — Audit and Anomalies are omitted (no backing endpoint; an empty
// nav item would imply hidden data). No theme switcher.
function navClass({ isActive }: { isActive: boolean }): string {
  return `nav-item ${isActive ? "active" : ""}`;
}

export function Sidebar() {
  const includeEval = useFilters((s) => s.includeEval);
  const setIncludeEval = useFilters((s) => s.setIncludeEval);
  // Close the mobile drawer after a nav tap (no-op on desktop). `setOpen` is a
  // stable store action, so the selector reference is stable across renders.
  const setNavOpen = useNav((s) => s.setOpen);
  const closeNav = (): void => setNavOpen(false);

  return (
    <nav className="sidebar">
      <div className="org">
        <span className="org-mark" aria-hidden="true">
          <span className="org-initial">O</span>
        </span>
        <span>
          <span className="org-name">Outrider</span>
          <span className="org-sub" style={{ display: "block" }}>
            review dashboard
          </span>
        </span>
      </div>

      <div className="sb-divider" aria-hidden="true" />

      <div className="nav-label">Workspace</div>
      <NavLink to="/" end className={navClass} onClick={closeNav}>
        <svg className="ico" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <rect x="1.5" y="1.5" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.3" />
          <rect x="9" y="1.5" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.3" />
          <rect x="1.5" y="9" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.3" />
          <rect x="9" y="9" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.3" />
        </svg>
        Overview
      </NavLink>
      <NavLink to="/reviews" className={navClass} onClick={closeNav}>
        <svg className="ico" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M2 3.5h12M2 8h12M2 12.5h7" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        </svg>
        Reviews
      </NavLink>

      <div className="sb-spacer" />

      <div className="sb-footer">
        <div className="eval-box">
          <div className="eval-row">
            <button
              type="button"
              className="switch"
              role="switch"
              aria-checked={includeEval}
              aria-label="Toggle eval data visibility"
              onClick={() => setIncludeEval(!includeEval)}
            >
              <span className="knob" aria-hidden="true" />
            </button>
            <div>
              <div className="eval-label">Show eval data</div>
              <div className="eval-hint">{includeEval ? "Visible" : "Hidden by default"}</div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
