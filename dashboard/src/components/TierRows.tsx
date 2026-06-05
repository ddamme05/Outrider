// Findings distribution by EVIDENCE TIER — three horizontal magnitude tracks
// (mockup .tier-rows) + the proof-boundary footnote. Tier is read off the deduped
// representative finding (display only; the endpoint computes the representative,
// per DECISIONS#039). Fill width = share of total findings (count/total), matching
// the severity total so the two distributions reconcile. The footnote renders the
// proof mapping verbatim — it's the proof-boundary contract, not decoration.
import { TIER_ORDER } from "../lib/metrics";

const TIER_PROOF: Record<string, string> = {
  observed: "query_match_id",
  inferred: "trace_path",
  judged: "model interpretation",
};

export function TierRows({ distribution }: { distribution: Record<string, number> }) {
  const rows = TIER_ORDER.map((tier) => ({ tier, count: distribution[tier] ?? 0 }));
  const total = rows.reduce((s, r) => s + r.count, 0);

  return (
    <div className="tier-block">
      <div className="tier-rows">
        {rows.map((r) => {
          const pct = total > 0 ? (r.count / total) * 100 : 0;
          return (
            <div className="tier-row" key={r.tier}>
              <span className="tier-name">
                <span className="sw" style={{ background: `var(--tier-${r.tier})` }} aria-hidden="true" />
                {r.tier}
              </span>
              <span className="tier-track">
                <span
                  className="tier-fill"
                  style={{ width: `${pct}%`, background: `var(--tier-${r.tier})` }}
                />
              </span>
              <span className="tier-val">{r.count}</span>
            </div>
          );
        })}
      </div>
      <div className="pipe-note tier-note">
        {TIER_ORDER.map((tier, i) => (
          <span key={tier}>
            {tier} → <code style={{ color: `var(--tier-${tier})` }}>{TIER_PROOF[tier]}</code>
            {i < TIER_ORDER.length - 1 ? " · " : ""}
          </span>
        ))}
      </div>
    </div>
  );
}
