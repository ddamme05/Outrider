// Display evidence-tier counts and their proof artifacts. The API chooses
// representative findings (DECISIONS#039); each track shows its share of that total.
import { TIER_ORDER } from "../lib/metrics";

const TIER_PROOF: Record<string, string> = {
  observed: "query_match_id",
  inferred: "trace_path",
  judged: "model interpretation",
};

export function TierRows({ distribution }: { distribution: Record<string, number> }) {
  const tierRows = TIER_ORDER.map((tier) => ({ tier, count: distribution[tier] ?? 0 }));
  const total = tierRows.reduce((count, tierRow) => count + tierRow.count, 0);

  return (
    <div className="tier-block">
      <div className="tier-rows">
        {tierRows.map((tierRow) => {
          const sharePercent = total > 0 ? (tierRow.count / total) * 100 : 0;
          return (
            <div className="tier-row" key={tierRow.tier}>
              <span className="tier-name">
                <span className="sw" style={{ background: `var(--tier-${tierRow.tier})` }} aria-hidden="true" />
                {tierRow.tier}
              </span>
              <span className="tier-track">
                <span
                  className="tier-fill"
                  style={{ width: `${sharePercent}%`, background: `var(--tier-${tierRow.tier})` }}
                />
              </span>
              <span className="tier-val">{tierRow.count}</span>
            </div>
          );
        })}
      </div>
      <div className="pipe-note tier-note">
        {TIER_ORDER.map((tier, index) => (
          <span key={tier}>
            {tier} → <code style={{ color: `var(--tier-${tier})` }}>{TIER_PROOF[tier]}</code>
            {index < TIER_ORDER.length - 1 ? " · " : ""}
          </span>
        ))}
      </div>
    </div>
  );
}
