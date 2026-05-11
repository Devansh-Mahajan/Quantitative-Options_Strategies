type ParityStatus = "migrated" | "partial" | "pending" | "legacy-only";

type ParityRow = {
  legacyTab: string;
  production: "protected";
  react: ParityStatus;
  endpoints: string;
};

const parityRows: ParityRow[] = [
  { legacyTab: "Live Perf", production: "protected", react: "partial", endpoints: "/api/equity /api/positions /api/risk/engine /api/strategies/status" },
  { legacyTab: "Options", production: "protected", react: "partial", endpoints: "/api/options/* /api/model_zoo/vae_vol/*" },
  { legacyTab: "Equities", production: "protected", react: "partial", endpoints: "/api/stocks/overview /api/stocks/*" },
  { legacyTab: "Trade Odds", production: "protected", react: "partial", endpoints: "/api/trades/odds" },
  { legacyTab: "Backtests", production: "protected", react: "partial", endpoints: "/api/backtest/list /api/backtest/*" },
  { legacyTab: "Parameters", production: "protected", react: "partial", endpoints: "/api/strategies/status /api/strategies/params" },
  { legacyTab: "Risk Engine", production: "protected", react: "partial", endpoints: "/api/risk/engine /api/risk/snapshot /api/risk/events" },
  { legacyTab: "Sim Lab", production: "protected", react: "partial", endpoints: "/api/equity/analytics plus labeled deterministic research projections" },
  { legacyTab: "Strategy Lab", production: "protected", react: "partial", endpoints: "/api/strategies/* /api/permutations/* /api/backtest/*" },
  { legacyTab: "Forward Test", production: "protected", react: "partial", endpoints: "/api/model_zoo/tft/* /api/equity/analytics" },
  { legacyTab: "ML Training", production: "protected", react: "partial", endpoints: "/api/training/* /api/model_zoo/*" },
  { legacyTab: "Intelligence", production: "protected", react: "partial", endpoints: "/api/intelligence /api/model_zoo/hmm/* /api/model_zoo/xgb/*" },
  { legacyTab: "Trade Analysis", production: "protected", react: "partial", endpoints: "/api/trades/analysis /api/trades" },
  { legacyTab: "Command Center", production: "protected", react: "partial", endpoints: "/api/health /api/system/* /api/strategies/* /api/risk/*" },
  { legacyTab: "GS Quant", production: "protected", react: "partial", endpoints: "/api/gsquant/*" },
  { legacyTab: "Sys Logs", production: "protected", react: "partial", endpoints: "/ws/logs/trader /ws/logs/risk" }
];

function statusClass(status: ParityStatus) {
  if (status === "migrated") return "text-terminal-emerald";
  if (status === "partial") return "text-terminal-cyan";
  if (status === "legacy-only") return "text-terminal-amber";
  return "text-terminal-muted";
}

export function FeatureParityMatrix() {
  const partialCount = parityRows.filter((row) => row.react === "partial" || row.react === "migrated").length;

  return (
    <section className="panel">
      <div className="panel-title">
        <span>Migration Feature Parity Matrix</span>
        <span className="text-terminal-amber">
          {partialCount}/{parityRows.length} endpoint hooks started - legacy stays production
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[10px]">
          <thead>
            <tr className="border-b border-terminal-grid bg-black text-left uppercase tracking-[0.14em] text-terminal-muted">
              <th className="px-2 py-2">Legacy Tab</th>
              <th className="px-2 py-2">Production State</th>
              <th className="px-2 py-2">React State</th>
              <th className="px-2 py-2">Primary Contract</th>
            </tr>
          </thead>
          <tbody>
            {parityRows.map((row) => (
              <tr key={row.legacyTab} className="border-b border-terminal-grid odd:bg-white/[0.015]">
                <td className="px-2 py-1.5 font-bold text-terminal-ink">{row.legacyTab}</td>
                <td className="px-2 py-1.5 uppercase text-terminal-emerald">{row.production}</td>
                <td className={`px-2 py-1.5 uppercase ${statusClass(row.react)}`}>{row.react}</td>
                <td className="px-2 py-1.5 text-terminal-muted">{row.endpoints}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="border-t border-terminal-grid px-2 py-2 text-[10px] uppercase tracking-[0.13em] text-terminal-muted">
        Release rule: partial means endpoint coverage exists, not full UI parity. Do not point production traffic at React until every tab is upgraded from partial to migrated and
        `verify:legacy` still passes for `dashboard/static/index.html`.
      </div>
    </section>
  );
}
