import { AgGridReact } from "ag-grid-react";
import type { ColDef } from "ag-grid-community";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { useApiResource } from "../../hooks/useApiResource";
import type {
  EquityResponse,
  PositionsResponse,
  RiskEngineResponse,
  StrategiesStatusResponse
} from "../../lib/api";

const GRID = "#1A1A1A";

type PositionGridRow = {
  symbol: string;
  asset: string;
  side: string;
  qty: number;
  entry: number;
  mark: number;
  marketValue: number;
  unrealizedPnl: number;
  source: string;
};

type StrategyGridRow = {
  strategy: string;
  enabled: string;
  params: number;
  riskMode: string;
};

function money(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "-";
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
}

function pct(value: number | null | undefined, decimals = 2) {
  if (value == null || Number.isNaN(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(decimals)}%`;
}

function num(value: number | null | undefined, decimals = 2) {
  if (value == null || Number.isNaN(value)) return "-";
  return value.toFixed(decimals);
}

function tone(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "text-terminal-muted";
  if (value > 0) return "text-terminal-emerald";
  if (value < 0) return "text-terminal-crimson";
  return "text-terminal-muted";
}

function statusTone(status?: string) {
  const normalized = String(status ?? "").toUpperCase();
  if (normalized.includes("HALT") || normalized.includes("KILL")) return "text-terminal-crimson";
  if (normalized.includes("STALE") || normalized.includes("WARN")) return "text-terminal-amber";
  if (normalized.includes("LIVE") || normalized.includes("OK")) return "text-terminal-emerald";
  return "text-terminal-muted";
}

function MetricCell({
  label,
  value,
  className = "text-terminal-ink",
  sub
}: {
  label: string;
  value: string;
  className?: string;
  sub?: string;
}) {
  return (
    <div className="metric-cell min-w-[112px]">
      <div className="text-[9px] uppercase tracking-[0.15em] text-terminal-muted">{label}</div>
      <div className={`mt-1 text-[14px] font-black ${className}`}>{value}</div>
      {sub ? <div className="mt-0.5 truncate text-[9px] text-terminal-muted">{sub}</div> : null}
    </div>
  );
}

function equityChartRows(data: EquityResponse | null) {
  const timestamps = data?.timestamps ?? [];
  return timestamps.map((timestamp, index) => ({
    timestamp,
    equity: Number(data?.equity?.[index] ?? 0),
    drawdown: Number(data?.drawdown?.[index] ?? 0) * 100
  }));
}

function positionRows(data: PositionsResponse | null): PositionGridRow[] {
  return (data?.positions ?? []).map((position) => {
    const qty = Number(position.quantity ?? position.qty ?? 0);
    const mark = Number(position.mark_price ?? position.current_price ?? position.price ?? 0);
    const entry = Number(position.entry_price ?? position.price ?? 0);
    const marketValue = Number(position.market_value ?? qty * mark);
    return {
      symbol: position.symbol ?? position.broker_symbol ?? "-",
      asset: position.asset_class ?? position.market ?? "-",
      side: position.side ?? (qty < 0 ? "SHORT" : "LONG"),
      qty,
      entry,
      mark,
      marketValue,
      unrealizedPnl: Number(position.unrealized_pnl ?? position.pnl ?? 0),
      source: position.source ?? data?.source ?? "broker"
    };
  });
}

function strategyRows(data: StrategiesStatusResponse | null): StrategyGridRow[] {
  return Object.entries(data?.strategies ?? {}).map(([strategy, status]) => ({
    strategy,
    enabled: status.enabled ? "ENABLED" : "DISABLED",
    params: Object.keys(status.params ?? {}).length,
    riskMode: data?.kill_switch ? "GLOBAL KILL" : "NORMAL"
  }));
}

function LiveEquityChart({ equity }: { equity: EquityResponse | null }) {
  const rows = equityChartRows(equity);

  return (
    <section className="panel lg:col-span-2">
      <div className="panel-title">
        <span>Live Equity / Drawdown</span>
        <span className={rows.length ? "text-terminal-emerald" : "text-terminal-crimson"}>
          {rows.length ? "/api/equity?window=1w" : "no equity samples"}
        </span>
      </div>
      {rows.length ? (
        <div className="h-[292px] px-1 py-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={rows} margin={{ top: 8, right: 14, bottom: 0, left: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis
                dataKey="timestamp"
                minTickGap={42}
                tick={{ fill: "#6B7280", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: GRID }}
              />
              <YAxis
                yAxisId="equity"
                tick={{ fill: "#6B7280", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: GRID }}
                tickFormatter={(value) => money(Number(value)).replace(".00", "")}
              />
              <YAxis
                yAxisId="drawdown"
                orientation="right"
                tick={{ fill: "#6B7280", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: GRID }}
                tickFormatter={(value) => `${Number(value).toFixed(1)}%`}
              />
              <Tooltip
                contentStyle={{ background: "#0F0F0F", border: `1px solid ${GRID}`, fontFamily: "JetBrains Mono" }}
                formatter={(value, name) => {
                  if (name === "equity") return [money(Number(value)), "Equity"];
                  return [`${Number(value).toFixed(2)}%`, "Drawdown"];
                }}
              />
              <Area
                yAxisId="equity"
                dataKey="equity"
                stroke="#00E5FF"
                fill="rgba(0,229,255,0.12)"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                yAxisId="drawdown"
                dataKey="drawdown"
                stroke="#FF335F"
                dot={false}
                strokeWidth={1.4}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="flex h-[292px] items-center justify-center px-4 text-center text-[11px] uppercase tracking-[0.14em] text-terminal-muted">
          Waiting for live equity data. This React panel does not fabricate portfolio history.
        </div>
      )}
    </section>
  );
}

function RiskStrip({ risk }: { risk: RiskEngineResponse | null }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <span>Risk Engine</span>
        <span className={risk?.kill_switch_active ? "text-terminal-crimson" : "text-terminal-emerald"}>
          {risk?.kill_switch_active ? "KILL ACTIVE" : "KILL CLEAR"}
        </span>
      </div>
      <div className="grid grid-cols-2 border-b border-terminal-grid xl:grid-cols-4">
        <MetricCell label="Status" value={risk?.trading_status ?? "-"} className={statusTone(risk?.trading_status)} />
        <MetricCell label="Risk Score" value={num(risk?.risk_score, 3)} />
        <MetricCell label="VaR / Eq" value={pct(risk?.var_pct_equity, 2)} className="text-terminal-amber" />
        <MetricCell label="CVaR / Eq" value={pct(risk?.cvar_pct_equity, 2)} className="text-terminal-crimson" />
        <MetricCell label="Stress / Eq" value={pct(risk?.stress_pct_equity, 2)} className="text-terminal-crimson" />
        <MetricCell label="Gross Exp" value={pct(risk?.gross_exposure_pct_equity, 1)} />
        <MetricCell label="Open Pos" value={num(risk?.open_positions, 0)} />
        <MetricCell label="VIX" value={num(risk?.vix, 2)} sub={risk?.macro_regime ?? undefined} />
      </div>
      <div className="grid grid-cols-2 xl:grid-cols-4">
        <MetricCell label="Delta" value={num(risk?.portfolio_delta, 2)} />
        <MetricCell label="Gamma" value={num(risk?.portfolio_gamma, 4)} />
        <MetricCell label="Vega" value={num(risk?.portfolio_vega, 2)} />
        <MetricCell label="Theta" value={num(risk?.portfolio_theta, 2)} />
      </div>
      {(risk?.breaches?.length || risk?.hard_kill_reasons?.length) ? (
        <div className="border-t border-terminal-grid px-2 py-2 text-[10px] uppercase tracking-[0.12em] text-terminal-crimson">
          {[...(risk.breaches ?? []), ...(risk.hard_kill_reasons ?? [])].join(" / ")}
        </div>
      ) : null}
    </section>
  );
}

function PositionsGrid({ positions, error }: { positions: PositionsResponse | null; error: string | null }) {
  const rows = positionRows(positions);
  const columns: ColDef<PositionGridRow>[] = [
    { field: "symbol", width: 104 },
    { field: "asset", width: 84 },
    { field: "side", width: 82 },
    { field: "qty", width: 90, type: "rightAligned", valueFormatter: ({ value }) => num(value, 4) },
    { field: "entry", width: 96, type: "rightAligned", valueFormatter: ({ value }) => num(value, 3) },
    { field: "mark", width: 96, type: "rightAligned", valueFormatter: ({ value }) => num(value, 3) },
    { field: "marketValue", width: 124, type: "rightAligned", valueFormatter: ({ value }) => money(value) },
    { field: "unrealizedPnl", width: 122, type: "rightAligned", valueFormatter: ({ value }) => money(value) },
    { field: "source", width: 112 }
  ];

  return (
    <section className="panel">
      <div className="panel-title">
        <span>Open Positions</span>
        <span className={rows.length ? "text-terminal-emerald" : "text-terminal-muted"}>
          {rows.length ? `${rows.length} live rows` : "flat / unavailable"}
        </span>
      </div>
      {rows.length ? (
        <div className="ag-theme-quartz-dark h-[282px] p-2">
          <AgGridReact rowData={rows} columnDefs={columns} rowHeight={27} headerHeight={28} suppressMovableColumns />
        </div>
      ) : (
        <div className="flex h-[282px] items-center justify-center px-4 text-center text-[11px] uppercase tracking-[0.14em] text-terminal-muted">
          {error ? `Positions unavailable: ${error}` : "No open positions returned by /api/positions."}
        </div>
      )}
    </section>
  );
}

function StrategyGrid({ strategies, error }: { strategies: StrategiesStatusResponse | null; error: string | null }) {
  const rows = strategyRows(strategies);
  const columns: ColDef<StrategyGridRow>[] = [
    { field: "strategy", flex: 1, minWidth: 160 },
    { field: "enabled", width: 104 },
    { field: "params", width: 90, type: "rightAligned" },
    { field: "riskMode", width: 118 }
  ];

  return (
    <section className="panel">
      <div className="panel-title">
        <span>Strategy Control Plane</span>
        <span className={strategies?.kill_switch ? "text-terminal-crimson" : "text-terminal-emerald"}>
          {strategies?.kill_switch ? "GLOBAL KILL" : "LIVE CONFIG"}
        </span>
      </div>
      {rows.length ? (
        <div className="ag-theme-quartz-dark h-[282px] p-2">
          <AgGridReact rowData={rows} columnDefs={columns} rowHeight={27} headerHeight={28} suppressMovableColumns />
        </div>
      ) : (
        <div className="flex h-[282px] items-center justify-center px-4 text-center text-[11px] uppercase tracking-[0.14em] text-terminal-muted">
          {error ? `Strategy status unavailable: ${error}` : "No strategy status returned by /api/strategies/status."}
        </div>
      )}
    </section>
  );
}

export function LiveOperationsPanel() {
  const equity = useApiResource<EquityResponse>("/api/equity?window=1w", 30_000);
  const positions = useApiResource<PositionsResponse>("/api/positions", 15_000);
  const risk = useApiResource<RiskEngineResponse>("/api/risk/engine", 15_000);
  const strategies = useApiResource<StrategiesStatusResponse>("/api/strategies/status", 20_000);

  const metrics = equity.data?.metrics;
  const dailyPnl = risk.data?.daily_pnl_pct;

  return (
    <div className="grid gap-3">
      <section className="panel">
        <div className="panel-title">
          <span>Production Live Ops Mirror</span>
          <span className="text-terminal-cyan">legacy dashboard remains active</span>
        </div>
        <div className="grid grid-cols-2 border-b border-terminal-grid md:grid-cols-4 xl:grid-cols-8">
          <MetricCell label="Equity" value={money(metrics?.current_equity ?? risk.data?.total_equity)} />
          <MetricCell label="Day PnL" value={pct(dailyPnl, 2)} className={tone(dailyPnl)} />
          <MetricCell label="Return" value={pct(metrics?.total_return_pct, 2)} className={tone(metrics?.total_return_pct)} />
          <MetricCell label="Sharpe" value={num(metrics?.sharpe, 2)} />
          <MetricCell label="Sortino" value={num(metrics?.sortino, 2)} />
          <MetricCell label="Calmar" value={num(metrics?.calmar, 2)} />
          <MetricCell label="Max DD" value={pct(metrics?.max_drawdown_pct, 2)} className="text-terminal-crimson" />
          <MetricCell label="Ann Vol" value={pct(metrics?.volatility_pct, 2)} />
        </div>
        <div className="px-2 py-1.5 text-[10px] uppercase tracking-[0.14em] text-terminal-muted">
          Real endpoint mirror only: `/api/equity`, `/api/positions`, `/api/risk/engine`, `/api/strategies/status`.
          {equity.error ? <span className="ml-2 text-terminal-crimson">equity: {equity.error}</span> : null}
        </div>
      </section>

      <div className="grid gap-3 xl:grid-cols-3">
        <LiveEquityChart equity={equity.data} />
        <RiskStrip risk={risk.data} />
      </div>

      <div className="grid gap-3 xl:grid-cols-2">
        <PositionsGrid positions={positions.data} error={positions.error} />
        <StrategyGrid strategies={strategies.data} error={strategies.error} />
      </div>
    </div>
  );
}
