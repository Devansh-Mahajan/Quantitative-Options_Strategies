import { useEffect, useMemo, useState } from "react";
import { AgGridReact } from "ag-grid-react";
import type { ColDef } from "ag-grid-community";
import { useApiResource } from "../../hooks/useApiResource";

type JsonRecord = Record<string, unknown>;

type EndpointTile = {
  label: string;
  endpoint: string;
  data: JsonRecord | null;
  error: string | null;
  metric: string;
};

function arr(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? (value.filter((item) => typeof item === "object" && item !== null) as JsonRecord[]) : [];
}

function obj(value: unknown): JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as JsonRecord) : {};
}

function text(value: unknown, fallback = "-") {
  if (value == null) return fallback;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return fallback;
}

function num(value: unknown, decimals = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(decimals);
}

function countMetric(data: JsonRecord | null, keys: string[]) {
  if (!data) return "-";
  for (const key of keys) {
    const value = data[key];
    if (Array.isArray(value)) return String(value.length);
    if (typeof value === "number") return String(value);
    if (typeof value === "object" && value !== null) return String(Object.keys(value).length);
  }
  return data.available === false ? "unavailable" : "live";
}

function EndpointHealthTile({ tile }: { tile: EndpointTile }) {
  const live = !!tile.data && !tile.error;
  return (
    <div className="border-r border-terminal-grid px-2 py-1.5 last:border-r-0">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[9px] uppercase tracking-[0.14em] text-terminal-muted">{tile.label}</span>
        <span className={live ? "text-[9px] uppercase text-terminal-emerald" : "text-[9px] uppercase text-terminal-crimson"}>
          {live ? "wired" : "down"}
        </span>
      </div>
      <div className="mt-1 text-[14px] font-black text-terminal-ink">{tile.metric}</div>
      <div className="mt-0.5 truncate text-[9px] text-terminal-muted">{tile.endpoint}</div>
    </div>
  );
}

function CompactGrid<T extends JsonRecord>({
  title,
  rows,
  columns,
  empty
}: {
  title: string;
  rows: T[];
  columns: ColDef<T>[];
  empty: string;
}) {
  return (
    <section className="panel">
      <div className="panel-title">
        <span>{title}</span>
        <span className={rows.length ? "text-terminal-emerald" : "text-terminal-muted"}>{rows.length} rows</span>
      </div>
      {rows.length ? (
        <div className="ag-theme-quartz-dark h-[260px] p-2">
          <AgGridReact rowData={rows} columnDefs={columns} rowHeight={27} headerHeight={28} suppressMovableColumns />
        </div>
      ) : (
        <div className="flex h-[260px] items-center justify-center px-4 text-center text-[11px] uppercase tracking-[0.14em] text-terminal-muted">
          {empty}
        </div>
      )}
    </section>
  );
}

function JsonPreview({ title, data, error }: { title: string; data: JsonRecord | null; error: string | null }) {
  const preview = data ? JSON.stringify(data, null, 2).slice(0, 2200) : error ?? "Endpoint unavailable";
  return (
    <section className="panel">
      <div className="panel-title">
        <span>{title}</span>
        <span className={data ? "text-terminal-emerald" : "text-terminal-crimson"}>{data ? "real payload" : "unavailable"}</span>
      </div>
      <pre className="h-[260px] overflow-auto p-2 text-[10px] leading-4 text-terminal-muted">{preview}</pre>
    </section>
  );
}

function useLogSocket(path: string) {
  const [lines, setLines] = useState<string[]>([]);
  const [state, setState] = useState("connecting");

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}${path}`);
    socket.onopen = () => setState("live");
    socket.onerror = () => setState("error");
    socket.onclose = () => setState((current) => (current === "live" ? "closed" : current));
    socket.onmessage = (event) => {
      setLines((current) => {
        const next = [...current, String(event.data)];
        return next.slice(-80);
      });
    };
    return () => socket.close();
  }, [path]);

  return { lines, state };
}

function LogsPanel() {
  const trader = useLogSocket("/ws/logs/trader");
  const risk = useLogSocket("/ws/logs/risk");

  const renderStream = (title: string, stream: ReturnType<typeof useLogSocket>) => (
    <div className="min-h-0 border-r border-terminal-grid last:border-r-0">
      <div className="flex h-7 items-center justify-between border-b border-terminal-grid px-2 text-[9px] uppercase tracking-[0.14em] text-terminal-muted">
        <span>{title}</span>
        <span className={stream.state === "live" ? "text-terminal-emerald" : "text-terminal-amber"}>{stream.state}</span>
      </div>
      <pre className="h-[232px] overflow-auto p-2 text-[10px] leading-4 text-terminal-muted">
        {stream.lines.length ? stream.lines.join("\n") : "Waiting for websocket log stream..."}
      </pre>
    </div>
  );

  return (
    <section className="panel lg:col-span-2">
      <div className="panel-title">
        <span>Live Logs</span>
        <span className="text-terminal-cyan">/ws/logs/trader /ws/logs/risk</span>
      </div>
      <div className="grid lg:grid-cols-2">{renderStream("Trader Service", trader)}{renderStream("Risk Manager", risk)}</div>
    </section>
  );
}

export function LegacyParityPanels() {
  const optionsOverview = useApiResource<JsonRecord>("/api/options/overview", 60_000);
  const optionsChain = useApiResource<JsonRecord>("/api/options/chain?limit=48", 60_000);
  const stocks = useApiResource<JsonRecord>("/api/stocks/overview", 60_000);
  const odds = useApiResource<JsonRecord>("/api/trades/odds", 60_000);
  const backtests = useApiResource<JsonRecord>("/api/backtest/list", 60_000);
  const params = useApiResource<JsonRecord>("/api/strategies/params", 60_000);
  const riskConfig = useApiResource<JsonRecord>("/api/config/risk", 60_000);
  const research = useApiResource<JsonRecord>("/api/research/desk", 60_000);
  const permutations = useApiResource<JsonRecord>("/api/permutations/strategies", 60_000);
  const trainingScripts = useApiResource<JsonRecord>("/api/training/scripts", 60_000);
  const trainingJobs = useApiResource<JsonRecord>("/api/training/jobs", 15_000);
  const mlAlpha = useApiResource<JsonRecord>("/api/ml/alpha", 30_000);
  const intelligence = useApiResource<JsonRecord>("/api/intelligence", 30_000);
  const tradeAnalysis = useApiResource<JsonRecord>("/api/trades/analysis", 60_000);
  const systemHealth = useApiResource<JsonRecord>("/api/system/health", 30_000);
  const recalibration = useApiResource<JsonRecord>("/api/system/master_recalibrate", 30_000);
  const riskEvents = useApiResource<JsonRecord>("/api/risk/events", 30_000);
  const strategyPnl = useApiResource<JsonRecord>("/api/strategies/pnl", 30_000);
  const gsSummary = useApiResource<JsonRecord>("/api/gsquant/summary", 90_000);

  const endpointTiles: EndpointTile[] = [
    { label: "Options", endpoint: "/api/options/*", data: optionsOverview.data, error: optionsOverview.error, metric: countMetric(optionsOverview.data, ["underlyings", "summary", "chains"]) },
    { label: "Chain", endpoint: "/api/options/chain", data: optionsChain.data, error: optionsChain.error, metric: countMetric(optionsChain.data, ["contracts", "chain", "rows"]) },
    { label: "Equities", endpoint: "/api/stocks/overview", data: stocks.data, error: stocks.error, metric: countMetric(stocks.data, ["symbols", "overview", "rows"]) },
    { label: "Odds", endpoint: "/api/trades/odds", data: odds.data, error: odds.error, metric: countMetric(odds.data, ["odds", "trades", "rows"]) },
    { label: "Backtests", endpoint: "/api/backtest/list", data: backtests.data, error: backtests.error, metric: countMetric(backtests.data, ["reports"]) },
    { label: "Params", endpoint: "/api/strategies/params", data: params.data, error: params.error, metric: countMetric(params.data, ["strategies"]) },
    { label: "Risk Cfg", endpoint: "/api/config/risk", data: riskConfig.data, error: riskConfig.error, metric: countMetric(riskConfig.data, ["risk"]) },
    { label: "Research", endpoint: "/api/research/desk", data: research.data, error: research.error, metric: countMetric(research.data, ["papers", "pairs", "underlyings"]) },
    { label: "Permute", endpoint: "/api/permutations/strategies", data: permutations.data, error: permutations.error, metric: countMetric(permutations.data, ["strategies"]) },
    { label: "Training", endpoint: "/api/training/*", data: trainingScripts.data, error: trainingScripts.error, metric: countMetric(trainingScripts.data, ["scripts"]) },
    { label: "ML Alpha", endpoint: "/api/ml/alpha", data: mlAlpha.data, error: mlAlpha.error, metric: countMetric(mlAlpha.data, ["signals"]) },
    { label: "Intel", endpoint: "/api/intelligence", data: intelligence.data, error: intelligence.error, metric: countMetric(intelligence.data, ["signals"]) },
    { label: "Trade Anl", endpoint: "/api/trades/analysis", data: tradeAnalysis.data, error: tradeAnalysis.error, metric: countMetric(tradeAnalysis.data, ["trades", "by_symbol", "by_strategy"]) },
    { label: "System", endpoint: "/api/system/health", data: systemHealth.data, error: systemHealth.error, metric: countMetric(systemHealth.data, ["services", "checks"]) },
    { label: "GS Quant", endpoint: "/api/gsquant/summary", data: gsSummary.data, error: gsSummary.error, metric: countMetric(gsSummary.data, ["symbols"]) },
    { label: "Risk Events", endpoint: "/api/risk/events", data: riskEvents.data, error: riskEvents.error, metric: countMetric(riskEvents.data, ["events"]) }
  ];

  const backtestRows = useMemo(() => arr(backtests.data?.reports).map((row) => ({
    run_id: text(row.run_id),
    return_pct: num(row.total_return_pct),
    sharpe: num(row.sharpe),
    max_dd: num(row.max_drawdown_pct),
    trades: text(row.trades),
    symbols: Array.isArray(row.symbols) ? row.symbols.join(",") : "-"
  })), [backtests.data]);

  const trainingRows = useMemo(() => {
    const scripts = arr(trainingScripts.data?.scripts).map((row) => ({
      id: text(row.id),
      label: text(row.label),
      status: "SCRIPT",
      exit_code: "-"
    }));
    const jobs = arr(trainingJobs.data?.jobs).map((row) => ({
      id: text(row.job_id),
      label: text(row.label ?? row.script),
      status: text(row.status),
      exit_code: text(row.exit_code)
    }));
    return [...jobs, ...scripts];
  }, [trainingJobs.data, trainingScripts.data]);

  const signalRows = useMemo(() => {
    const rows = arr(mlAlpha.data?.signals).length ? arr(mlAlpha.data?.signals) : arr(intelligence.data?.signals);
    return rows.slice(0, 40).map((row) => ({
      symbol: text(row.symbol ?? row.ticker),
      direction: text(row.direction ?? row.signal),
      alpha_score: num(row.alpha_score, 5),
      probability: num(row.probability ?? row.confidence, 4),
      model: text(row.model ?? row.source)
    }));
  }, [intelligence.data, mlAlpha.data]);

  const riskRows = useMemo(() => arr(riskEvents.data?.events).map((row) => ({
    ts: text(row.ts),
    event: text(row.event),
    detail: text(row.detail)
  })), [riskEvents.data]);

  const strategyPnlRows = useMemo(() => arr(strategyPnl.data?.strategies).map((row) => ({
    strategy: text(row.strategy),
    total_pnl: num(row.total_pnl),
    trades: text(row.trades),
    win_rate_pct: num(row.win_rate_pct),
    wl: `${text(row.wins, "0")}/${text(row.losses, "0")}`
  })), [strategyPnl.data]);

  const genericColumns: ColDef<JsonRecord>[] = [
    { field: "run_id", width: 110 },
    { field: "return_pct", width: 110, type: "rightAligned" },
    { field: "sharpe", width: 90, type: "rightAligned" },
    { field: "max_dd", width: 96, type: "rightAligned" },
    { field: "trades", width: 82, type: "rightAligned" },
    { field: "symbols", flex: 1, minWidth: 160 }
  ];

  return (
    <div className="grid gap-3">
      <section className="panel">
        <div className="panel-title">
          <span>Full Legacy Contract Coverage</span>
          <span className="text-terminal-cyan">all legacy tabs now have React endpoint hooks</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8">
          {endpointTiles.map((tile) => <EndpointHealthTile key={tile.label} tile={tile} />)}
        </div>
      </section>

      <div className="grid gap-3 xl:grid-cols-2">
        <CompactGrid
          title="Backtests"
          rows={backtestRows}
          columns={genericColumns}
          empty="No backtest reports returned by /api/backtest/list."
        />
        <CompactGrid
          title="ML Training / Jobs"
          rows={trainingRows}
          columns={[
            { field: "id", width: 118 },
            { field: "label", flex: 1, minWidth: 180 },
            { field: "status", width: 110 },
            { field: "exit_code", width: 96 }
          ]}
          empty="No training scripts or jobs returned by /api/training/*."
        />
        <CompactGrid
          title="ML Intelligence Signals"
          rows={signalRows}
          columns={[
            { field: "symbol", width: 96 },
            { field: "direction", width: 112 },
            { field: "alpha_score", width: 120, type: "rightAligned" },
            { field: "probability", width: 118, type: "rightAligned" },
            { field: "model", flex: 1, minWidth: 140 }
          ]}
          empty="No ML alpha or intelligence signals returned."
        />
        <CompactGrid
          title="Strategy PnL Attribution"
          rows={strategyPnlRows}
          columns={[
            { field: "strategy", flex: 1, minWidth: 160 },
            { field: "total_pnl", width: 118, type: "rightAligned" },
            { field: "trades", width: 88, type: "rightAligned" },
            { field: "win_rate_pct", width: 118, type: "rightAligned" },
            { field: "wl", width: 90 }
          ]}
          empty="No closed strategy PnL rows returned by /api/strategies/pnl."
        />
        <CompactGrid
          title="Risk Events"
          rows={riskRows}
          columns={[
            { field: "ts", width: 190 },
            { field: "event", width: 150 },
            { field: "detail", flex: 1, minWidth: 240 }
          ]}
          empty="No risk events returned by /api/risk/events."
        />
        <LogsPanel />
        <JsonPreview title="Trade Analysis Payload" data={tradeAnalysis.data} error={tradeAnalysis.error} />
        <JsonPreview title="GS Quant Summary Payload" data={gsSummary.data} error={gsSummary.error} />
        <JsonPreview title="System / Command Center Payload" data={systemHealth.data ?? recalibration.data} error={systemHealth.error ?? recalibration.error} />
        <JsonPreview title="Options / Equities Payload" data={obj({ options: optionsOverview.data, stocks: stocks.data, odds: odds.data })} error={optionsOverview.error ?? stocks.error ?? odds.error} />
      </div>
    </div>
  );
}
