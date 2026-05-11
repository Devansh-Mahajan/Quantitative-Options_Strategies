import { AgGridReact } from "ag-grid-react";
import type { ColDef } from "ag-grid-community";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { depthCurve } from "../../data/microstructureMockData";
import type { ExecutionLedgerFill, ExecutionLedgerResponse } from "../../lib/api";
import { useApiResource } from "../../hooks/useApiResource";

const GRID = "#1A1A1A";

function DepthChart() {
  return (
    <section className="panel min-h-[320px]">
      <div className="panel-title">
        <span>L3 Cumulative Depth</span>
        <span className="text-terminal-amber">synthetic depth until L3 feed is connected</span>
      </div>
      <div className="h-[276px] px-1 py-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={depthCurve} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="price" tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: GRID }} />
            <YAxis tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: GRID }} />
            <Tooltip contentStyle={{ background: "#0F0F0F", border: `1px solid ${GRID}`, fontFamily: "JetBrains Mono" }} />
            <Area dataKey="bidDepth" stroke="#00FF88" fill="rgba(0,255,136,0.16)" type="stepAfter" dot={false} isAnimationActive={false} />
            <Area dataKey="askDepth" stroke="#FF335F" fill="rgba(255,51,95,0.14)" type="stepBefore" dot={false} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function toFillPoints(liveFills: ExecutionLedgerFill[]) {
  return liveFills
    .filter((fill) => fill.is_filled || fill.partial_fill)
    .map((fill, index) => ({
      time: index,
      latencyMs: Number(fill.exec_score != null ? Math.max(0.15, (1 - fill.exec_score) * 8) : 0),
      size: Number(fill.qty ?? 0),
      slippageBps: 0,
      price: Number(fill.fill_price ?? 0)
    }))
    .filter((fill) => fill.size > 0 && fill.price > 0);
}

function LatencyScatter({ liveFills }: { liveFills: ExecutionLedgerFill[] }) {
  const plottedFills = toFillPoints(liveFills);

  return (
    <section className="panel min-h-[320px]">
      <div className="panel-title">
        <span>Latency / Execution Scatter</span>
        <span className={plottedFills.length ? "text-terminal-emerald" : "text-terminal-crimson"}>
          {plottedFills.length ? "broker-confirmed fills" : "no confirmed fills"}
        </span>
      </div>
      {plottedFills.length ? (
        <div className="h-[276px] px-1 py-2">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="time" type="number" name="sequence" tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: GRID }} />
              <YAxis dataKey="latencyMs" type="number" name="latency proxy" tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: GRID }} />
              <Tooltip contentStyle={{ background: "#0F0F0F", border: `1px solid ${GRID}`, fontFamily: "JetBrains Mono" }} />
              <Scatter
                data={plottedFills.map((fill) => ({ ...fill, z: Math.max(16, fill.size / 18) }))}
                fill="#00E5FF"
                isAnimationActive={false}
              />
              <Line dataKey="latencyMs" stroke="#F5C542" dot={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="flex h-[276px] items-center justify-center px-4 text-center text-[11px] uppercase tracking-[0.14em] text-terminal-muted">
          Execution scatter is intentionally empty until `/api/execution/ledger` returns broker-confirmed filled or partial-filled orders.
        </div>
      )}
    </section>
  );
}

type ExecutionGridRow = {
  id: string;
  symbol: string;
  side: string;
  size: number;
  fill: number;
  status: string;
  source: string;
  notional: number;
  filledAt: string;
};

function rowsFromLedger(fills: ExecutionLedgerFill[]): ExecutionGridRow[] {
  return fills
    .filter((fill) => fill.is_filled || fill.partial_fill)
    .map((fill, index) => ({
      id: fill.order_id ?? `FILL-${index + 1}`,
      symbol: fill.symbol,
      side: fill.side,
      size: Number(fill.qty ?? 0),
      fill: Number(fill.fill_price ?? 0),
      status: fill.status,
      source: fill.source ?? "broker-ledger",
      notional: Number(fill.notional ?? 0),
      filledAt: fill.filled_at ?? ""
    }));
}

function ExecutionGrid({ ledger, error }: { ledger: ExecutionLedgerResponse | null; error: string | null }) {
  const liveRows = rowsFromLedger(ledger?.fills ?? []);
  const columns: ColDef[] = [
    { field: "id", width: 92 },
    { field: "symbol", width: 100 },
    { field: "side", width: 80 },
    { field: "size", width: 86, type: "rightAligned" },
    { field: "fill", width: 105, type: "rightAligned", valueFormatter: ({ value }) => Number(value).toFixed(3) },
    { field: "notional", width: 110, type: "rightAligned", valueFormatter: ({ value }) => `$${Number(value).toFixed(2)}` },
    { field: "status", width: 112 },
    { field: "source", width: 120 },
    { field: "filledAt", width: 190 }
  ];

  return (
    <section className="panel lg:col-span-2">
      <div className="panel-title">
        <span>AG Grid Execution Ledger</span>
        <span className={liveRows.length ? "text-terminal-emerald" : "text-terminal-crimson"}>
          {liveRows.length ? `${liveRows.length} broker-confirmed fills` : "no synthetic trades displayed"}
        </span>
      </div>
      {liveRows.length ? (
        <div className="ag-theme-quartz-dark h-[320px] p-2">
          <AgGridReact rowData={liveRows} columnDefs={columns} rowHeight={27} headerHeight={28} suppressMovableColumns />
        </div>
      ) : (
        <div className="flex h-[320px] items-center justify-center px-4 text-center text-[11px] uppercase tracking-[0.14em] text-terminal-muted">
          {error ? `Execution ledger unavailable: ${error}` : "No broker-confirmed fills in /api/execution/ledger. This panel will not fabricate orders."}
        </div>
      )}
    </section>
  );
}

export function MicrostructureExecution() {
  const ledger = useApiResource<ExecutionLedgerResponse>("/api/execution/ledger", 10_000);

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <DepthChart />
      <LatencyScatter liveFills={ledger.data?.fills ?? []} />
      <ExecutionGrid ledger={ledger.data} error={ledger.error} />
    </div>
  );
}
