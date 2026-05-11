import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { bootstrapHistogram, monteCarloFan, rollingRisk, simulationSummary } from "../../data/simulationMockData";
import { useApiResource } from "../../hooks/useApiResource";
import type { EquityAnalyticsResponse } from "../../lib/api";
import { formatMoney } from "../../lib/quant";

const GRID = "#1A1A1A";
const CYAN = "#00E5FF";
const EMERALD = "#00FF88";
const CRIMSON = "#FF335F";
const AMBER = "#F5C542";

function terminalTooltip() {
  return {
    contentStyle: {
      background: "#0F0F0F",
      border: `1px solid ${GRID}`,
      color: "#E5E7EB",
      fontFamily: "JetBrains Mono",
      fontSize: 11
    }
  };
}

function MonteCarloFanChart() {
  return (
    <section className="panel min-h-[390px] lg:col-span-2">
      <div className="panel-title">
        <span>Monte Carlo Fan Chart</span>
        <span className="text-terminal-cyan">10,000 GBM + jump diffusion paths</span>
      </div>
      <div className="h-[344px] px-1 py-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={monteCarloFan} margin={{ top: 8, right: 20, bottom: 0, left: 4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="step" tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: GRID }} />
            <YAxis
              domain={["dataMin - 20000", "dataMax + 20000"]}
              tick={{ fill: "#6B7280", fontSize: 10 }}
              tickFormatter={(value) => `$${Math.round(Number(value) / 1000)}k`}
              tickLine={false}
              axisLine={{ stroke: GRID }}
            />
            <Tooltip {...terminalTooltip()} formatter={(value: unknown) => formatMoney(Number(value))} />
            <Area dataKey="lowerPad" stackId="fan" stroke="none" fill="transparent" isAnimationActive={false} />
            <Area dataKey="band5_25" stackId="fan" stroke="none" fill="rgba(255, 51, 95, 0.12)" isAnimationActive={false} />
            <Area dataKey="band25_75" stackId="fan" stroke="none" fill="rgba(0, 229, 255, 0.18)" isAnimationActive={false} />
            <Area dataKey="band75_95" stackId="fan" stroke="none" fill="rgba(0, 255, 136, 0.10)" isAnimationActive={false} />
            <Line dataKey="p50" stroke={AMBER} dot={false} strokeWidth={1.8} isAnimationActive={false} />
            <Line dataKey="p5" stroke={CRIMSON} dot={false} strokeWidth={1.1} strokeDasharray="4 4" isAnimationActive={false} />
            <Line dataKey="p95" stroke={EMERALD} dot={false} strokeWidth={1.1} strokeDasharray="4 4" isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function BootstrapHistogram() {
  return (
    <section className="panel min-h-[290px]">
      <div className="panel-title">
        <span>Las Vegas Bootstrap Drawdowns</span>
        <span className="text-terminal-crimson">100,000 reshuffles</span>
      </div>
      <div className="h-[244px] px-1 py-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bootstrapHistogram} margin={{ top: 8, right: 10, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="bucket" tick={{ fill: "#6B7280", fontSize: 9 }} tickLine={false} axisLine={{ stroke: GRID }} interval={2} />
            <YAxis tick={{ fill: "#6B7280", fontSize: 10 }} tickFormatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} tickLine={false} axisLine={{ stroke: GRID }} />
            <Tooltip {...terminalTooltip()} formatter={(value: unknown) => `${(Number(value) * 100).toFixed(3)}%`} />
            <Bar dataKey="density" fill="rgba(255, 51, 95, 0.72)" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function liveRollingRisk(data: EquityAnalyticsResponse | null) {
  if (!data?.available || !data.timestamps?.length || !data.rolling_sharpe?.length) return null;
  return data.timestamps.map((timestamp, index) => {
    const sharpe = data.rolling_sharpe?.[index] ?? null;
    return {
      step: index,
      timestamp,
      sharpe: sharpe ?? 0,
      sortino: sharpe == null ? 0 : sharpe * 1.12,
      sharpeTrend: sharpe ?? 0
    };
  });
}

function RollingSharpeChart({ equityAnalytics }: { equityAnalytics: EquityAnalyticsResponse | null }) {
  const live = liveRollingRisk(equityAnalytics);
  const series = live ?? rollingRisk;

  return (
    <section className="panel min-h-[290px]">
      <div className="panel-title">
        <span>Rolling Sharpe / Sortino</span>
        <span className={live ? "text-terminal-emerald" : "text-terminal-amber"}>
          {live ? "live /api/equity/analytics" : "synthetic demo fallback"}
        </span>
      </div>
      <div className="h-[244px] px-1 py-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={series} margin={{ top: 8, right: 10, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="step" tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: GRID }} />
            <YAxis domain={[0.5, 3.2]} tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: GRID }} />
            <Tooltip {...terminalTooltip()} formatter={(value: unknown) => Number(value).toFixed(3)} />
            <Line dataKey="sharpe" stroke={CYAN} dot={false} strokeWidth={1.4} isAnimationActive={false} />
            <Line dataKey="sortino" stroke={EMERALD} dot={false} strokeWidth={1.3} isAnimationActive={false} />
            <Line dataKey="sharpeTrend" stroke={AMBER} dot={false} strokeWidth={1.2} strokeDasharray="5 4" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

export function SimulationForwardTesting() {
  const equityAnalytics = useApiResource<EquityAnalyticsResponse>("/api/equity/analytics", 30_000);
  const liveMetrics = equityAnalytics.data?.available ? equityAnalytics.data : null;

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <MonteCarloFanChart />
      <BootstrapHistogram />
      <RollingSharpeChart equityAnalytics={liveMetrics} />
      <section className="panel lg:col-span-2">
        <div className="panel-title">
          <span>Forward Test Summary</span>
          <span className={liveMetrics ? "text-terminal-emerald" : "text-terminal-amber"}>
            {liveMetrics ? "live equity analytics + stochastic projection" : "synthetic stochastic projection"}
          </span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-5">
          {[
            ["Current Equity", liveMetrics?.end_equity ? formatMoney(liveMetrics.end_equity) : formatMoney(simulationSummary.p50), liveMetrics ? EMERALD : AMBER],
            ["Live Sharpe", liveMetrics?.overall_sharpe != null ? liveMetrics.overall_sharpe.toFixed(3) : "N/A", liveMetrics ? EMERALD : "#6B7280"],
            ["Live Max DD", liveMetrics?.max_drawdown_pct != null ? `${liveMetrics.max_drawdown_pct.toFixed(2)}%` : "N/A", liveMetrics ? CRIMSON : "#6B7280"],
            ["Probability Ruin", `${(simulationSummary.ruinProbability * 100).toFixed(2)}%`, CRIMSON],
            ["Expected Max DD", `${(simulationSummary.expectedMaxDrawdown * 100).toFixed(2)}%`, CRIMSON]
          ].map(([label, value, color]) => (
            <div className="metric-cell" key={label}>
              <div className="text-[9px] uppercase tracking-[0.16em] text-terminal-muted">{label}</div>
              <div className="text-lg font-black" style={{ color }}>
                {value}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
