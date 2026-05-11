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
import { conformalSeries, hmmHeatmap, hmmStates, shapWaterfall } from "../../data/alphaMockData";
import { useApiResource } from "../../hooks/useApiResource";
import type { HmmModelResponse, XgbModelResponse } from "../../lib/api";

const GRID = "#1A1A1A";
const CYAN = "#00E5FF";
const EMERALD = "#00FF88";
const AMBER = "#F5C542";

function probabilityColor(value: number) {
  const alpha = Math.max(0.08, Math.min(0.9, value));
  return `rgba(0, 229, 255, ${alpha})`;
}

function formatPct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function HmmRegimeHeatmap() {
  const times = [...new Set(hmmHeatmap.map((cell) => cell.time))];

  return (
    <section className="panel min-h-[276px]">
      <div className="panel-title">
        <span>HMM Regime Heatmap</span>
        <span className="text-terminal-cyan">Gaussian emissions · 4 latent states</span>
      </div>
      <div className="grid grid-cols-[118px_1fr] gap-1 px-2 py-2">
        <div />
        <div className="grid" style={{ gridTemplateColumns: `repeat(${times.length}, minmax(10px, 1fr))` }}>
          {times.map((time, index) => (
            <div key={time} className="truncate text-center text-[8px] text-terminal-muted">
              {index % 7 === 0 ? time : ""}
            </div>
          ))}
        </div>
        {hmmStates.map((state) => (
          <div key={state} className="contents">
            <div className="flex h-7 items-center border-r border-terminal-grid pr-2 text-[10px] text-terminal-muted">
              {state}
            </div>
            <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${times.length}, minmax(10px, 1fr))` }}>
              {times.map((time) => {
                const cell = hmmHeatmap.find((item) => item.time === time && item.state === state)!;
                return (
                  <div
                    key={`${time}-${state}`}
                    title={`${time} ${state}: ${formatPct(cell.probability)}`}
                    className="h-7 border border-black/70"
                    style={{ background: probabilityColor(cell.probability) }}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function liveShapFromXgb(xgb: XgbModelResponse | null) {
  const features = xgb?.signals?.[0]?.top_features;
  if (!xgb?.available || !features) return null;
  const entries = Array.isArray(features) ? features : Object.entries(features);
  let cumulative = 0;
  return entries.slice(0, 8).map(([feature, raw]) => {
    const shap = Number(raw) / 10_000;
    cumulative += shap;
    return { feature, value: Number(raw), shap, cumulative };
  });
}

function ShapWaterfallChart({ xgb }: { xgb: XgbModelResponse | null }) {
  const live = liveShapFromXgb(xgb);
  const rows = live ?? shapWaterfall;
  const min = Math.min(0, ...rows.map((row) => row.cumulative));
  const max = Math.max(...rows.map((row) => row.cumulative));
  const span = max - min || 1;

  return (
    <section className="panel min-h-[276px]">
      <div className="panel-title">
        <span>XGBoost SHAP Waterfall</span>
        <span className={live ? "text-terminal-emerald" : "text-terminal-amber"}>
          {live ? "live /api/model_zoo/xgb" : "synthetic feature demo"}
        </span>
      </div>
      <div className="space-y-1 px-2 py-2">
        {rows.map((row) => {
          const positive = row.shap >= 0;
          const width = `${Math.max(5, (Math.abs(row.shap) / span) * 86)}%`;
          return (
            <div key={row.feature} className="grid grid-cols-[156px_74px_1fr_76px] items-center gap-2 text-[10px]">
              <span className="truncate text-terminal-ink">{row.feature}</span>
              <span className="text-right text-terminal-muted">{row.value.toFixed(row.value < 1 ? 5 : 2)}</span>
              <div className="relative h-4 border border-terminal-grid bg-black">
                <div
                  className={positive ? "h-full bg-terminal-emerald" : "ml-auto h-full bg-terminal-crimson"}
                  style={{ width }}
                />
              </div>
              <span className={positive ? "text-right text-terminal-emerald" : "text-right text-terminal-crimson"}>
                {positive ? "+" : ""}
                {(row.shap * 10_000).toFixed(1)} bps
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ConformalPredictionBands() {
  return (
    <section className="panel min-h-[360px] lg:col-span-2">
      <div className="panel-title">
        <span>Conformal Prediction Bands</span>
        <span className="text-terminal-amber">95% adaptive uncertainty · online residual calibration</span>
      </div>
      <div className="h-[314px] px-1 py-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={conformalSeries} margin={{ top: 8, right: 18, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="time" tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: GRID }} />
            <YAxis domain={["dataMin - 2", "dataMax + 2"]} tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: GRID }} />
            <Tooltip
              contentStyle={{ background: "#0F0F0F", border: `1px solid ${GRID}`, color: "#E5E7EB", fontFamily: "JetBrains Mono" }}
              formatter={(value: unknown, name: unknown) => [Number(value).toFixed(3), String(name)]}
            />
            <Area dataKey="lower" stackId="band" stroke="none" fill="transparent" isAnimationActive={false} />
            <Area dataKey="band" stackId="band" stroke="none" fill="rgba(0, 229, 255, 0.14)" isAnimationActive={false} />
            <Line dataKey="price" name="realized" stroke={EMERALD} dot={false} strokeWidth={1.7} isAnimationActive={false} />
            <Line dataKey="prediction" name="model" stroke={CYAN} dot={false} strokeWidth={1.4} strokeDasharray="4 3" isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

export function AlphaRegimeMatrix() {
  const hmm = useApiResource<HmmModelResponse>("/api/model_zoo/hmm/SPY", 60_000);
  const xgb = useApiResource<XgbModelResponse>("/api/model_zoo/xgb/SPY", 60_000);
  const liveHmm = hmm.data?.available ? hmm.data : null;
  const liveXgb = xgb.data?.available ? xgb.data : null;
  const firstSignal = liveXgb?.signals?.[0];

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <ConformalPredictionBands />
      <HmmRegimeHeatmap />
      <ShapWaterfallChart xgb={liveXgb} />
      <section className="panel min-h-[132px] lg:col-span-2">
        <div className="panel-title">
          <span>Model Diagnostics</span>
          <span className={liveHmm || liveXgb ? "text-terminal-emerald" : "text-terminal-amber"}>
            {liveHmm || liveXgb ? "live model-zoo telemetry" : "synthetic model telemetry fallback"}
          </span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-5">
          {[
            ["HMM state", liveHmm?.current_regime ?? "High-Vol Trend", liveHmm ? EMERALD : AMBER],
            ["XGB expected return", firstSignal ? `${(firstSignal.predicted_return * 10_000).toFixed(1)} bps` : "+44.2 bps", firstSignal ? EMERALD : AMBER],
            ["Conformal width", "63.8 bps", CYAN],
            ["XGB probability", firstSignal ? `${(firstSignal.probability * 100).toFixed(1)}%` : "N/A", firstSignal ? EMERALD : "#6B7280"],
            ["Integrity", liveHmm || liveXgb ? "LIVE" : "DEMO", liveHmm || liveXgb ? EMERALD : AMBER]
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
