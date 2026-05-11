import { gaussian, mulberry32 } from "../lib/quant";

const random = mulberry32(1107);

export type HmmState = "Mean Revert" | "Breakout" | "High-Vol Trend" | "Illiquid";

export type HmmHeatCell = {
  time: string;
  state: HmmState;
  probability: number;
};

export type ShapContribution = {
  feature: string;
  value: number;
  shap: number;
  cumulative: number;
};

export type ConformalPoint = {
  time: string;
  price: number;
  lower: number;
  band: number;
  upper: number;
  prediction: number;
};

export const hmmStates: HmmState[] = ["Mean Revert", "Breakout", "High-Vol Trend", "Illiquid"];

export const hmmHeatmap: HmmHeatCell[] = Array.from({ length: 42 }, (_, t) => {
  const trend = 1 / (1 + Math.exp(-(t - 20) / 4));
  const meanRevert = 0.68 * (1 - trend) + 0.10 * trend + gaussian(random) * 0.015;
  const highVol = 0.10 * (1 - trend) + 0.48 * trend + gaussian(random) * 0.015;
  const breakout = 0.18 + Math.sin(t / 5) * 0.08 + gaussian(random) * 0.012;
  const illiquid = Math.max(0.03, 1 - meanRevert - highVol - breakout);
  const raw = [meanRevert, breakout, highVol, illiquid].map((value) => Math.max(0.02, value));
  const total = raw.reduce((sum, value) => sum + value, 0);
  return hmmStates.map((state, index) => ({
    time: `T-${String(42 - t).padStart(2, "0")}`,
    state,
    probability: raw[index] / total
  }));
}).flat();

const shapBase = [
  ["base_rate", 0.0, 0.0012],
  ["order_flow_imbalance", 0.63, 0.0028],
  ["vpin", 0.41, -0.0019],
  ["funding_rate_8h", 0.00042, 0.0011],
  ["rsi_14", 66.4, -0.0008],
  ["gamma_wall_distance", -0.018, -0.0013],
  ["hmm_breakout_prob", 0.58, 0.0016],
  ["spread_percentile", 0.72, -0.0010]
] as const;

let cumulative = 0;
export const shapWaterfall: ShapContribution[] = shapBase.map(([feature, value, shap]) => {
  cumulative += shap;
  return { feature, value, shap, cumulative };
});

export const conformalSeries: ConformalPoint[] = Array.from({ length: 96 }, (_, i) => {
  const price = 522.4 + Math.sin(i / 8) * 3.8 + i * 0.055 + gaussian(random) * 0.38;
  const prediction = price + Math.sin(i / 9) * 0.84;
  const interval = 1.7 + Math.abs(Math.sin(i / 13)) * 1.35 + (i > 68 ? 0.9 : 0);
  const lower = prediction - interval;
  const upper = prediction + interval;
  return {
    time: `${String(Math.floor(i / 4)).padStart(2, "0")}:${String((i % 4) * 15).padStart(2, "0")}`,
    price,
    prediction,
    lower,
    upper,
    band: upper - lower
  };
});
