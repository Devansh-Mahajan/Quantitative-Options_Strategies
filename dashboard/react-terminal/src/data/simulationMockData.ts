import { gaussian, movingAverage, mulberry32, quantile } from "../lib/quant";

const random = mulberry32(4242);

export type FanPoint = {
  step: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  lowerPad: number;
  band5_25: number;
  band25_75: number;
  band75_95: number;
};

export type DrawdownBucket = {
  bucket: string;
  density: number;
};

export type RollingRiskPoint = {
  step: number;
  sharpe: number;
  sortino: number;
  sharpeTrend: number;
};

function simulatePath(steps: number) {
  const values = [1_000_000];
  const mu = 0.14 / 252;
  const sigma = 0.19 / Math.sqrt(252);
  const jumpLambda = 0.028;
  const jumpMu = -0.014;
  const jumpSigma = 0.025;

  for (let i = 1; i < steps; i += 1) {
    const z = gaussian(random);
    const jump = random() < jumpLambda ? jumpMu + jumpSigma * gaussian(random) : 0;
    values.push(values[i - 1] * Math.exp(mu - 0.5 * sigma * sigma + sigma * z + jump));
  }
  return values;
}

const steps = 120;
const paths = Array.from({ length: 10_000 }, () => simulatePath(steps));

export const monteCarloFan: FanPoint[] = Array.from({ length: steps }, (_, step) => {
  const slice = paths.map((path) => path[step]);
  const p5 = quantile(slice, 0.05);
  const p25 = quantile(slice, 0.25);
  const p50 = quantile(slice, 0.5);
  const p75 = quantile(slice, 0.75);
  const p95 = quantile(slice, 0.95);
  return {
    step,
    p5,
    p25,
    p50,
    p75,
    p95,
    lowerPad: p5,
    band5_25: p25 - p5,
    band25_75: p75 - p25,
    band75_95: p95 - p75
  };
});

const drawdowns = Array.from({ length: 100_000 }, () => {
  const tradeCount = 180;
  let equity = 1;
  let peak = 1;
  let maxDrawdown = 0;
  for (let i = 0; i < tradeCount; i += 1) {
    const pnl = 0.0012 + gaussian(random) * 0.008 - (random() < 0.055 ? Math.abs(gaussian(random)) * 0.018 : 0);
    equity *= 1 + pnl;
    peak = Math.max(peak, equity);
    maxDrawdown = Math.min(maxDrawdown, equity / peak - 1);
  }
  return maxDrawdown;
});

export const bootstrapHistogram: DrawdownBucket[] = Array.from({ length: 22 }, (_, i) => {
  const left = -0.22 + i * 0.01;
  const right = left + 0.01;
  const count = drawdowns.filter((value) => value >= left && value < right).length;
  return {
    bucket: `${(left * 100).toFixed(0)}%`,
    density: count / drawdowns.length
  };
});

const sharpe = Array.from({ length: 170 }, (_, i) => 1.55 + Math.sin(i / 18) * 0.42 + gaussian(random) * 0.11);
const trend = movingAverage(sharpe, 20);

export const rollingRisk: RollingRiskPoint[] = sharpe.map((value, step) => ({
  step,
  sharpe: value,
  sortino: value + 0.42 + Math.max(0, Math.sin(step / 21)) * 0.25,
  sharpeTrend: trend[step]
}));

export const simulationSummary = {
  p5: monteCarloFan.at(-1)!.p5,
  p50: monteCarloFan.at(-1)!.p50,
  p95: monteCarloFan.at(-1)!.p95,
  ruinProbability: drawdowns.filter((value) => value < -0.18).length / drawdowns.length,
  expectedMaxDrawdown: drawdowns.reduce((sum, value) => sum + value, 0) / drawdowns.length
};
