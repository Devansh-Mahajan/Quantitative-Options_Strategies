export type VolSurfacePoint = {
  strike: number;
  expiry: number;
  iv: number;
  anomaly: boolean;
};

export type GreekExposure = {
  greek: "Delta" | "Gamma" | "Theta" | "Vega" | "Vanna";
  exposure: number;
  limit: number;
};

export type ScenarioCell = {
  spotShock: number;
  volShock: number;
  pnl: number;
};

export const volSurface: VolSurfacePoint[] = Array.from({ length: 15 }, (_, expiryIndex) => {
  const expiry = 7 + expiryIndex * 14;
  return Array.from({ length: 21 }, (_, strikeIndex) => {
    const moneyness = (strikeIndex - 10) / 10;
    const smile = 0.18 + 0.085 * moneyness * moneyness;
    const term = 0.04 * Math.log1p(expiry) / Math.log(220);
    const skew = -0.035 * moneyness;
    const localBump = Math.exp(-((moneyness + 0.35) ** 2) / 0.045) * Math.exp(-((expiry - 49) ** 2) / 1600) * 0.035;
    return {
      strike: 4600 + strikeIndex * 60,
      expiry,
      iv: smile + term + skew + localBump,
      anomaly: localBump > 0.018
    };
  });
}).flat();

export const greekExposure: GreekExposure[] = [
  { greek: "Delta", exposure: 0.38, limit: 0.65 },
  { greek: "Gamma", exposure: 0.72, limit: 0.80 },
  { greek: "Theta", exposure: -0.44, limit: -0.70 },
  { greek: "Vega", exposure: 0.58, limit: 0.76 },
  { greek: "Vanna", exposure: -0.31, limit: -0.55 }
];

export const scenarioMatrix: ScenarioCell[] = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10].flatMap((spotShock) =>
  [-15, -10, -5, 0, 5, 10, 15].map((volShock) => {
    const delta = 18200 * (spotShock / 100);
    const gamma = 0.5 * 92000 * (spotShock / 100) ** 2;
    const vega = 7400 * (volShock / 100);
    const skewPenalty = spotShock < -4 && volShock > 5 ? -11500 : 0;
    return {
      spotShock,
      volShock,
      pnl: delta + gamma + vega + skewPenalty
    };
  })
);
