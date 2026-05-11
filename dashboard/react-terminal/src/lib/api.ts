const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export type ApiState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
  source: "live" | "unavailable";
};

export async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export type ExecutionLedgerFill = {
  order_id?: string;
  symbol: string;
  side: "BUY" | "SELL" | string;
  qty: number | null;
  fill_price: number | null;
  filled_at?: string;
  status: string;
  is_filled: boolean;
  partial_fill: boolean;
  exec_score?: number | null;
  exec_tier?: string | null;
  source?: string | null;
  market?: string | null;
  notional?: number | null;
};

export type ExecutionLedgerResponse = {
  total: number;
  filled: number;
  records: ExecutionLedgerFill[];
  fills: ExecutionLedgerFill[];
  fill_rate: number;
  latest_fill_at_utc?: string | null;
};

export type EquityAnalyticsResponse = {
  available: boolean;
  notice?: string;
  timestamps?: string[];
  equity?: number[];
  rolling_sharpe?: Array<number | null>;
  drawdown_pct?: number[];
  overall_sharpe?: number | null;
  max_drawdown_pct?: number | null;
  end_equity?: number | null;
};

export type HmmModelResponse = {
  available: boolean;
  symbol?: string;
  current_regime?: string;
  probabilities?: Record<string, number>;
  error?: string;
};

export type XgbModelResponse = {
  available: boolean;
  symbol?: string;
  signals?: Array<{
    horizon_bars: number;
    direction: string;
    probability: number;
    predicted_return: number;
    top_features?: Record<string, number> | Array<[string, number]>;
  }>;
  error?: string;
};

export type HealthResponse = {
  status?: string;
  broker?: string;
  ts?: string;
  strategies_enabled?: string[];
};

export type EquityResponse = {
  timestamps: string[];
  equity: number[];
  drawdown: number[];
  metrics?: {
    current_equity?: number | null;
    peak_equity?: number | null;
    total_return_pct?: number | null;
    sharpe?: number | null;
    sortino?: number | null;
    calmar?: number | null;
    max_drawdown_pct?: number | null;
    volatility_pct?: number | null;
    annualised_return_pct?: number | null;
  };
};

export type PositionRow = {
  symbol?: string;
  broker_symbol?: string;
  asset_class?: string;
  market?: string;
  side?: string;
  quantity?: number | null;
  qty?: number | null;
  entry_price?: number | null;
  price?: number | null;
  mark_price?: number | null;
  current_price?: number | null;
  market_value?: number | null;
  unrealized_pnl?: number | null;
  pnl?: number | null;
  source?: string | null;
};

export type PositionsResponse = {
  positions: PositionRow[];
  count?: number;
  source?: string;
  summary?: Record<string, unknown>;
};

export type RiskEngineResponse = {
  generated_at_utc?: string;
  portfolio_delta?: number | null;
  portfolio_theta?: number | null;
  portfolio_vega?: number | null;
  portfolio_gamma?: number | null;
  risk_score?: number | null;
  var_pct_equity?: number | null;
  cvar_pct_equity?: number | null;
  stress_pct_equity?: number | null;
  gross_exposure_pct_equity?: number | null;
  kill_switch_active?: boolean;
  breaches?: string[];
  hard_kill_reasons?: string[];
  macro_regime?: string | null;
  vix?: number | null;
  open_positions?: number | null;
  buying_power?: number | null;
  total_equity?: number | null;
  daily_pnl_pct?: number | null;
  risk_snapshot_age_hours?: number | null;
  risk_snapshot_fresh?: boolean;
  trading_status?: string;
};

export type StrategyStatus = {
  enabled: boolean;
  params: Record<string, unknown>;
};

export type StrategiesStatusResponse = {
  strategies: Record<string, StrategyStatus>;
  kill_switch: boolean;
};
