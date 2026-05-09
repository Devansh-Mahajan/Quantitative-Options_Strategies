"""
VPIN + Aggressive Order Flow Imbalance.

VPIN (Volume-synchronized Probability of Informed trading):
  Easley, López de Prado, O'Hara (2012) — "Flow Toxicity and Liquidity in a
  High-Frequency World". The VPIN metric measures order-flow toxicity:
  how much of the trading volume is from informed traders vs noise traders.

  High VPIN → informed traders dominate → anticipate large directional moves
  Low VPIN  → mostly noise → mean-reversion likely

VPIN Formula:
  1. Define volume buckets of size V (V = avg_daily_vol / n_buckets)
  2. In each bucket τ, classify volume as buy (V^B) or sell (V^S) via:
       V^B_τ = Σ V_i · Φ((ΔP_i) / σ_ΔP)    [Φ = standard normal CDF]
       V^S_τ = V - V^B_τ
  3. VPIN = (1/n) · Σ |V^B_τ - V^S_τ| / V    [rolling over last n buckets]

Aggressive Flow Imbalance (Cartea et al. 2024, equations 11-14):
  I(V^b, V^a) = log(1 + V^b) - log(1 + V^a)    [multi-level LOB imbalance]

  Trade direction prediction via imbalance at multiple depth levels:
  - Level 1-5: most informative (market maker inventory)
  - Level 5-10: medium-term pressure
  - Level 10-20: institutional iceberg order proxy

  Net aggressive flow over multiple horizons [1, 5, 20, 60 bars]:
  OFI(T) = Σ_{t-T}^t (taker_buy_vol - taker_sell_vol)
"""

from __future__ import annotations
import logging
from collections import defaultdict, deque

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.vpin_flow")

# VPIN parameters
N_BUCKETS       = 50     # volume buckets per "day" (we'll use a rolling window)
VPIN_WINDOW     = 25     # number of buckets in rolling VPIN
VPIN_HIGH       = 0.65   # VPIN percentile above this → high toxicity → trend signal
VPIN_LOW        = 0.30   # VPIN percentile below this → low toxicity → reversion
VOL_BUCKET_FRAC = 0.002  # bucket size = 0.2% of lookback volume total

# Flow imbalance parameters
FLOW_HORIZONS   = [1, 5, 20, 60]     # bars for net flow aggregation
IMBALANCE_ENTRY = 0.40               # log-imbalance threshold for entry
SYMBOLS_FOCUS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]


class _VPINCalculator:
    """Online VPIN estimator using 1h candle volume classification."""

    def __init__(self, n_buckets: int, window: int, bucket_frac: float) -> None:
        self._n_buckets  = n_buckets
        self._window     = window
        self._bucket_frac = bucket_frac
        self._bucket_buffer: deque[float] = deque(maxlen=window)
        self._accumulated_vol: float = 0.0
        self._accumulated_buy: float = 0.0
        self._price_changes: deque[float] = deque(maxlen=100)
        self._bucket_size: float | None = None
        self._total_vol: float = 0.0

    def update(self, close: float, prev_close: float, volume: float,
               taker_buy_vol: float | None = None) -> float | None:
        """
        Update VPIN with a new bar. Returns current VPIN estimate or None if insufficient data.
        """
        delta_p = close - prev_close
        self._price_changes.append(delta_p)
        self._total_vol += volume

        # Determine bucket size from recent volume
        if self._bucket_size is None or len(self._price_changes) % 50 == 0:
            self._bucket_size = max(self._total_vol * self._bucket_frac, 1.0)

        # Classify volume as buy / sell
        if taker_buy_vol is not None:
            # Use actual taker flow if available (more accurate)
            buy_vol  = float(taker_buy_vol)
            sell_vol = volume - buy_vol
        else:
            # Bulk volume classification via normal CDF (Easley et al.)
            if len(self._price_changes) > 5:
                sigma_dp = float(np.std(list(self._price_changes)) + 1e-10)
                from scipy.special import ndtr  # standard normal CDF
                phi = float(ndtr(delta_p / sigma_dp))
            else:
                phi = 0.5
            buy_vol  = volume * phi
            sell_vol = volume * (1.0 - phi)

        self._accumulated_vol += volume
        self._accumulated_buy += buy_vol

        signals_out = []
        while self._accumulated_vol >= self._bucket_size:
            # Close this bucket
            imbalance = abs(self._accumulated_buy - (self._accumulated_vol - self._accumulated_buy))
            self._bucket_buffer.append(imbalance / (self._bucket_size + 1e-10))
            # Carry over excess
            excess_frac = (self._accumulated_vol - self._bucket_size) / volume
            self._accumulated_vol  = volume * excess_frac
            self._accumulated_buy  = buy_vol  * excess_frac

        if len(self._bucket_buffer) < self._window // 2:
            return None

        return float(np.mean(list(self._bucket_buffer)))


class VPINFlowStrategy(BaseStrategy):
    """
    VPIN toxicity + multi-horizon aggressive flow imbalance.

    High VPIN → informed flow dominant → follow momentum (trend signal).
    Low VPIN  → noise-driven market → fade moves (mean-reversion signal).
    Multi-horizon OFI (order flow imbalance) confirms direction.
    """

    name = "vpin_flow"
    market = "futures"

    def __init__(self, symbols: list[str] = SYMBOLS_FOCUS) -> None:
        self._symbols = symbols
        self._vpins: dict[str, _VPINCalculator] = {
            sym: _VPINCalculator(N_BUCKETS, VPIN_WINDOW, VOL_BUCKET_FRAC)
            for sym in symbols
        }
        self._prev_closes:  dict[str, float] = {}
        self._vpin_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._flow_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=max(FLOW_HORIZONS) + 5))

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals: list[Signal] = []

        regime_scale = {
            "volatile": 1.0,
            "bull":     0.9,
            "bear":     0.9,
            "ranging":  0.7,
        }.get(regime, 0.7)

        for sym in self._symbols:
            candle = store.candles.get((sym, "1h"))
            if candle is None:
                continue

            close  = candle.close
            volume = candle.volume
            prev   = self._prev_closes.get(sym, close)
            self._prev_closes[sym] = close

            # Get taker buy volume if available from history
            hist_df = store.get_history_df(sym, "1h")
            taker_buy = None
            if hist_df is not None and "taker_buy_base" in hist_df.columns and len(hist_df):
                taker_buy = float(hist_df["taker_buy_base"].iloc[-1])

            # ── Update VPIN ───────────────────────────────────────────────
            try:
                vpin = self._vpins[sym].update(close, prev, volume, taker_buy)
            except ImportError:
                # scipy not available — skip VPIN, use flow only
                vpin = None

            if vpin is not None:
                self._vpin_history[sym].append(vpin)

            # ── Order Flow Imbalance (Cartea et al. §11-14) ───────────────
            # Net aggressive flow = taker_buy - taker_sell (per bar)
            if taker_buy is not None:
                net_flow = taker_buy - (volume - taker_buy)
            else:
                # Proxy: volume × sign of return
                net_flow = volume * np.sign(close - prev)
            self._flow_history[sym].append(net_flow)

            # Multi-horizon flow signals
            flow_arr = np.array(list(self._flow_history[sym]), dtype=float)
            if len(flow_arr) < max(FLOW_HORIZONS):
                continue

            flow_scores = {}
            for horizon in FLOW_HORIZONS:
                horizon_flow = flow_arr[-horizon:].sum()
                horizon_vol  = np.abs(flow_arr[-horizon:]).sum() + 1e-10
                flow_scores[horizon] = horizon_flow / horizon_vol  # range (-1, +1)

            # Aggregate: weight shorter horizons more
            weights = [0.4, 0.3, 0.2, 0.1]
            agg_flow = sum(w * flow_scores[h] for w, h in zip(weights, FLOW_HORIZONS))

            # ── LOB Imbalance proxy (Cartea eq. 11) ──────────────────────
            book = store.book.get(sym)
            if book:
                # We only have best bid/ask, not depth — use spread as proxy
                mid  = (book.bid + book.ask) / 2
                spread_bps = (book.ask - book.bid) / (mid + 1e-10) * 10000
                # Tight spread = liquid = imbalance signal is reliable
                spread_factor = max(1.0 - spread_bps / 20.0, 0.1)
            else:
                spread_factor = 0.5

            # ── Signal generation ─────────────────────────────────────────
            if abs(agg_flow) < IMBALANCE_ENTRY:
                continue

            # Use VPIN to modulate confidence and direction
            vpin_hist = list(self._vpin_history[sym])
            if len(vpin_hist) > 20:
                # VPIN percentile rank
                vpin_pct = float(np.mean(np.array(vpin_hist[-20:]) <= (vpin or 0.5)))
            else:
                vpin_pct = 0.5

            if vpin_pct > VPIN_HIGH:
                # High toxicity: informed flow → follow the flow direction
                side = "BUY" if agg_flow > 0 else "SELL"
                mode = "toxicity_momentum"
                vpin_boost = 1.2
            elif vpin_pct < VPIN_LOW:
                # Low toxicity: noise-driven → fade the flow
                side = "SELL" if agg_flow > 0 else "BUY"
                mode = "noise_reversion"
                vpin_boost = 0.8
            else:
                # Neutral VPIN: follow flow with moderate confidence
                side = "BUY" if agg_flow > 0 else "SELL"
                mode = "neutral_flow"
                vpin_boost = 1.0

            confidence = min(abs(agg_flow) * vpin_boost * spread_factor * regime_scale, 0.88)
            if confidence < 0.25:
                continue

            signals.append(Signal(
                symbol=sym, market="futures", side=side,
                quantity=0.0, price=close,
                confidence=confidence,
                strategy=self.name,
                meta={
                    "mode": mode,
                    "agg_flow": round(agg_flow, 3),
                    "vpin": round(vpin or 0, 3),
                    "vpin_pct": round(vpin_pct, 2),
                },
            ))
            log.debug("VPIN %s %s  flow=%.3f  vpin=%.2f(%.0f%%)  spread=%.1fbps  mode=%s",
                      side, sym, agg_flow, vpin or 0, vpin_pct * 100, spread_bps if book else -1, mode)

        return signals
